from typing import Any

from .multi_step_planner import build_multi_step_plan

# 中文函数说明索引：
# - _multi_step_runtime_policy(...)：根据 Step Planner 和 bridge policy 生成 runtime 建议；核心状态包括 single_step_runtime、auto_runtime_experiment_candidate、explicit_runtime_only。
# - _planning_metadata(...)：汇总 task mode、置信度、clarification、execution policy 等 route_plan 顶层元信息。
# - _candidate_evidence_summary(...)：压缩每个候选 query_type 的证据，供前端 trace 和新对话排查路由原因。
# - _matching_schema_plan_proposal(...)：把 schema-aware proposal 与候选 query_type 对齐。
# - _proposal_backed_candidates(...)：按最终 route candidate 顺序生成 proposal-backed candidates，避免 proposal 原始顺序覆盖主路由。
# - _legacy_route_candidates(...)：没有 schema proposal 时生成兼容旧协议的 route candidates。
# - build_route_plan(...)：Routing Agent 的 route_plan 主构造函数，整合 evidence、candidate、task_analysis、multi_step_plan、runtime_policy。
# - build_keyword_route_plan(...)：纯关键词 fallback route_plan，用于 LLM/检索不可用时的保底。


ROUTE_PLAN_VERSION = 1
MAX_PROPOSAL_EXECUTION_CANDIDATES = 2
SUPPORTED_EXPERIMENTAL_RUNTIME_QUERY_TYPES = {"sql", "cypher"}


def _multi_step_runtime_policy(route_plan: dict[str, Any], multi_step_plan: dict[str, Any]) -> dict[str, Any]:
    steps = [step for step in multi_step_plan.get("steps", []) if step.get("step_type") == "query"]
    bridge_steps = multi_step_plan.get("bridge_steps", [])
    blockers: list[str] = []
    advisory_reasons: list[str] = []
    risk_level = "not_applicable"
    experimental_auto_candidate = False

    if multi_step_plan.get("status") != "planned_not_executed":
        return {
            "default_enable": False,
            "experimental_auto_candidate": False,
            "requires_explicit_enable": False,
            "recommended_mode": "single_step_runtime",
            "risk_level": risk_level,
            "reasons": ["multi_step_plan_not_required"],
        }

    if len(steps) == 0 or len(steps) > 2:
        blockers.append("step_count_outside_current_runtime_limit")
    if any(step.get("query_type") not in SUPPORTED_EXPERIMENTAL_RUNTIME_QUERY_TYPES for step in steps):
        blockers.append("unsupported_query_type_for_current_runtime")
    if any(not step.get("target_resources") for step in steps):
        blockers.append("missing_target_resource")

    bridge_statuses = [bridge.get("status", "") for bridge in bridge_steps]
    has_cross_source_bridge = any(status == "required_before_hard_binding" for status in bridge_statuses)
    has_unknown_bridge = any(status not in {"not_required", "required_before_hard_binding"} for status in bridge_statuses)
    if has_unknown_bridge:
        blockers.append("bridge_status_unknown")

    if route_plan.get("planning_status") == "low_confidence" or route_plan.get("uncertainty_source"):
        advisory_reasons.append("route_plan_low_confidence")

    if has_cross_source_bridge:
        risk_level = "medium"
        reasons = [
            *blockers,
            *advisory_reasons,
            "cross_source_bridge_requires_soft_context_or_resolver",
        ]
        recommended_mode = "explicit_runtime_only"
    elif blockers:
        risk_level = "high"
        reasons = [*blockers, *advisory_reasons]
        recommended_mode = "single_step_fallback"
    else:
        risk_level = "medium" if advisory_reasons else "low"
        experimental_auto_candidate = True
        reasons = [
            *advisory_reasons,
            "same_source_two_step_plan_with_hard_constraints",
        ]
        recommended_mode = "auto_runtime_experiment_candidate"

    return {
        "default_enable": False,
        "experimental_auto_candidate": experimental_auto_candidate,
        "requires_explicit_enable": not experimental_auto_candidate or has_cross_source_bridge,
        "recommended_mode": recommended_mode,
        "risk_level": risk_level,
        "reasons": reasons,
    }


def _planning_metadata(route_result: dict[str, Any], task_mode: str) -> dict[str, Any]:
    confidence = route_result.get("confidence", "low")
    uncertainty_source = route_result.get("uncertainty_source", "")
    task_analysis = route_result.get("task_analysis", {})
    keep_backup = bool(route_result.get("should_keep_backup_route", False))
    is_low_confidence = confidence == "low" or bool(uncertainty_source)
    is_multi_step_candidate = task_mode == "multi_step_candidate" or task_analysis.get("requires_system_multi_step", False)

    if is_multi_step_candidate:
        planning_status = "multi_step_candidate_single_step_fallback"
        policy_reason = "The question has multi-step signals; the default runtime keeps single-step compatibility unless multi-step runtime is explicitly enabled."
    elif task_analysis.get("query_strategy") == "single_query_with_nested_logic":
        planning_status = "single_query_nested_logic"
        policy_reason = "The question has nested logic that should be represented inside a single generated query."
    elif is_low_confidence:
        planning_status = "low_confidence"
        policy_reason = "Routing confidence is low or uncertain; execute the top candidate while preserving backup routes."
    else:
        planning_status = "ready"
        policy_reason = "Routing evidence is sufficient for single-step execution."

    return {
        "confidence": confidence,
        "planning_status": planning_status,
        "execution_policy": {
            "mode": "single_step_auto_execute",
            "preserve_backup_route": keep_backup or is_low_confidence,
            "allow_reroute_on_verification": True,
            "reason": policy_reason,
            "recommended_generation_strategy": task_analysis.get("recommended_generation_strategy", "direct_query"),
        },
        "clarification": {
            "recommended": is_low_confidence or is_multi_step_candidate,
            "required": False,
            "reason": (
                "Clarification would improve route certainty, but current demo mode continues automatically."
                if is_low_confidence or is_multi_step_candidate
                else ""
            ),
        },
    }


def _candidate_evidence_summary(candidate: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    query_type = candidate.get("query_type", "")
    scored = evidence.get("scored_modalities", {}).get(query_type, {})
    entity_hints = evidence.get("entity_hints", {})
    retrieval_hints = evidence.get("retrieval_hints", {}).get(query_type, {})
    signals = evidence.get("intent_signals", {})
    return {
        "reason_parts": scored.get("reason_parts", []),
        "surface_score": scored.get("surface_score"),
        "entity_score": scored.get("entity_score"),
        "structure_score": scored.get("structure_score"),
        "retrieval_prior_score": scored.get("retrieval_prior_score"),
        "matched_tokens": entity_hints.get(f"matched_{query_type}_tokens", []),
        "matched_entities": entity_hints.get(f"matched_{query_type}_entities", []),
        "retrieval_hint": retrieval_hints,
        "intent_flags": {
            key: value
            for key, value in signals.items()
            if key.startswith("has_") and value
        },
    }


def _matching_schema_plan_proposal(query_type: str, route_result: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    proposals = route_result.get("schema_plan_proposals") or evidence.get("schema_plan_proposals", [])
    for proposal in proposals:
        if proposal.get("query_type") == query_type:
            return proposal
    return {}


def _proposal_backed_candidates(route_result: dict[str, Any], evidence: dict[str, Any]) -> list[dict[str, Any]]:
    proposals = route_result.get("schema_plan_proposals") or evidence.get("schema_plan_proposals", [])
    candidates: list[dict[str, Any]] = []
    proposals_by_type: dict[str, dict[str, Any]] = {}
    for proposal in proposals:
        query_type = proposal.get("query_type", "")
        if query_type not in {"sql", "cypher", "mql", "vector"}:
            continue
        if query_type not in proposals_by_type:
            proposals_by_type[query_type] = proposal

    ordered_query_types: list[str] = []
    for candidate in route_result.get("candidates", []):
        query_type = candidate.get("query_type", "") if isinstance(candidate, dict) else str(candidate)
        if query_type in proposals_by_type and query_type not in ordered_query_types:
            ordered_query_types.append(query_type)
    selected = route_result.get("query_type", "")
    if selected in proposals_by_type and selected not in ordered_query_types:
        ordered_query_types.insert(0, selected)
    for proposal in proposals:
        query_type = proposal.get("query_type", "")
        if query_type in proposals_by_type and query_type not in ordered_query_types:
            ordered_query_types.append(query_type)

    for query_type in ordered_query_types:
        proposal = proposals_by_type[query_type]
        candidates.append(
            {
                "query_type": query_type,
                "score": None,
                "confidence": proposal.get("confidence", "low"),
                "reason": proposal.get("reason", ""),
                "plan_proposal": proposal,
                "candidate_source": "schema_plan_proposal",
            }
        )
        if len(candidates) >= MAX_PROPOSAL_EXECUTION_CANDIDATES:
            break
    return candidates


def _legacy_route_candidates(route_result: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for candidate in route_result.get("candidates", []):
        if not isinstance(candidate, dict):
            continue
        candidates.append({**candidate, "candidate_source": "route_candidate"})
    return candidates


def build_route_plan(route_result: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    """Build a structured route plan without changing routing decisions."""
    candidates = []
    candidate_sources = _proposal_backed_candidates(route_result, evidence)
    if len(candidate_sources) < MAX_PROPOSAL_EXECUTION_CANDIDATES:
        used_query_types = {candidate.get("query_type", "") for candidate in candidate_sources}
        for candidate in _legacy_route_candidates(route_result):
            if candidate.get("query_type", "") in used_query_types:
                continue
            candidate_sources.append(candidate)
            used_query_types.add(candidate.get("query_type", ""))
            if len(candidate_sources) >= MAX_PROPOSAL_EXECUTION_CANDIDATES:
                break

    for index, candidate in enumerate(candidate_sources, start=1):
        query_type = candidate.get("query_type", "")
        plan_proposal = candidate.get("plan_proposal") or _matching_schema_plan_proposal(query_type, route_result, evidence)
        candidates.append(
            {
                "candidate_id": f"R{index}",
                "rank": index,
                "query_type": query_type,
                "score": candidate.get("score"),
                "confidence": candidate.get("confidence", "low"),
                "reason": candidate.get("reason", ""),
                "evidence_summary": _candidate_evidence_summary(candidate, evidence),
                "plan_proposal": plan_proposal,
                "candidate_source": candidate.get("candidate_source", "route_candidate"),
                "status": "selected" if index == 1 else "backup",
            }
        )

    selected_candidate_id = candidates[0]["candidate_id"] if candidates else ""
    task_mode = route_result.get("task_mode", evidence.get("intent_signals", {}).get("task_mode_hint", "single_step"))
    task_analysis = evidence.get("task_analysis", evidence.get("intent_signals", {}).get("task_analysis", {}))
    route_result_with_analysis = {**route_result, "task_analysis": task_analysis}
    planning_metadata = _planning_metadata(route_result_with_analysis, task_mode)
    route_plan = {
        "plan_version": ROUTE_PLAN_VERSION,
        "task_mode": task_mode,
        "task_analysis": task_analysis,
        **planning_metadata,
        "proposal_execution_policy": {
            "enabled": bool(_proposal_backed_candidates(route_result, evidence)),
            "top_k": MAX_PROPOSAL_EXECUTION_CANDIDATES,
            "reason": "Execute top schema plan proposals first; fall back to legacy route candidates only when proposal slots are missing.",
        },
        "selected_candidate_id": selected_candidate_id,
        "selected_query_type": candidates[0]["query_type"] if candidates else route_result.get("query_type", ""),
        "should_keep_backup_route": route_result.get("should_keep_backup_route", False),
        "uncertainty_source": route_result.get("uncertainty_source", ""),
        "schema_plan_context": route_result.get("schema_plan_context") or evidence.get("schema_plan_context", {}),
        "schema_plan_proposals": route_result.get("schema_plan_proposals") or evidence.get("schema_plan_proposals", []),
        "candidates": candidates,
        "evidence": {
            "intent_signals": evidence.get("intent_signals", {}),
            "entity_hints": evidence.get("entity_hints", {}),
            "modality_scores": evidence.get("modality_scores", {}),
            "retrieval_hints": evidence.get("retrieval_hints", {}),
        },
    }
    route_plan["multi_step_plan"] = build_multi_step_plan(evidence.get("question", ""), route_plan)
    route_plan["multi_step_runtime_policy"] = _multi_step_runtime_policy(route_plan, route_plan["multi_step_plan"])
    return route_plan


def build_keyword_route_plan(route_result: dict[str, Any]) -> dict[str, Any]:
    candidates = []
    for index, candidate in enumerate(route_result.get("candidates", []), start=1):
        candidates.append(
            {
                "candidate_id": f"R{index}",
                "rank": index,
                "query_type": candidate.get("query_type", ""),
                "score": candidate.get("score"),
                "confidence": candidate.get("confidence", "low"),
                "reason": candidate.get("reason", ""),
                "evidence_summary": {},
                "plan_proposal": {},
                "status": "selected" if index == 1 else "backup",
            }
        )
    task_mode = route_result.get("task_mode", "single_step")
    planning_metadata = _planning_metadata(route_result, task_mode)
    return {
        "plan_version": ROUTE_PLAN_VERSION,
        "task_mode": task_mode,
        **planning_metadata,
        "proposal_execution_policy": {
            "enabled": False,
            "top_k": MAX_PROPOSAL_EXECUTION_CANDIDATES,
            "reason": "Keyword fallback has no schema plan proposals.",
        },
        "selected_candidate_id": candidates[0]["candidate_id"] if candidates else "",
        "selected_query_type": route_result.get("query_type", ""),
        "should_keep_backup_route": route_result.get("should_keep_backup_route", False),
        "uncertainty_source": route_result.get("uncertainty_source", ""),
        "schema_plan_context": route_result.get("schema_plan_context", {}),
        "schema_plan_proposals": route_result.get("schema_plan_proposals", []),
        "candidates": candidates,
        "evidence": route_result.get("signals", {}),
        "multi_step_plan": {
            "plan_version": 1,
            "status": "not_required",
            "reason": "Keyword fallback has no schema-aware multi-step plan.",
            "steps": [],
        },
        "multi_step_runtime_policy": {
            "default_enable": False,
            "experimental_auto_candidate": False,
            "requires_explicit_enable": False,
            "recommended_mode": "single_step_runtime",
            "risk_level": "not_applicable",
            "reasons": ["keyword_fallback_has_no_schema_aware_multi_step_plan"],
        },
    }
