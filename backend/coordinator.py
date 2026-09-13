import time
import uuid
from inspect import signature
from math import ceil
from typing import Any

from .cypher_pipeline_trace import run_cypher_pipeline
from .bridge_resolver import resolve_bridge_by_direct_lookup
from .generation_feedback import build_generation_retry_feedback
from .main import run_mql_pipeline, run_vector_pipeline
from .multi_step_runtime import run_multi_step_runtime
from .pipeline_compat import (
    get_pipeline_query,
    get_pipeline_selected_db,
    get_pipeline_selected_path,
    get_pipeline_working_schema,
)
from .router import QUERY_TYPES, route_query
from .schema_plan_proposer import resolve_sql_resource_path
from .sql_pipeline_trace_dail import run_sql_pipeline
from .verification import (
    VERDICT_FAIL_TERMINAL,
    VERDICT_LOCAL_REPAIR,
    VERDICT_PASS,
    VERDICT_REGENERATE,
    VERDICT_REROUTE,
    verify_single_step,
)

DEFAULT_MAX_GLOBAL_LOOPS = 6
DEFAULT_MAX_STEP_RETRIES = 2
DEFAULT_TIME_BUDGET_MS = 120_000
DEFAULT_TOKEN_BUDGET = 30_000
DEFAULT_VERIFICATION_GRACE_MS = 120_000
TOKEN_ESTIMATE_CHARS_PER_TOKEN = 4
RUNTIME_MODE_AUTO = "auto_policy"
RUNTIME_MODE_EXPLICIT_OVERRIDE = "explicit_override"

# 中文函数说明补充索引：
# - _run_pipeline_for(query_type, question, ...)：根据 route_proposal/target_resource 调用 SQL/Cypher/MQL/vector pipeline。
# - _proposal_target_resource(...)：读取 proposal 中指定的数据库资源，保证 Step Planner 选择的库传到 pipeline。
# - _route_candidate_id/_route_candidate_proposal：从 route_plan 中取当前候选 ID 和 proposal，供 attempt history/trace 使用。
# - _flatten_policy_text/_is_graph_event_question：把 route policy/问题文本压平，用于判断强图事件问题和弱 SQL fallback 保护。
# - run_runtime_step/verify_runtime_step/resolve_runtime_bridge：run_coordinated_query 内部传给 multi_step_runtime 的三个闭包；分别负责执行 step、验证 step、解析 bridge。
# - _should_allow_post_pipeline_verification(...)：pipeline 成功但时间预算刚超限时，允许一次 verification grace，避免正确结果因收尾预算被误判失败。
# 关键状态：
# - running：Coordinator 仍在执行当前任务。
# - succeeded：verification 接受最终结果。
# - failed：verification / retry / reroute 后仍无法接受。
# - budget_exceeded：预算保护触发；如果已有候选结果，final_answer 会说明其未 fully verified。
# - budget_grace_used：pipeline 成功后使用了一次 verification grace。
# - multi_step：Coordinator 进入 multi-step runtime，而不是普通单步 while loop。


def _now_ms() -> float:
    """返回当前高精度时间，单位毫秒。"""
    return time.perf_counter() * 1000


def _estimate_token_count(value: Any) -> int:
    """用字符数粗略估算 token 数。

    参数：
    - value：需要估算的对象，可以是字符串、列表、字典或其他可转字符串的值。

    说明：
    - 当前用于 Coordinator 预算控制，不是 tokenizer 的精确统计。
    """
    if value is None:
        return 0
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return 0
        return max(1, ceil(len(stripped) / TOKEN_ESTIMATE_CHARS_PER_TOKEN))
    if isinstance(value, dict):
        return sum(_estimate_token_count(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return sum(_estimate_token_count(item) for item in value)
    return _estimate_token_count(str(value))


def _refresh_budget_usage(state: dict[str, Any]) -> int:
    """刷新并返回当前任务的 token 估算用量。

    参数：
    - state：Coordinator 维护的任务状态字典。
    """
    token_usage = _estimate_token_count(state["memory"].get("llm_trace", []))
    state["control"]["token_usage_estimate"] = token_usage
    return token_usage


def _check_budget(state: dict[str, Any], started_ms: float) -> dict[str, Any] | None:
    """检查当前任务是否超过时间预算或 token 预算。

    参数：
    - state：Coordinator 任务状态。
    - started_ms：本次协调流程开始时的毫秒时间戳。

    返回：
    - 未超限返回 `None`；超限返回包含 reason、message、elapsed_ms 等信息的字典。
    """
    elapsed_ms = _now_ms() - started_ms
    state["control"]["elapsed_ms"] = round(elapsed_ms, 2)
    token_usage = _refresh_budget_usage(state)

    time_budget_ms = int(state["control"].get("time_budget_ms") or 0)
    if time_budget_ms > 0 and elapsed_ms >= time_budget_ms:
        return {
            "reason": "time_budget_exceeded",
            "message": f"Coordinator exceeded time budget: {round(elapsed_ms, 2)} ms >= {time_budget_ms} ms.",
            "elapsed_ms": round(elapsed_ms, 2),
            "token_usage_estimate": token_usage,
        }

    token_budget = int(state["control"].get("token_budget") or 0)
    if token_budget > 0 and token_usage >= token_budget:
        return {
            "reason": "token_budget_exceeded",
            "message": f"Coordinator exceeded token budget estimate: {token_usage} >= {token_budget}.",
            "elapsed_ms": round(elapsed_ms, 2),
            "token_usage_estimate": token_usage,
        }

    return None


def _should_allow_post_pipeline_verification(
    state: dict[str, Any],
    budget_status: dict[str, Any],
    pipeline_result: dict[str, Any],
) -> bool:
    """Allow one verification pass after a successful pipeline slightly exceeds time budget."""
    if budget_status.get("reason") != "time_budget_exceeded":
        return False
    if not pipeline_result.get("success", False):
        return False
    query = get_pipeline_query(pipeline_result)
    if not str(query or "").strip():
        return False
    if str(pipeline_result.get("error", "") or "").strip():
        return False

    elapsed_ms = float(budget_status.get("elapsed_ms", 0) or 0)
    time_budget_ms = int(state["control"].get("time_budget_ms") or 0)
    grace_ms = int(state["control"].get("verification_grace_ms") or 0)
    if time_budget_ms <= 0 or grace_ms <= 0:
        return False
    if elapsed_ms > time_budget_ms + grace_ms:
        return False
    return True


def _resolve_runtime_decision(
    *,
    multi_step_plan: dict[str, Any],
    runtime_policy: dict[str, Any],
    enable_multi_step_runtime: bool | None,
    enable_bridge_resolver: bool | None,
) -> dict[str, Any]:
    """Turn route-level runtime policy and optional overrides into effective execution flags."""
    plan_status = multi_step_plan.get("status", "")
    recommended_mode = runtime_policy.get("recommended_mode", "")
    risk_level = runtime_policy.get("risk_level", "")
    policy_reasons = runtime_policy.get("reasons", []) or []
    has_runtime_plan = plan_status == "planned_not_executed"

    if not has_runtime_plan:
        effective_multi_step = False
        multi_source = RUNTIME_MODE_EXPLICIT_OVERRIDE if enable_multi_step_runtime is not None else RUNTIME_MODE_AUTO
        multi_reason = "multi_step_plan_not_required"
    elif enable_multi_step_runtime is not None:
        effective_multi_step = bool(enable_multi_step_runtime)
        multi_source = RUNTIME_MODE_EXPLICIT_OVERRIDE
        multi_reason = "multi_step_runtime_explicitly_enabled" if effective_multi_step else "multi_step_runtime_explicitly_disabled"
    elif (
        recommended_mode == "auto_runtime_experiment_candidate"
        and risk_level == "low"
        and not runtime_policy.get("requires_explicit_enable", True)
    ):
        effective_multi_step = True
        multi_source = RUNTIME_MODE_AUTO
        multi_reason = "same_source_low_risk_runtime_auto_enabled"
    elif recommended_mode == "auto_runtime_experiment_candidate":
        effective_multi_step = False
        multi_source = RUNTIME_MODE_AUTO
        multi_reason = "auto_runtime_candidate_deferred_by_risk_policy"
    elif recommended_mode == "explicit_runtime_only":
        effective_multi_step = True
        multi_source = RUNTIME_MODE_AUTO
        multi_reason = "cross_source_runtime_auto_enabled_for_bridge_preflight"
    else:
        effective_multi_step = False
        multi_source = RUNTIME_MODE_AUTO
        multi_reason = "runtime_policy_does_not_enable_multi_step"

    if not effective_multi_step:
        effective_bridge = False
        bridge_source = RUNTIME_MODE_EXPLICIT_OVERRIDE if enable_bridge_resolver is not None else RUNTIME_MODE_AUTO
        bridge_reason = "bridge_resolver_disabled_because_runtime_not_active"
    elif enable_bridge_resolver is not None:
        effective_bridge = bool(enable_bridge_resolver)
        bridge_source = RUNTIME_MODE_EXPLICIT_OVERRIDE
        bridge_reason = "bridge_resolver_explicitly_enabled" if effective_bridge else "bridge_resolver_explicitly_disabled"
    elif recommended_mode == "explicit_runtime_only":
        effective_bridge = True
        bridge_source = RUNTIME_MODE_AUTO
        bridge_reason = "bridge_resolver_auto_enabled_with_high_confidence_gate"
    else:
        effective_bridge = False
        bridge_source = RUNTIME_MODE_AUTO
        bridge_reason = "bridge_resolver_not_auto_enabled"

    return {
        "effective_multi_step_runtime": effective_multi_step,
        "effective_bridge_resolver": effective_bridge,
        "decision_source": multi_source if multi_source == bridge_source else "mixed",
        "multi_step_decision_source": multi_source,
        "bridge_decision_source": bridge_source,
        "decision_reason": multi_reason,
        "bridge_decision_reason": bridge_reason,
        "policy_recommended_mode": recommended_mode,
        "policy_risk_level": risk_level,
        "policy_reasons": policy_reasons,
        "plan_status": plan_status,
        "explicit_multi_step_runtime": enable_multi_step_runtime,
        "explicit_bridge_resolver": enable_bridge_resolver,
    }


def _mark_budget_exceeded(state: dict[str, Any], budget_status: dict[str, Any]) -> None:
    """将任务状态标记为预算超限。

    参数：
    - state：需要修改的 Coordinator 任务状态。
    - budget_status：`_check_budget(...)` 返回的超限详情。
    """
    state["status"] = "budget_exceeded"
    state["plan"]["steps"][0]["status"] = "failed"
    state["control"]["budget_exceeded"] = True
    state["control"]["budget_reason"] = budget_status["reason"]
    state["verification"] = {
        "verdict": VERDICT_FAIL_TERMINAL,
        "failure_type": budget_status["reason"],
        "confidence": "high",
        "reason": budget_status["message"],
        "suggested_action": "terminate",
    }
    candidate_result = str(state.get("execution", {}).get("result_text", "") or "").strip()
    if candidate_result:
        result_text = (
            f"{budget_status['message']}\n\n"
            "A candidate query result was produced before the budget stop, but it was not fully verified:\n"
            f"{candidate_result}"
        )
    else:
        result_text = budget_status["message"]
    state["final_answer"] = {
        "success": False,
        "query": state["generation"]["current_query"],
        "result_text": result_text,
    }
    selected_candidate_id = state.get("routing", {}).get("selected_candidate_id", "")
    route_plan_candidates = state.get("routing", {}).get("route_plan", {}).get("candidates", [])
    for index, candidate in enumerate(route_plan_candidates):
        if candidate.get("candidate_id") == selected_candidate_id:
            _update_route_plan_candidate(
                state,
                index,
                status="budget_exceeded",
                detail=budget_status["message"],
                loop_count=state.get("control", {}).get("global_loop_count", 0),
                decision="budget_exceeded",
            )
            break
    _log_event(
        state,
        "budget",
        budget_status["message"],
        reason=budget_status["reason"],
        elapsed_ms=budget_status["elapsed_ms"],
        token_usage_estimate=budget_status["token_usage_estimate"],
    )


def _pipeline_for(query_type: str):
    """根据查询模态选择对应 pipeline。

    参数：
    - query_type：路由得到的查询类型，例如 `sql` 或 `cypher`。
    """
    return {
        "sql": run_sql_pipeline,
        "cypher": run_cypher_pipeline,
        "mql": run_mql_pipeline,
        "vector": run_vector_pipeline,
    }.get(query_type, run_sql_pipeline)


def _run_pipeline_for(
    query_type: str,
    question: str,
    generation_feedback: dict[str, Any] | None = None,
    route_proposal: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pipeline = _pipeline_for(query_type)
    route_proposal = route_proposal or {}
    target_resource = _proposal_target_resource(route_proposal, query_type)
    params = signature(pipeline).parameters
    kwargs: dict[str, Any] = {}
    if generation_feedback and generation_feedback.get("active") and "generation_feedback" in params:
        kwargs["generation_feedback"] = generation_feedback
    if query_type in {"sql", "cypher"}:
        if query_type == "sql" and target_resource.get("resource_path") and "db_path" in params:
            kwargs["db_path"] = target_resource["resource_path"]
        if query_type == "cypher" and target_resource.get("resource_id") and "db_name" in params:
            kwargs["db_name"] = target_resource["resource_id"]
        result = pipeline(question, **kwargs)
        result["route_proposal"] = route_proposal
        result["proposal_target_resource"] = target_resource
        return result
    result = pipeline(question)
    result["route_proposal"] = route_proposal
    result["proposal_target_resource"] = target_resource
    return result


def _proposal_target_resource(route_proposal: dict[str, Any], query_type: str) -> dict[str, Any]:
    resources = route_proposal.get("target_resources", [])
    if not isinstance(resources, list) or not resources:
        return {}
    first = resources[0] if isinstance(resources[0], dict) else {}
    resource_id = first.get("resource_id", "")
    if query_type == "sql" and resource_id:
        return {
            **first,
            "resource_path": first.get("resource_path") or resolve_sql_resource_path(resource_id),
        }
    return first


def _default_query_type_candidates(selected: str) -> list[str]:
    """当路由结果没有候选列表时，生成默认候选顺序。

    参数：
    - selected：路由层当前最优 query type，会被放在候选列表第一位。
    """
    ordered = [selected]
    for candidate in QUERY_TYPES:
        if candidate != selected:
            ordered.append(candidate)
    return ordered


def _route_candidates_from_plan(route_plan: dict[str, Any], fallback: list[str]) -> list[str]:
    """Read query-type candidates from route_plan while preserving old fallback behavior."""
    candidates = [
        candidate.get("query_type")
        for candidate in route_plan.get("candidates", [])
        if candidate.get("query_type") in QUERY_TYPES
    ]
    return candidates or fallback


def _route_candidate_id(state: dict[str, Any], route_index: int) -> str:
    candidates = state["routing"].get("route_plan", {}).get("candidates", [])
    if 0 <= route_index < len(candidates):
        return candidates[route_index].get("candidate_id", f"R{route_index + 1}")
    return f"R{route_index + 1}"


def _route_candidate_proposal(state: dict[str, Any], route_index: int) -> dict[str, Any]:
    candidates = state["routing"].get("route_plan", {}).get("candidates", [])
    if 0 <= route_index < len(candidates):
        return candidates[route_index].get("plan_proposal", {}) or {}
    return {}


def _update_route_plan_candidate(
    state: dict[str, Any],
    route_index: int,
    *,
    status: str,
    detail: str = "",
    loop_count: int | None = None,
    verdict: str = "",
    decision: str = "",
    error: str = "",
    generation_feedback: dict[str, Any] | None = None,
) -> None:
    """Update a route_plan candidate status and append an attempt event."""
    route_plan = state["routing"].get("route_plan", {})
    candidates = route_plan.get("candidates", [])
    if not (0 <= route_index < len(candidates)):
        return

    candidate = candidates[route_index]
    candidate["status"] = status
    if detail:
        candidate["last_detail"] = detail
    if loop_count is not None:
        candidate["last_loop_count"] = loop_count
    if verdict:
        candidate["last_verdict"] = verdict
    if decision:
        candidate["last_decision"] = decision
    if error:
        candidate["last_error"] = error
    if generation_feedback and generation_feedback.get("active"):
        candidate["last_generation_feedback"] = generation_feedback

    attempt = {
        "loop_count": loop_count if loop_count is not None else state["control"].get("global_loop_count", 0),
        "status": status,
        "detail": detail,
        "verdict": verdict,
        "decision": decision,
        "error": error,
        "generation_feedback": generation_feedback if generation_feedback and generation_feedback.get("active") else {},
    }
    candidate.setdefault("attempt_history", []).append(
        {key: value for key, value in attempt.items() if value not in ("", None)}
    )

    route_plan["selected_candidate_id"] = candidate.get("candidate_id", "")
    route_plan["selected_query_type"] = candidate.get("query_type", "")
    state["routing"]["route_plan"] = route_plan


def _attach_retrieval_to_route_plan_candidate(
    state: dict[str, Any],
    route_index: int,
    pipeline_result: dict[str, Any],
) -> None:
    """Attach actual retrieval output to the current route_plan candidate."""
    route_plan = state["routing"].get("route_plan", {})
    candidates = route_plan.get("candidates", [])
    if not (0 <= route_index < len(candidates)):
        return

    retrieval = pipeline_result.get("retrieval", {})
    if not retrieval:
        return

    candidates[route_index]["retrieval"] = {
        "method": retrieval.get("method", ""),
        "selected_db": get_pipeline_selected_db(pipeline_result) or retrieval.get("selected_db", ""),
        "selected_path": get_pipeline_selected_path(pipeline_result),
        "score": retrieval.get("score"),
        "top_candidates": retrieval.get("top_candidates", [])[:3],
    }
    route_plan["candidates"] = candidates
    state["routing"]["route_plan"] = route_plan


def _create_task_state(question: str) -> dict[str, Any]:
    """创建单步查询任务的统一状态结构。

    参数：
    - question：用户输入的自然语言问题。

    返回：
    - 包含路由、检索、schema、生成、执行、验证、修复、预算和未来多步预留字段的 task state。
    """
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    task_id = str(uuid.uuid4())
    return {
        "task_id": task_id,
        "user_question": question,
        "mode": "single_step",
        "status": "running",
        "created_at": now,
        "updated_at": now,
        "control": {
            "max_global_loops": DEFAULT_MAX_GLOBAL_LOOPS,
            "max_step_retries": DEFAULT_MAX_STEP_RETRIES,
            "global_loop_count": 0,
            "time_budget_ms": DEFAULT_TIME_BUDGET_MS,
            "token_budget": DEFAULT_TOKEN_BUDGET,
            "verification_grace_ms": DEFAULT_VERIFICATION_GRACE_MS,
            "elapsed_ms": 0,
            "token_usage_estimate": 0,
            "budget_exceeded": False,
            "budget_reason": "",
            "budget_grace_used": False,
        },
        "plan": {
            "plan_version": 1,
            "active_step_id": "S1",
            "steps": [
                {
                    "step_id": "S1",
                    "step_type": "query",
                    "step_goal": "Answer the user question directly with one query",
                    "status": "running",
                    "depends_on": [],
                    "input_vars": [],
                    "output_var": "final_result",
                    "expected_output_type": "table_or_scalar",
                    "modality_hint": [],
                    "db_hint": [],
                }
            ],
        },
        "routing": {
            "initial_query_type": "",
            "query_type_candidates": [],
            "route_plan": {},
            "route_candidates": [],
            "schema_plan_context": {},
            "schema_plan_proposals": [],
            "multi_step_plan": {},
            "multi_step_runtime": {},
            "signals": {},
            "selected_candidate_id": "",
            "selected_query_type": "",
            "planning_status": "",
            "execution_policy": {},
            "clarification": {},
            "low_confidence": False,
            "confidence": "",
            "reason": "",
            "uncertainty_source": "",
            "should_keep_backup_route": False,
            "needs_reroute": False,
            "reroute_count": 0,
        },
        "retrieval": {
            "db_candidates": [],
            "selected_db": "",
            "selected_path": "",
            "score": None,
            "needs_retrieve_again": False,
        },
        "schema_context": {
            "full_schema": "",
            "working_schema": "",
            "schema_notes": "",
        },
        "generation": {
            "current_query": "",
            "query_history": [],
            "cot_summary": "",
            "generator_notes": "",
            "retry_feedback": {},
            "last_retry_feedback": {},
        },
        "execution": {
            "success": False,
            "result_rows": [],
            "result_columns": [],
            "row_count": 0,
            "result_text": "",
            "error": "",
            "execution_history": [],
        },
        "verification": {
            "verdict": "",
            "failure_type": "",
            "confidence": "",
            "reason": "",
            "suggested_action": "",
        },
        "repair": {
            "repair_count": 0,
            "step_retry_count": 0,
            "repair_history": [],
        },
        "memory": {
            "llm_trace": [],
            "events": [],
        },
        "final_answer": {
            "success": False,
            "query": "",
            "result_text": "",
        },
        "reserved_for_future": {
            "variable_store": {},
            "cross_step_contracts": [],
            "branch_candidates": [],
        },
    }


def _log_event(state: dict[str, Any], stage: str, detail: str, **extra: Any) -> None:
    """向任务状态中追加一条事件日志。

    参数：
    - state：Coordinator 任务状态。
    - stage：事件所属阶段，例如 `route`、`verify`、`repair`。
    - detail：事件说明。
    - extra：其他结构化附加信息。
    """
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    state["memory"]["events"].append(
        {
            "ts": state["updated_at"],
            "stage": stage,
            "detail": detail,
            **extra,
        }
    )


def _sync_state_from_pipeline(state: dict[str, Any], pipeline_result: dict[str, Any]) -> None:
    """把 pipeline 输出同步到 Coordinator 的统一 state 中。

    参数：
    - state：Coordinator 任务状态，会被原地更新。
    - pipeline_result：SQL/Cypher pipeline 的执行结果。
    """
    state["retrieval"]["db_candidates"] = pipeline_result.get("retrieval", {}).get("top_candidates", [])
    state["retrieval"]["selected_db"] = get_pipeline_selected_db(pipeline_result)
    state["retrieval"]["selected_path"] = get_pipeline_selected_path(pipeline_result)
    state["retrieval"]["score"] = pipeline_result.get("retrieval", {}).get("score")

    state["schema_context"]["full_schema"] = pipeline_result.get("schema_full", "")
    state["schema_context"]["working_schema"] = get_pipeline_working_schema(pipeline_result)
    state["schema_context"]["schema_notes"] = pipeline_result.get("question_masked", "")

    current_query = get_pipeline_query(pipeline_result)
    state["generation"]["current_query"] = current_query
    if current_query:
        state["generation"]["query_history"].append(current_query)
    state["generation"]["cot_summary"] = pipeline_result.get("cot_raw", "")

    state["execution"]["success"] = pipeline_result.get("success", False)
    state["execution"]["result_rows"] = pipeline_result.get("result_rows", [])
    state["execution"]["result_columns"] = pipeline_result.get("result_columns", [])
    state["execution"]["row_count"] = pipeline_result.get("row_count", 0)
    state["execution"]["result_text"] = pipeline_result.get("result", "")
    state["execution"]["error"] = pipeline_result.get("error", "")
    state["execution"]["execution_history"].append(
        {
            "query": current_query,
            "success": pipeline_result.get("success", False),
            "error": pipeline_result.get("error", ""),
            "retrieved_db": state["retrieval"]["selected_db"],
        }
    )

    llm_trace = pipeline_result.get("llm_trace", [])
    if llm_trace:
        state["memory"]["llm_trace"].extend(llm_trace)


def _flatten_policy_text(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(_flatten_policy_text(item) for item in value.values())
    if isinstance(value, list):
        return " ".join(_flatten_policy_text(item) for item in value)
    return str(value or "").strip().lower()


def _is_graph_event_question(question: str) -> bool:
    normalized = str(question or "").strip().lower()
    has_event_measure = any(token in normalized for token in ["goal", "goals", "scored", "score", "scores"])
    has_graph_context = any(
        token in normalized
        for token in ["team", "teams", "player", "players", "match", "matches", "tournament", "tournaments"]
    )
    asks_quantity = any(
        token in normalized
        for token in ["total", "number of", "how many", "count", "calculate", "sum"]
    )
    return has_event_measure and has_graph_context and asks_quantity


def _should_block_weak_sql_backup_route(state: dict[str, Any], next_route_index: int) -> bool:
    """Avoid replacing a graph-event Cypher primary with an unevidenced SQL backup."""
    initial_query_type = str(state["routing"].get("initial_query_type") or "").lower()
    if initial_query_type != "cypher":
        return False
    if not _is_graph_event_question(state.get("user_question", "")):
        return False

    route_candidates = state["routing"].get("query_type_candidates", [])
    if not (0 <= next_route_index < len(route_candidates)):
        return False
    if route_candidates[next_route_index] != "sql":
        return False

    proposal = _route_candidate_proposal(state, next_route_index)
    confidence = str(proposal.get("confidence") or "").strip().lower()
    reason_text = _flatten_policy_text(proposal.get("reason", ""))
    target_text = _flatten_policy_text(proposal.get("target_resources", []))
    schema_text = _flatten_policy_text(proposal.get("schema_items", []))
    weak_reason_markers = ["fallback", "unrelated", "not fit", "does not fit", "not relevant", "mismatch", "backup"]
    weak_route = not proposal or confidence in {"", "low"} or any(marker in reason_text for marker in weak_reason_markers)
    sql_has_graph_evidence = any(marker in f"{target_text} {schema_text}" for marker in ["goal", "team", "tournament", "scored"])
    return weak_route and not sql_has_graph_evidence


def _decision_from_verdict(
    *,
    verdict: str,
    state: dict[str, Any],
    route_candidates: list[str],
    current_route_index: int,
) -> tuple[str, int]:
    """根据 Verification verdict 决定 Coordinator 下一步动作。

    参数：
    - verdict：Verification Agent 返回的判定类型。
    - state：当前任务状态，用于读取重试次数和上限。
    - route_candidates：路由候选 query type 列表。
    - current_route_index：当前正在尝试的候选下标。

    返回：
    - `(decision, next_route_index)`，decision 可能是 accept、retry_same_route、reroute_next_candidate 或 terminate。
    """
    max_step_retries = state["control"]["max_step_retries"]
    step_retry_count = state["repair"]["step_retry_count"]

    if verdict == VERDICT_PASS:
        return "accept", current_route_index

    if verdict == VERDICT_LOCAL_REPAIR:
        if step_retry_count < max_step_retries:
            return "retry_same_route", current_route_index
        if current_route_index + 1 < len(route_candidates):
            if _should_block_weak_sql_backup_route(state, current_route_index + 1):
                return "terminate", current_route_index
            return "reroute_next_candidate", current_route_index + 1
        return "terminate", current_route_index

    if verdict == VERDICT_REGENERATE:
        if step_retry_count < max_step_retries:
            return "retry_same_route", current_route_index
        return "terminate", current_route_index

    if verdict == VERDICT_REROUTE:
        if current_route_index + 1 < len(route_candidates):
            if _should_block_weak_sql_backup_route(state, current_route_index + 1):
                return "terminate", current_route_index
            return "reroute_next_candidate", current_route_index + 1
        return "terminate", current_route_index

    return "terminate", current_route_index


def run_coordinated_query(
    question: str,
    schema_summary: str = "",
    *,
    max_global_loops: int | None = None,
    max_step_retries: int | None = None,
    time_budget_ms: int | None = None,
    token_budget: int | None = None,
    enable_multi_step_runtime: bool | None = None,
    enable_bridge_resolver: bool | None = None,
) -> dict[str, Any]:
    """单步多 Agent 查询主入口。

    参数：
    - question：用户自然语言问题。
    - schema_summary：可选的全局 schema 摘要，主要给路由层提供先验。
    - max_global_loops：可选，覆盖 Coordinator 全局循环上限。
    - max_step_retries：可选，覆盖同一路由下的单步重试上限。
    - time_budget_ms：可选，覆盖本次请求时间预算，单位毫秒。
    - token_budget：可选，覆盖本次请求 token 估算预算。

    返回：
    - 包含 route、pipeline、verification、coordinator、final_answer、timing 和 task_state 的完整执行结果。
    """
    started_ms = _now_ms()
    state = _create_task_state(question)
    if max_global_loops is not None:
        state["control"]["max_global_loops"] = max_global_loops
    if max_step_retries is not None:
        state["control"]["max_step_retries"] = max_step_retries
    if time_budget_ms is not None:
        state["control"]["time_budget_ms"] = time_budget_ms
    if token_budget is not None:
        state["control"]["token_budget"] = token_budget

    route_started = _now_ms()
    initial_route = route_query(question, schema_summary=schema_summary)
    route_ms = _now_ms() - route_started

    fallback_route_candidates = [
        candidate.get("query_type")
        for candidate in initial_route.get("candidates", [])
        if candidate.get("query_type") in QUERY_TYPES
    ]
    if not fallback_route_candidates:
        fallback_route_candidates = _default_query_type_candidates(initial_route["query_type"])
    route_candidates = _route_candidates_from_plan(initial_route.get("route_plan", {}), fallback_route_candidates)
    current_route_index = 0
    state["mode"] = initial_route.get("task_mode", "single_step")
    selected_candidate_id = _route_candidate_id(
        {"routing": {"route_plan": initial_route.get("route_plan", {})}},
        current_route_index,
    )
    state["routing"].update(
        {
            "initial_query_type": initial_route["query_type"],
            "query_type_candidates": route_candidates,
            "route_plan": initial_route.get("route_plan", {}),
            "route_candidates": initial_route.get("candidates", []),
            "schema_plan_context": initial_route.get("schema_plan_context", {}),
            "schema_plan_proposals": initial_route.get("schema_plan_proposals", []),
            "multi_step_plan": initial_route.get("route_plan", {}).get("multi_step_plan", {}),
            "multi_step_runtime": {},
            "runtime_decision": {},
            "signals": initial_route.get("signals", {}),
            "selected_candidate_id": selected_candidate_id,
            "selected_query_type": route_candidates[current_route_index],
            "planning_status": initial_route.get("route_plan", {}).get("planning_status", ""),
            "execution_policy": initial_route.get("route_plan", {}).get("execution_policy", {}),
            "clarification": initial_route.get("route_plan", {}).get("clarification", {}),
            "low_confidence": initial_route.get("route_plan", {}).get("planning_status") == "low_confidence",
            "confidence": initial_route.get("confidence", ""),
            "reason": initial_route.get("reason", ""),
            "uncertainty_source": initial_route.get("uncertainty_source", ""),
            "should_keep_backup_route": initial_route.get("should_keep_backup_route", False),
        }
    )
    state["routing"]["runtime_decision"] = _resolve_runtime_decision(
        multi_step_plan=state["routing"]["multi_step_plan"],
        runtime_policy=state["routing"].get("route_plan", {}).get("multi_step_runtime_policy", {}),
        enable_multi_step_runtime=enable_multi_step_runtime,
        enable_bridge_resolver=enable_bridge_resolver,
    )
    _log_event(
        state,
        "route",
        "Initial routing completed.",
        selected_query_type=initial_route["query_type"],
        selected_candidate_id=selected_candidate_id,
        planning_status=state["routing"]["planning_status"],
        clarification=state["routing"]["clarification"],
        candidates=route_candidates,
        runtime_decision=state["routing"]["runtime_decision"],
    )

    budget_status = _check_budget(state, started_ms)
    if budget_status:
        _mark_budget_exceeded(state, budget_status)

    pipeline_result: dict[str, Any] = {}
    verification_result: dict[str, Any] = {}
    verification_ms = 0.0

    multi_step_plan = state["routing"].get("multi_step_plan", {})
    runtime_decision = state["routing"].get("runtime_decision", {})
    effective_multi_step_runtime = bool(runtime_decision.get("effective_multi_step_runtime", False))
    effective_bridge_resolver = bool(runtime_decision.get("effective_bridge_resolver", False))

    if effective_multi_step_runtime and multi_step_plan.get("status") == "planned_not_executed":
        state["mode"] = "multi_step"
        _log_event(
            state,
            "multi_step_runtime",
            "Starting multi-step runtime.",
            step_count=len(multi_step_plan.get("steps", [])),
        )

        def run_runtime_step(query_type: str, step_question: str, proposal: dict[str, Any]) -> dict[str, Any]:
            state["control"]["global_loop_count"] += 1
            step_pipeline_result = _run_pipeline_for(query_type, step_question, route_proposal=proposal)
            step_pipeline_result["llm_trace"] = step_pipeline_result.get("llm_trace", [])
            if step_pipeline_result["llm_trace"]:
                state["memory"]["llm_trace"].extend(step_pipeline_result["llm_trace"])
            return step_pipeline_result

        def verify_runtime_step(
            step_question: str,
            proposal: dict[str, Any],
            step_pipeline_result: dict[str, Any],
        ) -> dict[str, Any]:
            nonlocal verification_ms
            verify_started = _now_ms()
            result = verify_single_step(
                question=step_question,
                route_result={
                    "query_type": proposal.get("query_type", step_pipeline_result.get("query_type", "")),
                    "confidence": proposal.get("confidence", "low"),
                    "reason": proposal.get("reason", ""),
                    "initial_query_type": state["routing"]["initial_query_type"],
                    "candidates": state["routing"]["query_type_candidates"],
                    "route_plan": state["routing"]["route_plan"],
                    "route_proposal": proposal,
                    "bridge_evaluations": proposal.get("bridge_evaluations", []),
                },
                pipeline_result=step_pipeline_result,
                trace=state["memory"]["llm_trace"],
            )
            verification_ms += _now_ms() - verify_started
            return result

        def resolve_runtime_bridge(
            bridge: dict[str, Any],
            source_payload: dict[str, Any],
            target_step: dict[str, Any],
        ) -> dict[str, Any]:
            values = source_payload.get("primary_values", [])
            if not values:
                return {
                    "runtime_status": "missing_input",
                    "effective_consumption_policy": "soft_context",
                    "runtime_reason": "Bridge resolver had no source values to map.",
                }
            state["control"]["global_loop_count"] += 1
            direct_result = resolve_bridge_by_direct_lookup(bridge, source_payload, target_step)
            if direct_result.get("resolver_strategy") != "unsupported":
                return direct_result
            target_query_type = bridge.get("target_query_type", target_step.get("query_type", ""))
            bridge_question = (
                "Bridge mapping subtask.\n"
                f"Original user question: {question}\n"
                f"Source variable: {bridge.get('input_var', '')}\n"
                f"Source entity type: {bridge.get('source_entity_type', '')}\n"
                f"Source values to map: {values[:3]}\n"
                f"Target query type: {target_query_type}\n"
                f"Target schema items: {target_step.get('schema_items', [])}\n"
                "Return target-side mapping candidates only if the target schema and data support them. "
                "Do not invent a mapping."
            )
            bridge_proposal = {
                "proposal_id": bridge.get("bridge_id", "BR"),
                "query_type": target_query_type,
                "query_strategy": "bridge_mapping",
                "query_shape": "bridge_mapping",
                "schema_items": target_step.get("schema_items", []),
                "target_resources": target_step.get("target_resources", []),
                "source": "bridge_resolver",
                "reason": bridge_question,
            }
            state["control"]["global_loop_count"] += 1
            bridge_pipeline = _run_pipeline_for(target_query_type, bridge_question, route_proposal=bridge_proposal)
            bridge_pipeline["llm_trace"] = bridge_pipeline.get("llm_trace", [])
            if bridge_pipeline["llm_trace"]:
                state["memory"]["llm_trace"].extend(bridge_pipeline["llm_trace"])
            rows = bridge_pipeline.get("result_rows", [])
            columns = [str(column) for column in bridge_pipeline.get("result_columns", [])]
            target_field = columns[0] if columns else "value"
            target_values: list[Any] = []
            if isinstance(rows, list):
                for row in rows[:3]:
                    if isinstance(row, dict):
                        target_values.append(row.get(target_field, next(iter(row.values()), "")))
                    elif isinstance(row, (list, tuple)) and row:
                        target_values.append(row[0])
                    elif row:
                        target_values.append(row)
            if bridge_pipeline.get("success") and bridge_pipeline.get("row_count", 0) and target_values:
                return {
                    "runtime_status": "unresolved_soft_context",
                    "effective_consumption_policy": "soft_context",
                    "target_field": target_field,
                    "target_values": target_values,
                    "confidence": "medium",
                    "allow_hard_constraint": False,
                    "resolver_pipeline": {
                        "query_type": bridge_pipeline.get("query_type", target_query_type),
                        "selected_db": bridge_pipeline.get("selected_db", bridge_pipeline.get("retrieved_db", "")),
                        "query": bridge_pipeline.get("query", ""),
                        "row_count": bridge_pipeline.get("row_count", 0),
                    },
                    "runtime_reason": (
                        "Generated bridge resolver found target-side candidates, but they were not direct "
                        "high-confidence mappings; the candidates remain soft context."
                    ),
                }
            return {
                "runtime_status": "unresolved_soft_context",
                "effective_consumption_policy": "soft_context",
                "resolver_pipeline": {
                    "query_type": bridge_pipeline.get("query_type", target_query_type),
                    "selected_db": bridge_pipeline.get("selected_db", bridge_pipeline.get("retrieved_db", "")),
                    "query": bridge_pipeline.get("query", ""),
                    "row_count": bridge_pipeline.get("row_count", 0),
                    "error": bridge_pipeline.get("error", ""),
                },
                "runtime_reason": "Bridge resolver did not find target-side mapping candidates.",
            }

        runtime_result = run_multi_step_runtime(
            question=question,
            multi_step_plan=multi_step_plan,
            run_pipeline=run_runtime_step,
            verify_step=verify_runtime_step,
            resolve_bridge=resolve_runtime_bridge if effective_bridge_resolver else None,
            max_steps=2,
            block_unresolved_required_bridges=(
                runtime_decision.get("decision_reason") == "cross_source_runtime_auto_enabled_for_bridge_preflight"
                and runtime_decision.get("bridge_decision_reason") == "bridge_resolver_auto_enabled_with_high_confidence_gate"
            ),
        )
        state["routing"]["multi_step_runtime"] = runtime_result
        state["reserved_for_future"]["variable_store"] = runtime_result.get("variable_store", {})
        pipeline_result = runtime_result.get("final_pipeline", {})
        verification_result = runtime_result.get("final_verification", {})
        state["verification"] = {
            "verdict": verification_result.get("verdict", ""),
            "failure_type": verification_result.get("failure_type", ""),
            "confidence": verification_result.get("confidence", ""),
            "reason": verification_result.get("reason", ""),
            "suggested_action": verification_result.get("suggested_action", ""),
            "runtime_status": verification_result.get("runtime_status", ""),
        }
        if runtime_result.get("status") == "succeeded":
            state["status"] = "succeeded"
            state["plan"]["steps"][0]["status"] = "completed"
        else:
            state["status"] = "failed"
            state["plan"]["steps"][0]["status"] = "failed"
        state["final_answer"] = runtime_result.get("final_answer", state["final_answer"])
        _log_event(
            state,
            "multi_step_runtime",
            "Multi-step runtime completed.",
            status=runtime_result.get("status", ""),
            reason=runtime_result.get("reason", ""),
        )

    while (
        state["status"] == "running"
        and state["control"]["global_loop_count"] < state["control"]["max_global_loops"]
    ):
        budget_status = _check_budget(state, started_ms)
        if budget_status:
            _mark_budget_exceeded(state, budget_status)
            break

        state["control"]["global_loop_count"] += 1
        selected_query_type = route_candidates[current_route_index]
        selected_candidate_id = _route_candidate_id(state, current_route_index)
        state["routing"]["selected_candidate_id"] = selected_candidate_id
        state["routing"]["selected_query_type"] = selected_query_type
        _update_route_plan_candidate(
            state,
            current_route_index,
            status="running",
            detail="Coordinator dispatched this route candidate.",
            loop_count=state["control"]["global_loop_count"],
        )

        _log_event(
            state,
            "coordinator",
            "Dispatch pipeline.",
            loop_count=state["control"]["global_loop_count"],
            selected_candidate_id=selected_candidate_id,
            selected_query_type=selected_query_type,
        )

        generation_feedback = state["generation"].get("retry_feedback", {})
        state["generation"]["last_retry_feedback"] = generation_feedback if generation_feedback.get("active") else {}
        state["generation"]["retry_feedback"] = {}
        route_proposal = _route_candidate_proposal(state, current_route_index)

        # Pipeline outputs still expose legacy aliases, but Coordinator should consume the unified keys first.
        pipeline_result = _run_pipeline_for(selected_query_type, question, generation_feedback, route_proposal)
        pipeline_result["llm_trace"] = pipeline_result.get("llm_trace", [])
        _sync_state_from_pipeline(state, pipeline_result)
        _attach_retrieval_to_route_plan_candidate(state, current_route_index, pipeline_result)

        budget_status = _check_budget(state, started_ms)
        if budget_status:
            if _should_allow_post_pipeline_verification(state, budget_status, pipeline_result):
                state["control"]["budget_grace_used"] = True
                _log_event(
                    state,
                    "budget",
                    "Time budget exceeded after a successful pipeline; allowing one verification grace pass.",
                    reason=budget_status["reason"],
                    elapsed_ms=budget_status.get("elapsed_ms", 0),
                    verification_grace_ms=state["control"].get("verification_grace_ms", 0),
                )
            else:
                _mark_budget_exceeded(state, budget_status)
                break

        verify_started = _now_ms()
        verification_result = verify_single_step(
            question=question,
            route_result={
                "query_type": selected_query_type,
                "confidence": state["routing"]["confidence"],
                "reason": state["routing"]["reason"],
                "initial_query_type": state["routing"]["initial_query_type"],
                "candidates": route_candidates,
                "selected_candidate_id": selected_candidate_id,
                "route_plan": state["routing"]["route_plan"],
                "route_proposal": route_proposal,
            },
            pipeline_result=pipeline_result,
            trace=state["memory"]["llm_trace"],
        )
        verification_ms += _now_ms() - verify_started

        state["verification"] = {
            "verdict": verification_result.get("verdict", ""),
            "failure_type": verification_result.get("failure_type", ""),
            "confidence": verification_result.get("confidence", ""),
            "reason": verification_result.get("reason", ""),
            "suggested_action": verification_result.get("suggested_action", ""),
        }
        _log_event(
            state,
            "verify",
            "Verification completed.",
            selected_candidate_id=selected_candidate_id,
            verdict=verification_result.get("verdict", ""),
            failure_type=verification_result.get("failure_type", ""),
        )

        decision, next_route_index = _decision_from_verdict(
            verdict=verification_result.get("verdict", VERDICT_FAIL_TERMINAL),
            state=state,
            route_candidates=route_candidates,
            current_route_index=current_route_index,
        )
        _update_route_plan_candidate(
            state,
            current_route_index,
            status="verified",
            detail=verification_result.get("reason", ""),
            loop_count=state["control"]["global_loop_count"],
            verdict=verification_result.get("verdict", ""),
            decision=decision,
            error=pipeline_result.get("error", ""),
        )

        if decision == "accept":
            _update_route_plan_candidate(
                state,
                current_route_index,
                status="succeeded",
                detail="Accepted by Coordinator.",
                loop_count=state["control"]["global_loop_count"],
                verdict=verification_result.get("verdict", ""),
                decision=decision,
            )
            state["status"] = "succeeded"
            state["plan"]["steps"][0]["status"] = "completed"
            state["final_answer"] = {
                "success": True,
                "query": state["generation"]["current_query"],
                "result_text": state["execution"]["result_text"],
            }
            break

        if decision == "retry_same_route":
            generation_feedback = build_generation_retry_feedback(verification_result)
            if generation_feedback.get("active"):
                state["generation"]["retry_feedback"] = generation_feedback
            state["repair"]["step_retry_count"] += 1
            state["repair"]["repair_count"] += 1
            state["repair"]["repair_history"].append(
                {
                    "loop_count": state["control"]["global_loop_count"],
                    "candidate_id": selected_candidate_id,
                    "query_type": selected_query_type,
                    "verdict": verification_result.get("verdict", ""),
                    "reason": verification_result.get("reason", ""),
                    "generation_feedback": generation_feedback if generation_feedback.get("active") else {},
                }
            )
            _update_route_plan_candidate(
                state,
                current_route_index,
                status="retry_pending",
                detail="Coordinator will retry the same route candidate.",
                loop_count=state["control"]["global_loop_count"],
                verdict=verification_result.get("verdict", ""),
                decision=decision,
                generation_feedback=generation_feedback,
            )
            _log_event(
                state,
                "repair",
                "Retry same route.",
                selected_candidate_id=selected_candidate_id,
                query_type=selected_query_type,
                step_retry_count=state["repair"]["step_retry_count"],
                generation_feedback=generation_feedback if generation_feedback.get("active") else {},
            )
            continue

        if decision == "reroute_next_candidate":
            state["generation"]["retry_feedback"] = {}
            _update_route_plan_candidate(
                state,
                current_route_index,
                status="rerouted",
                detail="Coordinator switched away from this route candidate.",
                loop_count=state["control"]["global_loop_count"],
                verdict=verification_result.get("verdict", ""),
                decision=decision,
            )
            current_route_index = next_route_index
            state["routing"]["reroute_count"] += 1
            state["routing"]["needs_reroute"] = True
            state["repair"]["step_retry_count"] = 0
            next_candidate_id = _route_candidate_id(state, current_route_index)
            state["routing"]["selected_candidate_id"] = next_candidate_id
            _log_event(
                state,
                "reroute",
                "Switch to next query-type candidate.",
                previous_candidate_id=selected_candidate_id,
                next_candidate_id=next_candidate_id,
                next_query_type=route_candidates[current_route_index],
                reroute_count=state["routing"]["reroute_count"],
            )
            continue

        _update_route_plan_candidate(
            state,
            current_route_index,
            status="failed",
            detail=verification_result.get("reason", "Coordinator terminated this route candidate."),
            loop_count=state["control"]["global_loop_count"],
            verdict=verification_result.get("verdict", ""),
            decision=decision,
            error=pipeline_result.get("error", ""),
        )
        state["status"] = "failed"
        state["plan"]["steps"][0]["status"] = "failed"
        state["final_answer"] = {
            "success": False,
            "query": state["generation"]["current_query"],
            "result_text": pipeline_result.get("result", "") or verification_result.get("reason", "Execution failed."),
            "runtime_status": verification_result.get("runtime_status", ""),
        }
        break

    if state["status"] == "running":
        if state["routing"].get("query_type_candidates"):
            _update_route_plan_candidate(
                state,
                current_route_index,
                status="failed",
                detail="Coordinator exceeded the maximum number of loops.",
                loop_count=state["control"]["global_loop_count"],
                decision="loop_limit",
            )
        state["status"] = "failed"
        state["plan"]["steps"][0]["status"] = "failed"
        state["final_answer"] = {
            "success": False,
            "query": state["generation"]["current_query"],
            "result_text": "Coordinator exceeded the maximum number of loops.",
        }

    total_ms = _now_ms() - started_ms
    final_route_result = {
        "query_type": state["routing"]["selected_query_type"] or state["routing"]["initial_query_type"],
        "confidence": state["routing"]["confidence"],
        "reason": state["routing"]["reason"],
        "initial_query_type": state["routing"]["initial_query_type"],
        "selected_candidate_id": state["routing"]["selected_candidate_id"],
        "candidates": state["routing"]["route_candidates"] or state["routing"]["query_type_candidates"],
        "query_type_candidates": state["routing"]["query_type_candidates"],
        "route_plan": state["routing"]["route_plan"],
        "schema_plan_context": state["routing"]["schema_plan_context"],
        "schema_plan_proposals": state["routing"]["schema_plan_proposals"],
        "multi_step_plan": state["routing"]["multi_step_plan"],
        "multi_step_runtime": state["routing"]["multi_step_runtime"],
        "runtime_decision": state["routing"]["runtime_decision"],
        "planning_status": state["routing"]["planning_status"],
        "execution_policy": state["routing"]["execution_policy"],
        "clarification": state["routing"]["clarification"],
        "low_confidence": state["routing"]["low_confidence"],
        "signals": state["routing"]["signals"],
        "uncertainty_source": state["routing"]["uncertainty_source"],
        "should_keep_backup_route": state["routing"]["should_keep_backup_route"],
        "reroute_count": state["routing"]["reroute_count"],
    }
    coordinator_summary = {
        "task_id": state["task_id"],
        "mode": state["mode"],
        "status": state["status"],
        "active_step_id": state["plan"]["active_step_id"],
        "multi_step_plan_status": state["routing"].get("multi_step_plan", {}).get("status", ""),
        "multi_step_runtime_status": state["routing"].get("multi_step_runtime", {}).get("status", ""),
        "runtime_decision": state["routing"].get("runtime_decision", {}),
        "selected_candidate_id": state["routing"].get("selected_candidate_id", ""),
        "loop_count": state["control"]["global_loop_count"],
        "max_loops": state["control"]["max_global_loops"],
        "step_retry_count": state["repair"]["step_retry_count"],
        "reroute_count": state["routing"]["reroute_count"],
        "elapsed_ms": state["control"].get("elapsed_ms", 0),
        "time_budget_ms": state["control"].get("time_budget_ms", 0),
        "token_usage_estimate": state["control"].get("token_usage_estimate", 0),
        "token_budget": state["control"].get("token_budget", 0),
        "budget_exceeded": state["control"].get("budget_exceeded", False),
        "budget_reason": state["control"].get("budget_reason", ""),
        "budget_grace_used": state["control"].get("budget_grace_used", False),
        "verification_grace_ms": state["control"].get("verification_grace_ms", 0),
        "decision_history": state["memory"]["events"],
    }

    return {
        "question": question,
        "success": state["final_answer"]["success"],
        "route": final_route_result,
        "pipeline": pipeline_result,
        "verification": verification_result or state["verification"],
        "coordinator": coordinator_summary,
        "task_state": state,
        "timing": {
            "route_ms": round(route_ms, 2),
            "verification_ms": round(verification_ms, 2),
            "coordinator_ms": round(max(total_ms - route_ms, 0.0), 2),
            "total_ms": round(total_ms, 2),
        },
        "final_answer": state["final_answer"],
    }
