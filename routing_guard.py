import re
from typing import Any

from routing_common import (
    ALL_QUERY_TYPES,
    GENERIC_ENTITY_TOKENS,
    INTENT_KEYWORDS,
    QUERY_TYPES,
    ROUTING_STOPWORDS,
    TOP_K_CANDIDATES,
    normalize_text,
    tokenize,
)
from routing_evidence import extract_intent_signals
from routing_plan import build_keyword_route_plan, build_route_plan
from routing_scorer import confidence_from_scores

# 中文函数说明索引：
# - fallback_route(question, evidence, reason, uncertainty_source)：当 LLM 裁判缺失或不可用时，根据规则分数给出保守路由。
# - keyword_fallback(question, reason)：只用关键词和表面信号做最弱 fallback，主要保证极端情况下仍返回可执行 query_type。
# - guard_route(question, evidence, llm_result)：把 LLM 路由结果与规则 evidence 对齐，处理低置信、强图信号、强 SQL 信号、跨源信号等保护逻辑。


def fallback_route(question: str, evidence: dict[str, Any], reason: str, uncertainty_source: str) -> dict[str, Any]:
    ranked = sorted(
        evidence["scored_modalities"].items(),
        key=lambda item: (-item[1]["final_score"], item[0]),
    )
    top_candidates = ranked[:TOP_K_CANDIDATES]
    top1_score = top_candidates[0][1]["final_score"]
    top2_score = top_candidates[1][1]["final_score"] if len(top_candidates) > 1 else 0.0
    confidence = confidence_from_scores(
        top1_score,
        top2_score,
        evidence["intent_signals"]["has_cross_source_signal"],
    )
    task_analysis = evidence.get("task_analysis", evidence["intent_signals"].get("task_analysis", {}))
    task_mode = task_analysis.get("task_mode", evidence["intent_signals"]["task_mode_hint"])

    candidates = [
        {
            "query_type": modality,
            "score": round(payload["final_score"], 4),
            "confidence": confidence if index == 0 else "low",
            "reason": "；".join(payload["reason_parts"]),
        }
        for index, (modality, payload) in enumerate(top_candidates)
    ]
    should_keep_backup_route = (
        len(candidates) > 1
        and (top1_score - top2_score < 0.12 or confidence == "low")
    )
    result = {
        "task_mode": task_mode,
        "query_type": candidates[0]["query_type"],
        "candidates": candidates,
        "confidence": confidence,
        "reason": reason,
        "uncertainty_source": uncertainty_source,
        "should_keep_backup_route": should_keep_backup_route,
        "schema_plan_context": evidence.get("schema_plan_context", {}),
        "schema_plan_proposals": evidence.get("schema_plan_proposals", []),
        "signals": {
            "intent_signals": evidence["intent_signals"],
            "entity_hints": evidence["entity_hints"],
            "modality_scores": evidence["modality_scores"],
            "retrieval_hints": evidence["retrieval_hints"],
        },
    }
    result["route_plan"] = build_route_plan(result, evidence)
    return result


def keyword_fallback(question: str) -> dict[str, Any]:
    normalized = normalize_text(question)
    if any(keyword in normalized for keyword in INTENT_KEYWORDS["similarity"]):
        query_type = "vector"
    elif any(keyword in normalized for keyword in INTENT_KEYWORDS["document"]):
        query_type = "mql"
    elif any(keyword in normalized for keyword in INTENT_KEYWORDS["graph_relation"]):
        query_type = "cypher"
    else:
        query_type = "sql"

    candidates = [{"query_type": query_type, "score": 0.0, "confidence": "low", "reason": "关键词匹配兜底"}]
    result = {
        "task_mode": "multi_step_candidate" if any(keyword in normalized for keyword in INTENT_KEYWORDS["cross_source"]) else "single_step",
        "query_type": query_type,
        "candidates": candidates,
        "confidence": "low",
        "reason": "关键词匹配（兜底）",
        "uncertainty_source": "keyword_fallback",
        "should_keep_backup_route": False,
        "schema_plan_context": {},
        "schema_plan_proposals": [],
        "signals": {
            "intent_signals": extract_intent_signals(question),
            "entity_hints": {"counts": {"sql": 0, "cypher": 0, "mql": 0, "vector": 0}},
            "modality_scores": {mode: 0.0 for mode in ALL_QUERY_TYPES},
            "retrieval_hints": {"sql": {"best_resource": "", "score": 0.0}, "cypher": {"best_resource": "", "score": 0.0}},
        },
    }
    result["route_plan"] = build_keyword_route_plan(result)
    return result


def guard_route(route_result: dict[str, Any], evidence: dict[str, Any], context: dict[str, Any] | None = None) -> dict[str, Any]:
    context = context or {}
    normalized_question = normalize_text(evidence["question"])
    scored = evidence["scored_modalities"]
    signals = evidence["intent_signals"]
    retrieval_hints = evidence["retrieval_hints"]
    modality_scores = evidence["modality_scores"]
    ranked = sorted(scored.items(), key=lambda item: (-item[1]["final_score"], item[0]))
    if not ranked:
        return keyword_fallback(evidence["question"])

    top1_score = ranked[0][1]["final_score"]
    top2_score = ranked[1][1]["final_score"] if len(ranked) > 1 else 0.0
    rule_confidence = confidence_from_scores(
        top1_score,
        top2_score,
        evidence["intent_signals"]["has_cross_source_signal"],
    )
    task_analysis = evidence.get("task_analysis", evidence["intent_signals"].get("task_analysis", {}))
    task_mode = task_analysis.get("task_mode", route_result.get("task_mode", evidence["intent_signals"]["task_mode_hint"]))

    selected = route_result.get("query_type")
    if selected not in QUERY_TYPES:
        return fallback_route(
            evidence["question"],
            evidence,
            reason="Route Guard detected an invalid LLM routing result. Fallback to rule-based scoring.",
            uncertainty_source="guard_invalid_query_type",
        )

    fallback_candidates = [mode for mode, _ in ranked][:TOP_K_CANDIDATES]
    top_ranked_mode = ranked[0][0]
    top_ranked_score = ranked[0][1]["final_score"]
    selected_score = scored[selected]["final_score"]
    if selected != top_ranked_mode and top_ranked_score - selected_score >= 0.08:
        route_result["uncertainty_source"] = (
            route_result.get("uncertainty_source", "") or f"score_override:{selected}->{top_ranked_mode}"
        )
        selected = top_ranked_mode
        route_result["query_type"] = selected

    sql_score = modality_scores.get("sql", 0.0)
    cypher_score = modality_scores.get("cypher", 0.0)
    sql_retrieval = retrieval_hints.get("sql", {}).get("score", 0.0)
    cypher_retrieval = retrieval_hints.get("cypher", {}).get("score", 0.0)
    cypher_entity_count = evidence["entity_hints"].get("counts", {}).get("cypher", 0)
    sql_entity_count = evidence["entity_hints"].get("counts", {}).get("sql", 0)
    matched_sql_tokens = set(evidence["entity_hints"].get("matched_sql_tokens", []))
    matched_cypher_tokens = set(evidence["entity_hints"].get("matched_cypher_tokens", []))
    cypher_exclusive_schema_tokens = {
        token
        for token in matched_cypher_tokens - matched_sql_tokens
        if token not in GENERIC_ENTITY_TOKENS and token not in ROUTING_STOPWORDS
    }
    cypher_schema_evidence = (
        cypher_entity_count > sql_entity_count
        or (
            cypher_entity_count > 0
            and cypher_retrieval >= sql_retrieval + 0.05
            and cypher_score >= sql_score + 0.03
        )
        or (
            cypher_entity_count >= sql_entity_count
            and cypher_retrieval >= 0.80
            and cypher_retrieval >= sql_retrieval + 0.25
        )
    )

    if (
        selected == "cypher"
        and not signals["has_graph_relation"]
        and not signals["has_multi_hop_relation"]
        and not signals["has_similarity_search"]
        and not signals["has_document_pattern"]
        and not cypher_schema_evidence
        and sql_score + 0.02 >= cypher_score
    ):
        selected = "sql"
        route_result["query_type"] = selected
        route_result["uncertainty_source"] = (
            route_result.get("uncertainty_source", "") or "guard_neutral_sql_bias"
        )

    clear_cypher_support = (
        cypher_schema_evidence
        or cypher_entity_count > sql_entity_count
        or cypher_retrieval >= sql_retrieval + 0.10
        or cypher_score >= sql_score + 0.02
    )
    cypher_competitive = cypher_score + 0.05 >= sql_score or cypher_retrieval >= sql_retrieval + 0.25
    normalized_question = str(evidence.get("question") or "").lower()
    explicit_sql_table_request = bool(
        re.search(r"\bsql\s+(?:attribute\s+)?table\b", normalized_question)
        or re.search(r"\b(?:attribute|profile|roster|ranking)\s+(?:sql\s+)?table\b", normalized_question)
        or re.search(r"\b(?:attribute|profile|roster|ranking)\s+database\b", normalized_question)
    )
    cypher_aggregation_schema_support = (
        signals["has_aggregation"]
        and bool(cypher_exclusive_schema_tokens)
        and (
            cypher_entity_count > sql_entity_count
            or (cypher_entity_count >= sql_entity_count and len(cypher_exclusive_schema_tokens) >= 2)
        )
        and cypher_retrieval + 0.01 >= sql_retrieval
        and cypher_score + 0.20 >= sql_score
    )

    if (
        selected == "sql"
        and not explicit_sql_table_request
        and (signals["has_graph_relation"] or signals["has_multi_hop_relation"] or cypher_schema_evidence)
        and clear_cypher_support
        and (
            (signals["has_graph_relation"] and cypher_competitive)
            or (signals["has_multi_hop_relation"] and cypher_competitive)
            or (cypher_schema_evidence and cypher_competitive)
            or (cypher_schema_evidence and cypher_retrieval >= sql_retrieval + 0.25)
        )
    ):
        selected = "cypher"
        route_result["query_type"] = selected
        route_result["uncertainty_source"] = (
            route_result.get("uncertainty_source", "") or "guard_graph_schema_evidence"
        )

    if selected == "sql" and not explicit_sql_table_request and cypher_aggregation_schema_support:
        selected = "cypher"
        route_result["query_type"] = selected
        route_result["uncertainty_source"] = (
            route_result.get("uncertainty_source", "") or "guard_cypher_aggregation_schema_evidence"
        )

    llm_candidates = [candidate for candidate in route_result.get("candidates", []) if candidate in QUERY_TYPES]
    if selected not in llm_candidates:
        llm_candidates.insert(0, selected)
    else:
        llm_candidates = [selected] + [candidate for candidate in llm_candidates if candidate != selected]
    merged_candidates = []
    for candidate in llm_candidates + fallback_candidates:
        if candidate not in merged_candidates:
            merged_candidates.append(candidate)
    merged_candidates = merged_candidates[:TOP_K_CANDIDATES]

    candidate_payloads = []
    for candidate in merged_candidates:
        score = scored[candidate]["final_score"]
        candidate_payloads.append(
            {
                "query_type": candidate,
                "score": round(score, 4),
                "confidence": rule_confidence if candidate == merged_candidates[0] else "low",
                "reason": "；".join(scored[candidate]["reason_parts"]),
            }
        )

    if candidate_payloads and candidate_payloads[0]["query_type"] != ranked[0][0]:
        route_result["uncertainty_source"] = (
            route_result.get("uncertainty_source", "") or "llm-vs-score conflict"
        )

    should_keep_backup_route = (
        len(candidate_payloads) > 1
        and (top1_score - top2_score < 0.12 or rule_confidence == "low" or route_result.get("confidence") == "low")
    )
    if context.get("reroute_count", 0) > 0 and len(candidate_payloads) > 1:
        should_keep_backup_route = True

    result = {
        "task_mode": task_mode,
        "query_type": candidate_payloads[0]["query_type"],
        "candidates": candidate_payloads,
        "confidence": rule_confidence if route_result.get("confidence") not in {"high", "medium", "low"} else route_result["confidence"],
        "reason": route_result.get("reason", ""),
        "uncertainty_source": route_result.get("uncertainty_source", ""),
        "should_keep_backup_route": should_keep_backup_route,
        "schema_plan_context": evidence.get("schema_plan_context", {}),
        "schema_plan_proposals": route_result.get("schema_plan_proposals") or evidence.get("schema_plan_proposals", []),
        "signals": {
            "intent_signals": evidence["intent_signals"],
            "entity_hints": evidence["entity_hints"],
            "modality_scores": evidence["modality_scores"],
            "retrieval_hints": evidence["retrieval_hints"],
        },
    }
    result["route_plan"] = build_route_plan(result, evidence)
    return result
