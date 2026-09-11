import json
import re
from typing import Any

from openai import OpenAI

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL
from llm_utils import call_chat_completion


llm = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

# 中文函数说明索引：
# - refine_task_mode_with_schema(...)：基于规则初判、schema context 和路由证据调用 LLM 校正任务模式。
# - _guard_refined_task_analysis(...)：对 LLM 任务模式判断做确定性保护，避免强跨源/强顺序信号被错误降级。


QUERY_STRATEGIES = {
    "single_query",
    "single_query_with_nested_logic",
    "multi_step_candidate",
    "cross_source_candidate",
}

MULTI_STEP_STRATEGIES = {"multi_step_candidate", "cross_source_candidate"}
SINGLE_STEP_STRATEGIES = {"single_query", "single_query_with_nested_logic"}

STRONG_CROSS_SOURCE_SIGNALS = {
    "explicit_cross_database",
    "sql_then_graph",
    "graph_then_sql",
    "tables_then_graph",
    "graph_then_table",
    "use_target_source",
    "modality_named_sequence",
    "cross_source_keyword",
}

STRONG_SEQUENCE_SIGNALS = {
    "first_then",
    "then_find",
    "based_on_result",
    "use_previous_result",
}


def should_refine_task_mode_with_schema(
    question: str,
    rule_analysis: dict[str, Any],
    evidence: dict[str, Any],
    schema_context: dict[str, Any],
) -> tuple[bool, str]:
    """Return whether schema-aware LLM refinement is worth its latency cost."""
    normalized = question.lower()
    strategy = rule_analysis.get("query_strategy", "single_query")
    confidence = rule_analysis.get("confidence", "medium")
    rule_signals = _signals(rule_analysis)
    has_both_schema = _has_both_sql_and_cypher_schema_evidence(evidence, schema_context)

    if strategy == "cross_source_candidate":
        return False, "rule_already_cross_source_candidate"
    if strategy == "single_query_with_nested_logic":
        return False, "rule_already_nested_single_query"
    if (rule_signals & STRONG_SEQUENCE_SIGNALS) and confidence == "high" and not has_both_schema:
        return False, "high_confidence_sequence_without_cross_schema_evidence"

    average_boundary = (
        strategy == "single_query"
        and any(term in normalized for term in ["average", "avg"])
        and any(
            term in normalized
            for term in [
                "older than",
                "younger than",
                "more than",
                "less than",
                "higher than",
                "lower than",
                "above",
                "below",
            ]
        )
    )
    if average_boundary:
        return True, "average_comparison_boundary"

    explicit_modality_terms = any(
        term in normalized
        for term in [
            " sql",
            "sql ",
            "table",
            "tables",
            "database",
            "graph",
            "cypher",
            "neo4j",
        ]
    )
    if has_both_schema and (explicit_modality_terms or strategy == "multi_step_candidate"):
        return True, "schema_cross_source_boundary"

    if confidence == "low":
        return True, "low_confidence_task_mode"

    return False, "rule_confident_enough"


def _call_llm(prompt: str) -> str:
    return call_chat_completion(llm, prompt, label="task_mode_schema_judge")


def _compact_schema_context(schema_context: dict[str, Any]) -> dict[str, Any]:
    catalog = schema_context.get("catalog", {}) if isinstance(schema_context, dict) else {}
    sql_catalog = catalog.get("sql", []) if isinstance(catalog, dict) else []
    cypher_catalog = catalog.get("cypher", []) if isinstance(catalog, dict) else []
    return {
        "top_sql_schema": schema_context.get("sql", [])[:4] if isinstance(schema_context, dict) else [],
        "top_cypher_schema": schema_context.get("cypher", [])[:4] if isinstance(schema_context, dict) else [],
        "catalog_summary": {
            "sql_resource_count": len(sql_catalog) if isinstance(sql_catalog, list) else 0,
            "cypher_resource_count": len(cypher_catalog) if isinstance(cypher_catalog, list) else 0,
            "top_sql_resources": [
                item.get("resource_id", "")
                for item in (schema_context.get("sql", [])[:4] if isinstance(schema_context, dict) else [])
                if isinstance(item, dict)
            ],
            "top_cypher_resources": [
                item.get("resource_id", "")
                for item in (schema_context.get("cypher", [])[:4] if isinstance(schema_context, dict) else [])
                if isinstance(item, dict)
            ],
        },
        "limits": schema_context.get("limits", {}) if isinstance(schema_context, dict) else {},
    }


def _signals(task_analysis: dict[str, Any]) -> set[str]:
    return {
        item.get("signal", "")
        for item in task_analysis.get("signals", []) or []
        if isinstance(item, dict)
    }


def _task_mode_for_strategy(query_strategy: str) -> str:
    return "multi_step_candidate" if query_strategy in MULTI_STEP_STRATEGIES else "single_step"


def _strategy_from_payload(payload: dict[str, Any], fallback: str) -> str:
    strategy = str(payload.get("query_strategy") or fallback or "single_query").strip()
    return strategy if strategy in QUERY_STRATEGIES else fallback


def _bool_for_strategy(query_strategy: str) -> tuple[bool, bool, str]:
    if query_strategy == "single_query_with_nested_logic":
        return False, True, "use_nested_query_or_cte"
    if query_strategy == "multi_step_candidate":
        return True, False, "defer_to_multi_step_planner"
    if query_strategy == "cross_source_candidate":
        return True, False, "defer_to_multi_step_planner_with_bridge"
    return False, True, "direct_query"


def _normalize_llm_result(raw_payload: dict[str, Any], rule_analysis: dict[str, Any]) -> dict[str, Any]:
    fallback_strategy = rule_analysis.get("query_strategy", "single_query")
    query_strategy = _strategy_from_payload(raw_payload, fallback_strategy)
    requires_multi_step, can_use_single_query, generation_strategy = _bool_for_strategy(query_strategy)
    confidence = raw_payload.get("confidence", "low")
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"
    signals = raw_payload.get("signals", [])
    if not isinstance(signals, list):
        signals = []
    return {
        "task_mode": _task_mode_for_strategy(query_strategy),
        "query_strategy": query_strategy,
        "confidence": confidence,
        "requires_system_multi_step": requires_multi_step,
        "can_use_single_query": can_use_single_query,
        "recommended_generation_strategy": raw_payload.get("recommended_generation_strategy") or generation_strategy,
        "signals": signals,
        "reason": raw_payload.get("reason", "Schema-aware task mode judge returned no reason."),
        "source": "schema_aware_llm_judge",
        "llm_task_analysis": raw_payload,
    }


def _with_guard_reason(analysis: dict[str, Any], reason: str, llm_analysis: dict[str, Any] | None = None) -> dict[str, Any]:
    guarded = dict(analysis)
    guarded["source"] = "rule_guard_after_schema_aware_judge"
    guarded["schema_aware_guard_reason"] = reason
    if llm_analysis:
        guarded["llm_task_analysis"] = llm_analysis
    return guarded


def _has_both_sql_and_cypher_schema_evidence(evidence: dict[str, Any], schema_context: dict[str, Any]) -> bool:
    entity_counts = evidence.get("entity_hints", {}).get("counts", {})
    has_entity_overlap = entity_counts.get("sql", 0) > 0 and entity_counts.get("cypher", 0) > 0
    has_ranked_context = bool(schema_context.get("sql")) and bool(schema_context.get("cypher"))
    return has_entity_overlap or has_ranked_context


def _guard_refined_task_analysis(
    rule_analysis: dict[str, Any],
    refined_analysis: dict[str, Any],
    evidence: dict[str, Any],
    schema_context: dict[str, Any],
) -> dict[str, Any]:
    rule_strategy = rule_analysis.get("query_strategy", "single_query")
    refined_strategy = refined_analysis.get("query_strategy", "single_query")
    rule_signals = _signals(rule_analysis)

    has_strong_cross_source = bool(rule_signals & STRONG_CROSS_SOURCE_SIGNALS)
    has_strong_sequence = bool(rule_signals & STRONG_SEQUENCE_SIGNALS)
    refined_is_single = refined_strategy in SINGLE_STEP_STRATEGIES
    refined_is_multi = refined_strategy in MULTI_STEP_STRATEGIES

    if has_strong_cross_source and refined_is_single:
        return _with_guard_reason(
            rule_analysis,
            "LLM attempted to downgrade explicit cross-source signals to single-query.",
            refined_analysis,
        )

    if has_strong_sequence and refined_is_single and rule_analysis.get("confidence") == "high":
        return _with_guard_reason(
            rule_analysis,
            "LLM attempted to downgrade high-confidence sequential dependency signals.",
            refined_analysis,
        )

    if refined_is_multi and rule_strategy in SINGLE_STEP_STRATEGIES:
        has_schema_support = _has_both_sql_and_cypher_schema_evidence(evidence, schema_context)
        if refined_analysis.get("confidence") == "low" and not has_schema_support:
            return _with_guard_reason(
                rule_analysis,
                "Low-confidence LLM upgrade to multi-step lacked schema support.",
                refined_analysis,
            )

    merged = dict(refined_analysis)
    merged["rule_task_analysis"] = rule_analysis
    return merged


def refine_task_mode_with_schema(
    question: str,
    rule_analysis: dict[str, Any],
    evidence: dict[str, Any],
    schema_context: dict[str, Any],
) -> dict[str, Any]:
    """Refine rule-based task mode using compact schema evidence and an LLM judge."""
    compact_context = _compact_schema_context(schema_context)
    prompt = f"""You are the Task Mode Judge inside a multi-agent SQL/Cypher query system.
You must decide the execution shape, not the final query text.

Use the natural language question, rule-based preliminary analysis, modality evidence, and compact schema evidence.
Do not decide from the question alone.

Question:
{question}

Rule-based preliminary task analysis:
{json.dumps(rule_analysis, ensure_ascii=False)}

Intent signals:
{json.dumps(evidence.get("intent_signals", {}), ensure_ascii=False)}

Entity hints:
{json.dumps(evidence.get("entity_hints", {}), ensure_ascii=False)}

Modality scores:
{json.dumps(evidence.get("modality_scores", {}), ensure_ascii=False)}

Retrieval hints:
{json.dumps(evidence.get("retrieval_hints", {}), ensure_ascii=False)}

Compact schema context:
{json.dumps(compact_context, ensure_ascii=False)}

Task mode definitions:
- single_query: one SQL or one Cypher query can answer directly.
- single_query_with_nested_logic: one SQL/Cypher query can answer, but it needs nested logic, grouping, ranking, comparison to aggregate, or scoped aggregation.
- multi_step_candidate: the question requires sequential dependency where a previous result must be consumed by a later query step.
- cross_source_candidate: the question requires or strongly benefits from using both SQL and Cypher resources, with entity/attribute/relationship information passed across sources.

Important rules:
- Do not mark a normal aggregate, ranking, or nested SQL/Cypher query as system-level multi-step if one query can express it.
- Mark cross_source_candidate only when SQL-side and Cypher-side evidence are both needed, or when the question explicitly asks to cross sources.
- Use schema evidence to correct the rule-based preliminary analysis when the question alone is ambiguous.
- Prefer the safer mode when the evidence shows true dependency; prefer single_query_with_nested_logic for same-source nested logic.

Return JSON only:
{{
  "query_strategy": "single_query | single_query_with_nested_logic | multi_step_candidate | cross_source_candidate",
  "confidence": "high | medium | low",
  "requires_system_multi_step": true,
  "can_use_single_query": false,
  "recommended_generation_strategy": "direct_query | use_nested_query_or_cte | defer_to_multi_step_planner | defer_to_multi_step_planner_with_bridge",
  "signals": [{{"signal": "schema_cross_source_dependency", "text": "short evidence"}}],
  "reason": "one concise reason grounded in question and schema evidence"
}}"""
    raw = _call_llm(prompt)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError("Task Mode Judge did not return JSON.")
    payload = json.loads(match.group())
    if not isinstance(payload, dict):
        raise ValueError("Task Mode Judge returned non-object JSON.")
    refined = _normalize_llm_result(payload, rule_analysis)
    return _guard_refined_task_analysis(rule_analysis, refined, evidence, schema_context)
