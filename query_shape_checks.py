import re
from typing import Any

# 中文函数说明索引：
# - _normalize_query(query)：统一小写和空白，降低 SQL/Cypher 形态检查的词面差异。
# - _signal_names(signals)：抽取命中的结构 marker 名称，便于 verification reason 和回归报告解释。
# - _sql_nested_markers(query)：识别 SQL 中 CTE、子查询、grouped rank、ORDER BY LIMIT 等嵌套/排序 Top1 结构。
# - _cypher_nested_markers(query)：识别 Cypher WITH scope、collect、path 等复杂图查询结构。
# - _required_markers(generation_contract, task_analysis)：根据 task mode / generation contract 判断当前查询是否必须具备嵌套形态。
# - assess_generation_contract(...)：返回 required/satisfied/failure_type/confidence，供 Verification 决定是否 REGENERATE。


NESTED_QUERY_STRATEGY = "single_query_with_nested_logic"


def _normalize_query(query: str) -> str:
    lowered = query.lower()
    lowered = re.sub(r"'(?:''|[^'])*'", "''", lowered)
    lowered = re.sub(r'"(?:""|[^"])*"', '""', lowered)
    lowered = re.sub(r"`[^`]*`", "``", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _signal_names(task_analysis: dict[str, Any]) -> set[str]:
    return {
        str(item.get("signal", ""))
        for item in task_analysis.get("signals", []) or []
        if isinstance(item, dict)
    }


def _sql_nested_markers(normalized_query: str) -> list[str]:
    markers: list[str] = []
    if re.search(r"\bwith\b.+\bas\s*\(", normalized_query):
        markers.append("cte")
    if re.search(r"\(\s*select\b", normalized_query):
        markers.append("subquery")
    if re.search(r"\bover\s*\(", normalized_query):
        markers.append("window_function")
    if re.search(r"\bhaving\b", normalized_query):
        markers.append("having")
    if re.search(r"\b(join|from)\s*\(\s*select\b", normalized_query):
        markers.append("derived_table")
    if (
        re.search(r"\bgroup\s+by\b", normalized_query)
        and re.search(r"\border\s+by\b", normalized_query)
        and re.search(r"\blimit\b", normalized_query)
        and re.search(r"\b(count|sum|avg|min|max)\s*\(", normalized_query)
    ):
        markers.append("grouped_aggregate_rank")
    if re.search(r"\border\s+by\b", normalized_query) and re.search(r"\blimit\b", normalized_query):
        markers.append("order_limit")
    return markers


def _cypher_nested_markers(normalized_query: str) -> list[str]:
    markers: list[str] = []
    has_with = re.search(r"\bwith\b", normalized_query) is not None
    has_aggregate = re.search(r"\b(count|avg|sum|min|max|collect)\s*\(", normalized_query) is not None
    if has_with:
        markers.append("with_scope")
    if has_with and has_aggregate:
        markers.append("with_aggregation")
    if re.search(r"\bcollect\s*\(", normalized_query) or re.search(r"\bunwind\b", normalized_query):
        markers.append("collect_unwind")
    if re.search(r"\border\s+by\b.+\blimit\b", normalized_query):
        markers.append("order_limit")
    if re.search(r"\[[^\]]*\|[^\]]*\]", normalized_query):
        markers.append("pattern_comprehension")
    return markers


def _required_markers(query_type: str, task_analysis: dict[str, Any]) -> list[str]:
    signal_names = _signal_names(task_analysis)
    if query_type == "sql":
        if signal_names & {"average_comparison_dependency", "aggregate_reference_dependency"}:
            return ["subquery", "cte", "window_function", "having", "derived_table"]
        if "modifier_superlative_dependency" in signal_names:
            return [
                "subquery",
                "cte",
                "window_function",
                "having",
                "derived_table",
                "grouped_aggregate_rank",
                "order_limit",
            ]
        return ["subquery", "cte", "window_function", "having", "derived_table", "grouped_aggregate_rank"]
    if query_type == "cypher":
        if signal_names & {"average_comparison_dependency", "aggregate_reference_dependency"}:
            return ["with_aggregation", "collect_unwind", "pattern_comprehension"]
        return ["with_scope", "with_aggregation", "collect_unwind", "order_limit", "pattern_comprehension"]
    return []


def assess_generation_contract(
    *,
    query: str,
    query_type: str,
    generation_contract: dict[str, Any] | None = None,
    task_analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Check whether a generated query appears to satisfy its generation contract.

    This is a deterministic, conservative shape check. It does not try to prove
    semantic correctness; it only detects the common failure mode where the route
    and prompt requested nested logic but the generated query is a plain lookup.
    """
    contract = generation_contract or {}
    analysis = task_analysis or {}
    query_strategy = (
        contract.get("query_strategy")
        or analysis.get("query_strategy")
        or "single_query"
    )
    recommended = (
        contract.get("recommended_generation_strategy")
        or analysis.get("recommended_generation_strategy")
        or "direct_query"
    )
    normalized_type = (query_type or "").lower()

    if query_strategy != NESTED_QUERY_STRATEGY:
        return {
            "required": False,
            "satisfied": True,
            "query_type": normalized_type,
            "query_strategy": query_strategy,
            "recommended_generation_strategy": recommended,
            "detected_markers": [],
            "required_markers": [],
            "failure_type": "none",
            "confidence": "high",
            "reason": "No nested-logic generation contract is active.",
        }

    normalized_query = _normalize_query(query)
    if normalized_type == "sql":
        detected = _sql_nested_markers(normalized_query)
    elif normalized_type == "cypher":
        detected = _cypher_nested_markers(normalized_query)
    else:
        detected = []
    required = _required_markers(normalized_type, analysis)
    satisfied = bool(set(detected) & set(required))

    return {
        "required": True,
        "satisfied": satisfied,
        "query_type": normalized_type,
        "query_strategy": query_strategy,
        "recommended_generation_strategy": recommended,
        "detected_markers": detected,
        "required_markers": required,
        "failure_type": "none" if satisfied else "nested_logic_contract_unmet",
        "confidence": "medium" if satisfied else "high",
        "reason": (
            "Generated query contains a nested-logic shape marker."
            if satisfied
            else "Nested-logic contract is active, but the generated query looks like a flat query."
        ),
    }
