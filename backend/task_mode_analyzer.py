import re
from typing import Any

from .routing_common import INTENT_KEYWORDS, contains_any, normalize_text

# 中文函数说明索引：
# - _match_patterns(normalized, patterns)：用正则抽取顺序依赖、跨源、嵌套等自然语言信号。
# - analyze_task_mode(question)：规则初判入口，输出 task_mode、query_strategy、requires_system_multi_step 等字段。
#   路由证据构建阶段会在此基础上调用 schema-aware LLM judge 做校正。
# 状态说明：
# - single_query：普通单步问题，直接走单条 SQL/Cypher。
# - single_query_with_nested_logic：自然语言像单步，但查询内部需要嵌套/聚合排序/作用域结构。
# - multi_step_candidate：可能需要多个步骤或显式 first/then。
# - cross_source_candidate：可能跨 SQL/Cypher 或多个数据库，需要 Step Planner + bridge policy。


SUPERLATIVE_TERMS = [
    "most",
    "least",
    "highest",
    "lowest",
    "largest",
    "smallest",
    "greatest",
    "maximum",
    "minimum",
    "max",
    "min",
    "oldest",
    "youngest",
    "latest",
    "earliest",
]


def _match_patterns(normalized: str, patterns: list[tuple[str, str]]) -> list[dict[str, str]]:
    matches: list[dict[str, str]] = []
    for signal, pattern in patterns:
        for match in re.finditer(pattern, normalized):
            matches.append(
                {
                    "signal": signal,
                    "text": match.group(0),
                }
            )
    return matches


def analyze_task_mode(question: str) -> dict[str, Any]:
    """Classify task mode with deterministic natural-language signals.

    This function is the rule-based preliminary pass. The full routing flow may
    refine its output with compact schema evidence and an LLM judge.
    """
    normalized = normalize_text(question)
    _, cross_source_hits = contains_any(normalized, INTENT_KEYWORDS["cross_source"])

    cross_source_patterns = [
        ("explicit_cross_database", r"\bacross(?:\s+the)?(?:\s+\w+){0,2}\s+(?:databases|datasets|data\s+sources|sources)\b"),
        ("sql_then_graph", r"\b(?:sql|table|relational)\b.{0,40}\b(?:then|after that)\b.{0,40}\b(?:graph|cypher|neo4j)\b"),
        ("graph_then_sql", r"\b(?:graph|cypher|neo4j)\b.{0,40}\b(?:then|after that)\b.{0,40}\b(?:sql|table|relational)\b"),
        ("tables_then_graph", r"\b(?:tables?|database)\b.{0,80}\b(?:then|after that)\b.{0,80}\b(?:graph|cypher|neo4j)\b"),
        ("graph_then_table", r"\b(?:graph|cypher|neo4j)\b.{0,80}\b(?:then|after that)\b.{0,80}\b(?:tables?|database|attribute\s+table)\b"),
        ("use_target_source", r"\b(?:then|after that)\b.{0,20}\buse\s+the\s+[^.?!]{0,50}\b(?:graph|cypher|neo4j|sql|tables?|database)\b"),
        ("modality_named_sequence", r"\b(?:in sql|in graph|in cypher)\b.{0,80}\b(?:then|after that|based on)\b"),
    ]
    explicit_sequence_patterns = [
        ("first_then", r"\bfirst\b.{0,120}\b(?:then|after that|next)\b"),
        ("then_find", r"\bthen\b.{0,40}\b(?:find|return|show|list|get|count|calculate)\b"),
        ("based_on_result", r"\b(?:based on|according to)\s+(?:the\s+)?(?:result|results|output|answer)\b"),
        ("use_previous_result", r"\b(?:use|using)\s+(?:that|those|the previous|the result)\b"),
    ]
    nested_single_query_patterns = [
        (
            "modifier_superlative_dependency",
            r"\b(?:from|of|for|in|by|under|belonging to|associated with)\b.{0,60}\b(?:with|having|has|have)\s+the\s+(?:"
            + "|".join(SUPERLATIVE_TERMS)
            + r")\b",
        ),
        (
            "relative_superlative_dependency",
            r"\b(?:whose|that|which)\b.{0,60}\b(?:has|have|with|having)\s+the\s+(?:"
            + "|".join(SUPERLATIVE_TERMS)
            + r")\b",
        ),
        (
            "average_comparison_dependency",
            r"\b(?:more than|less than|higher than|lower than|above|below)\s+(?:the\s+)?average\b",
        ),
        (
            "aggregate_reference_dependency",
            r"\b(?:equal to|equals|greater than|less than|at least|at most)\s+(?:the\s+)?(?:avg|average|sum|total|count|maximum|minimum|max|min)\b",
        ),
        (
            "one_with_superlative_dependency",
            r"\b(?:the one|the ones|those|that)\s+with\s+the\s+(?:"
            + "|".join(SUPERLATIVE_TERMS)
            + r")\b",
        ),
    ]

    cross_source_matches = _match_patterns(normalized, cross_source_patterns)
    sequence_matches = _match_patterns(normalized, explicit_sequence_patterns)
    nested_matches = _match_patterns(normalized, nested_single_query_patterns)

    for hit in cross_source_hits:
        if hit in {"across databases", "in sql", "in graph"}:
            cross_source_matches.append({"signal": "cross_source_keyword", "text": hit})

    # Direct superlatives such as "Which singer has the highest age?" are still
    # single-query requests, not nested dependencies.
    direct_superlative = bool(
        re.search(r"\b(?:which|what|who)\b.{0,60}\b(?:has|have|is|are)\s+the\s+(?:" + "|".join(SUPERLATIVE_TERMS) + r")\b", normalized)
    )
    only_direct_relative_noise = (
        direct_superlative
        and nested_matches
        and all(match["signal"] == "relative_superlative_dependency" for match in nested_matches)
    )
    if direct_superlative and (not nested_matches or only_direct_relative_noise) and not sequence_matches and not cross_source_matches:
        return {
            "task_mode": "single_step",
            "query_strategy": "single_query",
            "confidence": "medium",
            "requires_system_multi_step": False,
            "can_use_single_query": True,
            "recommended_generation_strategy": "direct_query",
            "signals": [{"signal": "direct_superlative", "text": question.strip()}],
            "reason": "Direct superlative can usually be expressed as one ordered or aggregate query.",
        }

    if cross_source_matches:
        return {
            "task_mode": "multi_step_candidate",
            "query_strategy": "cross_source_candidate",
            "confidence": "high",
            "requires_system_multi_step": True,
            "can_use_single_query": False,
            "recommended_generation_strategy": "defer_to_future_planner",
            "signals": cross_source_matches + sequence_matches + nested_matches,
            "reason": "The question contains explicit cross-source or cross-modality dependency signals.",
        }

    if sequence_matches:
        return {
            "task_mode": "multi_step_candidate",
            "query_strategy": "multi_step_candidate",
            "confidence": "high",
            "requires_system_multi_step": True,
            "can_use_single_query": False,
            "recommended_generation_strategy": "defer_to_future_planner",
            "signals": sequence_matches + nested_matches,
            "reason": "The question asks for sequential work or references a previous result.",
        }

    if nested_matches:
        return {
            "task_mode": "single_step",
            "query_strategy": "single_query_with_nested_logic",
            "confidence": "medium",
            "requires_system_multi_step": False,
            "can_use_single_query": True,
            "recommended_generation_strategy": "use_nested_query_or_cte",
            "signals": nested_matches,
            "reason": "The question has an implicit dependency that should usually be represented inside one query.",
        }

    return {
        "task_mode": "single_step",
        "query_strategy": "single_query",
        "confidence": "medium",
        "requires_system_multi_step": False,
        "can_use_single_query": True,
        "recommended_generation_strategy": "direct_query",
        "signals": [],
        "reason": "No explicit cross-step or nested dependency signal was detected.",
    }
