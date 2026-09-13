import re
from typing import Any

from .routing_common import (
    GENERIC_ENTITY_TOKENS,
    INTENT_KEYWORDS,
    contains_any,
    load_schema_indexes,
    meaningful_tokens,
    normalize_text,
    parse_schema_summary_terms,
    tokenize,
)
from .schema_plan_proposer import build_deterministic_schema_plan_proposals, build_schema_plan_context
from .task_mode_analyzer import analyze_task_mode
from .task_mode_schema_judge import refine_task_mode_with_schema, should_refine_task_mode_with_schema

# 中文函数说明索引：
# - extract_intent_signals(question)：从自然语言中抽取聚合、排序、过滤、图关系、跨源等意图信号；这些信号是规则路由和 task mode 判断的基础。
# - extract_entity_hints(question, schema_summary)：根据问题 token 与 schema/index 术语的重合，推断 SQL 表/列、图节点/关系等实体线索。
# - build_route_evidence(question, schema_summary)：汇总 intent_signals、entity_hints、modality_scores、schema-aware task analysis 和 retrieval_hints，形成 route_plan 可解释 evidence。


def extract_intent_signals(question: str) -> dict[str, Any]:
    normalized = normalize_text(question)
    tokens = tokenize(question)

    has_aggregation, aggregation_hits = contains_any(normalized, INTENT_KEYWORDS["aggregation"])
    has_sorting, sorting_hits = contains_any(normalized, INTENT_KEYWORDS["sorting"])
    has_filtering, filtering_hits = contains_any(normalized, INTENT_KEYWORDS["filtering"])
    has_graph_relation, graph_hits = contains_any(normalized, INTENT_KEYWORDS["graph_relation"])
    has_similarity_search, similarity_hits = contains_any(normalized, INTENT_KEYWORDS["similarity"])
    has_document_pattern, document_hits = contains_any(normalized, INTENT_KEYWORDS["document"])
    has_cross_source_signal, cross_source_hits = contains_any(normalized, INTENT_KEYWORDS["cross_source"])

    if re.search(r"\bbetween\s+[-+]?\d+(?:\.\d+)?\s+and\s+[-+]?\d+(?:\.\d+)?\b", normalized):
        graph_hits = [hit for hit in graph_hits if hit != "between"]
        has_graph_relation = bool(graph_hits)

    relation_tokens = {
        "relationship",
        "relationships",
        "relation",
        "relations",
        "connected",
        "related",
        "linked",
        "associated",
        "through",
        "via",
        "between",
    }
    if "between" not in graph_hits:
        relation_tokens.discard("between")
    relation_token_count = sum(1 for token in tokens if token in relation_tokens)
    has_multi_hop_relation = relation_token_count >= 2 or ("who are the" in normalized and has_graph_relation)
    task_analysis = analyze_task_mode(question)
    has_nested_dependency = task_analysis["query_strategy"] in {
        "single_query_with_nested_logic",
        "multi_step_candidate",
        "cross_source_candidate",
    }
    task_mode_hint = task_analysis["task_mode"]

    return {
        "has_aggregation": has_aggregation,
        "has_sorting": has_sorting,
        "has_filtering": has_filtering,
        "has_graph_relation": has_graph_relation,
        "has_multi_hop_relation": has_multi_hop_relation,
        "has_similarity_search": has_similarity_search,
        "has_document_pattern": has_document_pattern,
        "has_nested_dependency": has_nested_dependency,
        "has_cross_source_signal": has_cross_source_signal,
        "task_mode_hint": task_mode_hint,
        "task_analysis": task_analysis,
        "matched_keywords": sorted(
            set(
                aggregation_hits
                + sorting_hits
                + filtering_hits
                + graph_hits
                + similarity_hits
                + document_hits
                + cross_source_hits
            )
        ),
    }


def extract_entity_hints(question: str, schema_summary: str = "") -> dict[str, Any]:
    question_tokens = meaningful_tokens(question)
    schema_indexes = load_schema_indexes()
    summary_terms = parse_schema_summary_terms(schema_summary)

    matched_sql_entities: set[str] = set()
    matched_cypher_entities: set[str] = set()
    matched_mql_entities: set[str] = set()
    matched_vector_entities: set[str] = set()
    matched_sql_tokens: set[str] = set()
    matched_cypher_tokens: set[str] = set()
    matched_mql_tokens: set[str] = set()
    matched_vector_tokens: set[str] = set()

    for terms in schema_indexes["sql"].values():
        for term in terms:
            term_tokens = meaningful_tokens(term)
            overlap = term_tokens & question_tokens
            if overlap:
                matched_sql_entities.add(term)
                matched_sql_tokens.update(overlap)

    for terms in schema_indexes["cypher"].values():
        for term in terms:
            term_tokens = meaningful_tokens(term)
            overlap = term_tokens & question_tokens
            if overlap:
                matched_cypher_entities.add(term)
                matched_cypher_tokens.update(overlap)

    for mode, target in [
        ("sql", matched_sql_entities),
        ("cypher", matched_cypher_entities),
        ("mql", matched_mql_entities),
        ("vector", matched_vector_entities),
    ]:
        for term in summary_terms[mode]:
            if term in question_tokens and term not in GENERIC_ENTITY_TOKENS:
                target.add(term)
                if mode == "sql":
                    matched_sql_tokens.add(term)
                elif mode == "cypher":
                    matched_cypher_tokens.add(term)
                elif mode == "mql":
                    matched_mql_tokens.add(term)
                elif mode == "vector":
                    matched_vector_tokens.add(term)

    counts = {
        "sql": len(matched_sql_tokens),
        "cypher": len(matched_cypher_tokens),
        "mql": len(matched_mql_tokens),
        "vector": len(matched_vector_tokens),
    }
    max_count = max(counts.values()) if counts else 0
    dominant_entities = sorted([mode for mode, count in counts.items() if count == max_count and count > 0])
    conflicts = []
    if counts["sql"] > 0 and counts["cypher"] > 0:
        conflicts.append("sql_vs_cypher_entity_overlap")

    return {
        "matched_sql_entities": sorted(matched_sql_entities)[:20],
        "matched_cypher_entities": sorted(matched_cypher_entities)[:20],
        "matched_mql_entities": sorted(matched_mql_entities)[:20],
        "matched_vector_entities": sorted(matched_vector_entities)[:20],
        "matched_sql_tokens": sorted(matched_sql_tokens)[:20],
        "matched_cypher_tokens": sorted(matched_cypher_tokens)[:20],
        "matched_mql_tokens": sorted(matched_mql_tokens)[:20],
        "matched_vector_tokens": sorted(matched_vector_tokens)[:20],
        "dominant_entities": dominant_entities,
        "conflicts": conflicts,
        "counts": counts,
    }


def build_route_evidence(
    question: str,
    signals: dict[str, Any],
    entity_hints: dict[str, Any],
    scored: dict[str, Any],
) -> dict[str, Any]:
    modality_scores = {
        modality: payload["final_score"]
        for modality, payload in scored["modalities"].items()
    }
    rule_task_analysis = signals.get("task_analysis", analyze_task_mode(question))
    evidence = {
        "question": question,
        "task_analysis": rule_task_analysis,
        "rule_task_analysis": rule_task_analysis,
        "intent_signals": signals,
        "entity_hints": entity_hints,
        "modality_scores": modality_scores,
        "scored_modalities": scored["modalities"],
        "retrieval_hints": scored["retrieval_hints"],
    }
    schema_plan_context = build_schema_plan_context(question, evidence)
    evidence["schema_plan_context"] = schema_plan_context
    should_refine, refine_reason = should_refine_task_mode_with_schema(
        question=question,
        rule_analysis=rule_task_analysis,
        evidence=evidence,
        schema_context=schema_plan_context,
    )
    if should_refine:
        try:
            task_analysis = refine_task_mode_with_schema(
                question=question,
                rule_analysis=rule_task_analysis,
                evidence=evidence,
                schema_context=schema_plan_context,
            )
            task_analysis["schema_aware_refine_trigger"] = refine_reason
        except Exception as exc:
            task_analysis = {
                **rule_task_analysis,
                "source": rule_task_analysis.get("source", "rule_task_mode_analyzer"),
                "schema_aware_refine_trigger": refine_reason,
                "schema_aware_judge_error": str(exc),
            }
    else:
        task_analysis = {
            **rule_task_analysis,
            "source": rule_task_analysis.get("source", "rule_task_mode_analyzer"),
            "schema_aware_refine_skipped": refine_reason,
        }
    evidence["task_analysis"] = task_analysis
    evidence["intent_signals"]["task_analysis"] = task_analysis
    evidence["intent_signals"]["task_mode_hint"] = task_analysis.get("task_mode", rule_task_analysis.get("task_mode", "single_step"))
    evidence["intent_signals"]["has_nested_dependency"] = task_analysis.get("query_strategy") in {
        "single_query_with_nested_logic",
        "multi_step_candidate",
        "cross_source_candidate",
    }
    evidence["schema_plan_proposals"] = build_deterministic_schema_plan_proposals(
        question,
        evidence,
        schema_plan_context,
    )
    return evidence
