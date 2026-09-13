from typing import Any

from .routing_common import (
    GRAPH_SURFACE_HINTS,
    MQL_SURFACE_HINTS,
    SQL_SURFACE_HINTS,
    VECTOR_SURFACE_HINTS,
    load_schema_indexes,
    meaningful_tokens,
    normalize_text,
)

# 中文函数说明索引：
# - _lexical_retrieval_hints(question)：根据问题 token 与 schema index 的重合度，给 SQL/Cypher 生成轻量 retrieval hint。
# - _clip(value, lower, upper)：把分数限制在指定范围内，避免单个信号把路由分数拉爆。
# - confidence_from_scores(scores)：根据 top1/top2 分差和绝对分数输出 high/medium/low 路由置信度。
# - score_modalities(question, evidence)：融合 intent、entity、retrieval 等 evidence，得到 SQL/Cypher/MQL/vector 的模态分数。


def _lexical_retrieval_hints(question: str) -> dict[str, Any]:
    question_tokens = meaningful_tokens(question)
    schema_indexes = load_schema_indexes()
    hints: dict[str, Any] = {
        "sql": {"best_resource": "", "score": 0.0},
        "cypher": {"best_resource": "", "score": 0.0},
    }

    for mode in ("sql", "cypher"):
        ranked: list[tuple[float, str]] = []
        for resource_id, terms in schema_indexes[mode].items():
            matched_tokens: set[str] = set()
            for term in terms:
                matched_tokens.update(meaningful_tokens(term) & question_tokens)
            if matched_tokens:
                ranked.append((float(len(matched_tokens)), resource_id))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        if ranked:
            best_score, best_resource = ranked[0]
            hints[mode] = {
                "best_resource": best_resource,
                "score": min(best_score / 3.0, 1.0),
            }
    return hints


def _clip(value: float) -> float:
    return max(0.0, min(value, 1.0))


def confidence_from_scores(top1_score: float, top2_score: float, has_cross_source_signal: bool) -> str:
    gap = top1_score - top2_score
    if top1_score >= 0.70 and gap >= 0.15 and not has_cross_source_signal:
        return "high"
    if top1_score >= 0.55 and gap >= 0.08:
        return "medium"
    return "low"


def score_modalities(question: str, signals: dict[str, Any], entity_hints: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_text(question)
    retrieval_hints = _lexical_retrieval_hints(question)
    results: dict[str, Any] = {}

    sql_surface = 0.0
    sql_surface += 0.35 if signals["has_aggregation"] else 0.0
    sql_surface += 0.10 if signals["has_sorting"] else 0.0
    sql_surface += 0.10 if signals["has_filtering"] else 0.0
    sql_surface += 0.10 if any(hint in normalized for hint in SQL_SURFACE_HINTS) else 0.0
    sql_surface -= 0.20 if signals["has_graph_relation"] else 0.0
    sql_surface -= 0.15 if signals["has_multi_hop_relation"] else 0.0
    sql_surface -= 0.40 if signals["has_similarity_search"] else 0.0
    sql_surface -= 0.30 if signals["has_document_pattern"] else 0.0
    sql_entity = min(entity_hints["counts"]["sql"] * 0.15, 1.0)
    sql_structure = 0.35 if signals["has_aggregation"] else 0.10
    if signals["has_graph_relation"]:
        sql_structure -= 0.20
    if signals["has_similarity_search"]:
        sql_structure -= 0.25
    sql_retrieval = retrieval_hints["sql"]["score"]
    sql_final = _clip(0.30 * _clip(sql_surface) + 0.30 * sql_entity + 0.25 * _clip(sql_structure) + 0.15 * sql_retrieval)
    results["sql"] = {
        "surface_score": round(_clip(sql_surface), 4),
        "entity_score": round(sql_entity, 4),
        "structure_score": round(_clip(sql_structure), 4),
        "retrieval_prior_score": round(sql_retrieval, 4),
        "final_score": round(sql_final, 4),
        "reason_parts": [
            "聚合/排序/筛选信号偏强" if signals["has_aggregation"] or signals["has_sorting"] else "结构化查询信号一般",
            "SQL 实体命中较多" if entity_hints["counts"]["sql"] > 0 else "SQL 实体命中较少",
        ],
    }

    cypher_surface = 0.0
    cypher_surface += 0.35 if signals["has_graph_relation"] else 0.0
    cypher_surface += 0.20 if signals["has_multi_hop_relation"] else 0.0
    cypher_surface += 0.10 if any(hint in normalized for hint in GRAPH_SURFACE_HINTS) else 0.0
    cypher_surface -= 0.20 if signals["has_aggregation"] and not signals["has_graph_relation"] else 0.0
    cypher_surface -= 0.35 if signals["has_similarity_search"] else 0.0
    cypher_surface -= 0.25 if signals["has_document_pattern"] else 0.0
    cypher_entity = min(entity_hints["counts"]["cypher"] * 0.15, 1.0)
    cypher_structure = 0.40 if signals["has_graph_relation"] else 0.10
    cypher_structure += 0.20 if signals["has_multi_hop_relation"] else 0.0
    if signals["has_aggregation"] and not signals["has_graph_relation"]:
        cypher_structure -= 0.20
    cypher_retrieval = retrieval_hints["cypher"]["score"]
    cypher_final = _clip(0.30 * _clip(cypher_surface) + 0.30 * cypher_entity + 0.25 * _clip(cypher_structure) + 0.15 * cypher_retrieval)
    results["cypher"] = {
        "surface_score": round(_clip(cypher_surface), 4),
        "entity_score": round(cypher_entity, 4),
        "structure_score": round(_clip(cypher_structure), 4),
        "retrieval_prior_score": round(cypher_retrieval, 4),
        "final_score": round(cypher_final, 4),
        "reason_parts": [
            "关系/路径信号偏强" if signals["has_graph_relation"] else "显式图关系信号较少",
            "Cypher 实体命中较多" if entity_hints["counts"]["cypher"] > 0 else "Cypher 实体命中较少",
        ],
    }

    mql_surface = 0.0
    mql_surface += 0.45 if signals["has_document_pattern"] else 0.0
    mql_surface += 0.10 if any(hint in normalized for hint in MQL_SURFACE_HINTS) else 0.0
    mql_surface -= 0.15 if signals["has_graph_relation"] else 0.0
    mql_surface -= 0.25 if signals["has_similarity_search"] else 0.0
    mql_entity = min(entity_hints["counts"]["mql"] * 0.20, 1.0)
    mql_structure = 0.45 if signals["has_document_pattern"] else 0.05
    mql_final = _clip(0.30 * _clip(mql_surface) + 0.30 * mql_entity + 0.25 * _clip(mql_structure))
    results["mql"] = {
        "surface_score": round(_clip(mql_surface), 4),
        "entity_score": round(mql_entity, 4),
        "structure_score": round(_clip(mql_structure), 4),
        "retrieval_prior_score": 0.0,
        "final_score": round(mql_final, 4),
        "reason_parts": [
            "文档/数组信号偏强" if signals["has_document_pattern"] else "文档信号较弱",
            "MQL 实体命中较少" if entity_hints["counts"]["mql"] == 0 else "MQL 实体命中较多",
        ],
    }

    vector_surface = 0.0
    vector_surface += 0.55 if signals["has_similarity_search"] else 0.0
    vector_surface += 0.10 if any(hint in normalized for hint in VECTOR_SURFACE_HINTS) else 0.0
    vector_surface -= 0.20 if signals["has_aggregation"] else 0.0
    vector_surface -= 0.20 if signals["has_graph_relation"] else 0.0
    vector_entity = min(entity_hints["counts"]["vector"] * 0.20, 1.0)
    vector_structure = 0.55 if signals["has_similarity_search"] else 0.05
    vector_final = _clip(0.30 * _clip(vector_surface) + 0.30 * vector_entity + 0.25 * _clip(vector_structure))
    results["vector"] = {
        "surface_score": round(_clip(vector_surface), 4),
        "entity_score": round(vector_entity, 4),
        "structure_score": round(_clip(vector_structure), 4),
        "retrieval_prior_score": 0.0,
        "final_score": round(vector_final, 4),
        "reason_parts": [
            "相似度检索信号偏强" if signals["has_similarity_search"] else "相似度信号较弱",
            "Vector 实体命中较少" if entity_hints["counts"]["vector"] == 0 else "Vector 实体命中较多",
        ],
    }

    return {
        "modalities": results,
        "retrieval_hints": retrieval_hints,
    }
