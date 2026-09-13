import pickle
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from .config import SCHEMA_INDEX_DIR
from .routing_common import meaningful_tokens
from .paired_benchmark_resources import (
    paired_cypher_metadata_records,
    paired_sql_metadata_records,
    resolve_paired_sql_resource_path,
)

# 中文函数说明索引：
# - _schema_plan_tokens/_token_overlap_score：对问题和 schema snippet 做轻量 token overlap 打分。
# - _load_sql_metadata/_load_cypher_metadata：读取 SQL/Cypher schema index metadata。
# - _clean_snippet/_schema_body/_score_text_without_resource_prefix：清洗 schema 文本，避免 resource id 前缀干扰排序。
# - _extract_names_before_type_parens/_relationship_pattern：从 Cypher schema 文本中抽取节点/关系名称。
# - build_global_schema_catalog()：构建跨 SQL/Cypher 的全局 schema catalog，供路由层和 Step Planner 看见数据库结构。
# - build_schema_plan_context(question)：给 LLM proposal 提供 top schema snippets 和资源候选，不包含具体数值。
# - build_deterministic_schema_plan_proposals(question, task_analysis)：规则生成 SQL/Cypher proposal，作为 LLM proposal 不稳定时的保底。
# - normalize_schema_plan_proposals(...)：规范化 proposal 字段，限制 top-K 并补齐 rank/source/reason。
# - resolve_sql_resource_path(resource_id)：把 SQL resource_id 映射为 sqlite 文件路径。
# - _filter_target_resources(...)：清理 proposal target_resources，避免不存在或重复资源传给 pipeline。


MAX_SCHEMA_SNIPPETS_PER_MODE = 4
MAX_PLAN_PROPOSALS = 3
MAX_CATALOG_ITEMS_PER_RESOURCE = 40
SCHEMA_PLAN_GENERIC_TOKENS = {
    "find",
    "first",
    "then",
    "most",
    "least",
    "top",
    "bottom",
    "popular",
    "related",
    "graph",
    "query",
    "show",
    "list",
}


def _schema_plan_tokens(text: str) -> set[str]:
    return meaningful_tokens(text) - SCHEMA_PLAN_GENERIC_TOKENS


def _token_overlap_score(question: str, text: str, preferred_tokens: set[str] | None = None) -> float:
    question_tokens = _schema_plan_tokens(question)
    text_tokens = _schema_plan_tokens(text)
    if not question_tokens or not text_tokens:
        return 0.0
    overlap = question_tokens & text_tokens
    preferred_overlap = (preferred_tokens or set()) & text_tokens
    return float(len(overlap)) + float(len(preferred_overlap)) * 1.5


@lru_cache(maxsize=1)
def _load_sql_metadata() -> list[dict[str, Any]]:
    path = SCHEMA_INDEX_DIR / "metadata.pkl"
    data = []
    if path.exists():
        with path.open("rb") as handle:
            raw = pickle.load(handle)
        data = raw if isinstance(raw, list) else []
    return data + paired_sql_metadata_records()


@lru_cache(maxsize=1)
def _load_cypher_metadata() -> list[dict[str, Any]]:
    path = SCHEMA_INDEX_DIR / "cypher_metadata.pkl"
    data = []
    if path.exists():
        with path.open("rb") as handle:
            raw = pickle.load(handle)
        data = raw if isinstance(raw, list) else []
    return data + paired_cypher_metadata_records()


def _clean_snippet(text: str, max_len: int = 260) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 3].rstrip() + "..."


def _schema_only_text(text: str) -> str:
    """Remove metadata sample values before sending route-level schema context."""
    raw = str(text or "")
    for marker in (" 样本:", "样本:", " Samples:", " samples:", " 示例:", " 样例:"):
        if marker in raw:
            return raw.split(marker, 1)[0]
    return raw


def _score_text_without_resource_prefix(text: str) -> str:
    return re.sub(r"^[^.:\s]+\.", "", text, count=1)


def _schema_body(text: str) -> str:
    cleaned = _schema_only_text(text)
    return cleaned.split(":", 1)[1] if ":" in cleaned else cleaned


def _extract_names_before_type_parens(text: str) -> list[str]:
    names: list[str] = []
    for match in re.finditer(r"(?:^|[,:\s])\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(", text):
        name = match.group(1)
        if name.lower() not in {"int", "integer", "text", "varchar", "char", "float", "double", "bool", "date", "str", "list"}:
            names.append(name)
    return names


def _relationship_pattern(text: str) -> dict[str, str]:
    match = re.search(
        r"\(:\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*-\s*\[:\s*([A-Za-z_][A-Za-z0-9_]*)\s*\]\s*->\s*\(:\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)",
        text,
    )
    if not match:
        return {}
    return {
        "from": match.group(1),
        "relationship": match.group(2),
        "to": match.group(3),
    }


@lru_cache(maxsize=1)
def build_global_schema_catalog() -> dict[str, Any]:
    sql_resources: dict[str, dict[str, Any]] = {}
    for item in _load_sql_metadata():
        db_id = item.get("db_id", "")
        table = item.get("table", "")
        if not db_id or not table:
            continue
        resource = sql_resources.setdefault(
            db_id,
            {
                "resource_id": db_id,
                "query_type": "sql",
                "tables": [],
            },
        )
        if len(resource["tables"]) >= MAX_CATALOG_ITEMS_PER_RESOURCE:
            continue
        resource["tables"].append(
            {
                "table": table,
                "columns": _extract_names_before_type_parens(_schema_body(item.get("text", ""))),
            }
        )

    cypher_resources: dict[str, dict[str, Any]] = {}
    for item in _load_cypher_metadata():
        db_name = item.get("db_name", "")
        label = item.get("label", "")
        item_type = item.get("type", "")
        if not db_name or not label:
            continue
        resource = cypher_resources.setdefault(
            db_name,
            {
                "resource_id": db_name,
                "query_type": "cypher",
                "node_labels": [],
                "relationship_types": [],
            },
        )
        text = _schema_only_text(item.get("text", ""))
        if item_type == "node" and len(resource["node_labels"]) < MAX_CATALOG_ITEMS_PER_RESOURCE:
            resource["node_labels"].append(
                {
                    "label": label,
                    "properties": _extract_names_before_type_parens(_schema_body(text)),
                }
            )
        elif item_type == "rel" and len(resource["relationship_types"]) < MAX_CATALOG_ITEMS_PER_RESOURCE:
            relationship = {"type": label}
            relationship.update(_relationship_pattern(text))
            resource["relationship_types"].append(relationship)

    return {
        "sql": sorted(sql_resources.values(), key=lambda item: item["resource_id"]),
        "cypher": sorted(cypher_resources.values(), key=lambda item: item["resource_id"]),
        "limits": {
            "contains_every_indexed_sql_database": True,
            "contains_every_indexed_cypher_database": True,
            "max_catalog_items_per_resource": MAX_CATALOG_ITEMS_PER_RESOURCE,
            "contains_table_rows": False,
            "contains_sample_values": False,
            "contains_full_create_schema": False,
        },
    }


def _rank_sql_snippets(question: str, evidence: dict[str, Any]) -> list[dict[str, Any]]:
    retrieval = evidence.get("retrieval_hints", {}).get("sql", {})
    best_resource = retrieval.get("best_resource", "")
    preferred_tokens = set(evidence.get("entity_hints", {}).get("matched_sql_tokens", [])) - SCHEMA_PLAN_GENERIC_TOKENS
    snippets: list[tuple[float, dict[str, Any]]] = []
    for item in _load_sql_metadata():
        db_id = item.get("db_id", "")
        text = _schema_only_text(item.get("text", ""))
        score = _token_overlap_score(question, _score_text_without_resource_prefix(text), preferred_tokens)
        if best_resource and db_id == best_resource and score > 0:
            score += 2.0
        if score <= 0:
            continue
        snippets.append(
            (
                score,
                {
                    "query_type": "sql",
                    "resource_id": db_id,
                    "schema_item": item.get("table", ""),
                    "snippet": _clean_snippet(text),
                    "score": round(score, 4),
                },
            )
        )
    snippets.sort(key=lambda row: (-row[0], row[1]["resource_id"], row[1]["schema_item"]))
    return [item for _, item in snippets[:MAX_SCHEMA_SNIPPETS_PER_MODE]]


def _rank_cypher_snippets(question: str, evidence: dict[str, Any]) -> list[dict[str, Any]]:
    retrieval = evidence.get("retrieval_hints", {}).get("cypher", {})
    best_resource = retrieval.get("best_resource", "")
    preferred_tokens = set(evidence.get("entity_hints", {}).get("matched_cypher_tokens", [])) - SCHEMA_PLAN_GENERIC_TOKENS
    snippets: list[tuple[float, dict[str, Any]]] = []
    for item in _load_cypher_metadata():
        db_name = item.get("db_name", "")
        text = _schema_only_text(item.get("text", ""))
        score = _token_overlap_score(question, _score_text_without_resource_prefix(text), preferred_tokens)
        if best_resource and db_name == best_resource and score > 0:
            score += 2.0
        if score <= 0:
            continue
        snippets.append(
            (
                score,
                {
                    "query_type": "cypher",
                    "resource_id": db_name,
                    "schema_item": f"{item.get('type', '')}:{item.get('label', '')}",
                    "snippet": _clean_snippet(text),
                    "score": round(score, 4),
                },
            )
        )
    snippets.sort(key=lambda row: (-row[0], row[1]["resource_id"], row[1]["schema_item"]))
    return [item for _, item in snippets[:MAX_SCHEMA_SNIPPETS_PER_MODE]]


def build_schema_plan_context(question: str, evidence: dict[str, Any]) -> dict[str, Any]:
    """Build a compact schema context for route-level planning."""
    return {
        "catalog": build_global_schema_catalog(),
        "sql": _rank_sql_snippets(question, evidence),
        "cypher": _rank_cypher_snippets(question, evidence),
        "limits": {
            "max_schema_snippets_per_mode": MAX_SCHEMA_SNIPPETS_PER_MODE,
            "contains_global_schema_catalog": True,
            "contains_full_database_schema": False,
            "contains_table_rows": False,
            "contains_sample_values": False,
        },
    }


def _schema_items_for(context: dict[str, Any], query_type: str) -> list[str]:
    return [
        item.get("schema_item", "")
        for item in context.get(query_type, [])[:3]
        if item.get("schema_item")
    ]


def _target_resources_for(context: dict[str, Any], query_type: str, schema_items: list[str]) -> list[dict[str, Any]]:
    resources: list[dict[str, Any]] = []
    seen: set[str] = set()
    schema_item_set = set(schema_items)
    for item in context.get(query_type, []):
        if schema_item_set and item.get("schema_item") not in schema_item_set:
            continue
        resource_id = item.get("resource_id", "")
        if not resource_id or resource_id in seen:
            continue
        seen.add(resource_id)
        resources.append(
            {
                "resource_id": resource_id,
                "schema_items": [
                    candidate.get("schema_item", "")
                    for candidate in context.get(query_type, [])
                    if candidate.get("resource_id") == resource_id and candidate.get("schema_item")
                ][:4],
            }
        )
        if len(resources) >= 2:
            break
    return _prefer_split_benchmark_resources(resources)


def _shape_from_task_analysis(query_type: str, task_analysis: dict[str, Any]) -> str:
    strategy = task_analysis.get("query_strategy", "single_query")
    signal_names = {
        item.get("signal", "")
        for item in task_analysis.get("signals", []) or []
        if isinstance(item, dict)
    }
    if query_type not in {"sql", "cypher"}:
        if strategy in {"multi_step_candidate", "cross_source_candidate"}:
            return "future_multi_step_plan"
        return "direct_query"
    if strategy == "single_query_with_nested_logic":
        if query_type == "cypher":
            return "with_scope_or_aggregation"
        if signal_names & {"average_comparison_dependency", "aggregate_reference_dependency"}:
            return "subquery_or_cte"
        return "subquery_cte_or_grouped_aggregate_rank"
    if strategy in {"multi_step_candidate", "cross_source_candidate"}:
        return "future_multi_step_plan"
    return "direct_query"


def build_deterministic_schema_plan_proposals(
    question: str,
    evidence: dict[str, Any],
    schema_context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build conservative schema-aware plan proposals without calling an LLM."""
    context = schema_context or build_schema_plan_context(question, evidence)
    task_analysis = evidence.get("task_analysis", {})
    scored = evidence.get("scored_modalities", {})
    ranked = sorted(scored.items(), key=lambda row: (-row[1].get("final_score", 0.0), row[0]))
    proposals: list[dict[str, Any]] = []
    for rank, (query_type, payload) in enumerate(ranked[:MAX_PLAN_PROPOSALS], start=1):
        if query_type not in {"sql", "cypher"} and payload.get("final_score", 0.0) <= 0:
            continue
        schema_items = _schema_items_for(context, query_type) if query_type in {"sql", "cypher"} else []
        proposals.append(
            {
                "proposal_id": f"P{rank}",
                "rank": rank,
                "query_type": query_type,
                "query_strategy": task_analysis.get("query_strategy", "single_query"),
                "query_shape": _shape_from_task_analysis(query_type, task_analysis),
                "schema_items": schema_items,
                "target_resources": _target_resources_for(context, query_type, schema_items),
                "confidence": "medium" if rank == 1 else "low",
                "source": "deterministic_schema_plan",
                "reason": "Schema-aware fallback proposal based on modality score, task analysis, and compact schema evidence.",
            }
        )
    return proposals


def normalize_schema_plan_proposals(raw: Any, fallback: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return fallback[:MAX_PLAN_PROPOSALS]
    proposals: list[dict[str, Any]] = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            continue
        query_type = item.get("query_type", "")
        if query_type not in {"sql", "cypher", "mql", "vector"}:
            continue
        fallback_match = next((plan for plan in fallback if plan.get("query_type") == query_type), {})
        schema_items = item.get("schema_items", []) if isinstance(item.get("schema_items", []), list) else []
        target_resources = item.get("target_resources") if isinstance(item.get("target_resources"), list) else []
        if not target_resources:
            target_resources = _filter_target_resources(fallback_match.get("target_resources", []), schema_items)
        target_resources = _prefer_split_benchmark_resources(target_resources)
        proposals.append(
            {
                "proposal_id": item.get("proposal_id") or f"P{index}",
                "rank": int(item.get("rank") or index),
                "query_type": query_type,
                "query_strategy": item.get("query_strategy", "single_query"),
                "query_shape": item.get("query_shape", "direct_query"),
                "schema_items": schema_items,
                "target_resources": target_resources,
                "confidence": item.get("confidence", "low"),
                "source": item.get("source", "llm_schema_plan"),
                "reason": item.get("reason", ""),
            }
        )
    return proposals[:MAX_PLAN_PROPOSALS] or fallback[:MAX_PLAN_PROPOSALS]


def _resource_stem(resource_id: str) -> str:
    for suffix in ("_sql", "_graph"):
        if resource_id.endswith(suffix):
            return resource_id[: -len(suffix)]
    return resource_id


def _prefer_split_benchmark_resources(resources: Any) -> list[dict[str, Any]]:
    """Prefer explicit paired split resources over legacy resources with the same stem.

    A proposal may contain both `wta_1` and `wta_1_sql` because the schema index
    includes Spider originals and paired split benchmark databases. When both are
    present, the split resource is the more specific target and must be tried first.
    """
    if not isinstance(resources, list):
        return []
    normalized = [resource for resource in resources if isinstance(resource, dict)]
    split_stems = {
        _resource_stem(str(resource.get("resource_id", "")))
        for resource in normalized
        if str(resource.get("resource_id", "")).endswith(("_sql", "_graph"))
    }

    def priority(resource: dict[str, Any]) -> tuple[int, str]:
        resource_id = str(resource.get("resource_id", ""))
        stem = _resource_stem(resource_id)
        if resource_id.endswith(("_sql", "_graph")):
            return (0, resource_id)
        if stem in split_stems:
            return (1, resource_id)
        return (2, resource_id)

    return sorted(normalized, key=priority)


def resolve_sql_resource_path(resource_id: str) -> str:
    paired_path = resolve_paired_sql_resource_path(resource_id)
    if paired_path:
        return paired_path
    for item in _load_sql_metadata():
        if item.get("db_id") == resource_id:
            return item.get("db_path", "")
    return ""


def _filter_target_resources(resources: Any, schema_items: list[str]) -> list[dict[str, Any]]:
    if not isinstance(resources, list):
        return []
    if not schema_items:
        return [resource for resource in resources if isinstance(resource, dict)]
    wanted = set(schema_items)
    filtered = []
    for resource in resources:
        if not isinstance(resource, dict):
            continue
        resource_items = set(resource.get("schema_items", []))
        if resource_items & wanted:
            filtered.append(resource)
    return filtered or [resource for resource in resources if isinstance(resource, dict)]
