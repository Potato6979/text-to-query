import re
from typing import Any

from .routing_common import meaningful_tokens

# 中文函数说明索引：
# - _schema_plan_proposals(route_plan)：读取 route_plan 中来自 schema-aware LLM/规则的数据库候选方案。
# - _future_plan_proposals(route_plan)：筛选 query_shape=future_multi_step_plan 的 proposal，作为 Step Planner 输入。
# - _token_variants/_expanded_tokens：生成 token 变体，处理复数、下划线等 schema 词面差异。
# - _proposal_text/_resource_text/_resource_schema_items：把 proposal/resource 转成可打分文本和 schema item 列表。
# - _catalog_names_for/_best_catalog_resource/_catalog_resource_by_id：从 schema catalog 中查找候选数据库资源。
# - _step_segments(question)：把自然语言按 first/then 等顺序词切成最多两步的执行片段。
# - _preferred_query_type(...)：根据片段、proposal、catalog 和上一段上下文决定当前 step 的 query_type。
# - _should_inherit_previous_query_type(...)：处理 “then find those matches” 这类承接表达；无显式切源信号时继承上一 query_type。
# - _infer_query_shape/_step_goal/_step_spec_from_segment：把片段转成 step 草案，包含目标、query_type、schema_items 和资源。
# - _build_step_planner_proposals/_ordered_proposals：把 top-K proposal 排序成真正 Step Planner 可执行顺序。
# - _proposal_schema_check/_planned_step_schema_check：记录 step 与 schema/proposal 的弱/强匹配情况，不直接决定答案正确性。
# - _build_output_contract/_build_input_contract：生成跨步骤 typed variable 契约，说明变量实体类型、字段和消费策略。
# - _bridge_requirement(...)：判断同源变量能否 hard_constraint，跨源变量是否 required_before_hard_binding。
# - build_multi_step_plan(...)：Step Planner 主入口，输出 steps、bridge_steps、plan_graph 和 planning_status。
# 状态说明：
# - planned_not_executed：已经规划出多步，但是否执行由 runtime_decision 决定。
# - not_required：问题被判断为单步或没有可执行多步计划。
# - hard_constraint：上一步变量可作为下一步硬过滤条件。
# - soft_context：上一步变量只能作为背景上下文，不能强行过滤目标库。
# - required_before_hard_binding：跨源变量必须先经过 Bridge Resolver 映射，才能升级为 hard_constraint。


MULTI_STEP_PLAN_VERSION = 1
MAX_MULTI_STEP_DRAFT_STEPS = 2
STEP_PLANNER_VERSION = 1
SQL_SOURCE_TOKENS = {"sql", "sqlite", "table", "tables", "relational"}
CYPHER_SOURCE_TOKENS = {"cypher", "graph", "neo4j", "node", "nodes", "relationship", "relationships"}
GENERIC_SCHEMA_ITEM_TOKENS = {
    "node",
    "nodes",
    "rel",
    "rels",
    "relationship",
    "relationships",
    "table",
    "tables",
    "column",
    "columns",
    "property",
    "properties",
}
ENTITY_STOP_TOKENS = GENERIC_SCHEMA_ITEM_TOKENS | {
    "id",
    "ids",
    "name",
    "names",
    "type",
    "types",
    "value",
    "values",
}
CONTINUATION_TOKENS = {
    "that",
    "those",
    "these",
    "this",
    "it",
    "them",
    "same",
    "result",
    "results",
    "returned",
    "previous",
    "related",
    "connected",
}
CONTINUATION_SURFACE_RE = re.compile(
    r"\b(that|those|these|this|it|them|same|previous|returned|result|results)\b",
    re.IGNORECASE,
)
PREFERRED_OUTPUT_ENTITY_TOKENS = [
    "player",
    "singer",
    "student",
    "highschooler",
    "character",
    "person",
    "team",
    "pilot",
    "concert",
    "match",
    "film",
    "planet",
]

RELATION_TARGET_ENTITY_BY_TOKEN = {
    "homeworld": "planet",
}


def _schema_plan_proposals(route_plan: dict[str, Any]) -> list[dict[str, Any]]:
    proposals = route_plan.get("schema_plan_proposals", [])
    return [
        proposal
        for proposal in proposals
        if proposal.get("query_type") in {"sql", "cypher", "mql", "vector"}
    ][:MAX_MULTI_STEP_DRAFT_STEPS]


def _future_plan_proposals(route_plan: dict[str, Any]) -> list[dict[str, Any]]:
    proposals = route_plan.get("schema_plan_proposals", [])
    return [
        proposal
        for proposal in proposals
        if proposal.get("query_shape") == "future_multi_step_plan"
    ][:MAX_MULTI_STEP_DRAFT_STEPS]


def _token_variants(token: str) -> set[str]:
    variants = {token}
    if token.endswith("ies") and len(token) > 4:
        variants.add(token[:-3] + "y")
    if token.endswith("es") and len(token) > 3:
        variants.add(token[:-2])
    if token.endswith("s") and len(token) > 3:
        variants.add(token[:-1])
    return variants


def _expanded_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for token in meaningful_tokens(text):
        tokens.update(_token_variants(token.lower()))
    return tokens - GENERIC_SCHEMA_ITEM_TOKENS


def _proposal_text(proposal: dict[str, Any]) -> str:
    target_ids = [str(resource.get("resource_id", "")) for resource in proposal.get("target_resources", [])]
    return " ".join(
        [
            str(proposal.get("query_type", "")),
            str(proposal.get("query_strategy", "")),
            str(proposal.get("query_shape", "")),
            " ".join(str(item) for item in proposal.get("schema_items", [])),
            " ".join(target_ids),
            str(proposal.get("reason", "")),
        ]
    )


def _resource_text(resource: dict[str, Any], query_type: str) -> str:
    parts = [str(resource.get("resource_id", ""))]
    if query_type == "sql":
        for table in resource.get("tables", []):
            parts.append(str(table.get("table", "")))
            parts.extend(str(column) for column in table.get("columns", []))
    elif query_type == "cypher":
        for node in resource.get("node_labels", []):
            parts.append(str(node.get("label", "")))
            parts.extend(str(prop) for prop in node.get("properties", []))
        for relationship in resource.get("relationship_types", []):
            parts.append(str(relationship.get("type", "")))
            parts.append(str(relationship.get("from", "")))
            parts.append(str(relationship.get("relationship", "")))
            parts.append(str(relationship.get("to", "")))
    return " ".join(parts)


def _resource_schema_items(resource: dict[str, Any], query_type: str, segment_tokens: set[str]) -> list[str]:
    items: list[tuple[int, str]] = []
    if query_type == "sql":
        for table in resource.get("tables", []):
            table_name = str(table.get("table", ""))
            tokens = _expanded_tokens(" ".join([table_name, *[str(column) for column in table.get("columns", [])]]))
            score = len(segment_tokens & tokens)
            if score > 0 or not items:
                items.append((score, table_name))
    elif query_type == "cypher":
        for node in resource.get("node_labels", []):
            label = str(node.get("label", ""))
            tokens = _expanded_tokens(" ".join([label, *[str(prop) for prop in node.get("properties", [])]]))
            score = len(segment_tokens & tokens)
            if score > 0 or not items:
                items.append((score, f"node:{label}"))
        for relationship in resource.get("relationship_types", []):
            rel_type = str(relationship.get("type", ""))
            tokens = _expanded_tokens(
                " ".join(
                    [
                        rel_type,
                        str(relationship.get("from", "")),
                        str(relationship.get("relationship", "")),
                        str(relationship.get("to", "")),
                    ]
                )
            )
            score = len(segment_tokens & tokens)
            if score > 0:
                items.append((score, f"rel:{rel_type}"))
    items.sort(key=lambda item: (-item[0], item[1]))
    return [item for _, item in items[:4]]


def _catalog_names_for(catalog: dict[str, Any], query_type: str, resource_ids: set[str] | None = None) -> set[str]:
    names: set[str] = set()
    for resource in catalog.get(query_type, []):
        resource_id = str(resource.get("resource_id", ""))
        if resource_ids and resource_id not in resource_ids:
            continue
        names.update(_expanded_tokens(resource_id))
        if query_type == "sql":
            for table in resource.get("tables", []):
                names.update(_expanded_tokens(table.get("table", "")))
                names.update(_expanded_tokens(" ".join(str(column) for column in table.get("columns", []))))
        elif query_type == "cypher":
            for node in resource.get("node_labels", []):
                names.update(_expanded_tokens(node.get("label", "")))
                names.update(_expanded_tokens(" ".join(str(prop) for prop in node.get("properties", []))))
            for relationship in resource.get("relationship_types", []):
                names.update(_expanded_tokens(relationship.get("type", "")))
                names.update(_expanded_tokens(relationship.get("from", "")))
                names.update(_expanded_tokens(relationship.get("relationship", "")))
                names.update(_expanded_tokens(relationship.get("to", "")))
    return names


def _step_segments(question: str) -> list[str]:
    lowered = str(question or "").strip()
    if not lowered:
        return []
    parts = re.split(r"\b(?:then|after that|afterwards|next)\b", lowered, maxsplit=1, flags=re.IGNORECASE)
    return [part.strip(" ,.;:") for part in parts if part.strip(" ,.;:")][:2]


def _catalog_query_type_score(segment: str, catalog: dict[str, Any], query_type: str) -> float:
    segment_tokens = _expanded_tokens(segment)
    if not segment_tokens:
        return 0.0
    score = 0.0
    if query_type == "sql":
        score += 3.0 * len(segment_tokens & SQL_SOURCE_TOKENS)
        score -= 2.0 * len(segment_tokens & CYPHER_SOURCE_TOKENS)
    elif query_type == "cypher":
        score += 3.0 * len(segment_tokens & CYPHER_SOURCE_TOKENS)
        score -= 2.0 * len(segment_tokens & SQL_SOURCE_TOKENS)
    for resource in catalog.get(query_type, []):
        score += float(len(segment_tokens & _expanded_tokens(_resource_text(resource, query_type)))) * 1.25
    return score


def _preferred_query_type(segment: str, catalog: dict[str, Any], proposals: list[dict[str, Any]]) -> str:
    segment_tokens = _expanded_tokens(segment)
    sql_explicit = bool(segment_tokens & SQL_SOURCE_TOKENS)
    cypher_explicit = bool(segment_tokens & CYPHER_SOURCE_TOKENS)
    if sql_explicit and not cypher_explicit:
        return "sql"
    if cypher_explicit and not sql_explicit:
        return "cypher"

    catalog_scores = {
        "sql": _catalog_query_type_score(segment, catalog, "sql"),
        "cypher": _catalog_query_type_score(segment, catalog, "cypher"),
    }
    if catalog_scores["sql"] > catalog_scores["cypher"] + 0.5:
        return "sql"
    if catalog_scores["cypher"] > catalog_scores["sql"] + 0.5:
        return "cypher"

    scores = dict(catalog_scores)
    for proposal in proposals:
        query_type = proposal.get("query_type", "")
        if query_type in scores:
            scores[query_type] += _segment_score(segment, proposal, catalog) * 0.5
    return "sql" if scores["sql"] >= scores["cypher"] else "cypher"


def _has_explicit_query_type_signal(segment: str) -> bool:
    tokens = _expanded_tokens(segment)
    return bool(tokens & SQL_SOURCE_TOKENS or tokens & CYPHER_SOURCE_TOKENS)


def _surface_continuation_signal(segment: str) -> bool:
    return bool(CONTINUATION_SURFACE_RE.search(str(segment or "")))


def _should_inherit_previous_query_type(previous_segment: str, current_segment: str, previous_query_type: str) -> bool:
    current_tokens = _expanded_tokens(current_segment)
    if not previous_query_type or _has_explicit_query_type_signal(current_segment):
        return False
    if _surface_continuation_signal(current_segment):
        return True
    previous_tokens = _expanded_tokens(previous_segment)
    return bool(
        current_tokens
        & previous_tokens
        & {
            "team",
            "match",
            "tournament",
            "person",
            "singer",
            "concert",
            "student",
            "highschooler",
            "character",
            "player",
            "film",
        }
    )


def _best_catalog_resource(segment: str, catalog: dict[str, Any], query_type: str) -> dict[str, Any]:
    segment_tokens = _expanded_tokens(segment)
    best_score = -1.0
    best_resource: dict[str, Any] = {}
    for resource in catalog.get(query_type, []):
        resource_tokens = _expanded_tokens(_resource_text(resource, query_type))
        score = float(len(segment_tokens & resource_tokens))
        if score > best_score:
            best_score = score
            best_resource = resource
    return best_resource


def _catalog_resource_by_id(catalog: dict[str, Any], query_type: str, resource_id: str) -> dict[str, Any]:
    for resource in catalog.get(query_type, []):
        if str(resource.get("resource_id", "")) == str(resource_id):
            return resource
    return {}


def _paired_resource_stem(resource_id: str) -> str:
    text = str(resource_id or "")
    for suffix in ("_sql", "_graph"):
        if text.endswith(suffix):
            return text[: -len(suffix)]
    return text


def _paired_resource_id(stem: str, query_type: str) -> str:
    if query_type == "sql":
        return f"{stem}_sql"
    if query_type == "cypher":
        return f"{stem}_graph"
    return ""


def _paired_split_exists(catalog: dict[str, Any], stem: str) -> bool:
    return bool(
        _catalog_resource_by_id(catalog, "sql", f"{stem}_sql")
        and _catalog_resource_by_id(catalog, "cypher", f"{stem}_graph")
    )


def _paired_candidate_stems(proposal: dict[str, Any], catalog: dict[str, Any]) -> list[str]:
    stems: list[str] = []
    seen: set[str] = set()
    for resource_id in _target_resource_ids(proposal):
        stem = _paired_resource_stem(resource_id)
        if not stem or stem in seen or not _paired_split_exists(catalog, stem):
            continue
        seen.add(stem)
        stems.append(stem)
    return stems


def _paired_alignment_score(stem: str, proposals: list[dict[str, Any]], catalog: dict[str, Any]) -> float:
    score = 0.0
    for proposal in proposals:
        query_type = proposal.get("query_type", "")
        resource_id = _paired_resource_id(stem, query_type)
        resource = _catalog_resource_by_id(catalog, query_type, resource_id)
        if not resource:
            return -1.0
        target_stems = {_paired_resource_stem(item) for item in _target_resource_ids(proposal)}
        if stem in target_stems:
            score += 20.0
        segment = str(proposal.get("step_segment") or proposal.get("reason", ""))
        segment_tokens = _expanded_tokens(segment)
        score += float(len(segment_tokens & _expanded_tokens(_resource_text(resource, query_type)))) * 2.0
    return score


def _align_proposal_to_paired_resource(
    proposal: dict[str, Any],
    catalog: dict[str, Any],
    stem: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    query_type = proposal.get("query_type", "")
    resource_id = _paired_resource_id(stem, query_type)
    resource = _catalog_resource_by_id(catalog, query_type, resource_id)
    if not resource:
        return proposal, {}

    previous_resources = _target_resource_ids(proposal)
    if previous_resources == [resource_id]:
        return proposal, {}

    updated = dict(proposal)
    segment = str(proposal.get("step_segment") or proposal.get("reason", ""))
    updated["target_resources"] = [{"resource_id": resource_id}]
    updated["schema_items"] = _resource_schema_items(resource, query_type, _expanded_tokens(segment)) or proposal.get("schema_items", [])
    return updated, {
        "type": "paired_resource_alignment",
        "status": "aligned",
        "query_type": query_type,
        "from_target_resources": previous_resources,
        "to_target_resources": [resource_id],
        "reason": "Adjacent cross-source steps share a paired benchmark stem, so the split SQL/graph resources are used as a matched pair.",
    }


def _align_paired_counterpart_resources(
    proposals: list[dict[str, Any]],
    route_plan: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    catalog = route_plan.get("schema_plan_context", {}).get("catalog", {})
    if len(proposals) < 2 or not catalog:
        return proposals, []

    aligned = [dict(proposal) for proposal in proposals]
    diagnostics: list[dict[str, Any]] = []
    for index in range(1, len(aligned)):
        previous = aligned[index - 1]
        current = aligned[index]
        query_pair = {previous.get("query_type", ""), current.get("query_type", "")}
        if query_pair != {"sql", "cypher"}:
            continue

        stems = _paired_candidate_stems(previous, catalog) + _paired_candidate_stems(current, catalog)
        unique_stems = sorted(set(stems))
        if not unique_stems:
            continue
        scored = [
            (_paired_alignment_score(stem, [previous, current], catalog), stem)
            for stem in unique_stems
        ]
        scored = [(score, stem) for score, stem in scored if score >= 0]
        if not scored:
            continue
        scored.sort(key=lambda item: (-item[0], item[1]))
        best_score, best_stem = scored[0]
        if best_score <= 0:
            continue

        updated_previous, previous_diag = _align_proposal_to_paired_resource(previous, catalog, best_stem)
        updated_current, current_diag = _align_proposal_to_paired_resource(current, catalog, best_stem)
        aligned[index - 1] = updated_previous
        aligned[index] = updated_current
        diagnostics.append(
            {
                "type": "paired_resource_alignment",
                "status": "applied",
                "from_step_index": index,
                "to_step_index": index + 1,
                "stem": best_stem,
                "score": round(best_score, 4),
                "reason": "A SQL/Cypher step pair had a shared paired benchmark counterpart.",
                "changes": [item for item in (previous_diag, current_diag) if item],
            }
        )

    return aligned, diagnostics


def _target_resource_score(segment: str, target: dict[str, Any], resource: dict[str, Any], query_type: str) -> float:
    segment_tokens = _expanded_tokens(segment)
    target_items = [str(item) for item in target.get("schema_items", []) if item]
    score = 0.0
    for item in target_items:
        score += 5.0 * len(segment_tokens & _schema_item_tokens(item))
    score += 1.0 * len(segment_tokens & _expanded_tokens(_resource_text(resource, query_type)))
    resource_id = str(resource.get("resource_id", ""))
    if query_type == "sql" and resource_id.endswith("_sql"):
        # In paired benchmark splits, *_sql is the allowed SQL half; the legacy
        # Spider database with the same stem must not win by having extra tables.
        score += 30.0
    if query_type == "cypher" and resource_id.endswith("_graph"):
        score += 30.0
    return score


def _supporting_proposal_resource(
    segment: str,
    catalog: dict[str, Any],
    query_type: str,
    proposal: dict[str, Any],
) -> dict[str, Any]:
    candidates: list[tuple[float, int, str, dict[str, Any]]] = []
    for index, target in enumerate(proposal.get("target_resources", [])):
        resource = _catalog_resource_by_id(catalog, query_type, str(target.get("resource_id", "")))
        if not resource:
            continue
        candidates.append(
            (
                -_target_resource_score(segment, target, resource, query_type),
                index,
                str(target.get("resource_id", "")),
                resource,
            )
        )
    if candidates:
        candidates.sort(key=lambda row: (row[0], row[1], row[2]))
        return candidates[0][3]
    return {}


def _best_supporting_proposal(segment: str, query_type: str, route_plan: dict[str, Any], proposals: list[dict[str, Any]]) -> dict[str, Any]:
    catalog = route_plan.get("schema_plan_context", {}).get("catalog", {})
    candidates = [proposal for proposal in proposals if proposal.get("query_type") == query_type]
    if not candidates:
        return {}
    candidates.sort(
        key=lambda proposal: (
            -_segment_score(segment, proposal, catalog),
            int(proposal.get("rank", 999)),
            str(proposal.get("proposal_id", "")),
        )
    )
    return candidates[0]


def _infer_query_shape(segment: str, proposal: dict[str, Any]) -> str:
    if proposal.get("query_shape"):
        return proposal.get("query_shape", "")
    tokens = _expanded_tokens(segment)
    if tokens & {"most", "least", "top", "highest", "lowest", "maximum", "minimum", "average"}:
        return "aggregation_query"
    if proposal.get("query_type") == "cypher":
        return "graph_traversal"
    return "direct_query"


def _step_goal(segment: str, index: int, total: int, query_type: str) -> str:
    role = "intermediate result for the next step" if index < total else "final result for the user request"
    return f"Use {query_type.upper()} to answer this subtask and produce the {role}: {segment}"


def _step_spec_from_segment(
    *,
    segment: str,
    index: int,
    total: int,
    route_plan: dict[str, Any],
    proposals: list[dict[str, Any]],
    forced_query_type: str | None = None,
) -> dict[str, Any]:
    catalog = route_plan.get("schema_plan_context", {}).get("catalog", {})
    query_type = forced_query_type or _preferred_query_type(segment, catalog, proposals)
    supporting_proposal = _best_supporting_proposal(segment, query_type, route_plan, proposals)
    resource = _supporting_proposal_resource(segment, catalog, query_type, supporting_proposal) or _best_catalog_resource(segment, catalog, query_type)
    segment_tokens = _expanded_tokens(segment)
    schema_items = _resource_schema_items(resource, query_type, segment_tokens)

    if supporting_proposal:
        schema_items = schema_items or supporting_proposal.get("schema_items", [])
        target_resources = [{"resource_id": resource.get("resource_id", "")}] if resource.get("resource_id") else supporting_proposal.get("target_resources", [])
        proposal_id = supporting_proposal.get("proposal_id", f"SP{index}")
    else:
        target_resources = [{"resource_id": resource.get("resource_id", "")}] if resource.get("resource_id") else []
        proposal_id = f"SP{index}"

    return {
        "proposal_id": proposal_id,
        "rank": index,
        "query_type": query_type,
        "query_strategy": "multi_step_candidate",
        "query_shape": _infer_query_shape(segment, supporting_proposal or {"query_type": query_type}),
        "schema_items": schema_items,
        "target_resources": target_resources,
        "confidence": supporting_proposal.get("confidence", "medium") if supporting_proposal else "medium",
        "source": "step_planner",
        "reason": _step_goal(segment, index, total, query_type),
        "step_segment": segment,
        "supporting_proposal_id": supporting_proposal.get("proposal_id", ""),
        "supporting_proposal_source": supporting_proposal.get("source", ""),
    }


def _build_step_planner_proposals(question: str, route_plan: dict[str, Any], proposals: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    segments = _step_segments(question)
    if len(segments) < 2:
        if len(proposals) < 2:
            return proposals, [
                {
                    "type": "step_planner",
                    "status": "fallback_to_proposals",
                    "reason": "The question could not be decomposed into two explicit segments.",
                }
            ]
        proposals, ordering = _ordered_proposals(question, route_plan, proposals)
        return proposals, [
            {
                "type": "step_planner",
                "status": "fallback_to_ordered_proposals",
                "reason": "The question could not be decomposed into two explicit segments; ordered proposals are used.",
                "ordering": ordering,
            }
        ]

    planned: list[dict[str, Any]] = []
    continuity_diagnostics: list[dict[str, Any]] = []
    for index, segment in enumerate(segments, start=1):
        forced_query_type = None
        if planned and _should_inherit_previous_query_type(
            segments[index - 2],
            segment,
            planned[-1].get("query_type", ""),
        ):
            forced_query_type = planned[-1].get("query_type", "")
            continuity_diagnostics.append(
                {
                    "type": "same_source_continuity",
                    "status": "query_type_inherited",
                    "from_step": f"MS{index - 1}",
                    "to_step": f"MS{index}",
                    "query_type": forced_query_type,
                    "reason": "The current segment uses continuation language and has no explicit source switch.",
                }
            )
        planned.append(
            _step_spec_from_segment(
                segment=segment,
                index=index,
                total=len(segments),
                route_plan=route_plan,
                proposals=proposals,
                forced_query_type=forced_query_type,
            )
        )
    return planned, [
        {
            "type": "step_planner",
            "status": "planned_from_question_and_catalog",
            "segments": segments,
            "planner_version": STEP_PLANNER_VERSION,
            "reason": "Steps were generated from question segments and schema catalog; proposals are retained as supporting evidence.",
            "supporting_proposal_ids": [step.get("supporting_proposal_id", "") for step in planned if step.get("supporting_proposal_id")],
        },
        *continuity_diagnostics,
    ]


def _segment_score(segment: str, proposal: dict[str, Any], catalog: dict[str, Any]) -> float:
    query_type = proposal.get("query_type", "")
    segment_tokens = _expanded_tokens(segment)
    if not query_type or not segment_tokens:
        return 0.0

    score = 0.0
    if query_type == "sql":
        score += 3.0 * len(segment_tokens & SQL_SOURCE_TOKENS)
        score -= 2.0 * len(segment_tokens & CYPHER_SOURCE_TOKENS)
    elif query_type == "cypher":
        score += 3.0 * len(segment_tokens & CYPHER_SOURCE_TOKENS)
        score -= 2.0 * len(segment_tokens & SQL_SOURCE_TOKENS)

    proposal_tokens = _expanded_tokens(_proposal_text(proposal))
    score += float(len(segment_tokens & proposal_tokens)) * 1.5

    target_ids = {str(resource.get("resource_id", "")) for resource in proposal.get("target_resources", []) if resource.get("resource_id")}
    target_names = _catalog_names_for(catalog, query_type, target_ids or None)
    score += float(len(segment_tokens & target_names)) * 2.0
    return score


def _ordered_proposals(question: str, route_plan: dict[str, Any], proposals: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(proposals) < 2:
        return proposals, []
    segments = _step_segments(question)
    if len(segments) < 2:
        return proposals, []

    catalog = route_plan.get("schema_plan_context", {}).get("catalog", {})
    first, second = proposals[0], proposals[1]
    original_score = _segment_score(segments[0], first, catalog) + _segment_score(segments[1], second, catalog)
    reversed_score = _segment_score(segments[0], second, catalog) + _segment_score(segments[1], first, catalog)
    diagnostic = {
        "type": "step_ordering",
        "status": "kept",
        "segments": segments,
        "original_order": [first.get("proposal_id", ""), second.get("proposal_id", "")],
        "original_score": round(original_score, 4),
        "reversed_score": round(reversed_score, 4),
        "reason": "The existing proposal order best matches the question segments.",
    }
    if reversed_score > original_score:
        diagnostic.update(
            {
                "status": "reordered",
                "applied_order": [second.get("proposal_id", ""), first.get("proposal_id", "")],
                "reason": "The reversed proposal order better matches the question segments and schema catalog.",
            }
        )
        return [second, first, *proposals[2:]], [diagnostic]
    return proposals, [diagnostic]


def _schema_item_tokens(item: str) -> set[str]:
    cleaned = re.sub(r"^(?:node|rel|relationship|table|column)\s*[:.]", "", str(item), flags=re.IGNORECASE)
    return _expanded_tokens(cleaned)


def _proposal_schema_check(proposal: dict[str, Any], route_plan: dict[str, Any]) -> dict[str, Any]:
    query_type = proposal.get("query_type", "")
    catalog = route_plan.get("schema_plan_context", {}).get("catalog", {})
    target_ids = {str(resource.get("resource_id", "")) for resource in proposal.get("target_resources", []) if resource.get("resource_id")}
    if not query_type or not target_ids or not catalog:
        return {
            "proposal_id": proposal.get("proposal_id", ""),
            "query_type": query_type,
            "status": "unknown",
            "reason": "Missing query type, target resource, or catalog.",
            "matched_items": [],
            "unmatched_items": [],
        }

    target_names = _catalog_names_for(catalog, query_type, target_ids)
    matched: list[str] = []
    unmatched: list[str] = []
    for item in proposal.get("schema_items", []):
        item_tokens = _schema_item_tokens(str(item))
        if not item_tokens:
            continue
        if item_tokens & target_names:
            matched.append(str(item))
        else:
            unmatched.append(str(item))

    if not matched and not unmatched:
        status = "unknown"
        reason = "Proposal has no schema items to validate."
    elif unmatched and not matched:
        status = "weak"
        reason = "No proposal schema items were found in the declared target resource."
    elif unmatched:
        status = "partial"
        reason = "Some proposal schema items were not found in the declared target resource."
    else:
        status = "pass"
        reason = "Proposal schema items match the declared target resource."

    return {
        "proposal_id": proposal.get("proposal_id", ""),
        "query_type": query_type,
        "target_resources": sorted(target_ids),
        "status": status,
        "reason": reason,
        "matched_items": matched,
        "unmatched_items": unmatched,
    }


def _planned_step_schema_check(step_spec: dict[str, Any], route_plan: dict[str, Any]) -> dict[str, Any]:
    return _proposal_schema_check(step_spec, route_plan)


def _schema_item_entity(item: str) -> str:
    raw = str(item or "").strip()
    match = re.match(r"^(node|table)\s*[:.]\s*([A-Za-z_][A-Za-z0-9_]*)", raw, flags=re.IGNORECASE)
    if match:
        return match.group(2).lower()
    if re.match(r"^(rel|relationship)\s*[:.]", raw, flags=re.IGNORECASE):
        return ""
    tokens = [token for token in _schema_item_tokens(raw) if token not in ENTITY_STOP_TOKENS]
    return sorted(tokens, key=lambda token: (len(token), token), reverse=True)[0] if tokens else ""


def _infer_entity_type(proposal: dict[str, Any]) -> str:
    reason_tokens = _expanded_tokens(proposal.get("reason", ""))
    for token, entity in RELATION_TARGET_ENTITY_BY_TOKEN.items():
        if token in reason_tokens:
            return entity
    for entity in PREFERRED_OUTPUT_ENTITY_TOKENS:
        if entity in reason_tokens:
            return entity
    for item in proposal.get("schema_items", []):
        entity = _schema_item_entity(str(item))
        if entity:
            return entity
    useful_tokens = sorted(reason_tokens - ENTITY_STOP_TOKENS, key=lambda token: (len(token), token), reverse=True)
    return useful_tokens[0] if useful_tokens else "record"


def _target_resource_ids(proposal: dict[str, Any]) -> list[str]:
    return [
        str(resource.get("resource_id", ""))
        for resource in proposal.get("target_resources", [])
        if resource.get("resource_id")
    ]


def _build_output_contract(step_id: str, output_var: str, proposal: dict[str, Any]) -> dict[str, Any]:
    entity_type = _infer_entity_type(proposal)
    return {
        "variable": output_var,
        "producer_step": step_id,
        "entity_type": entity_type,
        "query_type": proposal.get("query_type", ""),
        "target_resources": _target_resource_ids(proposal),
        "fields": [],
        "primary_value_policy": "prefer_name_or_title_else_first_column",
        "confidence": "draft",
    }


def _build_input_contract(
    *,
    previous_step: dict[str, Any],
    current_step_id: str,
    current_proposal: dict[str, Any],
    route_plan: dict[str, Any],
) -> dict[str, Any]:
    previous_output = previous_step.get("output_contract", {})
    variable = previous_step.get("output_var", "")
    source_query_type = previous_step.get("query_type", "")
    target_query_type = current_proposal.get("query_type", "")
    entity_type = previous_output.get("entity_type", "")
    catalog = route_plan.get("schema_plan_context", {}).get("catalog", {})
    target_ids = set(_target_resource_ids(current_proposal))
    target_names = _catalog_names_for(catalog, target_query_type, target_ids or None)
    proposal_tokens = _expanded_tokens(_proposal_text(current_proposal))
    entity_tokens = _expanded_tokens(entity_type)
    target_overlap = sorted(entity_tokens & target_names)
    proposal_overlap = sorted(entity_tokens & proposal_tokens)

    if not variable:
        compatibility = "unknown"
        policy = "soft_context"
        reason = "Previous step has no output variable."
    elif source_query_type == target_query_type:
        compatibility = "compatible_same_query_type"
        policy = "hard_constraint"
        reason = "Producer and consumer use the same query type."
    elif source_query_type != target_query_type and target_overlap:
        compatibility = "compatible_cross_source_name_overlap"
        policy = "soft_context"
        reason = "The previous entity type appears in the consumer target schema catalog, but cross-source hard binding still requires bridge resolution."
    elif target_overlap:
        compatibility = "compatible_same_source_name_overlap"
        policy = "hard_constraint"
        reason = "The previous entity type appears in the consumer target schema catalog."
    elif proposal_overlap:
        compatibility = "weak_cross_source_proposal_only_alignment"
        policy = "soft_context"
        reason = "The previous entity type appears in the consumer proposal text but not in the target schema catalog."
    else:
        compatibility = "weak_cross_source_entity_alignment"
        policy = "soft_context"
        reason = "The previous entity type does not appear in the consumer proposal or target schema catalog."

    return {
        "variable": variable,
        "consumer_step": current_step_id,
        "required": True,
        "expected_entity_type": entity_type,
        "source_query_type": source_query_type,
        "target_query_type": target_query_type,
        "compatibility": compatibility,
        "consumption_policy": policy,
        "matched_tokens": target_overlap,
        "proposal_only_tokens": proposal_overlap,
        "reason": reason,
    }


def _bridge_requirement(from_step: dict[str, Any], to_step: dict[str, Any], input_contract: dict[str, Any]) -> dict[str, Any]:
    compatibility = input_contract.get("compatibility", "")
    if input_contract.get("source_query_type", "") != input_contract.get("target_query_type", ""):
        status = "required_before_hard_binding"
        action = "Cross-source variables must be resolved by Bridge Resolver before hard binding."
    elif input_contract.get("consumption_policy") == "hard_constraint":
        status = "not_required"
        action = "No bridge step is needed before runtime execution."
    elif compatibility == "weak_cross_source_proposal_only_alignment":
        status = "required_before_hard_binding"
        action = "A future Step Planner should add a bridge or mapping step before this variable can be used as a hard constraint."
    elif compatibility == "weak_cross_source_entity_alignment":
        status = "required_before_hard_binding"
        action = "A future Step Planner should either add a bridge step or keep the variable as soft context."
    else:
        status = "unknown"
        action = "A future Step Planner should clarify the variable mapping."
    return {
        "bridge_id": f"BR{to_step.get('step_id', '')[2:] or '1'}",
        "step_type": "bridge_mapping",
        "from_step": from_step.get("step_id", ""),
        "to_step": to_step.get("step_id", ""),
        "input_var": input_contract.get("variable", ""),
        "source_entity_type": input_contract.get("expected_entity_type", ""),
        "source_query_type": input_contract.get("source_query_type", ""),
        "target_query_type": input_contract.get("target_query_type", ""),
        "status": status,
        "execution_enabled": False,
        "reason": input_contract.get("reason", ""),
        "suggested_action": action,
    }


def build_multi_step_plan(question: str, route_plan: dict[str, Any]) -> dict[str, Any]:
    task_analysis = route_plan.get("task_analysis", {})
    requires_multi_step = (
        route_plan.get("task_mode") == "multi_step_candidate"
        or task_analysis.get("requires_system_multi_step", False)
    )
    if not requires_multi_step:
        return {
            "plan_version": MULTI_STEP_PLAN_VERSION,
            "status": "not_required",
            "reason": "Task analysis does not require a system-level multi-step plan.",
            "steps": [],
        }

    raw_proposals = _future_plan_proposals(route_plan) or _schema_plan_proposals(route_plan)
    proposals, step_planner_diagnostics = _build_step_planner_proposals(question, route_plan, raw_proposals)
    proposals, paired_alignment_diagnostics = _align_paired_counterpart_resources(proposals, route_plan)
    ordered_raw_proposals, ordering_diagnostics = _ordered_proposals(question, route_plan, raw_proposals)
    proposal_checks = [_proposal_schema_check(proposal, route_plan) for proposal in ordered_raw_proposals]
    step_schema_checks = [_planned_step_schema_check(proposal, route_plan) for proposal in proposals]
    steps: list[dict[str, Any]] = []
    for index, proposal in enumerate(proposals, start=1):
        step_id = f"MS{index}"
        output_var = f"step_{index}_result"
        output_contract = _build_output_contract(step_id, output_var, proposal)
        input_contracts = (
            [
                _build_input_contract(
                    previous_step=steps[-1],
                    current_step_id=step_id,
                    current_proposal=proposal,
                    route_plan=route_plan,
                )
            ]
            if steps
            else []
        )
        steps.append(
            {
                "step_id": step_id,
                "step_type": "query",
                "step_goal": (
                    proposal.get("reason", "")
                    or (
                        "Produce an intermediate result for the next step."
                        if index == 1 and len(proposals) > 1
                        else "Use available context to answer the user request."
                    )
                ),
                "query_type": proposal.get("query_type", ""),
                "proposal_id": proposal.get("proposal_id", ""),
                "query_strategy": proposal.get("query_strategy", ""),
                "query_shape": proposal.get("query_shape", ""),
                "schema_items": proposal.get("schema_items", []),
                "target_resources": proposal.get("target_resources", []),
                "depends_on": [steps[-1]["step_id"]] if steps else [],
                "input_vars": [steps[-1]["output_var"]] if steps else [],
                "input_contracts": input_contracts,
                "output_var": output_var,
                "output_contract": output_contract,
                "status": "pending",
                "execution_enabled": True,
                "planner_source": proposal.get("source", "step_planner"),
                "supporting_proposal_id": proposal.get("supporting_proposal_id", ""),
                "step_segment": proposal.get("step_segment", ""),
            }
        )

    if not steps:
        steps.append(
            {
                "step_id": "MS1",
                "step_type": "planning_placeholder",
                "step_goal": "Decompose the user question into executable query steps.",
                "query_type": "",
                "proposal_id": "",
                "depends_on": [],
                "input_vars": [],
                "input_contracts": [],
                "output_var": "planned_steps",
                "output_contract": {
                    "variable": "planned_steps",
                    "producer_step": "MS1",
                    "entity_type": "planned_step",
                    "query_type": "",
                    "target_resources": [],
                    "fields": [],
                    "primary_value_policy": "not_applicable",
                    "confidence": "draft",
                },
                "status": "needs_planner_llm",
                "execution_enabled": False,
            }
        )

    return {
        "plan_version": MULTI_STEP_PLAN_VERSION,
        "status": "planned_not_executed",
        "question": question,
        "reason": "The task has system-level multi-step signals. Step Planner generated a step graph; the default runtime does not execute it unless multi-step runtime is explicitly enabled.",
        "execution_policy": {
            "mode": "execute_draft_steps_when_runtime_enabled",
            "top_k_proposals": MAX_MULTI_STEP_DRAFT_STEPS,
            "execute_in_current_runtime": bool(steps),
        },
        "planner_diagnostics": {
            "step_planner": step_planner_diagnostics,
            "paired_resource_alignment": paired_alignment_diagnostics,
            "ordering": ordering_diagnostics,
            "proposal_schema_checks": proposal_checks,
            "planned_step_schema_checks": step_schema_checks,
        },
        "steps": steps,
        "variable_store": {},
        "cross_step_contracts": [
            {
                "from_step": steps[index - 1]["step_id"],
                "to_step": steps[index]["step_id"],
                "variable": steps[index - 1]["output_var"],
                "status": "draft",
                "input_contract": (steps[index].get("input_contracts") or [{}])[0],
            }
            for index in range(1, len(steps))
        ],
        "bridge_steps": [
            _bridge_requirement(steps[index - 1], steps[index], (steps[index].get("input_contracts") or [{}])[0])
            for index in range(1, len(steps))
        ],
    }
