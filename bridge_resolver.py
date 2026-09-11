import sqlite3
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase

from config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER
from cypher_pipeline_roles import execute_cypher
from paired_benchmark_resources import resolve_actual_cypher_database
from schema_plan_proposer import resolve_sql_resource_path
from sqlite_utils import connect_sqlite
from sql_pipeline_roles import execute_sql

# 中文函数说明索引：
# - _normalize_bridge_values(source_payload)：从上一步 typed variable 中抽取去重后的 source primary values。
# - _target_resource(target_step)：读取目标 step 的数据库资源，决定 resolver 查哪个 SQL/Cypher 库。
# - _quote_cypher_literal/_quote_sql_identifier/_quote_sql_literal：构造 bridge lookup 查询时的安全转义。
# - _canonical_bridge_value/_unique_candidates：按 target id/value/field 对候选去重并判断唯一性。
# - target_labels_from_step(step)：从 output_contract/schema_items 推断 Cypher 目标 label，例如 Team/Player。
# - build_cypher_bridge_query(values, target_labels)：先 exact lookup 同名源节点，再沿短路径寻找目标 label 节点。
# - build_sql_bridge_query(conn, values)：扫描 SQL 目标库 name/title/label 类文本列，构造 UNION exact lookup。
# - _source_value_coverage/_bridge_resolution_assessment：评估候选数、唯一值数、source 覆盖率、match_type 和是否 allow_hard_constraint。
# - bridge_result_from_execution(...)：把 bridge query 执行结果转换为 runtime bridge result。
# - resolve_bridge_by_direct_lookup(...)：Coordinator 默认优先使用的 direct resolver；只在 high-confidence unique mapping 时硬绑定。
# 状态说明：
# - exact_unique：目标库 exact lookup 找到唯一候选，且 source values 覆盖完整。
# - relation_path_unique：图内短路径找到唯一目标候选。
# - ambiguous_exact_lookup / ambiguous_relation_expansion：多候选或不唯一，只能 ambiguous_soft_context。
# - partial_exact_lookup / none：覆盖不足或无候选，保持 unresolved_soft_context。


MAX_BRIDGE_VALUES = 25
MAX_BRIDGE_CANDIDATES = 50
NAME_FIELD_TOKENS = ("name", "title", "label")
ID_FIELD_TOKENS = ("id", "identifier", "key")
MAX_CYPHER_BRIDGE_HOPS = 2
ENTITY_LABEL_ALIASES = {
    "student": ["Highschooler", "Student"],
    "highschooler": ["Highschooler", "Student"],
}
ENTITY_TABLE_ALIASES = {
    "student": ["Highschooler"],
    "highschooler": ["Highschooler"],
    "player": ["players"],
    "character": ["Character"],
    "film": ["Film"],
    "planet": ["Planet"],
    "species": ["Species"],
    "starship": ["Starship"],
    "vehicle": ["Vehicle"],
}


def _normalize_value_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        values = [values]
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        lowered = text.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        normalized.append(text[:200])
        if len(normalized) >= MAX_BRIDGE_VALUES:
            break
    return normalized


def _expand_bridge_value_variants(values: list[str]) -> list[str]:
    variants: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        variants.append(text)
        parts = [part.strip() for part in text.split(":") if part.strip()]
        if len(parts) >= 2:
            terminal = parts[-1]
            if terminal and terminal.lower() != text.lower():
                variants.append(terminal)
    return _normalize_value_list(variants)


def _entity_type_from_bridge_value(value: Any) -> str:
    text = str(value or "").strip()
    parts = [part.strip() for part in text.split(":") if part.strip()]
    if len(parts) >= 3:
        return parts[-2].lower()
    return ""


def _source_value_entity_types(source_payload: dict[str, Any]) -> list[str]:
    values: list[Any] = []
    values.extend(_normalize_value_list(source_payload.get("primary_values", [])))

    def collect_bridge_ids(value: Any, field: str = "") -> None:
        lowered = str(field).lower()
        if lowered in {"bridge_id"} or lowered.endswith("_bridge_id"):
            values.append(value)
            return
        if isinstance(value, dict):
            for nested_field, nested_value in value.items():
                collect_bridge_ids(nested_value, str(nested_field))
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                collect_bridge_ids(item, field)

    collect_bridge_ids(source_payload.get("primary_values", []))
    records = source_payload.get("records", [])
    if isinstance(records, list):
        for record in records:
            collect_bridge_ids(record)
    entity_types: list[str] = []
    seen: set[str] = set()
    for value in values:
        entity_type = _entity_type_from_bridge_value(value)
        if entity_type and entity_type not in seen:
            seen.add(entity_type)
            entity_types.append(entity_type)
    return entity_types


def _record_field_values(source_payload: dict[str, Any], tokens: tuple[str, ...]) -> list[str]:
    records = source_payload.get("records", [])
    if not isinstance(records, list):
        return []

    values: list[Any] = []

    def bridge_id_terminal_value(value: Any) -> str:
        if not isinstance(value, str):
            return ""
        parts = [part.strip() for part in value.strip().split(":") if part.strip()]
        if len(parts) < 2:
            return ""
        return parts[-1]

    def collect(value: Any, field: str = "") -> None:
        lowered = str(field).lower()
        if lowered in {"bridge_id"} or lowered.endswith("_bridge_id"):
            terminal = bridge_id_terminal_value(value)
            if terminal:
                values.append(terminal)
            return
        if isinstance(value, dict):
            for nested_field, nested_value in value.items():
                collect(nested_value, str(nested_field))
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                collect(item, field)
            return
        if field and any(token in lowered for token in tokens):
            values.append(value)

    for record in records:
        if not isinstance(record, dict):
            continue
        for field, value in record.items():
            collect(value, str(field))
    return _normalize_value_list(values)


def _normalize_bridge_values(source_payload: dict[str, Any], *, prefer_identifiers: bool = False) -> list[str]:
    if prefer_identifiers:
        identifier_values = _record_field_values(source_payload, ID_FIELD_TOKENS)
        if identifier_values:
            expanded = _expand_bridge_value_variants(identifier_values)
            terminal_identifier_values = [
                value
                for value in expanded
                if any(str(raw).strip().endswith(f":{value}") for raw in identifier_values)
            ]
            return terminal_identifier_values or expanded
    primary_values = _normalize_value_list(source_payload.get("primary_values", []))
    if primary_values:
        return _expand_bridge_value_variants(primary_values)
    name_values = _record_field_values(source_payload, NAME_FIELD_TOKENS)
    if name_values:
        return _expand_bridge_value_variants(name_values)
    return _expand_bridge_value_variants(_record_field_values(source_payload, ID_FIELD_TOKENS))


def _target_resource(target_step: dict[str, Any]) -> dict[str, Any]:
    resources = target_step.get("target_resources", [])
    if isinstance(resources, list) and resources and isinstance(resources[0], dict):
        return resources[0]
    return {}


def _quote_cypher_literal(value: str) -> str:
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def _quote_sql_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _quote_sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _canonical_bridge_value(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _looks_like_identifier_values(values: list[str]) -> bool:
    normalized = [str(value or "").strip() for value in values if str(value or "").strip()]
    if not normalized:
        return False
    return all(
        value.isdigit()
        or ":" in value
        for value in normalized
    )


def _clean_label(value: str) -> str:
    cleaned = "".join(ch for ch in value.strip() if ch.isalnum() or ch == "_")
    if not cleaned:
        return ""
    return cleaned[:1].upper() + cleaned[1:]


def _label_candidates(value: Any) -> list[str]:
    raw = str(value or "").strip()
    if not raw:
        return []
    lowered = raw.lower()
    labels = ENTITY_LABEL_ALIASES.get(lowered, [])
    if not labels:
        labels = [_clean_label(raw)]
        if lowered.endswith("s") and len(lowered) > 1:
            labels.append(_clean_label(raw[:-1]))
    clean: list[str] = []
    seen: set[str] = set()
    for label in labels:
        cleaned = _clean_label(label)
        lowered_label = cleaned.lower()
        if not cleaned or lowered_label in seen:
            continue
        seen.add(lowered_label)
        clean.append(cleaned)
    return clean


def target_labels_from_step(target_step: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()

    def add_label(raw: str) -> None:
        label = _clean_label(raw)
        if not label or label.lower() in {"node", "relationship", "planned_step"}:
            return
        lowered = label.lower()
        if lowered in seen:
            return
        seen.add(lowered)
        labels.append(label)

    output_entity = target_step.get("output_contract", {}).get("entity_type", "")
    if output_entity:
        add_label(str(output_entity))
    for item in target_step.get("schema_items", []):
        text = str(item)
        lowered = text.lower()
        if lowered.startswith("node:"):
            add_label(text.split(":", 1)[1])
        elif lowered.startswith("label:"):
            add_label(text.split(":", 1)[1])
    return labels[:3]


def target_tables_from_step(target_step: dict[str, Any]) -> list[str]:
    """Infer SQL target tables from output contracts and schema hints."""
    tables: list[str] = []
    seen: set[str] = set()

    def add_table(raw: Any) -> None:
        text = str(raw or "").strip()
        if not text:
            return
        candidates = ENTITY_TABLE_ALIASES.get(text.lower(), [_clean_label(text)])
        for candidate in candidates:
            table = candidate.strip()
            lowered = table.lower()
            if not table or lowered in seen:
                continue
            seen.add(lowered)
            tables.append(table)

    output_entity = target_step.get("output_contract", {}).get("entity_type", "")
    add_table(output_entity)
    for item in target_step.get("schema_items", []):
        text = str(item)
        lowered = text.lower()
        if lowered.startswith("table:") or lowered.startswith("sql_table:"):
            add_table(text.split(":", 1)[1])
        elif target_step.get("query_type") == "sql" and ":" not in text and not lowered.startswith("rel"):
            table = text.strip()
            if table and table.lower() not in seen:
                seen.add(table.lower())
                tables.append(table)
    return tables[:3]


def narrow_sql_target_tables_for_source(
    target_tables: list[str],
    source_payload: dict[str, Any],
) -> list[str]:
    """Prefer the entity table encoded in typed bridge ids such as `domain:planet:88`."""
    source_entity_tables: list[str] = []
    seen: set[str] = set()
    for entity_type in _source_value_entity_types(source_payload):
        for table in ENTITY_TABLE_ALIASES.get(entity_type, [_clean_label(entity_type)]):
            lowered = table.lower()
            if table and lowered not in seen:
                seen.add(lowered)
                source_entity_tables.append(table)
    if not source_entity_tables:
        return target_tables
    if not target_tables:
        return source_entity_tables[:3]
    allowed = {table.lower() for table in source_entity_tables}
    narrowed = [table for table in target_tables if table.lower() in allowed]
    return narrowed or target_tables


def source_entity_labels_from_bridge(bridge: dict[str, Any], source_payload: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    for value in (
        bridge.get("source_entity_type", ""),
        source_payload.get("entity_type", ""),
        source_payload.get("contract", {}).get("entity_type", ""),
    ):
        for label in _label_candidates(value):
            if label.lower() not in {item.lower() for item in labels}:
                labels.append(label)
    return labels[:3]


def build_cypher_bridge_query(
    values: list[str],
    target_labels: list[str] | None = None,
    *,
    exact_labels: list[str] | None = None,
) -> str:
    """Build a conservative graph bridge query over name-like properties."""
    lowered_values = [_quote_cypher_literal(value.lower()) for value in values]
    value_list = ", ".join(lowered_values)
    exact = [label for label in (exact_labels or []) if _clean_label(label)]
    if exact:
        label_list = ", ".join(_quote_cypher_literal(_clean_label(label)) for label in exact)
        return (
            f"WITH [{value_list}] AS bridge_values, [{label_list}] AS exact_labels\n"
            "MATCH (n)\n"
            "WHERE any(label IN labels(n) WHERE label IN exact_labels)\n"
            "UNWIND keys(n) AS property\n"
            "WITH n, property, bridge_values, n[property] AS raw_value\n"
            "WHERE raw_value IS NOT NULL\n"
            "  AND property =~ '(?i).*(name|title|label|id|identifier|key).*'\n"
            "WITH n, property, raw_value, bridge_values, split(toString(raw_value), ':') AS raw_parts\n"
            "WITH n, property, raw_value, bridge_values, raw_parts[size(raw_parts) - 1] AS terminal_value\n"
            "WHERE toLower(toString(raw_value)) IN bridge_values\n"
            "   OR toLower(toString(terminal_value)) IN bridge_values\n"
            "WITH n, property,\n"
            "     CASE WHEN toLower(property) = 'bridge_id' OR toLower(property) ENDS WITH '_bridge_id'\n"
            "          THEN terminal_value ELSE raw_value END AS mapped_value\n"
            "RETURN labels(n) AS target_labels,\n"
            "       property AS target_field,\n"
            "       mapped_value AS target_value,\n"
            "       elementId(n) AS target_id,\n"
            "       labels(n) AS source_labels,\n"
            "       elementId(n) AS source_id,\n"
            "       0 AS bridge_hops\n"
            "ORDER BY target_field, target_value\n"
            f"LIMIT {MAX_BRIDGE_CANDIDATES}"
        )
    labels = [label for label in (target_labels or []) if _clean_label(label)]
    if labels:
        label_list = ", ".join(_quote_cypher_literal(_clean_label(label)) for label in labels)
        return (
            f"WITH [{value_list}] AS bridge_values, [{label_list}] AS target_labels\n"
            "MATCH (source)\n"
            "UNWIND keys(source) AS source_property\n"
            "WITH source, source_property, bridge_values, target_labels, source[source_property] AS source_value\n"
            "WHERE source_value IS NOT NULL\n"
            "  AND source_property =~ '(?i).*(name|title|label).*'\n"
            "  AND toLower(toString(source_value)) IN bridge_values\n"
            "WITH DISTINCT source, target_labels\n"
            f"MATCH path = (source)-[*0..{MAX_CYPHER_BRIDGE_HOPS}]-(target)\n"
            "WHERE any(label IN labels(target) WHERE label IN target_labels)\n"
            "UNWIND keys(target) AS target_property\n"
            "WITH source, target, target_property, target[target_property] AS target_value, length(path) AS bridge_hops\n"
            "WHERE target_value IS NOT NULL\n"
            "  AND target_property =~ '(?i).*(name|title|label).*'\n"
            "RETURN labels(target) AS target_labels,\n"
            "       target_property AS target_field,\n"
            "       target_value AS target_value,\n"
            "       elementId(target) AS target_id,\n"
            "       labels(source) AS source_labels,\n"
            "       elementId(source) AS source_id,\n"
            "       bridge_hops AS bridge_hops\n"
            "ORDER BY bridge_hops, target_field, target_value\n"
            f"LIMIT {MAX_BRIDGE_CANDIDATES}"
        )
    return (
        f"WITH [{value_list}] AS bridge_values\n"
        "MATCH (n)\n"
        "UNWIND keys(n) AS property\n"
        "WITH n, property, bridge_values, n[property] AS raw_value\n"
        "WHERE raw_value IS NOT NULL\n"
        "  AND property =~ '(?i).*(name|title|label).*'\n"
        "  AND toLower(toString(raw_value)) IN bridge_values\n"
        "RETURN labels(n) AS target_labels,\n"
        "       property AS target_field,\n"
        "       raw_value AS target_value,\n"
        "       elementId(n) AS target_id,\n"
        "       labels(n) AS source_labels,\n"
        "       elementId(n) AS source_id,\n"
        "       0 AS bridge_hops\n"
        f"LIMIT {MAX_BRIDGE_CANDIDATES}"
    )


def _sqlite_lookup_columns(
    conn: sqlite3.Connection,
    source_payload: dict[str, Any] | None = None,
    target_tables: list[str] | None = None,
    values: list[str] | None = None,
) -> list[tuple[str, str, bool]]:
    tables = [
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    ]
    if target_tables:
        allowed = {table.lower() for table in target_tables}
        tables = [table for table in tables if table.lower() in allowed]
    source_field = str((source_payload or {}).get("primary_field") or "").lower()
    prefer_identifier_only = (
        bool(source_field and any(token in source_field for token in ID_FIELD_TOKENS))
        or _looks_like_identifier_values(values or [])
    )
    columns: list[tuple[str, str, bool]] = []
    for table in tables:
        for column in conn.execute(f"PRAGMA table_info({_quote_sql_identifier(table)})").fetchall():
            column_name = str(column[1])
            column_type = str(column[2] or "").lower()
            lowered = column_name.lower()
            is_identifier = (
                any(token in lowered for token in ID_FIELD_TOKENS)
                or (source_field and lowered == source_field)
            )
            is_name_like = any(token in lowered for token in NAME_FIELD_TOKENS)
            is_text_like = "char" in column_type or "text" in column_type
            if prefer_identifier_only:
                if is_identifier:
                    columns.append((table, column_name, is_identifier))
                continue
            if is_identifier or is_name_like or is_text_like:
                columns.append((table, column_name, is_identifier))
    return columns


def _sqlite_compound_name_selects(
    conn: sqlite3.Connection,
    values: list[str],
    target_tables: list[str] | None = None,
) -> list[str]:
    value_set = {value.strip() for value in values if " " in value.strip()}
    if not value_set:
        return []
    lowered_values = ", ".join(_quote_sql_literal(value.lower()) for value in sorted(value_set))
    selects: list[str] = []
    tables = [
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    ]
    if target_tables:
        allowed = {table.lower() for table in target_tables}
        tables = [table for table in tables if table.lower() in allowed]
    for table in tables:
        columns = [str(column[1]) for column in conn.execute(f"PRAGMA table_info({_quote_sql_identifier(table)})").fetchall()]
        lowered_to_original = {column.lower(): column for column in columns}
        first_name = lowered_to_original.get("first_name")
        last_name = lowered_to_original.get("last_name")
        if not first_name or not last_name:
            continue
        id_columns = [
            column
            for column in columns
            if any(token in column.lower() for token in ID_FIELD_TOKENS)
            and not column.lower().endswith("_bridge_id")
        ]
        if not id_columns:
            continue
        id_column = sorted(id_columns, key=lambda column: (0 if column.lower().endswith("_id") else 1, column.lower()))[0]
        full_name = (
            f"lower(trim(CAST({_quote_sql_identifier(first_name)} AS TEXT) || ' ' || "
            f"CAST({_quote_sql_identifier(last_name)} AS TEXT)))"
        )
        reversed_name = (
            f"lower(trim(CAST({_quote_sql_identifier(last_name)} AS TEXT) || ' ' || "
            f"CAST({_quote_sql_identifier(first_name)} AS TEXT)))"
        )
        selects.append(
            "SELECT DISTINCT "
            f"{_quote_sql_literal(table)} AS target_table, "
            f"{_quote_sql_literal(id_column)} AS target_field, "
            f"CAST({_quote_sql_identifier(id_column)} AS TEXT) AS target_value, "
            f"CAST({_quote_sql_identifier(id_column)} AS TEXT) AS target_id "
            f"FROM {_quote_sql_identifier(table)} "
            f"WHERE {full_name} IN ({lowered_values}) OR {reversed_name} IN ({lowered_values})"
        )
    return selects


def build_sql_bridge_query(
    conn: sqlite3.Connection,
    values: list[str],
    source_payload: dict[str, Any] | None = None,
    target_tables: list[str] | None = None,
) -> str:
    """Build a conservative UNION query over ID and name-like SQLite columns."""
    lowered_values = ", ".join(_quote_sql_literal(value.lower()) for value in values)
    selects: list[str] = []
    for table, column, is_identifier in _sqlite_lookup_columns(
        conn,
        source_payload,
        target_tables=target_tables,
        values=values,
    ):
        target_id_expr = f"CAST({_quote_sql_identifier(column)} AS TEXT)" if is_identifier else "rowid"
        selects.append(
            "SELECT DISTINCT "
            f"{_quote_sql_literal(table)} AS target_table, "
            f"{_quote_sql_literal(column)} AS target_field, "
            f"CAST({_quote_sql_identifier(column)} AS TEXT) AS target_value, "
            f"{target_id_expr} AS target_id "
            f"FROM {_quote_sql_identifier(table)} "
            f"WHERE lower(CAST({_quote_sql_identifier(column)} AS TEXT)) IN ({lowered_values})"
        )
    selects.extend(_sqlite_compound_name_selects(conn, values, target_tables=target_tables))
    if not selects:
        return ""
    return " UNION ALL ".join(selects) + f" LIMIT {MAX_BRIDGE_CANDIDATES}"


def _row_item(row: Any, columns: list[str], name: str, index: int) -> Any:
    if isinstance(row, dict):
        return row.get(name, "")
    if isinstance(row, (list, tuple)):
        if name in columns:
            column_index = columns.index(name)
            if column_index < len(row):
                return row[column_index]
        if index < len(row):
            return row[index]
    return ""


def _unique_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for candidate in candidates:
        key = (
            _canonical_bridge_value(candidate.get("target_value", "")),
            _canonical_bridge_value(candidate.get("target_field", "")),
            _canonical_bridge_value(candidate.get("target_id", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _source_value_coverage(values: list[str], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    source_values = {_canonical_bridge_value(value) for value in values if _canonical_bridge_value(value)}
    matched_values = {
        _canonical_bridge_value(candidate.get("target_value", ""))
        for candidate in candidates
        if _canonical_bridge_value(candidate.get("target_value", "")) in source_values
    }
    return {
        "source_value_count": len(source_values),
        "matched_source_value_count": len(matched_values),
        "all_source_values_matched": bool(source_values) and matched_values == source_values,
    }


def _bridge_resolution_assessment(values: list[str], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    unique = _unique_candidates(candidates)
    unique_by_target_id: dict[str, dict[str, Any]] = {}
    for candidate in unique:
        target_id = _canonical_bridge_value(candidate.get("target_id", ""))
        if not target_id:
            continue
        existing = unique_by_target_id.get(target_id)
        if existing is None or _candidate_rank(candidate) < _candidate_rank(existing):
            unique_by_target_id[target_id] = candidate
    if len(unique_by_target_id) == 1 and len(unique) > 1:
        canonical_candidate = next(iter(unique_by_target_id.values()))
        unique = [canonical_candidate]
    unique_values = {
        _canonical_bridge_value(candidate.get("target_value", ""))
        for candidate in unique
        if _canonical_bridge_value(candidate.get("target_value", ""))
    }
    unique_target_ids = {
        _canonical_bridge_value(candidate.get("target_id", candidate.get("target_value", "")))
        for candidate in unique
        if _canonical_bridge_value(candidate.get("target_id", candidate.get("target_value", "")))
    }
    hop_values = [candidate.get("bridge_hops") for candidate in unique if candidate.get("bridge_hops") != ""]
    relation_expansion = any(str(value) not in {"0", "0.0"} for value in hop_values)
    exact_lookup = bool(unique) and not relation_expansion
    coverage = _source_value_coverage(values, unique)
    candidate_count = len(unique)
    unique_value_count = len(unique_values)
    unique_target_count = len(unique_target_ids)
    is_unique_target = candidate_count == 1 and unique_value_count == 1
    is_unique_value = (
        unique_value_count == 1
        and unique_target_count <= 1
        and coverage.get("all_source_values_matched")
    )
    is_exact_many_value_set = (
        exact_lookup
        and coverage.get("source_value_count", 0) > 1
        and coverage.get("all_source_values_matched")
        and unique_value_count == coverage.get("source_value_count")
        and unique_target_count == coverage.get("source_value_count")
    )

    if not unique:
        return {
            "candidates": unique,
            "candidate_count": 0,
            "unique_value_count": 0,
            "unique_target_count": 0,
            "is_unique_target": False,
            "match_type": "none",
            "confidence": "low",
            "allow_hard_constraint": False,
            "resolver_strategy": "exact_or_relation_lookup",
            "effective_consumption_policy": "soft_context",
            "runtime_status": "unresolved_soft_context",
            "runtime_reason": "Direct bridge resolver found no exact target-side candidates.",
            **coverage,
        }

    if is_exact_many_value_set:
        return {
            "candidates": unique,
            "candidate_count": candidate_count,
            "unique_value_count": unique_value_count,
            "unique_target_count": unique_target_count,
            "is_unique_target": False,
            "match_type": "exact_unique_many",
            "confidence": "high",
            "allow_hard_constraint": True,
            "resolver_strategy": "exact_node_lookup",
            "effective_consumption_policy": "hard_constraint",
            "runtime_status": "resolved",
            "runtime_reason": "Direct bridge resolver found a complete exact target-side set for all source values.",
            **coverage,
        }

    if exact_lookup and is_unique_value:
        return {
            "candidates": unique,
            "candidate_count": candidate_count,
            "unique_value_count": unique_value_count,
            "unique_target_count": unique_target_count,
            "is_unique_target": True,
            "match_type": "exact_unique" if candidate_count == 1 else "exact_unique_value_multi_field",
            "confidence": "high",
            "allow_hard_constraint": True,
            "resolver_strategy": "exact_node_lookup",
            "effective_consumption_policy": "hard_constraint",
            "runtime_status": "resolved",
            "runtime_reason": (
                "Direct bridge resolver found one exact target-side value."
                if candidate_count > 1
                else "Direct bridge resolver found one exact target-side candidate."
            ),
            **coverage,
        }

    if not is_unique_target:
        return {
            "candidates": unique,
            "candidate_count": candidate_count,
            "unique_value_count": unique_value_count,
            "unique_target_count": unique_target_count,
            "is_unique_target": False,
            "match_type": "ambiguous_relation_expansion" if relation_expansion else "ambiguous_exact_lookup",
            "confidence": "low",
            "allow_hard_constraint": False,
            "resolver_strategy": "relation_expansion" if relation_expansion else "exact_node_lookup",
            "effective_consumption_policy": "soft_context",
            "runtime_status": "ambiguous_soft_context",
            "runtime_reason": "Direct bridge resolver found multiple target-side candidates; hard binding is unsafe.",
            **coverage,
        }

    if relation_expansion and is_unique_target:
        return {
            "candidates": unique,
            "candidate_count": candidate_count,
            "unique_value_count": unique_value_count,
            "unique_target_count": unique_target_count,
            "is_unique_target": True,
            "match_type": "relation_path_unique",
            "confidence": "high",
            "allow_hard_constraint": True,
            "resolver_strategy": "relation_expansion",
            "effective_consumption_policy": "hard_constraint",
            "runtime_status": "resolved",
            "runtime_reason": "Direct bridge resolver found one target-side candidate through a short graph path.",
            **coverage,
        }

    return {
        "candidates": unique,
        "candidate_count": candidate_count,
        "unique_value_count": unique_value_count,
        "unique_target_count": unique_target_count,
        "is_unique_target": is_unique_target,
        "match_type": "partial_exact_lookup",
        "confidence": "medium",
        "allow_hard_constraint": False,
        "resolver_strategy": "exact_node_lookup",
        "effective_consumption_policy": "soft_context",
        "runtime_status": "unresolved_soft_context",
        "runtime_reason": "Direct bridge resolver did not cover all source values; hard binding is unsafe.",
        **coverage,
    }


def _candidate_rank(candidate: dict[str, Any]) -> tuple[int, str]:
    field = str(candidate.get("target_field", "")).lower()
    if field in {"id", "player_id", "singer_id", "student_id"}:
        return (0, field)
    if field.endswith("_id") and not any(role in field for role in ("winner", "loser", "match")):
        return (1, field)
    if any(token in field for token in NAME_FIELD_TOKENS):
        return (2, field)
    if any(token in field for token in ID_FIELD_TOKENS):
        return (3, field)
    return (4, field)


def _dedupe_target_values(candidates: list[dict[str, Any]]) -> list[Any]:
    values: list[Any] = []
    seen: set[str] = set()
    for candidate in sorted(candidates, key=_candidate_rank):
        value = candidate.get("target_value")
        key = _canonical_bridge_value(value)
        if not key or key in seen:
            continue
        seen.add(key)
        values.append(value)
        if len(values) >= MAX_BRIDGE_VALUES:
            break
    return values


def _best_candidate_per_target_id(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    passthrough: list[dict[str, Any]] = []
    for candidate in candidates:
        target_id = _canonical_bridge_value(candidate.get("target_id", ""))
        if not target_id:
            passthrough.append(candidate)
            continue
        existing = best.get(target_id)
        if existing is None or _candidate_rank(candidate) < _candidate_rank(existing):
            best[target_id] = candidate
    return sorted(list(best.values()) + passthrough, key=_candidate_rank)


def bridge_result_from_execution(
    *,
    bridge: dict[str, Any],
    values: list[str],
    query_type: str,
    selected_db: str,
    bridge_query: str,
    execution: dict[str, Any],
) -> dict[str, Any]:
    columns = [str(column) for column in execution.get("columns", [])]
    rows = execution.get("rows", [])
    target_values: list[Any] = []
    candidates: list[dict[str, Any]] = []
    if isinstance(rows, list):
        for row in rows[:MAX_BRIDGE_CANDIDATES]:
            target_value = _row_item(row, columns, "target_value", 2)
            if target_value == "":
                continue
            target_values.append(target_value)
            candidates.append(
                {
                    "target_value": target_value,
                    "target_field": _row_item(row, columns, "target_field", 1),
                    "target_id": _row_item(row, columns, "target_id", 3),
                    "target_labels": _row_item(row, columns, "target_labels", 0),
                    "source_labels": _row_item(row, columns, "source_labels", 4),
                    "source_id": _row_item(row, columns, "source_id", 5),
                    "bridge_hops": _row_item(row, columns, "bridge_hops", 6),
                }
            )
    assessment = _bridge_resolution_assessment(values, candidates)

    resolver_pipeline = {
        "query_type": query_type,
        "selected_db": selected_db,
        "query": bridge_query,
        "row_count": execution.get("row_count", 0),
        "error": execution.get("error") or "",
    }

    if execution.get("success") and assessment.get("allow_hard_constraint"):
        assessed_candidates = _best_candidate_per_target_id(assessment.get("candidates", []))
        return {
            "runtime_status": assessment["runtime_status"],
            "effective_consumption_policy": assessment["effective_consumption_policy"],
            "target_field": assessed_candidates[0].get("target_field", "target_value"),
            "target_values": _dedupe_target_values(assessed_candidates),
            "candidates": assessed_candidates,
            "candidate_count": assessment["candidate_count"],
            "unique_value_count": assessment["unique_value_count"],
            "unique_target_count": assessment["unique_target_count"],
            "is_unique_target": assessment["is_unique_target"],
            "match_type": assessment["match_type"],
            "allow_hard_constraint": assessment["allow_hard_constraint"],
            "confidence": assessment["confidence"],
            "resolver": "direct_exact_lookup_v1",
            "resolver_strategy": assessment["resolver_strategy"],
            "resolver_pipeline": resolver_pipeline,
            "runtime_reason": assessment["runtime_reason"],
            "source_value_count": assessment["source_value_count"],
            "matched_source_value_count": assessment["matched_source_value_count"],
            "all_source_values_matched": assessment["all_source_values_matched"],
        }

    return {
        "runtime_status": assessment["runtime_status"],
        "effective_consumption_policy": assessment["effective_consumption_policy"],
        "target_values": [],
        "candidates": assessment.get("candidates", []),
        "candidate_count": assessment["candidate_count"],
        "unique_value_count": assessment["unique_value_count"],
        "unique_target_count": assessment["unique_target_count"],
        "is_unique_target": assessment["is_unique_target"],
        "match_type": assessment["match_type"],
        "allow_hard_constraint": assessment["allow_hard_constraint"],
        "confidence": assessment["confidence"],
        "resolver": "direct_exact_lookup_v1",
        "resolver_strategy": assessment["resolver_strategy"],
        "resolver_pipeline": resolver_pipeline,
        "runtime_reason": assessment["runtime_reason"],
        "source_value_count": assessment["source_value_count"],
        "matched_source_value_count": assessment["matched_source_value_count"],
        "all_source_values_matched": assessment["all_source_values_matched"],
    }


def resolve_bridge_by_direct_lookup(
    bridge: dict[str, Any],
    source_payload: dict[str, Any],
    target_step: dict[str, Any],
) -> dict[str, Any]:
    target_query_type = bridge.get("target_query_type") or target_step.get("query_type", "")
    values = _normalize_bridge_values(
        source_payload,
        prefer_identifiers=target_query_type in {"sql", "cypher"},
    )
    if not values:
        return {
            "runtime_status": "missing_input",
            "effective_consumption_policy": "soft_context",
            "resolver": "direct_exact_lookup_v1",
            "runtime_reason": "Direct bridge resolver had no source values to map.",
        }

    resource = _target_resource(target_step)
    resource_id = resource.get("resource_id", "")
    if target_query_type == "cypher" and resource_id:
        exact_labels = source_entity_labels_from_bridge(bridge, source_payload)
        bridge_query = build_cypher_bridge_query(
            values,
            target_labels=target_labels_from_step(target_step),
            exact_labels=exact_labels,
        )
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        try:
            execution = execute_cypher(driver, bridge_query, db_name=resolve_actual_cypher_database(resource_id))
        finally:
            driver.close()
        result = bridge_result_from_execution(
            bridge=bridge,
            values=values,
            query_type="cypher",
            selected_db=resource_id,
            bridge_query=bridge_query,
            execution=execution,
        )
        target_entity_type = target_step.get("output_contract", {}).get("entity_type", "")
        if target_entity_type:
            result["target_entity_type"] = target_entity_type
        return result

    if target_query_type == "sql":
        resource_path = resource.get("resource_path") or resolve_sql_resource_path(resource_id)
        if not resource_path:
            return {
                "runtime_status": "unresolved_soft_context",
                "effective_consumption_policy": "soft_context",
                "resolver": "direct_exact_lookup_v1",
                "runtime_reason": "Direct bridge resolver could not resolve the target SQL resource path.",
            }
        conn = connect_sqlite(Path(resource_path))
        try:
            target_tables = narrow_sql_target_tables_for_source(
                target_tables_from_step(target_step),
                source_payload,
            )
            bridge_query = build_sql_bridge_query(
                conn,
                values,
                source_payload=source_payload,
                target_tables=target_tables,
            )
            if not bridge_query:
                return {
                    "runtime_status": "unresolved_soft_context",
                    "effective_consumption_policy": "soft_context",
                    "resolver": "direct_exact_lookup_v1",
                    "runtime_reason": "Direct bridge resolver found no name-like SQL target columns.",
                }
            execution = execute_sql(conn, bridge_query)
        finally:
            conn.close()
        result = bridge_result_from_execution(
            bridge=bridge,
            values=values,
            query_type="sql",
            selected_db=resource_id,
            bridge_query=bridge_query,
            execution=execution,
        )
        target_entity_type = target_step.get("output_contract", {}).get("entity_type", "")
        if target_entity_type:
            result["target_entity_type"] = target_entity_type
        return result

    return {
        "runtime_status": "unresolved_soft_context",
        "effective_consumption_policy": "soft_context",
        "resolver": "direct_exact_lookup_v1",
        "resolver_strategy": "unsupported",
        "runtime_reason": f"Direct bridge resolver does not support target query type: {target_query_type}.",
    }
