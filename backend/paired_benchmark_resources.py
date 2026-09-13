from __future__ import annotations

import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
PAIRED_DIR = ROOT / "data" / "paired_benchmark"
DEFAULT_MANIFEST = PAIRED_DIR / "manifest.json"
DEFAULT_INDEX = ROOT / "schema_index" / "paired_benchmark_resources.json"
DEFAULT_NEO4J_DATABASE = os.getenv("PAIRED_BENCHMARK_NEO4J_DATABASE", "pairedbenchmark")
DEFAULT_GRAPH_DATABASES = {
    "concert_singer": "pbconcertsinger",
    "network_1": "pbnetwork1",
    "wta_1": "pbwta1",
    "star_wars": "pbstarwars",
}


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _abs(path_text: str) -> str:
    path = Path(path_text)
    if path.is_absolute():
        return str(path)
    return str((ROOT / path).resolve())


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _domain_env_key(domain_name: str) -> str:
    normalized = re.sub(r"[^A-Z0-9]+", "_", domain_name.upper()).strip("_")
    return f"PAIRED_BENCHMARK_{normalized}_NEO4J_DATABASE"


def _default_graph_database_name(domain_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "", domain_name.lower())
    return f"pb{slug or 'graph'}"


def database_name_for_domain(domain_name: str) -> str:
    default_name = DEFAULT_GRAPH_DATABASES.get(domain_name, _default_graph_database_name(domain_name))
    return os.getenv(_domain_env_key(domain_name), default_name)


def _sqlite_table_records(resource_id: str, db_path: Path) -> list[dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
        ]
        records: list[dict[str, Any]] = []
        for table in tables:
            columns = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
            column_items = [
                {"name": row[1], "type": row[2] or "TEXT"}
                for row in columns
            ]
            column_text = ", ".join(f"{item['name']}({item['type']})" for item in column_items)
            sample = conn.execute(f'SELECT * FROM "{table}" LIMIT 1').fetchone()
            sample_text = ""
            if sample:
                sample_text = " sample: " + ", ".join(
                    f"{key}={sample[key]}" for key in sample.keys() if sample[key] not in (None, "")
                )
            records.append(
                {
                    "resource_id": resource_id,
                    "table": table,
                    "columns": column_items,
                    "text": f"{resource_id}.{table}: {column_text}{sample_text}",
                }
            )
        return records
    finally:
        conn.close()


def _graph_schema_records(resource_id: str, graph_records_path: Path) -> list[dict[str, Any]]:
    label_props: dict[str, set[str]] = {}
    label_samples: dict[str, dict[str, Any]] = {}
    id_to_label: dict[str, str] = {}
    rel_patterns: set[tuple[str, str, str]] = set()
    records = []
    with graph_records_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            records.append(item)
            if item.get("kind") == "node":
                label = str(item.get("label") or "")
                props = item.get("properties") or {}
                label_props.setdefault(label, set()).update(props.keys())
                label_samples.setdefault(label, props)
                id_to_label[str(item.get("id"))] = label

    for item in records:
        if item.get("kind") != "relationship":
            continue
        rel_type = str(item.get("type") or "")
        start = id_to_label.get(str(item.get("start")), "BridgeEntity")
        end = id_to_label.get(str(item.get("end")), "BridgeEntity")
        rel_patterns.add((start, rel_type, end))

    schema_records: list[dict[str, Any]] = []
    for label in sorted(label_props):
        props = sorted(label_props[label])
        sample = label_samples.get(label, {})
        sample_text = ", ".join(f"{key}={value}" for key, value in list(sample.items())[:4])
        schema_records.append(
            {
                "resource_id": resource_id,
                "type": "node",
                "label": label,
                "properties": props,
                "text": f"{resource_id}.{label}: " + ", ".join(f"{prop}(property)" for prop in props)
                + (f" sample: {sample_text}" if sample_text else ""),
            }
        )
    for start, rel_type, end in sorted(rel_patterns):
        schema_records.append(
            {
                "resource_id": resource_id,
                "type": "rel",
                "label": rel_type,
                "start": start,
                "end": end,
                "text": f"{resource_id} [{rel_type}]: (:{start})-[:{rel_type}]->(:{end})",
            }
        )
    return schema_records


def build_paired_resource_index(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    neo4j_database: str | None = None,
) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    resources: dict[str, dict[str, Any]] = {}
    graph_databases: dict[str, str] = {}

    for domain in manifest.get("domains", []):
        domain_name = str(domain["domain"])
        sql_resource_id = f"{domain_name}_sql"
        graph_resource_id = f"{domain_name}_graph"
        graph_database = neo4j_database or database_name_for_domain(domain_name)
        graph_databases[domain_name] = graph_database

        sql_path = ROOT / domain["sql_resource"]
        sql_schema_records = _sqlite_table_records(sql_resource_id, sql_path)
        resources[sql_resource_id] = {
            "resource_id": sql_resource_id,
            "domain": domain_name,
            "query_type": "sql",
            "resource_path": _rel(sql_path),
            "schema_records": sql_schema_records,
            "tables": [
                {
                    "table": item["table"],
                    "columns": item["columns"],
                }
                for item in sql_schema_records
            ],
        }

        graph_records_path = ROOT / domain["graph_records"]
        graph_schema_records = _graph_schema_records(graph_resource_id, graph_records_path)
        resources[graph_resource_id] = {
            "resource_id": graph_resource_id,
            "domain": domain_name,
            "query_type": "cypher",
            "neo4j_database": graph_database,
            "cypher_import": domain["cypher_import"],
            "graph_records": domain["graph_records"],
            "schema_records": graph_schema_records,
            "node_labels": [
                {
                    "label": item["label"],
                    "properties": item.get("properties", []),
                }
                for item in graph_schema_records
                if item["type"] == "node"
            ],
            "relationship_types": [
                {
                    "type": item["label"],
                    "from": item.get("start", ""),
                    "to": item.get("end", ""),
                }
                for item in graph_schema_records
                if item["type"] == "rel"
            ],
        }

    return {
        "version": "paired_benchmark_resource_index_v1",
        "source_manifest": _rel(manifest_path),
        "graph_database_strategy": "single_database_override" if neo4j_database else "domain_isolated",
        "aggregate_neo4j_database": DEFAULT_NEO4J_DATABASE,
        "graph_databases": graph_databases,
        "resources": resources,
    }


def write_paired_resource_index(index_path: Path = DEFAULT_INDEX) -> dict[str, Any]:
    index = build_paired_resource_index()
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return index


def load_paired_resource_index(index_path: Path = DEFAULT_INDEX) -> dict[str, Any]:
    if not index_path.exists():
        return build_paired_resource_index()
    return _load_json(index_path)


def paired_resources(index_path: Path = DEFAULT_INDEX) -> dict[str, dict[str, Any]]:
    return load_paired_resource_index(index_path).get("resources", {})


def get_paired_resource(resource_id: str) -> dict[str, Any] | None:
    return paired_resources().get(resource_id)


def resolve_paired_sql_resource_path(resource_id: str) -> str:
    resource = get_paired_resource(resource_id)
    if not resource or resource.get("query_type") != "sql":
        return ""
    return _abs(str(resource.get("resource_path", "")))


def resolve_paired_cypher_database(resource_id: str) -> str:
    resource = get_paired_resource(resource_id)
    if not resource or resource.get("query_type") != "cypher":
        return ""
    return str(resource.get("neo4j_database") or DEFAULT_NEO4J_DATABASE)


def resolve_actual_cypher_database(resource_id: str) -> str:
    resolved = resolve_paired_cypher_database(resource_id)
    return resolved or resource_id


def rank_paired_cypher_resources(question: str) -> list[dict[str, Any]]:
    tokens = {token for token in question.lower().split() if token}
    ranked: list[tuple[float, dict[str, Any]]] = []
    for resource in paired_resources().values():
        if resource.get("query_type") != "cypher":
            continue
        text_blob = " ".join(
            [resource.get("resource_id", ""), resource.get("domain", "")]
            + [item.get("text", "") for item in resource.get("schema_records", [])[:20]]
        ).lower()
        overlap = sum(1 for token in tokens if token in text_blob)
        if overlap <= 0:
            continue
        candidate = {
            "db_name": resource["resource_id"],
            "score": float(overlap),
            "labels": [item["label"] for item in resource.get("schema_records", []) if item.get("type") == "node"][:8],
            "dataset_dir": f"paired_benchmark/{resource['domain']}",
            "actual_db_name": resource.get("neo4j_database", DEFAULT_NEO4J_DATABASE),
        }
        ranked.append((float(overlap), candidate))
    ranked.sort(key=lambda item: (-item[0], item[1]["db_name"]))
    return [candidate for _, candidate in ranked]


def paired_sql_metadata_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for resource in paired_resources().values():
        if resource.get("query_type") != "sql":
            continue
        for item in resource.get("schema_records", []):
            records.append(
                {
                    "db_id": resource["resource_id"],
                    "table": item["table"],
                    "text": item["text"],
                    "db_path": _abs(resource["resource_path"]),
                }
            )
    return records


def paired_cypher_metadata_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for resource in paired_resources().values():
        if resource.get("query_type") != "cypher":
            continue
        for item in resource.get("schema_records", []):
            records.append(
                {
                    "db_name": resource["resource_id"],
                    "actual_db_name": resource.get("neo4j_database", DEFAULT_NEO4J_DATABASE),
                    "dataset_dir": f"paired_benchmark/{resource['domain']}",
                    "type": item["type"],
                    "label": item["label"],
                    "text": item["text"],
                }
            )
    return records
