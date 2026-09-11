import json
import re

from paired_benchmark_resources import resolve_actual_cypher_database


SUPPORTED_SCHEMA_FORMATS = {
    "enhanced",
    "base",
    "json",
    "cypher_create",
    "exact_match_pruned",
}

# 中文函数说明补充索引：
# - normalize_schema_format(schema_format)：规范化图 schema 输出格式。
# - extract_graph_schema_data(driver, db_name)：从 Neo4j 抽取 labels、relationships、properties 等结构化 schema 数据。
# - extract_graph_schema(driver, db_name, schema_format)：按指定格式渲染图 schema。
# - filter_graph_schema(schema_text, question)：根据问题做轻量图 schema 过滤，减少 Cypher prompt 噪声。


def normalize_schema_format(schema_format: str | None) -> str:
    """规范化并校验图 schema 输出格式。

    参数：
    - schema_format：调用方传入的格式名称，空值默认 `cypher_create`。
    """
    normalized = (schema_format or "cypher_create").strip().lower()
    if normalized not in SUPPORTED_SCHEMA_FORMATS:
        raise ValueError(
            f"Unsupported schema format: {schema_format}. "
            f"Expected one of {sorted(SUPPORTED_SCHEMA_FORMATS)}."
        )
    return normalized


def extract_graph_schema_data(driver, db_name: str) -> tuple[list[dict], list[dict]]:
    """从 Neo4j 中抽取结构化节点和关系 schema。

    参数：
    - driver：Neo4j driver。
    - db_name：Neo4j 数据库名。
    """
    nodes_data: list[dict] = []
    relationships_data: list[dict] = []

    actual_db = resolve_actual_cypher_database(db_name)

    with driver.session(database=actual_db) as session:
        labels = [
            row["label"]
            for row in session.run("CALL db.labels()")
            if row["label"] != "BridgeEntity"
        ]

        for label in labels:
            sample = session.run(f"MATCH (n:{label}) RETURN n LIMIT 1").single()
            if not sample:
                continue

            sample_props = dict(sample["n"])
            samples = []
            for item in session.run(f"MATCH (n:{label}) RETURN n LIMIT 2"):
                node_props = dict(item["n"])
                samples.append({key: value for key, value in list(node_props.items())[:4]})

            nodes_data.append(
                {
                    "label": label,
                    "properties": {
                        key: type(value).__name__ for key, value in sample_props.items()
                    },
                    "samples": samples,
                }
            )

        relations = session.run(
            """
            MATCH (a)-[r]->(b)
            RETURN DISTINCT
                [label IN labels(a) WHERE label <> 'BridgeEntity'][0] AS from_label,
                type(r) AS rel_type,
                [label IN labels(b) WHERE label <> 'BridgeEntity'][0] AS to_label,
                keys(r) AS rel_keys
            ORDER BY rel_type, from_label, to_label
            """
        )
        for row in relations:
            if not row["from_label"] or not row["to_label"]:
                continue
            relationships_data.append(
                {
                    "start": row["from_label"],
                    "type": row["rel_type"],
                    "end": row["to_label"],
                    "properties": list(row["rel_keys"] or []),
                }
            )

    return nodes_data, relationships_data


def _format_enhanced(nodes_data: list[dict], relationships_data: list[dict]) -> str:
    """把图 schema 格式化为 enhanced 文本。

    参数：
    - nodes_data：节点 schema 数据。
    - relationships_data：关系 schema 数据。
    """
    lines = ["[Nodes]"]
    for node in nodes_data:
        props = ", ".join(
            f"{key}({value})" for key, value in node["properties"].items()
        )
        lines.append(f"  {node['label']}: {props}")
        for sample in node.get("samples", []):
            sample_text = ", ".join(f"{key}={value}" for key, value in sample.items())
            lines.append(f"    Sample: {sample_text}")

    lines.append("")
    lines.append("[Relationships]")
    for relation in relationships_data:
        rel_props = relation.get("properties", [])
        if rel_props:
            lines.append(
                f"  (:{relation['start']})-[:{relation['type']} "
                f"{{{', '.join(rel_props)}}}]->(:{relation['end']})"
            )
        else:
            lines.append(
                f"  (:{relation['start']})-[:{relation['type']}]->(:{relation['end']})"
            )

    return "\n".join(lines)


def _format_base(nodes_data: list[dict], relationships_data: list[dict]) -> str:
    """把图 schema 格式化为基础说明文本。

    参数：
    - nodes_data：节点 schema 数据。
    - relationships_data：关系 schema 数据。
    """
    lines = ["Node labels and properties:"]
    for node in nodes_data:
        props = ", ".join(
            f"{key} ({value})" for key, value in node["properties"].items()
        ) or "(no properties)"
        lines.append(f"- {node['label']}: {props}")

    lines.append("")
    lines.append("Relationship types:")
    for relation in relationships_data:
        rel_props = relation.get("properties", [])
        if rel_props:
            lines.append(
                f"- (:{relation['start']})-[:{relation['type']} "
                f"{{{', '.join(rel_props)}}}]->(:{relation['end']})"
            )
        else:
            lines.append(
                f"- (:{relation['start']})-[:{relation['type']}]->(:{relation['end']})"
            )

    return "\n".join(lines)


def _format_json(nodes_data: list[dict], relationships_data: list[dict]) -> str:
    """把图 schema 格式化为 JSON 字符串。

    参数：
    - nodes_data：节点 schema 数据。
    - relationships_data：关系 schema 数据。
    """
    payload = {
        "nodes": [
            {"label": node["label"], "properties": dict(node["properties"])}
            for node in nodes_data
        ],
        "relationships": [
            {
                "start": relation["start"],
                "type": relation["type"],
                "end": relation["end"],
                "properties": list(relation.get("properties", [])),
            }
            for relation in relationships_data
        ],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _format_cypher_create(nodes_data: list[dict], relationships_data: list[dict]) -> str:
    """把图 schema 格式化为接近 Cypher CREATE 示例的文本。

    参数：
    - nodes_data：节点 schema 数据。
    - relationships_data：关系 schema 数据。
    """
    lines: list[str] = []
    for node in nodes_data:
        props = ", ".join(
            f"{key} ({value})" for key, value in node["properties"].items()
        ) or "(no properties)"
        lines.append(f"// Node: {node['label']}")
        lines.append(f"// Properties: {props}")
        if node.get("samples"):
            sample = node["samples"][0]
            sample_pairs = ", ".join(
                f"{key}: '{value}'" if isinstance(value, str) else f"{key}: {value}"
                for key, value in sample.items()
            )
            alias = node["label"][:1].lower()
            lines.append(
                f"// Example: CREATE ({alias}:{node['label']} {{{sample_pairs}}})"
            )
        lines.append("")

    lines.append("// Relationships:")
    for relation in relationships_data:
        rel_props = relation.get("properties", [])
        if rel_props:
            lines.append(
                f"(:{relation['start']})-[:{relation['type']} "
                f"{{{', '.join(rel_props)}}}]->(:{relation['end']})"
            )
        else:
            lines.append(
                f"(:{relation['start']})-[:{relation['type']}]->(:{relation['end']})"
            )

    return "\n".join(lines).rstrip()


def _normalize_exact_match_text(value: str) -> str:
    """把 label/property/relation 名称规范化为可精确匹配的问题文本。

    参数：
    - value：schema 名称或问题文本。
    """
    text = re.sub(r"(?<!^)(?=[A-Z])", " ", value)
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^0-9a-zA-Z\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def _question_mentions_label(question: str, label: str) -> bool:
    """判断问题是否提到某个节点 label。

    参数：
    - question：用户自然语言问题。
    - label：图节点 label。
    """
    question_terms = set(_normalize_exact_match_text(question).split())
    label_terms = _normalize_exact_match_text(label).split()
    for term in label_terms:
        if term in question_terms:
            return True
        if term.endswith("y") and term[:-1] + "ies" in question_terms:
            return True
        if term + "s" in question_terms:
            return True
    return False


def _build_question_terms(question: str) -> tuple[set[str], set[str]]:
    """构建问题 token 集和短语集。

    参数：
    - question：用户自然语言问题。
    """
    normalized = _normalize_exact_match_text(question)
    tokens = [token for token in normalized.split() if token]
    token_set = set(tokens)
    phrase_set = set()
    for start in range(len(tokens)):
        for width in range(1, min(4, len(tokens) - start) + 1):
            phrase_set.add(" ".join(tokens[start:start + width]))
    return token_set, phrase_set


def _schema_name_matches(name: str, token_set: set[str], phrase_set: set[str]) -> bool:
    """判断 schema 名称是否与问题 token/phrase 匹配。

    参数：
    - name：label、relationship type 或 property 名称。
    - token_set：问题 token 集。
    - phrase_set：问题短语集。
    """
    normalized = _normalize_exact_match_text(name)
    if not normalized:
        return False
    if normalized in phrase_set:
        return True
    parts = normalized.split()
    if len(parts) == 1:
        return parts[0] in token_set
    return all(part in token_set for part in parts)


def exact_match_filter_graph_schema(
    question: str,
    nodes_data: list[dict],
    relationships_data: list[dict],
    schema_format: str = "enhanced",
) -> str:
    """基于问题和 schema 名称的精确匹配过滤图 schema。

    参数：
    - question：用户自然语言问题。
    - nodes_data：完整节点 schema。
    - relationships_data：完整关系 schema。
    - schema_format：输出格式。
    """
    token_set, phrase_set = _build_question_terms(question)

    matched_nodes: set[str] = set()
    matched_relations: set[str] = set()

    for node in nodes_data:
        if _schema_name_matches(node["label"], token_set, phrase_set):
            matched_nodes.add(node["label"])
            continue
        for property_name in node.get("properties", {}).keys():
            if _schema_name_matches(property_name, token_set, phrase_set):
                matched_nodes.add(node["label"])
                break

    for relation in relationships_data:
        if _schema_name_matches(relation["type"], token_set, phrase_set):
            matched_relations.add(relation["type"])
            matched_nodes.add(relation["start"])
            matched_nodes.add(relation["end"])
            continue
        for property_name in relation.get("properties", []):
            if _schema_name_matches(property_name, token_set, phrase_set):
                matched_relations.add(relation["type"])
                matched_nodes.add(relation["start"])
                matched_nodes.add(relation["end"])
                break

    if not matched_nodes and not matched_relations:
        return format_graph_schema(nodes_data, relationships_data, schema_format="enhanced")

    filtered_relationships = [
        relation
        for relation in relationships_data
        if relation["type"] in matched_relations
        or relation["start"] in matched_nodes
        or relation["end"] in matched_nodes
    ]

    for relation in filtered_relationships:
        matched_nodes.add(relation["start"])
        matched_nodes.add(relation["end"])

    filtered_nodes = [
        node for node in nodes_data if node["label"] in matched_nodes
    ]

    render_format = "enhanced" if schema_format == "exact_match_pruned" else schema_format
    return format_graph_schema(
        filtered_nodes,
        filtered_relationships,
        schema_format=render_format,
    )


def format_graph_schema(
    nodes_data: list[dict],
    relationships_data: list[dict],
    schema_format: str = "cypher_create",
) -> str:
    """按指定格式渲染图 schema。

    参数：
    - nodes_data：节点 schema 数据。
    - relationships_data：关系 schema 数据。
    - schema_format：输出格式。
    """
    normalized = normalize_schema_format(schema_format)
    if normalized == "enhanced":
        return _format_enhanced(nodes_data, relationships_data)
    if normalized == "base":
        return _format_base(nodes_data, relationships_data)
    if normalized == "json":
        return _format_json(nodes_data, relationships_data)
    if normalized == "exact_match_pruned":
        return _format_enhanced(nodes_data, relationships_data)
    return _format_cypher_create(nodes_data, relationships_data)


def extract_graph_schema(driver, db_name: str = "wwc2019", format: str = "cypher_create") -> str:
    """抽取并格式化指定 Neo4j 数据库的图 schema。

    参数：
    - driver：Neo4j driver。
    - db_name：Neo4j 数据库名。
    - format：输出格式。
    """
    nodes_data, relationships_data = extract_graph_schema_data(driver, db_name)
    return format_graph_schema(nodes_data, relationships_data, schema_format=format)


def filter_graph_schema(
    nodes_data: list[dict],
    relationships_data: list[dict],
    relevant_nodes: list[str] | None = None,
    relevant_relations: list[str] | None = None,
    schema_format: str = "cypher_create",
    question: str = "",
) -> str:
    """根据相关节点和关系过滤图 schema。

    参数：
    - nodes_data：完整节点 schema。
    - relationships_data：完整关系 schema。
    - relevant_nodes：LLM schema filtering 选出的相关节点。
    - relevant_relations：LLM schema filtering 选出的相关关系。
    - schema_format：输出格式。
    - question：用户问题，用于补充 question-aware schema closure。
    """
    node_set = {node for node in (relevant_nodes or []) if node}
    relation_set = {relation for relation in (relevant_relations or []) if relation}

    filtered_relationships = [
        relation
        for relation in relationships_data
        if relation["type"] in relation_set
    ]

    referenced_nodes = {
        relation["start"] for relation in filtered_relationships
    } | {
        relation["end"] for relation in filtered_relationships
    }
    effective_nodes = node_set | referenced_nodes

    # Keep question-aware schema closure among selected nodes so the generator can bind
    # paths through relationships whose endpoint labels are both explicitly mentioned.
    for relation in relationships_data:
        if relation in filtered_relationships:
            continue
        if (
            relation["start"] in effective_nodes
            and relation["end"] in effective_nodes
            and question
            and _question_mentions_label(question, relation["start"])
            and _question_mentions_label(question, relation["end"])
        ):
            filtered_relationships.append(relation)

    filtered_nodes = [
        node for node in nodes_data if node["label"] in effective_nodes
    ]

    if not filtered_nodes and not filtered_relationships:
        return format_graph_schema(nodes_data, relationships_data, schema_format=schema_format)

    return format_graph_schema(
        filtered_nodes,
        filtered_relationships,
        schema_format=schema_format,
    )
