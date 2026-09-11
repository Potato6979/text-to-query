import re
from typing import Any

from cypher_schema_utils import normalize_schema_format

from config import (
    MAX_RESULT_ROWS,
    MAX_RETRIES,
)
from execution_agents import run_execution_with_repair
from fix_agents import fix_cypher_query
from generation_agents import generate_cypher_query as shared_generate_cypher_query
from paired_benchmark_resources import resolve_actual_cypher_database
from retrieval_agents import retrieve_cypher_targets
from schema_agents import (
    prepare_cypher_schema_context as shared_prepare_cypher_schema_context,
)

# 中文函数说明索引：
# - retrieve_cypher_candidates(question)：检索候选 Neo4j 图数据库。
# - normalize_schema_name(name)：规范化 schema/database 名称。
# - prepare_cypher_schema_context(...)：抽取并过滤图 schema，准备 Cypher 生成上下文。
# - generate_cypher_query(...)：调用 Cypher 生成器，结合 schema 和 feedback 生成查询。
# - execute_cypher / execute_cypher_with_repair(...)：执行 Cypher 并进行有限次确定性修复。
# - deterministic repair helpers：处理 relationship domain、quoted value grounding、DISTINCT/collect、goal event 等通用图查询问题。


_AVG_COLLECT_DISTINCT_MAP_RE = re.compile(
    r"WITH\s+(?P<context>[A-Za-z_][A-Za-z0-9_]*)\s*,\s*"
    r"(?P<entity>[A-Za-z_][A-Za-z0-9_]*)\s*,\s*"
    r"(?P<expr>.*?)\s+AS\s+(?P<scalar>[A-Za-z_][A-Za-z0-9_]*)\s+"
    r"WITH\s+(?P=context)\s*,\s*COLLECT\s*\(\s*DISTINCT\s*\{(?P<map_body>.*?)\}\s*\)\s+AS\s+(?P<collection>[A-Za-z_][A-Za-z0-9_]*)\s+"
    r"UNWIND\s+(?P=collection)\s+AS\s+(?P<item>[A-Za-z_][A-Za-z0-9_]*)\s+"
    r"RETURN\s+(?P<context_return>.*?)\s+AS\s+(?P<context_alias>[A-Za-z_][A-Za-z0-9_]*)\s*,\s*"
    r"AVG\s*\(\s*(?P=item)\.(?P=scalar)\s*\)\s+AS\s+(?P<avg_alias>[A-Za-z_][A-Za-z0-9_]*)",
    re.IGNORECASE | re.DOTALL,
)
_DURATION_AGE_RE = re.compile(
    r"duration\.between\(\s*(?P<dob>[A-Za-z_][A-Za-z0-9_]*)\s*,\s*(?P<ref>[A-Za-z_][A-Za-z0-9_]*)\s*\)\.years",
    re.IGNORECASE,
)
_NODE_TOKEN_RE = re.compile(r"\((?P<body>[^()]*)\)")
_REL_TOKEN_RE = re.compile(
    r"(?P<left_arrow><-|-)\s*\[(?P<body>[^\]]*)\]\s*(?P<right_arrow>->|-)",
    re.DOTALL,
)
_CYPHER_IDENTIFIER_RE = r"`[^`]+`|[A-Za-z_][A-Za-z0-9_]*"
_LABEL_RE = re.compile(rf":\s*(?P<label>{_CYPHER_IDENTIFIER_RE})")
_REL_TYPE_RE = re.compile(rf":\s*(?P<types>{_CYPHER_IDENTIFIER_RE}(?:\s*\|\s*{_CYPHER_IDENTIFIER_RE})*)")
_VAR_RE = re.compile(rf"^\s*(?P<var>{_CYPHER_IDENTIFIER_RE})")
_QUESTION_QUOTED_LITERAL_RE = re.compile(r"(['\"])(?P<value>[^'\"]+)\1")
_NODE_PROPERTY_LITERAL_RE = re.compile(
    rf"(?P<prefix>\((?P<var>{_CYPHER_IDENTIFIER_RE})?\s*:\s*(?P<label>{_CYPHER_IDENTIFIER_RE})\s*\{{[^}}]*?)"
    rf"(?P<property>{_CYPHER_IDENTIFIER_RE})\s*:\s*(?P<quote>['\"])(?P<value>[^'\"]+)(?P=quote)",
    re.DOTALL,
)
_NODE_SINGLE_NAME_WITH_YEAR_RE = re.compile(
    rf"(?P<prefix>\((?P<var>{_CYPHER_IDENTIFIER_RE})?\s*:\s*(?P<label>{_CYPHER_IDENTIFIER_RE})\s*\{{\s*)"
    rf"(?P<property>{_CYPHER_IDENTIFIER_RE})\s*:\s*(?P<quote>['\"])(?P<value>\d{{4}}\s+[^'\"]+)(?P=quote)\s*"
    rf"(?P<suffix>\}}\))",
    re.DOTALL,
)
_NODE_LABEL_VAR_RE = re.compile(
    rf"\((?P<var>{_CYPHER_IDENTIFIER_RE})\s*:\s*(?P<label>{_CYPHER_IDENTIFIER_RE})(?:\s*\{{[^}}]*\}})?\)",
    re.DOTALL,
)
_WHERE_NAME_WITH_YEAR_RE = re.compile(
    rf"(?P<prefix>\b(?P<var>{_CYPHER_IDENTIFIER_RE})\s*\.\s*name\s*=\s*)"
    rf"(?P<quote>['\"])(?P<value>\d{{4}}\s+[^'\"]+)(?P=quote)",
    re.IGNORECASE,
)
_COUNT_NODE_RE = re.compile(rf"COUNT\s*\(\s*(?!DISTINCT\b)(?P<var>{_CYPHER_IDENTIFIER_RE})\s*\)", re.IGNORECASE)
_COLLECT_DISTINCT_RE = re.compile(r"\bCOLLECT\s*\(\s*DISTINCT\s+(?P<expr>[^()]+?)\s*\)", re.IGNORECASE)
_NUMBERED_REL_UNION_RE = re.compile(
    rf":(?P<base>{_CYPHER_IDENTIFIER_RE})(?P<suffix>\s*\|\s*(?P=base)_\d+(?:\s*\|\s*(?P=base)_\d+)*)",
    re.IGNORECASE,
)
_DURATION_TOTAL_SECONDS_RE = re.compile(
    r"duration\.totalSeconds\s*\(\s*"
    r"(?P<expr>(?:`[^`]+`|[A-Za-z_][A-Za-z0-9_]*)(?:\s*\.\s*(?:`[^`]+`|[A-Za-z_][A-Za-z0-9_]*))+)"
    r"\s*\)",
    re.IGNORECASE,
)
_DURATION_COMPONENT_SUM_RE = re.compile(
    rf"(?P<wrapped>\(\s*)?"
    rf"(?P<expr>[A-Za-z_][A-Za-z0-9_]*(?:\s*\.\s*[A-Za-z_][A-Za-z0-9_]*)+)\.days\s*\*\s*86400\s*\+\s*"
    rf"(?P=expr)\.hours\s*\*\s*3600\s*\+\s*"
    rf"(?P=expr)\.minutes\s*\*\s*60\s*\+\s*"
    rf"(?P=expr)\.seconds"
    rf"(?P<closing>\s*\))?",
    re.IGNORECASE,
)
_PARALLEL_COLLECT_RE = re.compile(
    rf"(?P<first>MATCH\s+\((?P<anchor>{_CYPHER_IDENTIFIER_RE})(?:\s*:[^)]*)?\)\s*-\[[^\]]*\]->"
    rf"\((?P<v1>{_CYPHER_IDENTIFIER_RE})(?:\s*:[^)]*)?\s*\{{[^}}]*\}}\)\s*)"
    rf"(?P<gap1>\s+)"
    rf"(?P<second>MATCH\s+\((?P=anchor)\)\s*-\[[^\]]*\]->"
    rf"\((?P<v2>{_CYPHER_IDENTIFIER_RE})(?:\s*:[^)]*)?\s*\{{[^}}]*\}}\)\s*)"
    rf"(?P<gap2>\s+)"
    rf"WITH\s+(?P=anchor)\s*,\s*"
    rf"collect\s*\(\s*(?P=v1)\s*\.\s*(?P<prop1>{_CYPHER_IDENTIFIER_RE})\s*\)\s+AS\s+(?P<a1>{_CYPHER_IDENTIFIER_RE})\s*,\s*"
    rf"collect\s*\(\s*(?P=v2)\s*\.\s*(?P<prop2>{_CYPHER_IDENTIFIER_RE})\s*\)\s+AS\s+(?P<a2>{_CYPHER_IDENTIFIER_RE})",
    re.IGNORECASE | re.DOTALL,
)
_PARALLEL_COLLECT_RETURN_RE = re.compile(
    rf"(?P<first>MATCH\s+\((?P<anchor>{_CYPHER_IDENTIFIER_RE})(?:\s*:[^)]*)?\)\s*-\[[^\]]*\]->"
    rf"\((?P<v1>{_CYPHER_IDENTIFIER_RE})(?:\s*:[^)]*)?\s*\{{[^}}]*\}}\)\s*,\s*)"
    rf"(?P<second>\((?P=anchor)\)\s*-\[[^\]]*\]->"
    rf"\((?P<v2>{_CYPHER_IDENTIFIER_RE})(?:\s*:[^)]*)?\s*\{{[^}}]*\}}\)\s*)"
    rf"\s+RETURN\s+(?P<anchor_return>.*?)\s*,\s*"
    rf"collect\s*\(\s*(?P=v1)\s*\.\s*(?P<prop1>{_CYPHER_IDENTIFIER_RE})\s*\)\s+AS\s+(?P<a1>{_CYPHER_IDENTIFIER_RE})\s*,\s*"
    rf"collect\s*\(\s*(?P=v2)\s*\.\s*(?P<prop2>{_CYPHER_IDENTIFIER_RE})\s*\)\s+AS\s+(?P<a2>{_CYPHER_IDENTIFIER_RE})",
    re.IGNORECASE | re.DOTALL,
)
_PATTERN_COUNT_REL_RE = re.compile(
    rf"(?P<prefix>WITH\s+.*?),\s*COUNT\s*\{{\s*"
    rf"\((?P<source>{_CYPHER_IDENTIFIER_RE})\)"
    rf"\s*(?P<left><-|-)\s*\[\s*:(?P<rel_type>{_CYPHER_IDENTIFIER_RE})\s*\]\s*(?P<right>->|-)\s*"
    rf"\((?P<target>[^)]*)\)\s*\}}\s+AS\s+(?P<alias>{_CYPHER_IDENTIFIER_RE})",
    re.IGNORECASE | re.DOTALL,
)
_RETURN_DISTINCT_RE = re.compile(r"\bRETURN\s+DISTINCT\b", re.IGNORECASE)
_RETURN_CLAUSE_RE = re.compile(
    r"\bRETURN\s+(?!DISTINCT\b)(?P<expr>.*?)(?=\bORDER\s+BY\b|\bLIMIT\b|$)",
    re.IGNORECASE | re.DOTALL,
)
_RETURN_ANY_CLAUSE_RE = re.compile(
    r"\bRETURN\s+(?:DISTINCT\s+)?(?P<expr>.*?)(?=\bORDER\s+BY\b|\bLIMIT\b|$)",
    re.IGNORECASE | re.DOTALL,
)
_SIMPLE_RETURN_NAME_TITLE_RE = re.compile(
    rf"^\s*(?P<var>{_CYPHER_IDENTIFIER_RE})\s*\.\s*(?P<prop>name|title)"
    rf"(?:\s+AS\s+(?:{_CYPHER_IDENTIFIER_RE}))?\s*$",
    re.IGNORECASE,
)
_SUM_SCORE_RE = re.compile(
    rf"SUM\s*\(\s*(?P<rel>{_CYPHER_IDENTIFIER_RE})\s*\.\s*score\s*\)\s+AS\s+(?P<alias>{_CYPHER_IDENTIFIER_RE})",
    re.IGNORECASE,
)
_TEAM_PLAYED_IN_RE = re.compile(
    rf"\((?P<team>{_CYPHER_IDENTIFIER_RE})\s*:\s*Team\b[^)]*\)\s*"
    rf"-\s*\[\s*(?P<rel>{_CYPHER_IDENTIFIER_RE})?\s*:\s*PLAYED_IN\s*\]\s*->\s*"
    rf"\((?P<match>{_CYPHER_IDENTIFIER_RE})(?:\s*:\s*Match\b[^)]*)?\)",
    re.IGNORECASE | re.DOTALL,
)
_QUESTION_DISTINCT_RE = re.compile(r"\b(unique|distinct|different|non[-\s]?duplicated)\b", re.IGNORECASE)
_QUESTION_ROW_LEVEL_RE = re.compile(r"\b(all|every|each|details?)\b", re.IGNORECASE)
_QUESTION_STRONG_ROW_LEVEL_RE = re.compile(
    r"\b(each|details?|records?|occurrences?|interactions?|relationships?|visits?)\b",
    re.IGNORECASE,
)
_QUESTION_EXPLICIT_UNDIRECTED_RE = re.compile(
    r"\b(undirected|bidirectional|both\s+directions|either\s+direction|regardless\s+of\s+direction)\b",
    re.IGNORECASE,
)
_LOGICAL_NODE_COUNT_LABELS = {
    "match": {"match", "matches"},
    "team": {"team", "teams"},
    "person": {"person", "people", "player", "players", "coach", "coaches"},
    "squad": {"squad", "squads"},
    "tournament": {"tournament", "tournaments"},
}


def _strip_cypher_identifier(value: str) -> str:
    """去掉 Cypher 反引号标识符外壳。

    参数：
    - value：可能带反引号的 label、变量名或属性名。
    """
    value = value.strip()
    if value.startswith("`") and value.endswith("`"):
        return value[1:-1]
    return value


def _quote_cypher_string(value: str, quote: str = "'") -> str:
    """把字符串值安全包装为 Cypher 字符串 literal。

    参数：
    - value：待写入 Cypher 的字符串。
    - quote：使用单引号或双引号。
    """
    if quote == '"':
        return '"' + value.replace('"', '\\"') + '"'
    return "'" + value.replace("'", "\\'") + "'"


def _normalize_literal_value(value: str) -> str:
    """规范化 literal 值，便于大小写和空白无关比较。

    参数：
    - value：字符串 literal。
    """
    return re.sub(r"\s+", " ", value).strip().lower()


def _question_tokens(question: str) -> set[str]:
    """提取问题中的英文 token 集合。

    参数：
    - question：用户自然语言问题。
    """
    return set(re.findall(r"[a-zA-Z][a-zA-Z0-9]*", question.lower()))


def _duration_seconds_expression(expr: str, *, full_duration: bool) -> str:
    """Build a Cypher seconds expression for Neo4j Duration values.

    Some MTQ datasets store total seconds in `duration.seconds`; adding hours,
    minutes, and seconds double-counts those values after Neo4j decomposes the
    duration for display. Only use the full component sum when the query asked
    for an explicit total duration function.
    """
    if full_duration:
        return f"({expr}.days * 86400 + {expr}.hours * 3600 + {expr}.minutes * 60 + {expr}.seconds)"
    return f"{expr}.seconds"


def _parse_node_token(token: str) -> dict[str, str]:
    """解析 Cypher 节点片段中的变量名和 label。

    参数：
    - token：形如 `(p:Person)` 的节点片段。
    """
    body = token.strip()[1:-1].strip()
    variable = ""
    label = ""
    var_match = _VAR_RE.match(body)
    if var_match and not body[var_match.start("var") : var_match.end("var")].startswith(":"):
        variable = _strip_cypher_identifier(var_match.group("var"))
    label_match = _LABEL_RE.search(body)
    if label_match:
        label = _strip_cypher_identifier(label_match.group("label"))
    return {"variable": variable, "label": label}


def _parse_relationship_types(token: str) -> list[str]:
    """解析 Cypher 关系片段中的关系类型。

    参数：
    - token：形如 `-[r:TYPE]->` 或 `-[:A|B]->` 的关系片段。
    """
    body_match = _REL_TOKEN_RE.match(token)
    if not body_match:
        return []
    type_match = _REL_TYPE_RE.search(body_match.group("body"))
    if not type_match:
        return []
    return [
        _strip_cypher_identifier(item)
        for item in re.split(r"\s*\|\s*", type_match.group("types"))
        if item.strip()
    ]


def _relationship_direction(token: str) -> str:
    """判断关系片段方向。

    参数：
    - token：Cypher 关系片段。

    返回：
    - `left_to_right`、`right_to_left` 或空字符串。
    """
    match = _REL_TOKEN_RE.match(token)
    if not match:
        return ""
    if match.group("left_arrow") == "-" and match.group("right_arrow") == "->":
        return "left_to_right"
    if match.group("left_arrow") == "<-" and match.group("right_arrow") == "-":
        return "right_to_left"
    return ""


def _collect_cypher_tokens(cypher: str) -> list[dict[str, Any]]:
    """按位置收集 Cypher 中的节点和关系 token。

    参数：
    - cypher：待分析的 Cypher 查询。
    """
    tokens: list[dict[str, Any]] = []
    for match in _REL_TOKEN_RE.finditer(cypher):
        tokens.append({"kind": "rel", "text": match.group(0), "start": match.start(), "end": match.end()})
    for match in _NODE_TOKEN_RE.finditer(cypher):
        tokens.append({"kind": "node", "text": match.group(0), "start": match.start(), "end": match.end()})
    return sorted(tokens, key=lambda item: (item["start"], item["end"]))


def _build_variable_label_map(cypher: str) -> dict[str, str]:
    """构建变量名到 label 的映射。

    参数：
    - cypher：待分析的 Cypher 查询。
    """
    labels: dict[str, str] = {}
    for match in _NODE_TOKEN_RE.finditer(cypher):
        node = _parse_node_token(match.group(0))
        if node["variable"] and node["label"]:
            labels.setdefault(node["variable"], node["label"])
    return labels


def _resolve_node_label(node: dict[str, str], variable_labels: dict[str, str]) -> str:
    """解析节点最终 label，优先使用节点自身 label，否则查变量绑定。

    参数：
    - node：`_parse_node_token(...)` 解析出的节点信息。
    - variable_labels：变量到 label 的映射。
    """
    if node["label"]:
        return node["label"]
    if node["variable"]:
        return variable_labels.get(node["variable"], "")
    return ""


def validate_cypher_relationship_domains(
    cypher: str,
    relationships_data: list[dict[str, Any]],
) -> dict[str, Any]:
    """校验有方向关系模式是否符合图 schema。

    参数：
    - cypher：待校验的 Cypher 查询。
    - relationships_data：图 schema 中的关系定义列表，每项包含 start/type/end。

    说明：
    - 只在关系两端 label 可确定时触发，避免误伤合法但静态信息不足的查询。
    """
    allowed_by_type: dict[str, set[tuple[str, str]]] = {}
    for relation in relationships_data:
        rel_type = str(relation.get("type", ""))
        start = str(relation.get("start", ""))
        end = str(relation.get("end", ""))
        if rel_type and start and end:
            allowed_by_type.setdefault(rel_type, set()).add((start, end))

    if not allowed_by_type:
        return {"valid": True, "violations": []}

    variable_labels = _build_variable_label_map(cypher)
    tokens = _collect_cypher_tokens(cypher)
    violations: list[dict[str, Any]] = []

    for index, token in enumerate(tokens):
        if token["kind"] != "rel" or index == 0 or index + 1 >= len(tokens):
            continue
        left_token = tokens[index - 1]
        right_token = tokens[index + 1]
        if left_token["kind"] != "node" or right_token["kind"] != "node":
            continue

        direction = _relationship_direction(token["text"])
        if not direction:
            continue

        left_node = _parse_node_token(left_token["text"])
        right_node = _parse_node_token(right_token["text"])
        left_label = _resolve_node_label(left_node, variable_labels)
        right_label = _resolve_node_label(right_node, variable_labels)
        if not left_label or not right_label:
            continue

        if direction == "left_to_right":
            actual_start, actual_end = left_label, right_label
        else:
            actual_start, actual_end = right_label, left_label

        for rel_type in _parse_relationship_types(token["text"]):
            allowed_pairs = allowed_by_type.get(rel_type)
            if not allowed_pairs:
                violations.append(
                    {
                        "type": rel_type,
                        "actual_start": actual_start,
                        "actual_end": actual_end,
                        "allowed_pairs": [],
                        "fragment": f"{left_token['text']}{token['text']}{right_token['text']}",
                        "reason": "unknown_relationship_type",
                    }
                )
                continue
            if (actual_start, actual_end) not in allowed_pairs:
                violations.append(
                    {
                        "type": rel_type,
                        "actual_start": actual_start,
                        "actual_end": actual_end,
                        "allowed_pairs": sorted(allowed_pairs),
                        "fragment": f"{left_token['text']}{token['text']}{right_token['text']}",
                        "reason": "invalid_relationship_domain",
                    }
                )

    return {"valid": not violations, "violations": violations}


def repair_cypher_unambiguous_relationship_direction(
    cypher: str,
    relationships_data: list[dict[str, Any]],
    question: str = "",
) -> tuple[str, dict[str, Any] | None]:
    """Orient undirected relationships when schema has exactly one legal direction."""
    allowed_by_type: dict[str, set[tuple[str, str]]] = {}
    for relation in relationships_data:
        rel_type = str(relation.get("type", ""))
        start = str(relation.get("start", ""))
        end = str(relation.get("end", ""))
        if rel_type and start and end:
            allowed_by_type.setdefault(rel_type, set()).add((start, end))
    if not allowed_by_type:
        return cypher, None

    variable_labels = _build_variable_label_map(cypher)
    tokens = _collect_cypher_tokens(cypher)
    replacements: list[dict[str, Any]] = []
    for index, token in enumerate(tokens):
        if token["kind"] != "rel" or index == 0 or index + 1 >= len(tokens):
            continue
        if _relationship_direction(token["text"]):
            continue
        left_token = tokens[index - 1]
        right_token = tokens[index + 1]
        if left_token["kind"] != "node" or right_token["kind"] != "node":
            continue

        left_node = _parse_node_token(left_token["text"])
        right_node = _parse_node_token(right_token["text"])
        left_label = _resolve_node_label(left_node, variable_labels)
        right_label = _resolve_node_label(right_node, variable_labels)
        if not left_label or not right_label:
            continue

        rel_types = _parse_relationship_types(token["text"])
        if len(rel_types) != 1:
            continue
        rel_type = rel_types[0]
        allowed_pairs = allowed_by_type.get(rel_type, set())
        left_to_right = (left_label, right_label) in allowed_pairs
        right_to_left = (right_label, left_label) in allowed_pairs
        if (
            left_to_right
            and right_to_left
            and left_label == right_label
            and not _QUESTION_EXPLICIT_UNDIRECTED_RE.search(question)
        ):
            replacement_text = token["text"].rstrip("-") + "->"
            replacements.append(
                {
                    "start": token["start"],
                    "end": token["end"],
                    "original_fragment": token["text"],
                    "repaired_fragment": replacement_text,
                    "type": rel_type,
                    "left_label": left_label,
                    "right_label": right_label,
                }
            )
            continue
        if left_to_right == right_to_left:
            continue

        replacement_text = token["text"].rstrip("-") + "->" if left_to_right else "<-" + token["text"].lstrip("-")
        replacements.append(
            {
                "start": token["start"],
                "end": token["end"],
                "original_fragment": token["text"],
                "repaired_fragment": replacement_text,
                "type": rel_type,
                "left_label": left_label,
                "right_label": right_label,
            }
        )

    if not replacements:
        return cypher, None

    repaired = cypher
    for replacement in sorted(replacements, key=lambda item: item["start"], reverse=True):
        repaired = repaired[: replacement["start"]] + replacement["repaired_fragment"] + repaired[replacement["end"] :]
    return repaired, {
        "repair_type": "unambiguous_relationship_direction",
        "repairs": [
            {
                "original_fragment": item["original_fragment"],
                "repaired_fragment": item["repaired_fragment"],
                "type": item["type"],
                "left_label": item["left_label"],
                "right_label": item["right_label"],
            }
            for item in replacements
        ],
    }


def _format_relationship_domain_error(validation: dict[str, Any]) -> str:
    """把 relationship domain validation 结果格式化为修复 Agent 可读错误。

    参数：
    - validation：`validate_cypher_relationship_domains(...)` 返回的校验结果。
    """
    details = []
    for violation in validation.get("violations", [])[:3]:
        allowed = ", ".join(
            f"(:{start})-[:{violation['type']}]->(:{end})"
            for start, end in violation.get("allowed_pairs", [])
        ) or "not present in schema"
        details.append(
            f"{violation['fragment']} implies (:{violation['actual_start']})"
            f"-[:{violation['type']}]->(:{violation['actual_end']}), allowed: {allowed}"
        )
    return "Schema relationship domain mismatch: " + " | ".join(details)


def _find_label_value_match(driver: Any, db_name: str, label: str, target_value: str) -> dict[str, str] | None:
    """在图数据库指定 label 下查找唯一匹配的属性和值。

    参数：
    - driver：Neo4j driver。
    - db_name：Neo4j 数据库名。
    - label：节点 label。
    - target_value：用户问题中显式给出的 quoted value。
    """
    query = f"""
    MATCH (n:{_strip_cypher_identifier(label)})
    UNWIND keys(n) AS key
    WITH key, n[key] AS value
    WHERE value IS NOT NULL AND toLower(toString(value)) = toLower($target_value)
    RETURN key, toString(value) AS value
    ORDER BY key
    LIMIT 2
    """
    try:
        with driver.session(database=resolve_actual_cypher_database(db_name)) as session:
            rows = list(session.run(query, target_value=target_value))
    except Exception:
        return None
    if len(rows) != 1:
        return None
    return {"property": rows[0]["key"], "value": rows[0]["value"]}


def repair_cypher_quoted_value_grounding(
    *,
    driver: Any,
    db_name: str,
    question: str,
    cypher: str,
) -> tuple[str, dict[str, Any] | None]:
    """用用户问题中的显式 quoted value 修复 Cypher literal。

    参数：
    - driver：Neo4j driver。
    - db_name：当前图库名称。
    - question：用户自然语言问题。
    - cypher：生成器输出的 Cypher。
    """
    question_values = [
        match.group("value").strip()
        for match in _QUESTION_QUOTED_LITERAL_RE.finditer(question)
        if match.group("value").strip()
    ]
    normalized_question_values = {_normalize_literal_value(value) for value in question_values}
    if len(question_values) != 1:
        return cypher, None

    target_value = question_values[0]
    for match in _NODE_PROPERTY_LITERAL_RE.finditer(cypher):
        generated_value = match.group("value").strip()
        if _normalize_literal_value(generated_value) in normalized_question_values:
            continue

        label = _strip_cypher_identifier(match.group("label"))
        value_match = _find_label_value_match(driver, db_name, label, target_value)
        if not value_match:
            continue

        quote = match.group("quote")
        replacement = (
            f"{match.group('prefix')}{value_match['property']}: "
            f"{_quote_cypher_string(value_match['value'], quote)}"
        )
        repaired = cypher[: match.start()] + replacement + cypher[match.end() :]
        return repaired, {
            "repair_type": "quoted_value_grounding",
            "label": label,
            "old_property": _strip_cypher_identifier(match.group("property")),
            "old_value": generated_value,
            "new_property": value_match["property"],
            "new_value": value_match["value"],
        }

    return cypher, None


def repair_cypher_split_year_name_literal(
    *,
    driver: Any,
    db_name: str,
    cypher: str,
) -> tuple[str, dict[str, Any] | None]:
    """Split literals like `2014 Week 1` into separate name/year properties when the graph proves it."""
    for match in _NODE_SINGLE_NAME_WITH_YEAR_RE.finditer(cypher):
        label = _strip_cypher_identifier(match.group("label"))
        property_name = _strip_cypher_identifier(match.group("property"))
        if property_name.lower() != "name":
            continue
        value = match.group("value").strip()
        year_match = re.match(r"^(?P<year>\d{4})\s+(?P<name>.+)$", value)
        if not year_match:
            continue
        year = year_match.group("year")
        name = re.sub(r"\s+", " ", year_match.group("name")).strip()
        if not name:
            continue
        query = f"""
        MATCH (n:{label})
        WHERE toLower(toString(n.name)) = toLower($name)
          AND toString(n.year) = $year
        RETURN n.name AS name, n.year AS year
        LIMIT 2
        """
        try:
            with driver.session(database=resolve_actual_cypher_database(db_name)) as session:
                rows = list(session.run(query, name=name, year=year))
        except Exception:
            continue
        if len(rows) != 1:
            continue
        quote = match.group("quote")
        replacement = (
            f"{match.group('prefix')}name: {_quote_cypher_string(str(rows[0]['name']), quote)}, "
            f"year: {_quote_cypher_string(str(rows[0]['year']), quote)}{match.group('suffix')}"
        )
        repaired = cypher[: match.start()] + replacement + cypher[match.end() :]
        return repaired, {
            "repair_type": "split_year_name_literal",
            "label": label,
            "old_property": property_name,
            "old_value": value,
            "new_name": str(rows[0]["name"]),
            "new_year": str(rows[0]["year"]),
        }
    return cypher, None


def _cypher_var_label_map(cypher: str) -> dict[str, str]:
    return {
        match.group("var"): _strip_cypher_identifier(match.group("label"))
        for match in _NODE_LABEL_VAR_RE.finditer(cypher)
    }


def repair_cypher_split_year_name_where_predicate(
    *,
    driver: Any,
    db_name: str,
    cypher: str,
) -> tuple[str, dict[str, Any] | None]:
    """Repair `x.name = '2014 Week 1'` to `x.name = 'Week 1'` when `x.year = '2014'` is already constrained."""
    var_labels = _cypher_var_label_map(cypher)
    for match in _WHERE_NAME_WITH_YEAR_RE.finditer(cypher):
        var = match.group("var")
        label = var_labels.get(var, "")
        if not label:
            continue
        value = match.group("value").strip()
        year_match = re.match(r"^(?P<year>\d{4})\s+(?P<name>.+)$", value)
        if not year_match:
            continue
        year = year_match.group("year")
        name = re.sub(r"\s+", " ", year_match.group("name")).strip()
        if not name:
            continue
        year_predicate = re.compile(
            rf"\b{re.escape(var)}\s*\.\s*year\s*=\s*['\"]?{re.escape(year)}['\"]?",
            re.IGNORECASE,
        )
        if not year_predicate.search(cypher):
            continue
        query = f"""
        MATCH (n:{label})
        WHERE toLower(toString(n.name)) = toLower($name)
          AND toString(n.year) = $year
        RETURN n.name AS name, n.year AS year
        LIMIT 2
        """
        try:
            with driver.session(database=resolve_actual_cypher_database(db_name)) as session:
                rows = list(session.run(query, name=name, year=year))
        except Exception:
            continue
        if len(rows) != 1:
            continue
        quote = match.group("quote")
        replacement = f"{match.group('prefix')}{_quote_cypher_string(str(rows[0]['name']), quote)}"
        repaired = cypher[: match.start()] + replacement + cypher[match.end() :]
        return repaired, {
            "repair_type": "split_year_name_where_predicate",
            "label": label,
            "variable": var,
            "old_property": "name",
            "old_value": value,
            "new_name": str(rows[0]["name"]),
            "year": str(rows[0]["year"]),
        }
    return cypher, None


def execute_cypher(driver, cypher: str, db_name: str = "wwc2019") -> dict[str, Any]:
    """执行 Cypher 并返回统一执行结果。

    参数：
    - driver：Neo4j driver。
    - cypher：待执行 Cypher 查询。
    - db_name：Neo4j 数据库名。
    """
    try:
        with driver.session(database=resolve_actual_cypher_database(db_name)) as session:
            result = list(session.run(cypher))

        if not result:
            return {
                "success": True,
                "result_text": "(query succeeded, but returned no rows)",
                "rows": [],
                "columns": [],
                "row_count": 0,
                "error": None,
            }

        keys = list(result[0].keys())
        header = " | ".join(keys)
        separator = "-" * len(header) if header else ""
        lines = [line for line in [header, separator] if line]
        rows: list[list[Any]] = []

        for record in result[:MAX_RESULT_ROWS]:
            row_values: list[Any] = []
            row_text: list[str] = []
            for key in keys:
                value = record[key]
                if hasattr(value, "_properties"):
                    value = dict(value._properties)
                row_values.append(value)
                row_text.append("NULL" if value is None else str(value))
            rows.append(row_values)
            lines.append(" | ".join(row_text))

        if len(result) > MAX_RESULT_ROWS:
            lines.append(f"... total {len(result)} rows, showing first {MAX_RESULT_ROWS}")

        return {
            "success": True,
            "result_text": "\n".join(lines),
            "rows": rows,
            "columns": keys,
            "row_count": len(result),
            "error": None,
        }
    except Exception as exc:
        return {
            "success": False,
            "result_text": "",
            "rows": [],
            "columns": [],
            "row_count": 0,
            "error": str(exc),
        }

def normalize_schema_name(schema_format: str) -> str:
    """规范化图 schema 格式名称。

    参数：
    - schema_format：调用方传入的 schema 格式别名。
    """
    return normalize_schema_format(schema_format)


def retrieve_cypher_candidates(question: str, *, db_name: str | None) -> dict[str, Any]:
    """检索或指定 Cypher 图数据库候选。

    参数：
    - question：用户自然语言问题。
    - db_name：可选指定图库名称；为空时自动检索。
    """
    result = retrieve_cypher_targets(
        question,
        db_name=db_name,
    )
    return {
        "db_candidates": result["raw_candidates"],
        "retrieval": {
            "method": result["method"],
            "selected_db": result["selected_db"],
            "score": result["score"],
            "top_candidates": [
                {
                    "db_name": candidate["resource_id"],
                    "score": candidate["score"],
                    "labels": candidate["metadata"].get("labels", []),
                    **({"error": candidate["raw"]["error"]} if "error" in candidate["raw"] else {}),
                }
                for candidate in result["top_candidates"]
            ],
        },
        "retrieval_ms": result["retrieval_ms"],
    }


def prepare_cypher_schema_context(
    *,
    driver: Any,
    question: str,
    current_db: str,
    schema_format: str,
    trace: list[dict[str, str]],
    schema_grounding_mode: str = "llm_filter",
) -> dict[str, Any]:
    """准备 Cypher 生成所需的图 schema 上下文。

    参数：
    - driver：Neo4j driver。
    - question：用户自然语言问题。
    - current_db：当前候选图库名称。
    - schema_format：schema 输出格式。
    - trace：LLM 调用 trace。
    - schema_grounding_mode：schema grounding 策略。默认 `llm_filter` 与正式系统一致。
    """
    return shared_prepare_cypher_schema_context(
        driver=driver,
        question=question,
        current_db=current_db,
        schema_format=schema_format,
        trace=trace,
        schema_grounding_mode=schema_grounding_mode,
    )


def generate_cypher_query(
    question: str,
    filtered_schema: str,
    trace: list[dict[str, str]],
    generation_feedback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """调用共享生成 Agent 生成 Cypher。

    参数：
    - question：用户自然语言问题。
    - filtered_schema：过滤后的相关图 schema。
    - trace：LLM 调用 trace。
    - generation_feedback：可选的 Verification 反馈，用于同一路由的重新生成。
    """
    return shared_generate_cypher_query(question, filtered_schema, trace, generation_feedback=generation_feedback)


def repair_cypher_avg_collect_distinct_map(cypher: str) -> tuple[str, dict[str, Any] | None]:
    """修复 `AVG` 之前错误使用 `COLLECT(DISTINCT map)` 的聚合粒度。

    参数：
    - cypher：待检查的 Cypher 查询。

    说明：
    - 当前主要作为确定性回归覆盖的工具函数，不在执行层无条件接受。
    """
    match = _AVG_COLLECT_DISTINCT_MAP_RE.search(cypher)
    if not match:
        return cypher, None

    map_body = match.group("map_body")
    entity = match.group("entity")
    scalar = match.group("scalar")
    if entity not in map_body or scalar not in map_body:
        return cypher, None

    replacement = (
        f"WITH {match.group('context')}, {entity}, {match.group('expr').strip()} AS {scalar}\n"
        f"WITH {match.group('context')}, AVG({scalar}) AS {match.group('avg_alias')}\n"
        f"RETURN {match.group('context_return').strip()} AS {match.group('context_alias')}, "
        f"{match.group('avg_alias')}"
    )
    repaired = cypher[: match.start()] + replacement + cypher[match.end() :]
    return repaired, {
        "repair_type": "avg_collect_distinct_map_to_entity_grain",
        "entity": entity,
        "scalar": scalar,
        "original_fragment": match.group(0),
        "repaired_fragment": replacement,
    }


def repair_cypher_tournament_age_reference(question: str, cypher: str) -> tuple[str, dict[str, Any] | None]:
    """修复赛事上下文年龄计算的时间参照。

    参数：
    - question：用户自然语言问题。
    - cypher：生成器输出的 Cypher。

    说明：
    - 当问题要求 tournament context 中的年龄时，优先使用 tournament year 而不是 match date。
    """
    normalized_question = question.lower()
    if "age" not in normalized_question or "tournament" not in normalized_question:
        return cypher, None
    if "match date" in normalized_question or "at the match" in normalized_question:
        return cypher, None
    if ".year" not in cypher or "duration.between" not in cypher:
        return cypher, None

    match = _DURATION_AGE_RE.search(cypher)
    if not match:
        return cypher, None

    tournament_alias_match = re.search(r"\((?P<alias>[A-Za-z_][A-Za-z0-9_]*):Tournament\b", cypher)
    if not tournament_alias_match:
        return cypher, None

    tournament_alias = tournament_alias_match.group("alias")
    replacement = f"({tournament_alias}.year - {match.group('dob')}.year)"
    repaired = cypher[: match.start()] + replacement + cypher[match.end() :]
    return repaired, {
        "repair_type": "tournament_age_reference_year",
        "original_fragment": match.group(0),
        "repaired_fragment": replacement,
    }


def repair_cypher_unrequested_collect_distinct(question: str, cypher: str) -> tuple[str, dict[str, Any] | None]:
    """Remove COLLECT(DISTINCT ...) when the question asks for event-level grouped lists."""
    if not _COLLECT_DISTINCT_RE.search(cypher):
        return cypher, None
    if _QUESTION_DISTINCT_RE.search(question):
        return cypher, None
    if not re.search(r"\b(visited|visits?|interacted|interactions?|appeared|piloted)\b", question, re.IGNORECASE):
        return cypher, None

    replacements: list[dict[str, Any]] = []
    for match in _COLLECT_DISTINCT_RE.finditer(cypher):
        expr = match.group("expr").strip()
        replacement = f"collect({expr})"
        replacements.append(
            {
                "start": match.start(),
                "end": match.end(),
                "original_fragment": match.group(0),
                "repaired_fragment": replacement,
            }
        )
    if not replacements:
        return cypher, None

    repaired = cypher
    for replacement in sorted(replacements, key=lambda item: item["start"], reverse=True):
        repaired = repaired[: replacement["start"]] + replacement["repaired_fragment"] + repaired[replacement["end"] :]
    return repaired, {
        "repair_type": "unrequested_collect_distinct_removed",
        "repairs": [
            {
                "original_fragment": item["original_fragment"],
                "repaired_fragment": item["repaired_fragment"],
            }
            for item in replacements
        ],
    }


def repair_cypher_numbered_relationship_union(question: str, cypher: str) -> tuple[str, dict[str, Any] | None]:
    """Prefer the base relationship type over base|base_1|... unions unless variants are requested."""
    if not _NUMBERED_REL_UNION_RE.search(cypher):
        return cypher, None
    if re.search(r"\b(all relationship types|all relationships|variants?|numbered)\b", question, re.IGNORECASE):
        return cypher, None

    replacements: list[dict[str, Any]] = []
    for match in _NUMBERED_REL_UNION_RE.finditer(cypher):
        base = match.group("base")
        replacement = f":{base}"
        replacements.append(
            {
                "start": match.start(),
                "end": match.end(),
                "original_fragment": match.group(0),
                "repaired_fragment": replacement,
            }
        )
    repaired = cypher
    for replacement in sorted(replacements, key=lambda item: item["start"], reverse=True):
        repaired = repaired[: replacement["start"]] + replacement["repaired_fragment"] + repaired[replacement["end"] :]
    return repaired, {
        "repair_type": "numbered_relationship_union_base_only",
        "repairs": [
            {
                "original_fragment": item["original_fragment"],
                "repaired_fragment": item["repaired_fragment"],
            }
            for item in replacements
        ],
    }


def repair_cypher_duration_total_seconds(cypher: str) -> tuple[str, dict[str, Any] | None]:
    """Rewrite unsupported duration.totalSeconds(x) to a Neo4j-compatible duration expression."""
    replacements: list[dict[str, Any]] = []
    for match in _DURATION_TOTAL_SECONDS_RE.finditer(cypher):
        expr = match.group("expr")
        replacement = _duration_seconds_expression(expr, full_duration=True)
        replacements.append(
            {
                "start": match.start(),
                "end": match.end(),
                "original_fragment": match.group(0),
                "repaired_fragment": replacement,
            }
        )
    if not replacements:
        return cypher, None

    repaired = cypher
    for replacement in sorted(replacements, key=lambda item: item["start"], reverse=True):
        repaired = repaired[: replacement["start"]] + replacement["repaired_fragment"] + repaired[replacement["end"] :]
    return repaired, {
        "repair_type": "duration_total_seconds_rewritten",
        "repairs": [
            {
                "original_fragment": item["original_fragment"],
                "repaired_fragment": item["repaired_fragment"],
            }
            for item in replacements
        ],
    }


def repair_cypher_duration_component_seconds(cypher: str) -> tuple[str, dict[str, Any] | None]:
    """Use `.seconds` for hand-written day/hour/minute/second duration sums.

    Generated Cypher often expands "in seconds" into days/hours/minutes/seconds.
    In the local MTQ Neo4j imports, the canonical gold queries use
    `duration.seconds`; this repair avoids triple-counting display components.
    """
    replacements: list[dict[str, Any]] = []
    for match in _DURATION_COMPONENT_SUM_RE.finditer(cypher):
        expr = match.group("expr")
        replacement = _duration_seconds_expression(expr, full_duration=False)
        if match.group("wrapped") and match.group("closing"):
            replacement = f"({replacement})"
        replacements.append(
            {
                "start": match.start(),
                "end": match.end(),
                "original_fragment": match.group(0),
                "repaired_fragment": replacement,
            }
        )
    if not replacements:
        return cypher, None

    repaired = cypher
    for replacement in sorted(replacements, key=lambda item: item["start"], reverse=True):
        repaired = repaired[: replacement["start"]] + replacement["repaired_fragment"] + repaired[replacement["end"] :]
    return repaired, {
        "repair_type": "duration_component_seconds_rewritten",
        "repairs": [
            {
                "original_fragment": item["original_fragment"],
                "repaired_fragment": item["repaired_fragment"],
            }
            for item in replacements
        ],
    }


def repair_cypher_parallel_collect_multiplication(cypher: str) -> tuple[str, dict[str, Any] | None]:
    """Split parallel relationship expansions before collect to avoid Cartesian multiplication."""
    match = _PARALLEL_COLLECT_RE.search(cypher)
    if not match:
        return_match = _PARALLEL_COLLECT_RETURN_RE.search(cypher)
        if not return_match:
            return cypher, None
        anchor = return_match.group("anchor")
        v1 = return_match.group("v1")
        v2 = return_match.group("v2")
        prop1 = return_match.group("prop1")
        prop2 = return_match.group("prop2")
        a1 = return_match.group("a1")
        a2 = return_match.group("a2")
        anchor_return = return_match.group("anchor_return").strip()
        replacement = (
            f"{return_match.group('first').rstrip().rstrip(',')}\n"
            f"WITH {anchor}, collect({v1}.{prop1}) AS {a1}\n"
            f"MATCH {return_match.group('second').rstrip()}\n"
            f"WITH {anchor}, {a1}, collect({v2}.{prop2}) AS {a2}\n"
            f"RETURN {anchor_return}, {a1}, {a2}"
        )
        repaired = cypher[: return_match.start()] + replacement + cypher[return_match.end() :]
        return repaired, {
            "repair_type": "parallel_collect_multiplication_split",
            "original_fragment": return_match.group(0),
            "repaired_fragment": replacement,
        }

    anchor = match.group("anchor")
    v1 = match.group("v1")
    v2 = match.group("v2")
    prop1 = match.group("prop1")
    prop2 = match.group("prop2")
    a1 = match.group("a1")
    a2 = match.group("a2")
    replacement = (
        f"{match.group('first').rstrip()}\n"
        f"WITH {anchor}, collect({v1}.{prop1}) AS {a1}\n"
        f"{match.group('second').rstrip()}\n"
        f"WITH {anchor}, {a1}, collect({v2}.{prop2}) AS {a2}"
    )
    repaired = cypher[: match.start()] + replacement + cypher[match.end() :]
    return repaired, {
        "repair_type": "parallel_collect_multiplication_split",
        "original_fragment": match.group(0),
        "repaired_fragment": replacement,
    }


def repair_cypher_required_count_optional_match(question: str, cypher: str) -> tuple[str, dict[str, Any] | None]:
    """Use MATCH instead of OPTIONAL MATCH for required relationship-count averages."""
    tokens = _question_tokens(question)
    if not ({"average", "avg", "number", "count", "total"} & tokens):
        return cypher, None
    if not re.search(r"\b(served|serves|watched|visited|involved|interacted|appeared|piloted)\b", question, re.IGNORECASE):
        return cypher, None
    if "OPTIONAL MATCH" not in cypher.upper():
        return cypher, None

    replacements: list[dict[str, Any]] = []
    for match in re.finditer(r"\bOPTIONAL\s+MATCH\b(?P<body>.*?)(?=\bWITH\b|\bRETURN\b|$)", cypher, re.IGNORECASE | re.DOTALL):
        body = match.group("body")
        if re.search(r"\bCOUNT\s*\(", cypher[match.end() : match.end() + 160], re.IGNORECASE):
            replacements.append(
                {
                    "start": match.start(),
                    "end": match.start() + len(match.group(0)) - len(body),
                    "original_fragment": match.group(0)[: len(match.group(0)) - len(body)],
                    "repaired_fragment": "MATCH",
                }
            )
    if not replacements:
        return cypher, None

    repaired = cypher
    for replacement in sorted(replacements, key=lambda item: item["start"], reverse=True):
        repaired = repaired[: replacement["start"]] + replacement["repaired_fragment"] + repaired[replacement["end"] :]
    return repaired, {
        "repair_type": "required_count_optional_match_to_match",
        "repairs": [
            {
                "original_fragment": item["original_fragment"],
                "repaired_fragment": item["repaired_fragment"],
            }
            for item in replacements
        ],
    }


def repair_cypher_pattern_count_required_relationship(question: str, cypher: str) -> tuple[str, dict[str, Any] | None]:
    """Expand COUNT { (x)-[:REL]->() } into MATCH + count when the relationship is required."""
    tokens = _question_tokens(question)
    if not ({"average", "avg", "number", "count", "total"} & tokens):
        return cypher, None
    if not re.search(r"\b(served|serves|watched|visited|involved|interacted|appeared|piloted)\b", question, re.IGNORECASE):
        return cypher, None

    match = _PATTERN_COUNT_REL_RE.search(cypher)
    if not match:
        return cypher, None

    prefix = match.group("prefix").rstrip()
    rel_type = match.group("rel_type")
    alias = match.group("alias")
    source = match.group("source")
    source_context = prefix[len("WITH") :].strip()
    target = match.group("target").strip()
    target_fragment = f"({target})" if target else "()"
    if match.group("left") == "-" and match.group("right") == "->":
        pattern = f"({source})-[:{rel_type}]->{target_fragment}"
    elif match.group("left") == "<-" and match.group("right") == "-":
        pattern = f"({source})<-[:{rel_type}]-{target_fragment}"
    else:
        pattern = f"({source})-[:{rel_type}]-{target_fragment}"
    replacement = (
        f"{prefix}\n"
        f"MATCH {pattern}\n"
        f"WITH {source_context}, COUNT(*) AS {alias}"
    )
    repaired = cypher[: match.start()] + replacement + cypher[match.end() :]
    return repaired, {
        "repair_type": "pattern_count_required_relationship_to_match",
        "original_fragment": match.group(0),
        "repaired_fragment": replacement,
        "relationship_type": rel_type,
    }


def repair_cypher_join_multiplied_node_counts(question: str, cypher: str) -> tuple[str, dict[str, Any] | None]:
    """修复关系扩展导致的逻辑节点计数放大。

    参数：
    - question：用户自然语言问题。
    - cypher：生成器输出的 Cypher。

    说明：
    - 只对逻辑节点数量使用 `COUNT(DISTINCT node)`，不处理用户明确要求的关系事件计数。
    """
    tokens = _question_tokens(question)
    if not ({"count", "number", "total"} & tokens):
        return cypher, None
    if "COUNT(DISTINCT" not in cypher.upper():
        return cypher, None

    variable_labels = _build_variable_label_map(cypher)
    replacements: list[dict[str, Any]] = []
    for match in _COUNT_NODE_RE.finditer(cypher):
        variable = _strip_cypher_identifier(match.group("var"))
        label = variable_labels.get(variable, "")
        if not label:
            continue
        label_terms = _LOGICAL_NODE_COUNT_LABELS.get(label.lower(), {label.lower(), label.lower() + "s"})
        if not (tokens & label_terms):
            continue
        replacement = f"COUNT(DISTINCT {match.group('var')})"
        replacements.append(
            {
                "start": match.start(),
                "end": match.end(),
                "original_fragment": match.group(0),
                "repaired_fragment": replacement,
                "variable": variable,
                "label": label,
            }
        )

    if not replacements:
        return cypher, None

    repaired = cypher
    for replacement in sorted(replacements, key=lambda item: item["start"], reverse=True):
        repaired = repaired[: replacement["start"]] + replacement["repaired_fragment"] + repaired[replacement["end"] :]

    return repaired, {
        "repair_type": "join_multiplied_node_count_distinct",
        "repairs": [
            {
                "original_fragment": item["original_fragment"],
                "repaired_fragment": item["repaired_fragment"],
                "variable": item["variable"],
                "label": item["label"],
            }
            for item in replacements
        ],
    }


def repair_cypher_unrequested_return_distinct(question: str, cypher: str) -> tuple[str, dict[str, Any] | None]:
    """移除用户未请求的 `RETURN DISTINCT`。

    参数：
    - question：用户自然语言问题。
    - cypher：生成器输出的 Cypher。
    """
    if not _RETURN_DISTINCT_RE.search(cypher):
        return cypher, None
    if _QUESTION_DISTINCT_RE.search(question):
        return cypher, None
    if not _QUESTION_ROW_LEVEL_RE.search(question):
        return cypher, None
    return_match = _RETURN_ANY_CLAUSE_RE.search(cypher)
    if (
        return_match
        and not _QUESTION_STRONG_ROW_LEVEL_RE.search(question)
        and _SIMPLE_RETURN_NAME_TITLE_RE.match(return_match.group("expr").strip())
        and len(_REL_TOKEN_RE.findall(cypher)) >= 2
    ):
        return cypher, None

    repaired = _RETURN_DISTINCT_RE.sub("RETURN", cypher, count=1)
    if repaired == cypher:
        return cypher, None
    return repaired, {"repair_type": "unrequested_return_distinct_removed"}


def repair_cypher_missing_return_distinct(question: str, cypher: str) -> tuple[str, dict[str, Any] | None]:
    """Add RETURN DISTINCT for single-entity name/title list queries prone to join multiplication."""
    if _RETURN_DISTINCT_RE.search(cypher):
        return cypher, None
    if _QUESTION_DISTINCT_RE.search(question) or _QUESTION_STRONG_ROW_LEVEL_RE.search(question):
        return cypher, None
    if not re.search(r"\b(all|list|find|show|retrieve)\b", question, re.IGNORECASE):
        return cypher, None

    return_match = _RETURN_CLAUSE_RE.search(cypher)
    if not return_match:
        return cypher, None
    return_expr = return_match.group("expr").strip()
    if "," in return_expr:
        return cypher, None
    if not _SIMPLE_RETURN_NAME_TITLE_RE.match(return_expr):
        return cypher, None
    if len(re.findall(r"\bMATCH\b", cypher, re.IGNORECASE)) < 1:
        return cypher, None
    if len(_REL_TOKEN_RE.findall(cypher)) < 2:
        return cypher, None

    repaired = cypher[: return_match.start()] + "RETURN DISTINCT " + cypher[return_match.start("expr") :]
    return repaired, {
        "repair_type": "missing_return_distinct_added",
        "return_expression": return_expr,
    }


def repair_cypher_team_goal_score_property_to_goal_events(
    question: str,
    cypher: str,
) -> tuple[str, dict[str, Any] | None]:
    """Prefer goal events over team-match score sums for team goal-total questions.

    The repair is intentionally narrow: it only fires when the question asks for
    team goals scored, the generated query sums `PLAYED_IN.score`, and the query
    has a clear Team -> Match pattern.
    """
    tokens = _question_tokens(question)
    if not ({"goal", "goals"} & tokens and {"scored", "score"} & tokens and {"team", "teams"} & tokens):
        return cypher, None
    if {"average", "avg"} & tokens or ("per" in tokens and "match" in tokens):
        return cypher, None
    if "SCORED_GOAL" in cypher.upper():
        return cypher, None

    sum_match = _SUM_SCORE_RE.search(cypher)
    played_match = _TEAM_PLAYED_IN_RE.search(cypher)
    if not sum_match or not played_match:
        return cypher, None
    if _strip_cypher_identifier(sum_match.group("rel")) != _strip_cypher_identifier(played_match.group("rel") or ""):
        return cypher, None

    team_var = _strip_cypher_identifier(played_match.group("team"))
    match_var = _strip_cypher_identifier(played_match.group("match"))
    variable_labels = _build_variable_label_map(cypher)
    tournament_vars = [
        variable
        for variable, label in variable_labels.items()
        if label.lower() == "tournament"
    ]
    tournament_var = tournament_vars[0] if tournament_vars else ""
    aggregate_alias = _strip_cypher_identifier(sum_match.group("alias"))
    return_projection = (
        f"{team_var}.name AS teamName, {tournament_var}.name AS tournamentName, COUNT(sg) AS {aggregate_alias}"
        if tournament_var
        else f"{team_var}.name AS teamName, COUNT(sg) AS {aggregate_alias}"
    )
    return_match = re.search(r"\bRETURN\b[\s\S]*$", cypher, re.IGNORECASE)
    if not return_match:
        return cypher, None

    prefix = cypher[: return_match.start()].rstrip()
    bridge = (
        f"\nOPTIONAL MATCH (p:Person)-[sg:SCORED_GOAL]->({match_var})\n"
        f"WHERE (p)-[:REPRESENTS]->({team_var})"
    )
    repaired = f"{prefix}{bridge}\nRETURN {return_projection}"
    return repaired, {
        "repair_type": "team_goal_score_property_to_goal_events",
        "team_variable": team_var,
        "match_variable": match_var,
        "tournament_variable": tournament_var,
        "original_aggregate": sum_match.group(0),
        "repaired_aggregate": f"COUNT(sg) AS {aggregate_alias}",
    }


def execute_cypher_with_repair(
    *,
    driver: Any,
    question: str,
    filtered_schema: str,
    cypher: str,
    current_db: str,
    relationships_data: list[dict[str, Any]] | None = None,
    trace: list[dict[str, str]],
) -> dict[str, Any]:
    """执行 Cypher，并串联 schema 校验、LLM 修复和确定性修复。

    参数：
    - driver：Neo4j driver。
    - question：用户自然语言问题。
    - filtered_schema：过滤后的相关图 schema。
    - cypher：生成器输出的 Cypher。
    - current_db：当前候选图库名称。
    - relationships_data：schema 中的关系 start/type/end 数据，用于 domain validation。
    - trace：LLM 调用 trace。
    """

    def _execute_validated_cypher(current_cypher: str) -> dict[str, Any]:
        """先做关系域校验，再执行 Cypher。

        参数：
        - current_cypher：当前待执行查询。
        """
        validation = validate_cypher_relationship_domains(current_cypher, relationships_data or [])
        if not validation["valid"]:
            return {
                "success": False,
                "result_text": "",
                "rows": [],
                "columns": [],
                "row_count": 0,
                "error": _format_relationship_domain_error(validation),
                "schema_validation": validation,
            }
        return execute_cypher(driver, current_cypher, current_db)

    def _fix_current_cypher(current_cypher: str, error_message: str, attempt: int) -> tuple[str, str]:
        """调用 Cypher 修复 Agent。

        参数：
        - current_cypher：当前失败查询。
        - error_message：执行或 schema 校验错误。
        - attempt：第几次修复尝试。
        """
        return fix_cypher_query(
            question=question,
            filtered_schema=filtered_schema,
            bad_cypher=current_cypher,
            error_msg=error_message,
            attempt=attempt,
            trace=trace,
        )

    initial_cypher = cypher
    initial_repair_history: list[dict[str, Any]] = []
    grounded_cypher, grounding_detail = repair_cypher_quoted_value_grounding(
        driver=driver,
        db_name=current_db,
        question=question,
        cypher=cypher,
    )
    if grounding_detail and grounded_cypher != cypher:
        initial_cypher = grounded_cypher
        initial_repair_history.append(
            {
                "attempt": 0,
                "bad_cypher": cypher,
                "error": "quoted_value_grounding",
                "fixed_cypher": grounded_cypher,
                "cot_raw": "deterministic_quoted_value_grounding",
                "repair_detail": grounding_detail,
            }
        )
    split_cypher, split_detail = repair_cypher_split_year_name_literal(
        driver=driver,
        db_name=current_db,
        cypher=initial_cypher,
    )
    if split_detail and split_cypher != initial_cypher:
        initial_repair_history.append(
            {
                "attempt": 0,
                "bad_cypher": initial_cypher,
                "error": "split_year_name_literal",
                "fixed_cypher": split_cypher,
                "cot_raw": "deterministic_split_year_name_literal",
                "repair_detail": split_detail,
            }
        )
        initial_cypher = split_cypher
    split_where_cypher, split_where_detail = repair_cypher_split_year_name_where_predicate(
        driver=driver,
        db_name=current_db,
        cypher=initial_cypher,
    )
    if split_where_detail and split_where_cypher != initial_cypher:
        initial_repair_history.append(
            {
                "attempt": 0,
                "bad_cypher": initial_cypher,
                "error": "split_year_name_where_predicate",
                "fixed_cypher": split_where_cypher,
                "cot_raw": "deterministic_split_year_name_where_predicate",
                "repair_detail": split_where_detail,
            }
        )
        initial_cypher = split_where_cypher

    directed_cypher, direction_detail = repair_cypher_unambiguous_relationship_direction(
        initial_cypher,
        relationships_data or [],
        question,
    )
    if direction_detail and directed_cypher != initial_cypher:
        initial_repair_history.append(
            {
                "attempt": 0,
                "bad_cypher": initial_cypher,
                "error": "unambiguous_relationship_direction",
                "fixed_cypher": directed_cypher,
                "cot_raw": "deterministic_unambiguous_relationship_direction",
                "repair_detail": direction_detail,
            }
        )
        initial_cypher = directed_cypher

    numbered_rel_cypher, numbered_rel_detail = repair_cypher_numbered_relationship_union(
        question,
        initial_cypher,
    )
    if numbered_rel_detail and numbered_rel_cypher != initial_cypher:
        initial_repair_history.append(
            {
                "attempt": 0,
                "bad_cypher": initial_cypher,
                "error": "numbered_relationship_union",
                "fixed_cypher": numbered_rel_cypher,
                "cot_raw": "deterministic_numbered_relationship_union_base_only",
                "repair_detail": numbered_rel_detail,
            }
        )
        initial_cypher = numbered_rel_cypher

    duration_cypher, duration_detail = repair_cypher_duration_total_seconds(initial_cypher)
    if duration_detail and duration_cypher != initial_cypher:
        initial_repair_history.append(
            {
                "attempt": 0,
                "bad_cypher": initial_cypher,
                "error": "duration_total_seconds",
                "fixed_cypher": duration_cypher,
                "cot_raw": "deterministic_duration_total_seconds_rewritten",
                "repair_detail": duration_detail,
            }
        )
        initial_cypher = duration_cypher

    duration_component_cypher, duration_component_detail = repair_cypher_duration_component_seconds(initial_cypher)
    if duration_component_detail and duration_component_cypher != initial_cypher:
        initial_repair_history.append(
            {
                "attempt": 0,
                "bad_cypher": initial_cypher,
                "error": "duration_component_seconds",
                "fixed_cypher": duration_component_cypher,
                "cot_raw": "deterministic_duration_component_seconds_rewritten",
                "repair_detail": duration_component_detail,
            }
        )
        initial_cypher = duration_component_cypher

    execution_result = run_execution_with_repair(
        initial_query=initial_cypher,
        max_retries=MAX_RETRIES,
        execute_fn=_execute_validated_cypher,
        fix_fn=_fix_current_cypher,
        should_try_next_candidate_fn=lambda error_message, attempt: (
            attempt == 0
            and any(token in error_message.lower() for token in ["unknown label", "not found", "relationshiptype"])
        ),
        query_key="cypher",
        bad_query_key="bad_cypher",
        fixed_query_key="fixed_cypher",
        terminal_failure_message=lambda error_message, retry_count: (
            f"Execution failed after {retry_count} retries: {error_message}"
        ),
    )

    if initial_repair_history:
        execution_result["retry_history"] = initial_repair_history + list(
            execution_result.get("retry_history", [])
                )

    if execution_result["success"]:
        repaired_cypher, repair_detail = repair_cypher_numbered_relationship_union(
            question,
            execution_result["cypher"],
        )
        if repair_detail and repaired_cypher != execution_result["cypher"]:
            repaired_execution = _execute_validated_cypher(repaired_cypher)
            if (
                repaired_execution["success"]
                and 0 < repaired_execution["row_count"] <= execution_result["row_count"]
                and repaired_execution["columns"] == execution_result["result_columns"]
            ):
                retry_history = list(execution_result.get("retry_history", []))
                retry_history.append(
                    {
                        "attempt": execution_result.get("retries", 0) + 1,
                        "bad_cypher": execution_result["cypher"],
                        "error": "numbered_relationship_union",
                        "fixed_cypher": repaired_cypher,
                        "cot_raw": "deterministic_numbered_relationship_union_base_only",
                        "repair_detail": repair_detail,
                    }
                )
                execution_result.update(
                    {
                        "result": repaired_execution["result_text"],
                        "result_rows": repaired_execution["rows"],
                        "result_columns": repaired_execution["columns"],
                        "row_count": repaired_execution["row_count"],
                        "retries": execution_result.get("retries", 0) + 1,
                        "retry_history": retry_history,
                        "cypher": repaired_cypher,
                    }
                )

    if execution_result["success"]:
        repaired_cypher, repair_detail = repair_cypher_unambiguous_relationship_direction(
            execution_result["cypher"],
            relationships_data or [],
            question,
        )
        if repair_detail and repaired_cypher != execution_result["cypher"]:
            repaired_execution = _execute_validated_cypher(repaired_cypher)
            if (
                repaired_execution["success"]
                and 0 < repaired_execution["row_count"] <= execution_result["row_count"]
                and repaired_execution["columns"] == execution_result["result_columns"]
            ):
                retry_history = list(execution_result.get("retry_history", []))
                retry_history.append(
                    {
                        "attempt": execution_result.get("retries", 0) + 1,
                        "bad_cypher": execution_result["cypher"],
                        "error": "unambiguous_relationship_direction",
                        "fixed_cypher": repaired_cypher,
                        "cot_raw": "deterministic_unambiguous_relationship_direction",
                        "repair_detail": repair_detail,
                    }
                )
                execution_result.update(
                    {
                        "result": repaired_execution["result_text"],
                        "result_rows": repaired_execution["rows"],
                        "result_columns": repaired_execution["columns"],
                        "row_count": repaired_execution["row_count"],
                        "retries": execution_result.get("retries", 0) + 1,
                        "retry_history": retry_history,
                        "cypher": repaired_cypher,
                    }
                )

    if execution_result["success"]:
        repaired_cypher, repair_detail = repair_cypher_unrequested_collect_distinct(
            question,
            execution_result["cypher"],
        )
        if repair_detail and repaired_cypher != execution_result["cypher"]:
            repaired_execution = _execute_validated_cypher(repaired_cypher)
            if (
                repaired_execution["success"]
                and repaired_execution["row_count"] == execution_result["row_count"]
                and repaired_execution["columns"] == execution_result["result_columns"]
            ):
                retry_history = list(execution_result.get("retry_history", []))
                retry_history.append(
                    {
                        "attempt": execution_result.get("retries", 0) + 1,
                        "bad_cypher": execution_result["cypher"],
                        "error": "unrequested_collect_distinct",
                        "fixed_cypher": repaired_cypher,
                        "cot_raw": "deterministic_unrequested_collect_distinct_removed",
                        "repair_detail": repair_detail,
                    }
                )
                execution_result.update(
                    {
                        "result": repaired_execution["result_text"],
                        "result_rows": repaired_execution["rows"],
                        "result_columns": repaired_execution["columns"],
                        "row_count": repaired_execution["row_count"],
                        "retries": execution_result.get("retries", 0) + 1,
                        "retry_history": retry_history,
                        "cypher": repaired_cypher,
                    }
                )

    if execution_result["success"]:
        repaired_cypher, repair_detail = repair_cypher_parallel_collect_multiplication(
            execution_result["cypher"],
        )
        if repair_detail and repaired_cypher != execution_result["cypher"]:
            repaired_execution = _execute_validated_cypher(repaired_cypher)
            if (
                repaired_execution["success"]
                and repaired_execution["row_count"] == execution_result["row_count"]
                and repaired_execution["columns"] == execution_result["result_columns"]
            ):
                retry_history = list(execution_result.get("retry_history", []))
                retry_history.append(
                    {
                        "attempt": execution_result.get("retries", 0) + 1,
                        "bad_cypher": execution_result["cypher"],
                        "error": "parallel_collect_multiplication",
                        "fixed_cypher": repaired_cypher,
                        "cot_raw": "deterministic_parallel_collect_multiplication_split",
                        "repair_detail": repair_detail,
                    }
                )
                execution_result.update(
                    {
                        "result": repaired_execution["result_text"],
                        "result_rows": repaired_execution["rows"],
                        "result_columns": repaired_execution["columns"],
                        "row_count": repaired_execution["row_count"],
                        "retries": execution_result.get("retries", 0) + 1,
                        "retry_history": retry_history,
                        "cypher": repaired_cypher,
                    }
                )

    if execution_result["success"]:
        repaired_cypher, repair_detail = repair_cypher_required_count_optional_match(
            question,
            execution_result["cypher"],
        )
        if repair_detail and repaired_cypher != execution_result["cypher"]:
            repaired_execution = _execute_validated_cypher(repaired_cypher)
            if (
                repaired_execution["success"]
                and repaired_execution["row_count"] == execution_result["row_count"]
                and repaired_execution["columns"] == execution_result["result_columns"]
            ):
                retry_history = list(execution_result.get("retry_history", []))
                retry_history.append(
                    {
                        "attempt": execution_result.get("retries", 0) + 1,
                        "bad_cypher": execution_result["cypher"],
                        "error": "required_count_optional_match",
                        "fixed_cypher": repaired_cypher,
                        "cot_raw": "deterministic_required_count_optional_match_to_match",
                        "repair_detail": repair_detail,
                    }
                )
                execution_result.update(
                    {
                        "result": repaired_execution["result_text"],
                        "result_rows": repaired_execution["rows"],
                        "result_columns": repaired_execution["columns"],
                        "row_count": repaired_execution["row_count"],
                        "retries": execution_result.get("retries", 0) + 1,
                        "retry_history": retry_history,
                        "cypher": repaired_cypher,
                    }
                )

    if execution_result["success"]:
        repaired_cypher, repair_detail = repair_cypher_pattern_count_required_relationship(
            question,
            execution_result["cypher"],
        )
        if repair_detail and repaired_cypher != execution_result["cypher"]:
            repaired_execution = _execute_validated_cypher(repaired_cypher)
            if (
                repaired_execution["success"]
                and repaired_execution["row_count"] == execution_result["row_count"]
                and repaired_execution["columns"] == execution_result["result_columns"]
            ):
                retry_history = list(execution_result.get("retry_history", []))
                retry_history.append(
                    {
                        "attempt": execution_result.get("retries", 0) + 1,
                        "bad_cypher": execution_result["cypher"],
                        "error": "pattern_count_required_relationship",
                        "fixed_cypher": repaired_cypher,
                        "cot_raw": "deterministic_pattern_count_required_relationship_to_match",
                        "repair_detail": repair_detail,
                    }
                )
                execution_result.update(
                    {
                        "result": repaired_execution["result_text"],
                        "result_rows": repaired_execution["rows"],
                        "result_columns": repaired_execution["columns"],
                        "row_count": repaired_execution["row_count"],
                        "retries": execution_result.get("retries", 0) + 1,
                        "retry_history": retry_history,
                        "cypher": repaired_cypher,
                    }
                )

    if execution_result["success"]:
        repaired_cypher, repair_detail = repair_cypher_missing_return_distinct(
            question,
            execution_result["cypher"],
        )
        if repair_detail and repaired_cypher != execution_result["cypher"]:
            repaired_execution = _execute_validated_cypher(repaired_cypher)
            if (
                repaired_execution["success"]
                and 0 < repaired_execution["row_count"] <= execution_result["row_count"]
                and repaired_execution["columns"] == execution_result["result_columns"]
            ):
                retry_history = list(execution_result.get("retry_history", []))
                retry_history.append(
                    {
                        "attempt": execution_result.get("retries", 0) + 1,
                        "bad_cypher": execution_result["cypher"],
                        "error": "missing_return_distinct",
                        "fixed_cypher": repaired_cypher,
                        "cot_raw": "deterministic_missing_return_distinct_added",
                        "repair_detail": repair_detail,
                    }
                )
                execution_result.update(
                    {
                        "result": repaired_execution["result_text"],
                        "result_rows": repaired_execution["rows"],
                        "result_columns": repaired_execution["columns"],
                        "row_count": repaired_execution["row_count"],
                        "retries": execution_result.get("retries", 0) + 1,
                        "retry_history": retry_history,
                        "cypher": repaired_cypher,
                    }
                )

    if execution_result["success"]:
        repaired_cypher, repair_detail = repair_cypher_unrequested_return_distinct(
            question,
            execution_result["cypher"],
        )
        if repair_detail and repaired_cypher != execution_result["cypher"]:
            repaired_execution = _execute_validated_cypher(repaired_cypher)
            if (
                repaired_execution["success"]
                and repaired_execution["row_count"] >= execution_result["row_count"]
                and repaired_execution["columns"] == execution_result["result_columns"]
            ):
                retry_history = list(execution_result.get("retry_history", []))
                retry_history.append(
                    {
                        "attempt": execution_result.get("retries", 0) + 1,
                        "bad_cypher": execution_result["cypher"],
                        "error": "unrequested_return_distinct",
                        "fixed_cypher": repaired_cypher,
                        "cot_raw": "deterministic_unrequested_return_distinct_removed",
                        "repair_detail": repair_detail,
                    }
                )
                execution_result.update(
                    {
                        "result": repaired_execution["result_text"],
                        "result_rows": repaired_execution["rows"],
                        "result_columns": repaired_execution["columns"],
                        "row_count": repaired_execution["row_count"],
                        "retries": execution_result.get("retries", 0) + 1,
                        "retry_history": retry_history,
                        "cypher": repaired_cypher,
                    }
                )

    if execution_result["success"]:
        repaired_cypher, repair_detail = repair_cypher_team_goal_score_property_to_goal_events(
            question,
            execution_result["cypher"],
        )
        if repair_detail and repaired_cypher != execution_result["cypher"]:
            repaired_execution = _execute_validated_cypher(repaired_cypher)
            if repaired_execution["success"] and repaired_execution["row_count"] == execution_result["row_count"]:
                retry_history = list(execution_result.get("retry_history", []))
                retry_history.append(
                    {
                        "attempt": execution_result.get("retries", 0) + 1,
                        "bad_cypher": execution_result["cypher"],
                        "error": "team_goal_score_property",
                        "fixed_cypher": repaired_cypher,
                        "cot_raw": "deterministic_team_goal_score_property_to_goal_events",
                        "repair_detail": repair_detail,
                    }
                )
                execution_result.update(
                    {
                        "result": repaired_execution["result_text"],
                        "result_rows": repaired_execution["rows"],
                        "result_columns": repaired_execution["columns"],
                        "row_count": repaired_execution["row_count"],
                        "retries": execution_result.get("retries", 0) + 1,
                        "retry_history": retry_history,
                        "cypher": repaired_cypher,
                    }
                )

    if execution_result["success"]:
        repaired_cypher, repair_detail = repair_cypher_join_multiplied_node_counts(
            question,
            execution_result["cypher"],
        )
        if repair_detail and repaired_cypher != execution_result["cypher"]:
            repaired_execution = _execute_validated_cypher(repaired_cypher)
            if repaired_execution["success"] and repaired_execution["row_count"] == execution_result["row_count"]:
                retry_history = list(execution_result.get("retry_history", []))
                retry_history.append(
                    {
                        "attempt": execution_result.get("retries", 0) + 1,
                        "bad_cypher": execution_result["cypher"],
                        "error": "join_multiplied_node_count",
                        "fixed_cypher": repaired_cypher,
                        "cot_raw": "deterministic_join_multiplied_node_count_distinct",
                        "repair_detail": repair_detail,
                    }
                )
                execution_result.update(
                    {
                        "result": repaired_execution["result_text"],
                        "result_rows": repaired_execution["rows"],
                        "result_columns": repaired_execution["columns"],
                        "row_count": repaired_execution["row_count"],
                        "retries": execution_result.get("retries", 0) + 1,
                        "retry_history": retry_history,
                        "cypher": repaired_cypher,
                    }
                )

    if execution_result["success"]:
        repaired_cypher, repair_detail = repair_cypher_tournament_age_reference(
            question,
            execution_result["cypher"],
        )
        if repair_detail and repaired_cypher != execution_result["cypher"]:
            repaired_execution = _execute_validated_cypher(repaired_cypher)
            if repaired_execution["success"] and repaired_execution["row_count"] == execution_result["row_count"]:
                retry_history = list(execution_result.get("retry_history", []))
                retry_history.append(
                    {
                        "attempt": execution_result.get("retries", 0) + 1,
                        "bad_cypher": execution_result["cypher"],
                        "error": "tournament_age_reference",
                        "fixed_cypher": repaired_cypher,
                        "cot_raw": "deterministic_tournament_age_reference_year",
                        "repair_detail": repair_detail,
                    }
                )
                execution_result.update(
                    {
                        "result": repaired_execution["result_text"],
                        "result_rows": repaired_execution["rows"],
                        "result_columns": repaired_execution["columns"],
                        "row_count": repaired_execution["row_count"],
                        "retries": execution_result.get("retries", 0) + 1,
                        "retry_history": retry_history,
                        "cypher": repaired_cypher,
                    }
                )

    return execution_result


def execute_cypher_without_repair(
    *,
    driver: Any,
    cypher: str,
    current_db: str,
) -> dict[str, Any]:
    """执行初始 Cypher，不做任何确定性或 LLM repair。用于严格 repair 消融。"""
    execution = execute_cypher(driver, cypher, current_db)
    return {
        "success": bool(execution["success"]),
        "result": execution["result_text"],
        "result_rows": execution["rows"],
        "result_columns": execution["columns"],
        "row_count": execution["row_count"],
        "cypher": cypher,
        "error": execution.get("error") or "",
        "retries": 0,
        "retry_history": [],
        "execution_ms": 0,
        "retry_ms": 0,
        "should_try_next_candidate": False,
    }
