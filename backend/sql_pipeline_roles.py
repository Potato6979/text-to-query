import sqlite3
import re
from typing import Any

from .config import (
    MAX_RESULT_ROWS,
    MAX_RETRIES,
)
from .execution_agents import run_execution_with_repair
from .fix_agents import fix_sql_query
from .generation_agents import generate_sql_query as shared_generate_sql_query
from .retrieval_agents import retrieve_sql_targets
from .schema_agents import extract_sql_schema, prepare_sql_schema_context as shared_prepare_sql_schema_context
from .sql_example_pool_dail import ExamplePoolItem

# 中文函数说明索引：
# - retrieve_sql_candidates(question)：检索 SQL 候选数据库。
# - extract_schema(db_path)：读取 SQLite schema。
# - prepare_sql_schema_context(...)：准备 DAIL 示例选择、masked question、完整 schema 和工作上下文。
# - generate_sql_query(...)：调用 SQL 生成器，结合 schema/example/feedback 生成 SQL。
# - execute_sql / execute_sql_with_repair(...)：执行 SQL 并进行有限次确定性修复。
# - deterministic repair helpers：处理 value grounding、DISTINCT、projection、superlative 等通用 SQL 形态问题，避免写死具体数据集。


_STRING_EQ_RE = re.compile(
    r"(?P<left>(?:(?P<alias>[A-Za-z_][A-Za-z0-9_]*)\.)?[\"`]?(?P<column>[A-Za-z_][A-Za-z0-9_]*)[\"`]?)"
    r"\s*=\s*(?P<quote>['\"])(?P<value>.*?)(?P=quote)",
    re.IGNORECASE | re.DOTALL,
)
_TABLE_ALIAS_RE = re.compile(
    r"\b(?:FROM|JOIN)\s+[\"`]?(?P<table>[A-Za-z_][A-Za-z0-9_]*)[\"`]?"
    r"(?:\s+(?:AS\s+)?(?P<alias>"
    r"(?!(?:ON|WHERE|JOIN|LEFT|RIGHT|INNER|OUTER|FULL|CROSS|GROUP|ORDER|HAVING|LIMIT)\b)"
    r"[A-Za-z_][A-Za-z0-9_]*))?",
    re.IGNORECASE,
)
_SELECT_RE = re.compile(r"^\s*SELECT\s+", re.IGNORECASE)
_SELECT_DISTINCT_RE = re.compile(r"^\s*SELECT\s+DISTINCT\b", re.IGNORECASE)
_SQL_AGG_RE = re.compile(r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\(", re.IGNORECASE)
_SELECT_FROM_RE = re.compile(r"^\s*SELECT\s+(?P<select>[\s\S]+?)\s+FROM\s", re.IGNORECASE)
_QUESTION_DISTINCT_RE = re.compile(r"\b(unique|distinct|different|non[-\s]?duplicated)\b", re.IGNORECASE)
_QUESTION_ALL_ROWS_RE = re.compile(r"\b(all|every)\b", re.IGNORECASE)
_QUESTION_DETAILS_RE = re.compile(r"\bdetails?\b", re.IGNORECASE)
_FULL_RECORD_REQUEST_RE = re.compile(
    r"\b(all columns|all fields|full records?|complete records?|entire records?|everything)\b",
    re.IGNORECASE,
)
_QUESTION_ENTITY_LIST_RE = re.compile(r"\b(which|what|list|show|find)\b", re.IGNORECASE)
_QUESTION_NON_NAME_ATTRIBUTE_RE = re.compile(
    r"\b(code|codes|id|ids|identifier|identifiers|city|cities|country|countries|detail|details|full|record|records)\b",
    re.IGNORECASE,
)
REQUEST_TERM_ALIASES = {
    "director": {"directed", "directed_by"},
    "directed": {"director", "directed_by"},
}
_NUMERIC_CAST_RE = re.compile(
    r"CAST\s*\(\s*(?P<expr>(?:(?P<alias>[A-Za-z_][A-Za-z0-9_]*)\.)?[\"`]?(?P<column>[A-Za-z_][A-Za-z0-9_]*)[\"`]?)\s+AS\s+(?:INTEGER|REAL|FLOAT|NUMERIC|DECIMAL)\s*\)",
    re.IGNORECASE,
)
_DATE_STRING_EQ_RE = re.compile(
    r"(?P<left>(?:(?P<alias>[A-Za-z_][A-Za-z0-9_]*)\.)?(?P<column>[A-Za-z_][A-Za-z0-9_]*date[A-Za-z0-9_]*))"
    r"\s*=\s*(?P<quote>['\"])(?P<value>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})(?P=quote)",
    re.IGNORECASE,
)
_COUNTRY_CODE_COLUMN_RE = re.compile(r"\b(country|nationality|nation|country_code|countrycode)\b", re.IGNORECASE)
_COUNTRY_CODE_LITERAL_RE = re.compile(r"^[A-Z]{2,3}$")
_BOOLEAN_LITERAL_EQ_RE = re.compile(
    r"(?P<left>(?:(?P<alias>[A-Za-z_][A-Za-z0-9_]*)\.)?[\"`]?(?P<column>[A-Za-z_][A-Za-z0-9_]*)[\"`]?)"
    r"\s*=\s*(?P<value>TRUE|FALSE|[01])\b",
    re.IGNORECASE,
)
COUNTRY_TERM_TO_CODES = {
    "american": ["USA", "US"],
    "argentinian": ["ARG", "AR"],
    "australian": ["AUS", "AU"],
    "austrian": ["AUT", "AT"],
    "belarus": ["BLR", "BY"],
    "belarusian": ["BLR", "BY"],
    "belgian": ["BEL", "BE"],
    "belgium": ["BEL", "BE"],
    "britain": ["GBR", "GB", "UK"],
    "british": ["GBR", "GB", "UK"],
    "canada": ["CAN", "CA"],
    "canadian": ["CAN", "CA"],
    "china": ["CHN", "CN"],
    "chinese": ["CHN", "CN"],
    "croatia": ["CRO", "HR"],
    "croatian": ["CRO", "HR"],
    "czech": ["CZE", "CZ"],
    "denmark": ["DEN", "DK"],
    "danish": ["DEN", "DK"],
    "france": ["FRA", "FR"],
    "french": ["FRA", "FR"],
    "german": ["GER", "DEU", "DE"],
    "germany": ["GER", "DEU", "DE"],
    "italian": ["ITA", "IT"],
    "italy": ["ITA", "IT"],
    "japan": ["JPN", "JP"],
    "japanese": ["JPN", "JP"],
    "poland": ["POL", "PL"],
    "polish": ["POL", "PL"],
    "russia": ["RUS", "RU"],
    "russian": ["RUS", "RU"],
    "serbia": ["SRB", "RS"],
    "serbian": ["SRB", "RS"],
    "spain": ["ESP", "ES"],
    "spanish": ["ESP", "ES"],
    "sweden": ["SWE", "SE"],
    "swedish": ["SWE", "SE"],
    "swiss": ["SUI", "CHE", "CH"],
    "ukraine": ["UKR", "UA"],
    "ukrainian": ["UKR", "UA"],
    "united states": ["USA", "US"],
}


def extract_schema(conn: sqlite3.Connection) -> str:
    """抽取当前 SQLite 连接的 schema 文本。

    参数：
    - conn：已打开的 SQLite 连接。
    """
    return extract_sql_schema(conn)


def execute_sql(conn: sqlite3.Connection, sql: str) -> dict[str, Any]:
    """执行 SQL 并返回统一执行结果。

    参数：
    - conn：当前 SQLite 数据库连接。
    - sql：待执行的 SQL 查询语句。
    """
    try:
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description] if cur.description else []
        cur.close()

        if not rows:
            return {
                "success": True,
                "result_text": "(query succeeded, but returned no rows)",
                "rows": [],
                "columns": columns,
                "row_count": 0,
                "error": None,
            }

        header = " | ".join(columns)
        separator = "-" * len(header) if header else ""
        lines = [line for line in [header, separator] if line]
        for row in rows[:MAX_RESULT_ROWS]:
            lines.append(" | ".join("NULL" if value is None else str(value) for value in row))
        if len(rows) > MAX_RESULT_ROWS:
            lines.append(f"... total {len(rows)} rows, showing first {MAX_RESULT_ROWS}")

        return {
            "success": True,
            "result_text": "\n".join(lines),
            "rows": [list(row) for row in rows[:MAX_RESULT_ROWS]],
            "columns": columns,
            "row_count": len(rows),
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


def _quote_identifier(identifier: str) -> str:
    """把 SQLite 标识符安全包装为双引号形式。

    参数：
    - identifier：表名或列名。
    """
    return '"' + identifier.replace('"', '""') + '"'


def _quote_literal(value: str) -> str:
    """把字符串值安全包装为 SQL 单引号 literal。

    参数：
    - value：需要写入 SQL 的字符串值。
    """
    return "'" + value.replace("'", "''") + "'"


def _normalize_identifier(identifier: str) -> str:
    """规范化 SQL 标识符，便于大小写无关比较。

    参数：
    - identifier：表名、列名或别名。
    """
    return identifier.strip().strip('"`[]').lower()


def _normalize_text_token(value: str) -> str:
    """把文本压缩为只含小写字母数字的 token。

    参数：
    - value：待规范化的文本。
    """
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _identifier_words(value: str) -> list[str]:
    """Split a SQL result column into lower-case semantic words."""
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value.strip().strip('"`[]'))
    return [word.lower() for word in re.findall(r"[A-Za-z0-9]+", spaced)]


def _singularize_token(token: str) -> str:
    """Very small English singularizer for entity-name projection checks."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def _requested_question_terms(question: str) -> set[str]:
    """Return terms from the explicit requested-output span of the question."""
    question_lower = question.lower()
    request_match = re.search(
        r"\b(?:list|show|return|retrieve|find|give|provide|what(?:\s+are|\s+is)?)\b(?P<span>.*)",
        question_lower,
        re.IGNORECASE | re.DOTALL,
    )
    request_span = request_match.group("span") if request_match else question_lower
    request_span = re.split(
        r"\b(?:ordered\s+by|order\s+by|sorted\s+by|sort\s+by|where|whose|who|that|which|with|having|group(?:ed)?\s+by|for\s+each)\b",
        request_span,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    return {
        _singularize_token(token)
        for token in re.findall(r"[A-Za-z][A-Za-z0-9]*", request_span)
    }


def _column_explicitly_requested(question: str, column: str) -> bool:
    """Check whether a non-name result column is explicitly requested by the user."""
    request_terms = _requested_question_terms(question)
    column_terms = {
        _singularize_token(word)
        for word in _identifier_words(column)
        if word not in {"name", "title"}
    }
    if not column_terms:
        return False
    expanded_request_terms = set(request_terms)
    for term in request_terms:
        expanded_request_terms.update(REQUEST_TERM_ALIASES.get(term, set()))
    expanded_column_terms = set(column_terms)
    for term in column_terms:
        expanded_column_terms.update(REQUEST_TERM_ALIASES.get(term, set()))
    return bool(column_terms <= expanded_request_terms or expanded_column_terms & request_terms)


def _table_names(conn: sqlite3.Connection) -> list[str]:
    """读取当前 SQLite 数据库中的用户表名。

    参数：
    - conn：SQLite 连接。
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type='table' AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        """
    )
    names = [row[0] for row in cur.fetchall()]
    cur.close()
    return names


def _column_info(conn: sqlite3.Connection, table: str) -> list[tuple[str, str]]:
    """读取指定表的列名和 SQLite 类型。

    参数：
    - conn：SQLite 连接。
    - table：表名。
    """
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({_quote_identifier(table)})")
    info = [(row[1], row[2] or "") for row in cur.fetchall()]
    cur.close()
    return info


def _is_text_like(type_name: str) -> bool:
    """判断 SQLite 类型是否可按文本列处理。

    参数：
    - type_name：SQLite schema 中的列类型字符串。
    """
    normalized = (type_name or "").lower()
    return any(token in normalized for token in ["char", "text", "clob", "varchar", "date", "time"])


def _build_alias_map(sql: str, conn: sqlite3.Connection) -> dict[str, str]:
    """从 SQL 中提取表别名到真实表名的映射。

    参数：
    - sql：待分析的 SQL。
    - conn：SQLite 连接，用于确认真实表名。
    """
    known_tables = {_normalize_identifier(table): table for table in _table_names(conn)}
    aliases: dict[str, str] = {}
    for match in _TABLE_ALIAS_RE.finditer(sql):
        table_raw = match.group("table")
        alias_raw = match.group("alias") or table_raw
        table = known_tables.get(_normalize_identifier(table_raw))
        if table:
            aliases[_normalize_identifier(alias_raw)] = table
            aliases[_normalize_identifier(table_raw)] = table
    return aliases


def _find_case_insensitive_value_matches(conn: sqlite3.Connection, value: str) -> list[dict[str, str]]:
    """在所有文本列中查找大小写无关匹配的真实取值。

    参数：
    - conn：SQLite 连接。
    - value：生成 SQL 中出现的字符串 literal。
    """
    matches: list[dict[str, str]] = []
    value_lower = value.lower()
    for table in _table_names(conn):
        for column, type_name in _column_info(conn, table):
            if not _is_text_like(type_name):
                continue
            try:
                cur = conn.cursor()
                cur.execute(
                    f"""
                    SELECT DISTINCT {_quote_identifier(column)}
                    FROM {_quote_identifier(table)}
                    WHERE LOWER({_quote_identifier(column)}) = LOWER(?)
                    LIMIT 5
                    """,
                    (value,),
                )
                rows = [str(row[0]) for row in cur.fetchall() if row[0] is not None]
                cur.close()
            except Exception:
                continue
            for row_value in rows:
                if row_value.lower() == value_lower:
                    matches.append({"table": table, "column": column, "value": row_value})
    return matches


def _column_names(conn: sqlite3.Connection, table: str) -> set[str]:
    """读取指定表的规范化列名集合。

    参数：
    - conn：SQLite 连接。
    - table：表名。
    """
    return {_normalize_identifier(column) for column, _ in _column_info(conn, table)}


def _resolve_predicate_table(
    *,
    conn: sqlite3.Connection,
    alias: str,
    column: str,
    alias_map: dict[str, str],
) -> str:
    """根据 SQL 谓词中的别名和列名推断其所属表。

    参数：
    - conn：SQLite 连接。
    - alias：谓词左侧的表别名，可能为空。
    - column：谓词左侧列名。
    - alias_map：`_build_alias_map(...)` 得到的别名映射。
    """
    if alias:
        return alias_map.get(_normalize_identifier(alias), "")

    candidate_tables = sorted(set(alias_map.values()))
    matching_tables = [
        table
        for table in candidate_tables
        if _normalize_identifier(column) in _column_names(conn, table)
    ]
    if len(matching_tables) == 1:
        return matching_tables[0]
    return ""


def _best_value_grounding_match(
    *,
    matches: list[dict[str, str]],
    original_table: str,
    original_column: str,
    original_value: str,
) -> tuple[int, dict[str, str] | None]:
    """为 value grounding 候选匹配打分并选择最佳项。

    参数：
    - matches：真实数据库中找到的候选取值列表。
    - original_table：原谓词推断出的表名。
    - original_column：原谓词列名。
    - original_value：原谓词字符串值。
    """
    ranked_matches = []
    for match in matches:
        score = 0
        if original_table and _normalize_identifier(match["table"]) == _normalize_identifier(original_table):
            score += 4
        if _normalize_identifier(match["column"]) == _normalize_identifier(original_column):
            score += 2
        if match["value"] == original_value:
            score += 1
        ranked_matches.append((score, match))

    ranked_matches.sort(key=lambda item: (-item[0], item[1]["table"], item[1]["column"], item[1]["value"]))
    return ranked_matches[0] if ranked_matches else (0, None)


def repair_sql_empty_result_with_value_lookup(conn: sqlite3.Connection, sql: str) -> tuple[str, dict[str, Any] | None]:
    """用数据库真实值修复空结果 SQL。

    参数：
    - conn：SQLite 连接。
    - sql：执行成功但返回空结果的 SQL。

    说明：
    - 这是确定性修复，不调用 LLM。
    - 只在字符串等值谓词能被当前数据库真实值明确 grounding 时改写。
    - 主要处理大小写不一致或同表过滤字段选择错误。
    """
    alias_map = _build_alias_map(sql, conn)
    replacements: list[dict[str, Any]] = []
    for predicate in _STRING_EQ_RE.finditer(sql):
        alias = predicate.group("alias") or ""
        original_column = predicate.group("column")
        original_value = predicate.group("value")
        original_table = _resolve_predicate_table(
            conn=conn,
            alias=alias,
            column=original_column,
            alias_map=alias_map,
        )
        if not original_value.strip():
            continue

        matches = _find_case_insensitive_value_matches(conn, original_value)
        if not matches:
            continue

        best_score, best = _best_value_grounding_match(
            matches=matches,
            original_table=original_table,
            original_column=original_column,
            original_value=original_value,
        )
        if not best or best_score < 4:
            continue

        new_left = predicate.group("left")
        if original_table and _normalize_identifier(best["table"]) == _normalize_identifier(original_table):
            if alias:
                new_left = f"{alias}.{best['column']}"
            else:
                new_left = best["column"]
        elif _normalize_identifier(best["column"]) != _normalize_identifier(original_column):
            continue

        replacement = f"{new_left} = {_quote_literal(best['value'])}"
        if replacement != predicate.group(0):
            replacements.append(
                {
                    "start": predicate.start(),
                    "end": predicate.end(),
                    "replacement": replacement,
                    "repair_type": "empty_result_value_lookup",
                    "original_predicate": predicate.group(0),
                    "repaired_predicate": replacement,
                    "matched_table": best["table"],
                    "matched_column": best["column"],
                    "matched_value": best["value"],
                }
            )

    if not replacements:
        return sql, None

    repaired_sql = sql
    for replacement in sorted(replacements, key=lambda item: item["start"], reverse=True):
        repaired_sql = (
            repaired_sql[: replacement["start"]]
            + replacement["replacement"]
            + repaired_sql[replacement["end"] :]
        )

    if repaired_sql != sql:
        details = [
            {
                "repair_type": item["repair_type"],
                "original_predicate": item["original_predicate"],
                "repaired_predicate": item["replacement"],
                "matched_table": item["matched_table"],
                "matched_column": item["matched_column"],
                "matched_value": item["matched_value"],
            }
            for item in replacements
        ]
        return repaired_sql, {
            "repair_type": "empty_result_value_lookup",
            "repairs": details,
            "repair_count": len(details),
        }

    return sql, None


def repair_sql_empty_result_with_date_format(conn: sqlite3.Connection, sql: str) -> tuple[str, dict[str, Any] | None]:
    """Rewrite date string literals to compact YYYYMMDD when that exact value exists.

    This is intentionally narrow: it only applies to equality predicates on
    date-like columns after a query returned no rows.
    """
    alias_map = _build_alias_map(sql, conn)
    replacements: list[dict[str, Any]] = []
    for predicate in _DATE_STRING_EQ_RE.finditer(sql):
        alias = predicate.group("alias") or ""
        column = predicate.group("column")
        table = _resolve_predicate_table(
            conn=conn,
            alias=alias,
            column=column,
            alias_map=alias_map,
        )
        if not table:
            continue
        compact_value = f"{predicate.group('value')}{predicate.group('month')}{predicate.group('day')}"
        try:
            cur = conn.cursor()
            cur.execute(
                f"""
                SELECT 1
                FROM {_quote_identifier(table)}
                WHERE CAST({_quote_identifier(column)} AS TEXT) = ?
                LIMIT 1
                """,
                (compact_value,),
            )
            has_exact_value = cur.fetchone() is not None
            cur.close()
        except Exception:
            continue
        if not has_exact_value:
            continue

        replacement = f"{predicate.group('left')} = {compact_value}"
        replacements.append(
            {
                "start": predicate.start(),
                "end": predicate.end(),
                "replacement": replacement,
                "original_predicate": predicate.group(0),
                "repaired_predicate": replacement,
                "matched_table": table,
                "matched_column": column,
                "matched_value": compact_value,
            }
        )

    if not replacements:
        return sql, None

    repaired_sql = sql
    for replacement in sorted(replacements, key=lambda item: item["start"], reverse=True):
        repaired_sql = (
            repaired_sql[: replacement["start"]]
            + replacement["replacement"]
            + repaired_sql[replacement["end"] :]
        )

    return repaired_sql, {
        "repair_type": "empty_result_date_format",
        "repairs": [
            {
                "original_predicate": item["original_predicate"],
                "repaired_predicate": item["repaired_predicate"],
                "matched_table": item["matched_table"],
                "matched_column": item["matched_column"],
                "matched_value": item["matched_value"],
            }
            for item in replacements
        ],
        "repair_count": len(replacements),
    }


def _question_country_code_candidates(question: str) -> list[tuple[str, list[str]]]:
    normalized = " ".join(re.findall(r"[a-z]+", question.lower()))
    candidates: list[tuple[str, list[str]]] = []
    for term, codes in sorted(COUNTRY_TERM_TO_CODES.items(), key=lambda item: (-len(item[0]), item[0])):
        if re.search(rf"(?<![a-z]){re.escape(term)}(?![a-z])", normalized):
            candidates.append((term, codes))
    return candidates


def _column_has_exact_text_value(conn: sqlite3.Connection, table: str, column: str, value: str) -> bool:
    try:
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT 1
            FROM {_quote_identifier(table)}
            WHERE CAST({_quote_identifier(column)} AS TEXT) = ?
            LIMIT 1
            """,
            (value,),
        )
        has_value = cur.fetchone() is not None
        cur.close()
        return has_value
    except Exception:
        return False


def repair_sql_country_code_from_question(
    conn: sqlite3.Connection,
    question: str,
    sql: str,
) -> tuple[str, dict[str, Any] | None]:
    """Repair country/nationality code literals when the question explicitly names the country.

    The repair is deliberately narrow: it only rewrites short uppercase code
    literals on country-like columns, and only to a code that exists in the
    same target column.
    """
    question_codes = _question_country_code_candidates(question)
    if not question_codes:
        return sql, None
    primary_codes = {codes[0] for _, codes in question_codes if codes}
    if len(primary_codes) != 1:
        return sql, None

    alias_map = _build_alias_map(sql, conn)
    replacements: list[dict[str, Any]] = []
    for predicate in _STRING_EQ_RE.finditer(sql):
        alias = predicate.group("alias") or ""
        column = predicate.group("column")
        original_value = predicate.group("value").strip()
        if not _COUNTRY_CODE_COLUMN_RE.search(column):
            continue
        if not _COUNTRY_CODE_LITERAL_RE.match(original_value):
            continue

        table = _resolve_predicate_table(
            conn=conn,
            alias=alias,
            column=column,
            alias_map=alias_map,
        )
        if not table:
            continue

        replacement_value = ""
        matched_term = ""
        for term, codes in question_codes:
            for code in codes:
                if code == original_value:
                    replacement_value = ""
                    break
                if _column_has_exact_text_value(conn, table, column, code):
                    replacement_value = code
                    matched_term = term
                    break
            if replacement_value:
                break

        if not replacement_value:
            continue

        replacement = f"{predicate.group('left')} = {_quote_literal(replacement_value)}"
        replacements.append(
            {
                "start": predicate.start(),
                "end": predicate.end(),
                "replacement": replacement,
                "original_predicate": predicate.group(0),
                "repaired_predicate": replacement,
                "matched_table": table,
                "matched_column": column,
                "matched_question_term": matched_term,
                "original_value": original_value,
                "matched_value": replacement_value,
            }
        )

    if not replacements:
        return sql, None

    repaired_sql = sql
    for replacement in sorted(replacements, key=lambda item: item["start"], reverse=True):
        repaired_sql = (
            repaired_sql[: replacement["start"]]
            + replacement["replacement"]
            + repaired_sql[replacement["end"] :]
        )

    return repaired_sql, {
        "repair_type": "country_code_from_question",
        "repairs": [
            {
                "original_predicate": item["original_predicate"],
                "repaired_predicate": item["repaired_predicate"],
                "matched_table": item["matched_table"],
                "matched_column": item["matched_column"],
                "matched_question_term": item["matched_question_term"],
                "original_value": item["original_value"],
                "matched_value": item["matched_value"],
            }
            for item in replacements
        ],
        "repair_count": len(replacements),
    }


def _is_boolean_like_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    type_name = ""
    for existing_column, existing_type in _column_info(conn, table):
        if _normalize_identifier(existing_column) == _normalize_identifier(column):
            type_name = str(existing_type or "").lower()
            break
    lowered_column = column.lower()
    return (
        "bool" in type_name
        or lowered_column.startswith("is_")
        or lowered_column.startswith("has_")
        or lowered_column in {"male", "female", "active", "official"}
    )


def _boolean_literal_values(conn: sqlite3.Connection, table: str, column: str) -> set[str]:
    try:
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT DISTINCT CAST({_quote_identifier(column)} AS TEXT)
            FROM {_quote_identifier(table)}
            WHERE {_quote_identifier(column)} IS NOT NULL
            LIMIT 10
            """
        )
        values = {str(row[0]).strip().upper() for row in cur.fetchall() if str(row[0]).strip()}
        cur.close()
        return values
    except Exception:
        return set()


def repair_sql_empty_result_boolean_literal(
    conn: sqlite3.Connection,
    sql: str,
) -> tuple[str, dict[str, Any] | None]:
    """Rewrite boolean predicates to T/F when a boolean-like text column uses those values."""
    alias_map = _build_alias_map(sql, conn)
    replacements: list[dict[str, Any]] = []
    for predicate in _BOOLEAN_LITERAL_EQ_RE.finditer(sql):
        alias = predicate.group("alias") or ""
        column = predicate.group("column")
        table = _resolve_predicate_table(conn=conn, alias=alias, column=column, alias_map=alias_map)
        if not table or not _is_boolean_like_column(conn, table, column):
            continue
        values = _boolean_literal_values(conn, table, column)
        if not {"T", "F"} <= values:
            continue
        raw_value = predicate.group("value").upper()
        replacement_value = "T" if raw_value in {"1", "TRUE"} else "F"
        replacement = f"{predicate.group('left')} = '{replacement_value}'"
        replacements.append(
            {
                "start": predicate.start(),
                "end": predicate.end(),
                "replacement": replacement,
                "original_predicate": predicate.group(0),
                "repaired_predicate": replacement,
                "matched_table": table,
                "matched_column": column,
                "matched_value": replacement_value,
            }
        )

    if not replacements:
        return sql, None

    repaired_sql = sql
    for replacement in sorted(replacements, key=lambda item: item["start"], reverse=True):
        repaired_sql = (
            repaired_sql[: replacement["start"]]
            + replacement["replacement"]
            + repaired_sql[replacement["end"] :]
        )

    return repaired_sql, {
        "repair_type": "empty_result_boolean_literal",
        "repairs": [
            {
                "original_predicate": item["original_predicate"],
                "repaired_predicate": item["repaired_predicate"],
                "matched_table": item["matched_table"],
                "matched_column": item["matched_column"],
                "matched_value": item["matched_value"],
            }
            for item in replacements
        ],
        "repair_count": len(replacements),
    }


def _has_duplicate_rows(rows: list[list[Any]]) -> bool:
    """判断结果集中是否存在完全重复的行。

    参数：
    - rows：执行结果行列表。
    """
    seen = set()
    for row in rows:
        key = tuple(row)
        if key in seen:
            return True
        seen.add(key)
    return False


def repair_sql_duplicate_rows_with_distinct(sql: str) -> tuple[str, dict[str, Any] | None]:
    """为简单非聚合查询补充 DISTINCT。

    参数：
    - sql：已执行成功但返回重复行的 SQL。

    说明：
    - 只处理非聚合、无 GROUP BY、未已有 DISTINCT 的简单 SELECT。
    """
    normalized = sql.strip()
    if not _SELECT_RE.search(normalized):
        return sql, None
    if _SELECT_DISTINCT_RE.search(normalized):
        return sql, None
    if re.search(r"\bGROUP\s+BY\b", normalized, re.IGNORECASE):
        return sql, None
    if _SQL_AGG_RE.search(normalized):
        return sql, None

    repaired_sql = _SELECT_RE.sub("SELECT DISTINCT ", sql, count=1)
    if repaired_sql == sql:
        return sql, None
    return repaired_sql, {"repair_type": "duplicate_result_distinct"}


def _question_preserves_row_level_results(question: str) -> bool:
    """判断问题是否明确要求保留行级结果。

    参数：
    - question：用户自然语言问题。
    """
    if _QUESTION_DISTINCT_RE.search(question):
        return False
    return bool(
        _QUESTION_ALL_ROWS_RE.search(question)
        or _QUESTION_DETAILS_RE.search(question)
        or (
            _QUESTION_ENTITY_LIST_RE.search(question)
            and not _QUESTION_NON_NAME_ATTRIBUTE_RE.search(question)
        )
    )


def repair_sql_unrequested_distinct(question: str, sql: str) -> tuple[str, dict[str, Any] | None]:
    """移除用户未请求的 SELECT DISTINCT。

    参数：
    - question：用户自然语言问题，用于判断是否有去重意图或行级结果意图。
    - sql：生成器输出的 SQL。
    """
    normalized = sql.strip()
    if not _SELECT_DISTINCT_RE.search(normalized):
        return sql, None
    if _QUESTION_DISTINCT_RE.search(question):
        return sql, None
    if not _question_preserves_row_level_results(question):
        return sql, None

    repaired_sql = _SELECT_DISTINCT_RE.sub("SELECT", sql, count=1)
    if repaired_sql == sql:
        return sql, None
    return repaired_sql, {"repair_type": "unrequested_distinct_removed"}


def _select_projection_target_columns(question: str, columns: list[str]) -> list[str]:
    """根据问题显式语义选择可安全裁剪的目标列。

    参数：
    - question：用户自然语言问题。
    - columns：当前查询实际返回的列名列表。

    说明：
    - 当前只对 details 类显式字段做保守裁剪，不根据当前 benchmark 题面猜测 name 列。
    """
    if len(columns) <= 1 or _FULL_RECORD_REQUEST_RE.search(question):
        return []

    question_lower = question.lower()
    if "details" in question_lower:
        detail_columns = [
            column
            for column in columns
            if "detail" in _normalize_text_token(column)
        ]
        if len(detail_columns) == 1:
            return detail_columns

    if _QUESTION_ENTITY_LIST_RE.search(question) and not _QUESTION_NON_NAME_ATTRIBUTE_RE.search(question):
        explicitly_requested_non_name_columns = [
            column
            for column in columns
            if not ({"name", "title"} & set(_identifier_words(column)))
            and _column_explicitly_requested(question, column)
        ]
        if explicitly_requested_non_name_columns:
            return []

        question_terms = {
            _singularize_token(token)
            for token in re.findall(r"[A-Za-z][A-Za-z0-9]*", question_lower)
        }
        name_columns: list[str] = []
        for column in columns:
            words = _identifier_words(column)
            if not ({"name", "title"} & set(words)):
                continue
            entity_words = {_singularize_token(word) for word in words if word not in {"name", "title"}}
            if entity_words and not (entity_words & question_terms):
                continue
            name_columns.append(column)
        if len(name_columns) == 1:
            return name_columns

    return []


def repair_sql_overwide_projection(
    question: str,
    sql: str,
    columns: list[str],
) -> tuple[str, dict[str, Any] | None]:
    """把返回列过宽的 SQL 包成子查询并在外层裁剪列。

    参数：
    - question：用户自然语言问题。
    - sql：原始生成 SQL。
    - columns：原 SQL 执行后返回的列名。
    """
    target_columns = _select_projection_target_columns(question, columns)
    if not target_columns or len(target_columns) >= len(columns):
        return sql, None

    inner_sql = sql.strip().rstrip(";")
    projection = ", ".join(_quote_identifier(column) for column in target_columns)
    repaired_sql = f"SELECT {projection} FROM ({inner_sql}) AS projected_result"
    return repaired_sql, {
        "repair_type": "overwide_projection_trimmed",
        "target_columns": target_columns,
        "original_columns": columns,
    }


def _question_requires_entity_attribution(question: str) -> bool:
    normalized = question.lower()
    return bool(
        re.search(r"\beach\s+(?:player|student|singer|character|planet|entity|record)\b", normalized)
        or re.search(r"\beach\s+of\s+those\s+(?:players|students|singers|characters|planets|entities|records)\b", normalized)
        or re.search(r"\btheir\s+(?:ids?|names?|countries|country|hand|ranking|points|ages?|song|songs|attributes|metrics)\b", normalized)
    )


def _select_list_expressions(sql: str) -> list[str]:
    match = _SELECT_FROM_RE.search(sql.strip().rstrip(";"))
    if not match:
        return []
    select_text = match.group("select")
    expressions: list[str] = []
    current: list[str] = []
    depth = 0
    quote = ""
    for char in select_text:
        if quote:
            current.append(char)
            if char == quote:
                quote = ""
            continue
        if char in {"'", '"', "`"}:
            quote = char
            current.append(char)
            continue
        if char == "(":
            depth += 1
        elif char == ")" and depth > 0:
            depth -= 1
        if char == "," and depth == 0:
            expressions.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if current:
        expressions.append("".join(current).strip())
    return expressions


def _selected_column_names(sql: str) -> set[str]:
    names: set[str] = set()
    for expression in _select_list_expressions(sql):
        alias_match = re.search(r"\bAS\s+[\"`]?(?P<alias>[A-Za-z_][A-Za-z0-9_]*)[\"`]?\s*$", expression, re.IGNORECASE)
        if alias_match:
            names.add(_normalize_identifier(alias_match.group("alias")))
        column_match = re.search(r"(?:^|\.|[\"`])(?P<column>[A-Za-z_][A-Za-z0-9_]*)(?:[\"`]?)\s*$", expression)
        if column_match:
            names.add(_normalize_identifier(column_match.group("column")))
    return names


def _is_numeric_text_column(conn: sqlite3.Connection, alias: str, column: str, alias_map: dict[str, str]) -> bool:
    table = _resolve_predicate_table(conn=conn, alias=alias, column=column, alias_map=alias_map)
    if not table:
        return False
    for candidate_column, type_name in _column_info(conn, table):
        if _normalize_identifier(candidate_column) != _normalize_identifier(column):
            continue
        return _is_text_like(type_name)
    return False


def _sqlite_integer_text_guard(expr: str) -> str:
    """Return a SQLite predicate that keeps only non-empty integer text values."""
    return f"({expr} GLOB '[0-9]*' AND {expr} NOT GLOB '*[^0-9]*' AND {expr} != '')"


def _sqlite_integer_text_guard_present(sql: str, expr: str) -> bool:
    lowered = re.sub(r"\s+", " ", sql.lower())
    expr_lower = expr.lower()
    return (
        f"{expr_lower} glob '[0-9]*'" in lowered
        and f"{expr_lower} not glob '*[^0-9]*'" in lowered
        and f"{expr_lower} != ''" in lowered
    )


def _add_numeric_guard_to_aggregate_subqueries(sql: str, *, expr: str, table: str, guard: str) -> tuple[str, int]:
    """Add the same numeric-text guard inside aggregate subqueries using the cast."""
    table_pattern = rf"[\"`]?(?:{re.escape(table)})[\"`]?"
    cast_pattern = rf"CAST\s*\(\s*{re.escape(expr)}\s+AS\s+(?:INTEGER|REAL|FLOAT|NUMERIC|DECIMAL)\s*\)"
    pattern = re.compile(
        rf"(?P<prefix>\(\s*SELECT\s+(?:AVG|SUM|MIN|MAX)\s*\(\s*{cast_pattern}\s*\)\s+FROM\s+{table_pattern}\s*)"
        rf"(?P<where>WHERE\b)?",
        re.IGNORECASE,
    )
    replacements: list[dict[str, Any]] = []
    for match in pattern.finditer(sql):
        close_at = sql.find(")", match.end())
        scope = sql[match.start() : close_at if close_at != -1 else len(sql)]
        if guard.lower() in scope.lower():
            continue
        if match.group("where"):
            replacement = f"{match.group('prefix')}WHERE {guard} AND "
        else:
            replacement = f"{match.group('prefix')}WHERE {guard} "
        replacements.append({"start": match.start(), "end": match.end(), "replacement": replacement})

    if not replacements:
        return sql, 0

    repaired = sql
    for replacement in sorted(replacements, key=lambda item: item["start"], reverse=True):
        repaired = repaired[: replacement["start"]] + replacement["replacement"] + repaired[replacement["end"] :]
    return repaired, len(replacements)


def repair_sql_numeric_text_cast_filter(
    conn: sqlite3.Connection,
    question: str,
    sql: str,
) -> tuple[str, dict[str, Any] | None]:
    """Constrain numeric casts on text columns when the question asks for numeric values.

    SQLite casts non-numeric strings such as "unknown" to 0, which silently
    changes AVG/COUNT semantics. If a question explicitly says "numeric" and
    the generated SQL casts a text column, add SQLite integer-text guards.
    """
    if "numeric" not in question.lower():
        return sql, None
    matches = list(_NUMERIC_CAST_RE.finditer(sql))
    if not matches:
        return sql, None
    alias_map = _build_alias_map(sql, conn)
    guard_specs: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for match in matches:
        expr = match.group("expr")
        alias = match.group("alias") or ""
        column = match.group("column")
        table = _resolve_predicate_table(conn=conn, alias=alias, column=column, alias_map=alias_map)
        if not table or not _is_numeric_text_column(conn, alias, column, alias_map):
            continue
        guard = _sqlite_integer_text_guard(expr)
        guard_key = (guard.lower(), table.lower())
        if guard_key in seen:
            continue
        seen.add(guard_key)
        guard_specs.append({"guard": guard, "expr": expr, "table": table})
    if not guard_specs:
        return sql, None

    repaired_sql = sql
    changed = False
    for spec in guard_specs:
        guard = spec["guard"]
        if not _sqlite_integer_text_guard_present(repaired_sql, spec["expr"]):
            where_match = re.search(r"\bWHERE\b", repaired_sql, re.IGNORECASE)
            if where_match:
                insert_at = where_match.end()
                repaired_sql = repaired_sql[:insert_at] + f" {guard} AND" + repaired_sql[insert_at:]
            else:
                trailing_semicolon = ";" if repaired_sql.rstrip().endswith(";") else ""
                body = repaired_sql.rstrip().rstrip(";")
                repaired_sql = f"{body} WHERE {guard}{trailing_semicolon}"
            changed = True
        repaired_sql, subquery_repairs = _add_numeric_guard_to_aggregate_subqueries(
            repaired_sql,
            expr=spec["expr"],
            table=spec["table"],
            guard=guard,
        )
        changed = changed or bool(subquery_repairs)

    if not changed:
        return sql, None

    return repaired_sql, {
        "repair_type": "numeric_text_cast_filter",
        "guards": [spec["guard"] for spec in guard_specs],
    }


def retrieve_sql_candidates(
    question: str,
    *,
    db_path: str | None,
    use_retrieval: bool,
) -> dict[str, Any]:
    """检索或指定 SQL 数据库候选。

    参数：
    - question：用户自然语言问题。
    - db_path：可选指定数据库路径。
    - use_retrieval：是否启用向量/词面检索。
    """
    result = retrieve_sql_targets(
        question,
        db_path=db_path,
        use_retrieval=use_retrieval,
    )
    return {
        "db_candidates": result["raw_candidates"],
        "retrieval": {
            "method": result["method"],
            "selected_db": result["selected_db"],
            "selected_path": result["selected_path"],
            "score": result["score"],
            "top_candidates": [
                {
                    "db_id": candidate["resource_id"],
                    "db_path": candidate["resource_path"],
                    "score": candidate["score"],
                    "tables": candidate["metadata"].get("tables", []),
                    **({"error": candidate["raw"]["error"]} if "error" in candidate["raw"] else {}),
                }
                for candidate in result["top_candidates"]
            ],
        },
        "retrieval_ms": result["retrieval_ms"],
    }


def prepare_sql_schema_context(
    *,
    conn: sqlite3.Connection,
    question: str,
    current_db_path: str,
    selector_type: str,
    example_num: int,
    include_rule: bool,
    force_bow_selector: bool,
    max_seq_len: int,
    max_ans_len: int,
    scope_factor: int,
    tokenizer_model: str,
    trace: list[dict[str, str]],
    use_value_hints: bool = True,
) -> dict[str, Any]:
    """准备 SQL 生成所需的完整 schema、示例和 prompt 上下文。

    参数：
    - conn：当前候选 SQLite 数据库连接。
    - question：用户自然语言问题。
    - current_db_path：当前候选数据库路径。
    - selector_type：DAIL 示例选择策略。
    - example_num：示例数量。
    - include_rule：是否包含 SQL 生成规则。
    - force_bow_selector：是否强制使用 BOW 选择器。
    - max_seq_len：prompt 长度预算。
    - max_ans_len：答案长度预算。
    - scope_factor：schema/example 截断范围因子。
    - tokenizer_model：token 估算模型名。
    - trace：LLM 调用 trace。

    重要约定：
    - SQL 链路不做 table/column 级 schema 裁剪。
    - 这里调用的 shared schema context 会保留完整 CREATE TABLE schema，只做 DAIL 示例选择、value hints 和调试预览。
    """
    return shared_prepare_sql_schema_context(
        conn=conn,
        question=question,
        current_db_path=current_db_path,
        selector_type=selector_type,
        example_num=example_num,
        include_rule=include_rule,
        force_bow_selector=force_bow_selector,
        max_seq_len=max_seq_len,
        max_ans_len=max_ans_len,
        scope_factor=scope_factor,
        tokenizer_model=tokenizer_model,
        trace=trace,
        use_value_hints=use_value_hints,
    )


def generate_sql_query(prompt: str, trace: list[dict[str, str]]) -> dict[str, Any]:
    """调用共享生成 Agent 生成 SQL。

    参数：
    - prompt：完整 SQL DAIL prompt。
    - trace：LLM 调用 trace。
    """
    return shared_generate_sql_query(prompt, trace)


def execute_sql_with_repair(
    *,
    conn: sqlite3.Connection,
    question: str,
    schema_sql: str,
    examples: list[ExamplePoolItem],
    sql: str,
    trace: list[dict[str, str]],
    include_rule: bool,
    use_fix: bool,
    use_result_shape_repair: bool = False,
) -> dict[str, Any]:
    """执行 SQL，并串联 LLM 修复与确定性修复。

    参数：
    - conn：SQLite 连接。
    - question：用户自然语言问题。
    - schema_sql：当前数据库 schema 文本。
    - examples：DAIL few-shot 示例。
    - sql：生成器输出的 SQL。
    - trace：LLM 调用 trace。
    - include_rule：修复 prompt 是否包含生成规则。
    - use_fix：是否启用修复流程。
    - use_result_shape_repair：是否启用执行成功后的结果形态改写。默认关闭，只保留执行失败和空结果/value grounding 类修复。
    """
    attempts = MAX_RETRIES if use_fix else 0

    def _fix_sql(current_sql: str, error_message: str, attempt: int) -> tuple[str, str]:
        """调用 SQL 修复 Agent。

        参数：
        - current_sql：当前执行失败的 SQL。
        - error_message：SQLite 错误信息。
        - attempt：第几次修复尝试。
        """
        return fix_sql_query(
            question=question,
            schema_sql=schema_sql,
            examples=examples,
            bad_sql=current_sql,
            error_msg=error_message,
            attempt=attempt,
            include_rule=include_rule,
            trace=trace,
        )

    execution_result = run_execution_with_repair(
        initial_query=sql,
        max_retries=attempts,
        execute_fn=lambda current_sql: execute_sql(conn, current_sql),
        fix_fn=_fix_sql,
        should_try_next_candidate_fn=lambda error_message, attempt: (
            attempt == 0 and "no such table" in error_message.lower()
        ),
        query_key="sql",
        bad_query_key="bad_sql",
        fixed_query_key="fixed_sql",
        terminal_failure_message=lambda error_message, retry_count: (
            f"Execution failed after {retry_count} retries: {error_message}"
        ),
    )

    if (
        use_fix
        and execution_result["success"]
        and execution_result.get("row_count", 0) == 0
        and _STRING_EQ_RE.search(execution_result.get("sql", ""))
    ):
        repaired_sql, repair_detail = repair_sql_empty_result_with_date_format(
            conn,
            execution_result["sql"],
        )
        if repair_detail and repaired_sql != execution_result["sql"]:
            repaired_execution = execute_sql(conn, repaired_sql)
            if repaired_execution["success"] and repaired_execution["row_count"] > 0:
                retry_history = list(execution_result.get("retry_history", []))
                retry_history.append(
                    {
                        "attempt": execution_result.get("retries", 0) + 1,
                        "bad_sql": execution_result["sql"],
                        "error": "empty_result",
                        "fixed_sql": repaired_sql,
                        "cot_raw": "deterministic_empty_result_date_format",
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
                        "sql": repaired_sql,
                    }
                )

    if use_fix and execution_result["success"]:
        repaired_sql, repair_detail = repair_sql_country_code_from_question(
            conn,
            question,
            execution_result["sql"],
        )
        if repair_detail and repaired_sql != execution_result["sql"]:
            repaired_execution = execute_sql(conn, repaired_sql)
            if repaired_execution["success"] and repaired_execution["row_count"] > 0:
                retry_history = list(execution_result.get("retry_history", []))
                retry_history.append(
                    {
                        "attempt": execution_result.get("retries", 0) + 1,
                        "bad_sql": execution_result["sql"],
                        "error": "country_code_from_question",
                        "fixed_sql": repaired_sql,
                        "cot_raw": "deterministic_country_code_from_question",
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
                        "sql": repaired_sql,
                    }
                )

    if (
        use_fix
        and execution_result["success"]
        and execution_result.get("row_count", 0) == 0
        and _BOOLEAN_LITERAL_EQ_RE.search(execution_result.get("sql", ""))
    ):
        repaired_sql, repair_detail = repair_sql_empty_result_boolean_literal(
            conn,
            execution_result["sql"],
        )
        if repair_detail and repaired_sql != execution_result["sql"]:
            repaired_execution = execute_sql(conn, repaired_sql)
            if repaired_execution["success"] and repaired_execution["row_count"] > 0:
                retry_history = list(execution_result.get("retry_history", []))
                retry_history.append(
                    {
                        "attempt": execution_result.get("retries", 0) + 1,
                        "bad_sql": execution_result["sql"],
                        "error": "empty_result_boolean_literal",
                        "fixed_sql": repaired_sql,
                        "cot_raw": "deterministic_empty_result_boolean_literal",
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
                        "sql": repaired_sql,
                    }
                )

    if (
        use_fix
        and execution_result["success"]
        and execution_result.get("row_count", 0) == 0
        and _STRING_EQ_RE.search(execution_result.get("sql", ""))
    ):
        repaired_sql, repair_detail = repair_sql_empty_result_with_value_lookup(
            conn,
            execution_result["sql"],
        )
        if repair_detail and repaired_sql != execution_result["sql"]:
            repaired_execution = execute_sql(conn, repaired_sql)
            if repaired_execution["success"] and repaired_execution["row_count"] > 0:
                retry_history = list(execution_result.get("retry_history", []))
                retry_history.append(
                    {
                        "attempt": execution_result.get("retries", 0) + 1,
                        "bad_sql": execution_result["sql"],
                        "error": "empty_result",
                        "fixed_sql": repaired_sql,
                        "cot_raw": "deterministic_empty_result_value_lookup",
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
                        "sql": repaired_sql,
                    }
                )

    if (
        use_fix
        and use_result_shape_repair
        and execution_result["success"]
        and execution_result.get("row_count", 0) > 0
    ):
        repaired_sql, repair_detail = repair_sql_numeric_text_cast_filter(
            conn,
            question,
            execution_result["sql"],
        )
        if repair_detail and repaired_sql != execution_result["sql"]:
            repaired_execution = execute_sql(conn, repaired_sql)
            if repaired_execution["success"] and repaired_execution["row_count"] > 0:
                retry_history = list(execution_result.get("retry_history", []))
                retry_history.append(
                    {
                        "attempt": execution_result.get("retries", 0) + 1,
                        "bad_sql": execution_result["sql"],
                        "error": "numeric_text_cast_filter",
                        "fixed_sql": repaired_sql,
                        "cot_raw": "deterministic_numeric_text_cast_filter",
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
                        "sql": repaired_sql,
                    }
                )

    if (
        use_fix
        and use_result_shape_repair
        and execution_result["success"]
        and execution_result.get("row_count", 0) > 0
    ):
        repaired_sql, repair_detail = repair_sql_unrequested_distinct(question, execution_result["sql"])
        if repair_detail and repaired_sql != execution_result["sql"]:
            repaired_execution = execute_sql(conn, repaired_sql)
            if (
                repaired_execution["success"]
                and repaired_execution["row_count"] >= execution_result["row_count"]
                and repaired_execution["columns"] == execution_result["result_columns"]
            ):
                retry_history = list(execution_result.get("retry_history", []))
                retry_history.append(
                    {
                        "attempt": execution_result.get("retries", 0) + 1,
                        "bad_sql": execution_result["sql"],
                        "error": "unrequested_distinct",
                        "fixed_sql": repaired_sql,
                        "cot_raw": "deterministic_unrequested_distinct_removed",
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
                        "sql": repaired_sql,
                    }
                )

    if (
        use_fix
        and use_result_shape_repair
        and execution_result["success"]
        and execution_result.get("row_count", 0) > 0
        and _has_duplicate_rows(execution_result.get("result_rows", []))
        and not _question_preserves_row_level_results(question)
    ):
        repaired_sql, repair_detail = repair_sql_duplicate_rows_with_distinct(execution_result["sql"])
        if repair_detail and repaired_sql != execution_result["sql"]:
            repaired_execution = execute_sql(conn, repaired_sql)
            if (
                repaired_execution["success"]
                and repaired_execution["row_count"] > 0
                and repaired_execution["row_count"] < execution_result["row_count"]
            ):
                retry_history = list(execution_result.get("retry_history", []))
                retry_history.append(
                    {
                        "attempt": execution_result.get("retries", 0) + 1,
                        "bad_sql": execution_result["sql"],
                        "error": "duplicate_result_rows",
                        "fixed_sql": repaired_sql,
                        "cot_raw": "deterministic_duplicate_result_distinct",
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
                        "sql": repaired_sql,
                    }
                )

    if (
        use_fix
        and use_result_shape_repair
        and execution_result["success"]
        and execution_result.get("row_count", 0) > 0
        and len(execution_result.get("result_columns", [])) > 1
    ):
        repaired_sql, repair_detail = repair_sql_overwide_projection(
            question,
            execution_result["sql"],
            execution_result.get("result_columns", []),
        )
        if repair_detail and repaired_sql != execution_result["sql"]:
            repaired_execution = execute_sql(conn, repaired_sql)
            if (
                repaired_execution["success"]
                and repaired_execution["row_count"] == execution_result["row_count"]
                and len(repaired_execution["columns"]) < len(execution_result["result_columns"])
            ):
                retry_history = list(execution_result.get("retry_history", []))
                retry_history.append(
                    {
                        "attempt": execution_result.get("retries", 0) + 1,
                        "bad_sql": execution_result["sql"],
                        "error": "overwide_projection",
                        "fixed_sql": repaired_sql,
                        "cot_raw": "deterministic_overwide_projection_trimmed",
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
                        "sql": repaired_sql,
                    }
                )

    return execution_result
