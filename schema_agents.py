import json
import re
import sqlite3
import time
from typing import Any

from openai import OpenAI

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
)
from llm_utils import call_chat_completion
from nested_logic_hints import build_nested_logic_guidance
from paired_benchmark_resources import resolve_actual_cypher_database
from cypher_schema_utils import (
    extract_graph_schema as render_graph_schema,
    extract_graph_schema_data,
    filter_graph_schema,
)
from sql_example_pool_dail import ExamplePoolItem
from sql_pipeline_dail import _build_target_info, _find_db_id_by_path, _rank_examples
from sql_prompt_dail import build_dail_prompt_with_budget, extract_schema_create_table

llm = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

# 中文函数说明补充索引：
# - call_llm(prompt, trace, label)：Schema 阶段 LLM 调用入口。
# - extract_sql_schema(conn)：抽取 SQLite CREATE TABLE schema。
# - prepare_sql_schema_context(...)：为 SQL DAIL 准备完整 schema、linked schema、masked question、示例和 nested guidance。
# - extract_graph_schema(driver, db_name, format)：抽取 Neo4j 图 schema。
# - prepare_cypher_schema_context(...)：过滤图 schema，生成 Cypher prompt 所需上下文。


def _now_ms() -> float:
    """返回当前高精度时间，单位毫秒。"""
    return time.perf_counter() * 1000


def call_llm(prompt: str, trace: list[dict[str, str]] | None = None, label: str = "") -> str:
    """调用 schema 阶段使用的 LLM。

    参数：
    - prompt：schema linking 或 schema filtering 提示词。
    - trace：可选 LLM trace。
    - label：本次调用标签。
    """
    return call_chat_completion(llm, prompt, trace=trace, label=label or "llm_call")


def extract_sql_schema(conn: sqlite3.Connection) -> str:
    """抽取 SQLite schema 的 CREATE TABLE 文本。

    参数：
    - conn：SQLite 连接。
    """
    return extract_schema_create_table(conn)


def _quote_sqlite_identifier(identifier: str) -> str:
    """安全引用 SQLite 表名或列名。

    参数：
    - identifier：表名或列名。
    """
    return '"' + identifier.replace('"', '""') + '"'


def _is_text_like_sql_type(type_name: str) -> bool:
    """判断 SQLite 列类型是否适合作为文本值提示来源。

    参数：
    - type_name：SQLite 列类型。
    """
    normalized = (type_name or "").lower()
    return any(token in normalized for token in ["char", "text", "clob", "varchar", "date", "time"])


def _format_hint_value(value: Any, max_len: int = 80) -> str:
    """格式化 value hint 中展示的数据库真实值。

    参数：
    - value：数据库中的原始取值。
    - max_len：最长展示字符数。
    """
    text = str(value).replace("\n", " ").strip()
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def extract_sql_value_hints(
    conn: sqlite3.Connection,
    question: str,
    *,
    max_lines: int = 40,
    max_values_per_line: int = 12,
    scan_limit_per_column: int = 500,
) -> str:
    """为 SQL 生成构造紧凑的真实值提示。

    参数：
    - conn：SQLite 连接。
    - question：用户自然语言问题。
    - max_lines：最多输出多少行提示。
    - max_values_per_line：每行最多展示多少个值。
    - scan_limit_per_column：每个文本列最多扫描多少个 distinct 值。

    说明：
    - 该函数从当前数据库运行时读取值，不写死具体实体或数据库。
    """
    question_lower = question.lower()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type='table' AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        """
    )
    table_names = [row[0] for row in cur.fetchall()]

    matched_lines: list[str] = []
    enum_lines: list[str] = []
    seen_lines: set[str] = set()

    for table in table_names:
        cur.execute(f"PRAGMA table_info({_quote_sqlite_identifier(table)})")
        columns = cur.fetchall()
        for column in columns:
            column_name = column[1]
            type_name = column[2] or ""
            if not _is_text_like_sql_type(type_name):
                continue

            table_sql = _quote_sqlite_identifier(table)
            column_sql = _quote_sqlite_identifier(column_name)
            try:
                cur.execute(
                    f"""
                    SELECT DISTINCT {column_sql}
                    FROM {table_sql}
                    WHERE {column_sql} IS NOT NULL
                    LIMIT ?
                    """,
                    (scan_limit_per_column,),
                )
                values = [_format_hint_value(row[0]) for row in cur.fetchall() if str(row[0]).strip()]
            except Exception:
                continue

            matched_values = []
            for value in values:
                value_lower = value.lower()
                if len(value_lower) >= 2 and (value_lower in question_lower or question_lower in value_lower):
                    matched_values.append(value)
                if len(matched_values) >= max_values_per_line:
                    break

            if matched_values:
                line = f"- {table}.{column_name} matching values: " + ", ".join(matched_values)
                if line not in seen_lines:
                    matched_lines.append(line)
                    seen_lines.add(line)

            if len(values) <= max_values_per_line and values:
                line = f"- {table}.{column_name} low-cardinality values: " + ", ".join(values)
                if line not in seen_lines:
                    enum_lines.append(line)
                    seen_lines.add(line)

            if len(matched_lines) + len(enum_lines) >= max_lines:
                break
        if len(matched_lines) + len(enum_lines) >= max_lines:
            break

    cur.close()
    lines = (matched_lines + enum_lines)[:max_lines]
    return "\n".join(lines)


def extract_graph_schema(driver, db_name: str = "wwc2019", format: str = "cypher_create") -> str:
    """抽取指定 Neo4j 图数据库的 schema 文本。

    参数：
    - driver：Neo4j driver。
    - db_name：Neo4j 数据库名。
    - format：schema 输出格式。
    """
    return render_graph_schema(driver, resolve_actual_cypher_database(db_name), format=format)


def _extract_sql(raw: str) -> str:
    """从 LLM 输出中提取 SQL。

    参数：
    - raw：模型原始输出。
    """
    tagged = re.search(r"<sql>(.*?)</sql>", raw, re.DOTALL | re.IGNORECASE)
    if tagged:
        return tagged.group(1).strip()
    code_block = re.search(r"```sql\s*(.*?)```", raw, re.DOTALL | re.IGNORECASE)
    if code_block:
        return code_block.group(1).strip()
    fallback = re.search(r"(SELECT[\s\S]+?;)", raw, re.IGNORECASE)
    if fallback:
        return fallback.group(1).strip()
    return raw.strip()


def _serialize_examples(examples: list[ExamplePoolItem]) -> list[dict[str, str]]:
    """把 DAIL 示例对象序列化为可放入 trace 的字典。

    参数：
    - examples：选中的 SQL few-shot 示例。
    """
    return [
        {
            "db_id": item.db_id,
            "question": item.question,
            "query": item.query,
        }
        for item in examples
    ]


def _build_selection_preview(
    *,
    selector_type: str,
    question_masked: str,
    examples: list[ExamplePoolItem],
    prompt_tokens: int,
) -> str:
    """构造 SQL schema linking / example selection 的可读预览。

    参数：
    - selector_type：示例选择策略。
    - question_masked：DAIL mask 后的问题。
    - examples：最终选中的示例。
    - prompt_tokens：prompt token 估算值。
    """
    lines = [
        f"selector_type: {selector_type}",
        f"question_masked: {question_masked}",
        f"selected_examples: {len(examples)}",
        f"prompt_tokens: {prompt_tokens}",
    ]
    lines.append("")
    lines.append("[Selected Examples]")
    for idx, item in enumerate(examples, 1):
        lines.append(f"{idx}. ({item.db_id}) {item.question}")
        lines.append(f"   SQL: {item.query}")
    return "\n".join(lines)


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
    """准备 SQL 生成所需的完整 schema、value hints、DAIL 示例和 prompt。

    参数：
    - conn：当前候选 SQLite 数据库连接。
    - question：用户自然语言问题。
    - current_db_path：当前数据库路径，用于定位 db_id 和示例池。
    - selector_type：DAIL 示例选择策略。
    - example_num：目标示例数量。
    - include_rule：是否加入 SQL 生成规则。
    - force_bow_selector：是否强制使用 BOW 示例选择。
    - max_seq_len：prompt 最大长度预算。
    - max_ans_len：答案长度预算。
    - scope_factor：候选示例召回倍数。
    - tokenizer_model：token 估算模型名。
    - trace：LLM 调用 trace。

    重要约定：
    - SQL 链路当前不做 table/column 级 schema 裁剪。
    - 既往实验表明 SQL schema 裁剪会降低最终 EX，因此这里保留完整 CREATE TABLE schema。
    - `working_schema` / `schema_linked` 仅作为 DAIL 示例选择、question masking 和调试预览，不代表被裁剪后的 schema。
    """
    schema_started = _now_ms()
    schema_sql = extract_sql_schema(conn)
    value_hints = extract_sql_value_hints(conn, question) if use_value_hints else ""
    nested_guidance = build_nested_logic_guidance(question, "sql")
    schema_extraction_ms = round(_now_ms() - schema_started, 2)

    db_id = _find_db_id_by_path(current_db_path)
    target_info = _build_target_info(question, db_id)

    selection_started = _now_ms()
    recall_size = max(example_num, example_num * scope_factor)
    ranked_examples = _rank_examples(
        selector_type=selector_type,
        question=question,
        question_masked=target_info["question_masked"],
        db_id=db_id,
        force_bow=force_bow_selector,
    )
    prompt, examples, prompt_tokens = build_dail_prompt_with_budget(
        question=question,
        schema_sql=schema_sql,
        value_hints=value_hints,
        candidate_examples=ranked_examples[:recall_size],
        include_rule=include_rule,
        target_example_num=example_num,
        max_seq_len=max_seq_len,
        max_ans_len=max_ans_len,
        tokenizer_model=tokenizer_model,
    )

    return {
        "schema_sql": schema_sql,
        "schema_full": schema_sql,
        "working_schema": _build_selection_preview(
            selector_type=selector_type,
            question_masked=target_info["question_masked"],
            examples=examples,
            prompt_tokens=prompt_tokens,
        ),
        "question_masked": target_info["question_masked"],
        "examples": examples,
        "examples_serialized": _serialize_examples(examples),
        "prompt_tokens": prompt_tokens,
        "prompt": prompt,
        "schema_linked": _build_selection_preview(
            selector_type=selector_type,
            question_masked=target_info["question_masked"],
            examples=examples,
            prompt_tokens=prompt_tokens,
        ),
        "schema_extraction_ms": schema_extraction_ms,
        "schema_linking_ms": round(_now_ms() - selection_started, 2),
        "value_hints": value_hints,
        "task_analysis": nested_guidance["analysis"],
        "nested_logic_guidance": nested_guidance["guidance"],
    }


def schema_filtering(
    question: str,
    full_schema: str,
    nodes_data: list[dict],
    relationships_data: list[dict],
    schema_format: str = "cypher_create",
    trace: list[dict[str, str]] | None = None,
) -> str:
    """根据问题过滤图 schema，保留相关节点和关系。

    参数：
    - question：用户自然语言问题。
    - full_schema：完整图 schema 文本。
    - nodes_data：图节点 schema 结构化数据。
    - relationships_data：图关系 schema 结构化数据。
    - schema_format：输出 schema 格式。
    - trace：可选 LLM trace。
    """
    prompt = f"""You are a graph database expert. Given a user question, identify the relevant node labels and relationship types.

Graph schema:
{full_schema}

Question:
{question}

Return JSON only:
{{
  "relevant_nodes": ["NodeLabel"],
  "relevant_relations": ["REL_TYPE"]
}}"""
    raw = call_llm(prompt, trace=trace, label="cypher_schema_filtering")
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return full_schema

    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return full_schema

    return filter_graph_schema(
        nodes_data,
        relationships_data,
        relevant_nodes=data.get("relevant_nodes", []),
        relevant_relations=data.get("relevant_relations", []),
        schema_format=schema_format,
        question=question,
    )


def _schema_name_mentioned(question: str, schema_name: str) -> bool:
    text = re.sub(r"(?<!^)(?=[A-Z])", " ", str(schema_name or ""))
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^0-9a-zA-Z\s]+", " ", text)
    name_terms = re.sub(r"\s+", " ", text).strip().lower().split()
    question_terms = set(re.findall(r"[0-9a-zA-Z]+", str(question or "").lower()))
    for term in name_terms:
        if term in question_terms:
            return True
        if term.endswith("y") and term[:-1] + "ies" in question_terms:
            return True
        if term + "s" in question_terms:
            return True
    return False


def prepare_cypher_schema_context(
    *,
    driver: Any,
    question: str,
    current_db: str,
    schema_format: str,
    trace: list[dict[str, str]],
    schema_grounding_mode: str = "llm_filter",
) -> dict[str, Any]:
    """准备 Cypher 生成所需的完整 schema、过滤 schema 和关系结构数据。

    参数：
    - driver：Neo4j driver。
    - question：用户自然语言问题。
    - current_db：当前候选图库名称。
    - schema_format：schema 输出格式。
    - trace：LLM 调用 trace。
    - schema_grounding_mode：schema grounding 策略。`llm_filter` 是正式系统默认行为；
      `full_schema` 不裁剪 schema；`exact_match` 使用问题词面与 schema 名称匹配裁剪。
    """
    grounding_mode = (schema_grounding_mode or "llm_filter").strip().lower()
    schema_started = _now_ms()
    actual_db = resolve_actual_cypher_database(current_db)
    nodes_data, relationships_data = extract_graph_schema_data(driver, actual_db)
    full_schema = extract_graph_schema(driver, actual_db, format=schema_format)
    schema_extraction_ms = round(_now_ms() - schema_started, 2)

    filtering_started = _now_ms()
    if grounding_mode == "full_schema":
        filtered_schema = full_schema
    elif grounding_mode == "exact_match":
        filtered_schema = filter_graph_schema(
            nodes_data,
            relationships_data,
            relevant_nodes=[
                str(node.get("label") or "")
                for node in nodes_data
                if _schema_name_mentioned(question, str(node.get("label") or ""))
            ],
            relevant_relations=[
                str(relation.get("type") or "")
                for relation in relationships_data
                if _schema_name_mentioned(question, str(relation.get("type") or ""))
            ],
            schema_format=schema_format,
            question=question,
        )
    elif grounding_mode == "llm_filter":
        filtered_schema = schema_filtering(
            question,
            full_schema,
            nodes_data,
            relationships_data,
            schema_format=schema_format,
            trace=trace,
        )
    else:
        raise ValueError(f"Unsupported schema_grounding_mode: {schema_grounding_mode}")

    return {
        "schema_full": full_schema,
        "working_schema": filtered_schema,
        "schema_filtered": filtered_schema,
        "relationships_data": relationships_data,
        "schema_grounding_mode": grounding_mode,
        "schema_extraction_ms": schema_extraction_ms,
        "schema_linking_ms": round(_now_ms() - filtering_started, 2),
    }
