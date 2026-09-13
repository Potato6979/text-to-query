import re
import sqlite3
import threading
from typing import Any

from openai import OpenAI

from .config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    LLM_REQUEST_TIMEOUT,
    LLM_SEED,
    LLM_TEMPERATURE,
    MAX_RESULT_ROWS,
    MAX_RETRIES,
    SQLITE_DB_PATH,
)
from .sqlite_utils import connect_sqlite
from .sql_example_pool_dail import (
    ExamplePoolItem,
    get_spider_db_path,
    load_spider_dev_targets,
    load_spider_example_pool,
    load_spider_tables,
)
from .sql_example_selector_dail import DAILExampleSelector
from .sql_prompt_dail import (
    build_dail_prompt,
    build_dail_prompt_with_budget,
    extract_schema_create_table,
)
from .sql_schema_masking_dail import mask_question

llm = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
_selector_cache: dict[bool, DAILExampleSelector] = {}
_selector_lock = threading.Lock()
DEFAULT_TOKENIZER_MODEL = "gpt-3.5-turbo"
DEFAULT_MAX_SEQ_LEN = 4096
DEFAULT_MAX_ANS_LEN = 200
DEFAULT_SCOPE_FACTOR = 100


def _get_selector(force_bow: bool = False) -> DAILExampleSelector:
    """获取并缓存 DAIL 示例选择器。

    参数：
    - force_bow：是否强制使用 BOW 后备向量，而不是 sentence-transformer。
    """
    if force_bow not in _selector_cache:
        with _selector_lock:
            if force_bow not in _selector_cache:
                _selector_cache[force_bow] = DAILExampleSelector(
                    load_spider_example_pool(),
                    force_bow=force_bow,
                )
    return _selector_cache[force_bow]


def call_llm(prompt: str) -> str:
    """调用 SQL DAIL 链路中的 LLM。

    参数：
    - prompt：完整 SQL 生成或修复提示词。
    """
    response = llm.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=LLM_TEMPERATURE,
        seed=LLM_SEED,
        timeout=LLM_REQUEST_TIMEOUT,
    )
    return (response.choices[0].message.content or "").strip()


def _extract_sql(raw: str) -> str:
    """从模型输出中提取 SQL。

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


def execute_sql(conn: sqlite3.Connection, sql: str) -> tuple[bool, str]:
    """执行 SQL 并返回是否成功和文本结果。

    参数：
    - conn：SQLite 连接。
    - sql：待执行 SQL。
    """
    try:
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description] if cur.description else []
        cur.close()

        if not rows:
            return True, "（查询成功，结果为空）"

        header = " | ".join(columns)
        separator = "-" * len(header) if header else ""
        lines = [line for line in [header, separator] if line]
        for row in rows[:MAX_RESULT_ROWS]:
            lines.append(" | ".join("NULL" if value is None else str(value) for value in row))
        if len(rows) > MAX_RESULT_ROWS:
            lines.append(f"... 共 {len(rows)} 行，只显示前 {MAX_RESULT_ROWS} 行")
        return True, "\n".join(lines)
    except Exception as exc:
        return False, str(exc)


def _find_db_id_by_path(db_path: str) -> str | None:
    """根据 SQLite 文件路径反查 Spider db_id。

    参数：
    - db_path：SQLite 数据库路径。
    """
    normalized = db_path.lower().replace("/", "\\")
    table_map = load_spider_tables()
    for db_id in table_map:
        if get_spider_db_path(db_id).lower() == normalized:
            return db_id
    return None


def _build_target_info(question: str, db_id: str | None) -> dict[str, Any]:
    """构造 DAIL 示例选择所需的目标库和问题 mask 信息。

    参数：
    - question：用户自然语言问题。
    - db_id：Spider 数据库 id，可能为空。
    """
    table_map = load_spider_tables()
    table_json = table_map.get(db_id) if db_id else None
    table_names = table_json.get("table_names_original", []) if table_json else []
    column_names = [name for _, name in table_json.get("column_names_original", [])] if table_json else []
    linked_targets = load_spider_dev_targets()
    linked_target = linked_targets.get((db_id, question.strip())) if db_id else None
    return {
        "db_id": db_id,
        "question": question.strip(),
        "question_masked": (
            linked_target["question_masked"]
            if linked_target is not None
            else mask_question(question, table_names, column_names, mask_tag="<mask>", value_tag="<unk>")
        ),
        "question_pattern": linked_target.get("question_pattern", "") if linked_target else "",
        "table_json": table_json,
    }


def _rank_examples(
    *,
    selector_type: str,
    question: str,
    question_masked: str,
    db_id: str | None,
    force_bow: bool,
) -> list[ExamplePoolItem]:
    """根据选择策略对 DAIL 示例池排序。

    参数：
    - selector_type：示例选择策略。
    - question：原始问题。
    - question_masked：schema/value mask 后的问题。
    - db_id：当前目标数据库 id。
    - force_bow：是否强制使用 BOW selector。
    """
    selector = _get_selector(force_bow=force_bow)
    return selector.rank(
        selector_type=selector_type,
        question=question,
        question_masked=question_masked,
        db_id=db_id,
        cross_domain_only=True,
    )


def _build_prompt_from_ranked_examples(
    *,
    question: str,
    schema_sql: str,
    ranked_examples: list[ExamplePoolItem],
    include_rule: bool,
    example_num: int,
    max_seq_len: int,
    max_ans_len: int,
    tokenizer_model: str,
) -> tuple[str, list[ExamplePoolItem], int]:
    """从已排序示例中构造受 token 预算约束的 DAIL prompt。

    参数：
    - question：用户自然语言问题。
    - schema_sql：当前数据库 schema。
    - ranked_examples：排序后的候选示例。
    - include_rule：是否包含 SQL 生成规则。
    - example_num：目标示例数量。
    - max_seq_len：prompt 长度预算。
    - max_ans_len：答案长度预算。
    - tokenizer_model：token 估算模型名。
    """
    return build_dail_prompt_with_budget(
        question=question,
        schema_sql=schema_sql,
        candidate_examples=ranked_examples,
        include_rule=include_rule,
        target_example_num=example_num,
        max_seq_len=max_seq_len,
        max_ans_len=max_ans_len,
        tokenizer_model=tokenizer_model,
    )


def fix_sql_dail(
    *,
    question: str,
    schema_sql: str,
    examples: list[ExamplePoolItem],
    bad_sql: str,
    error_msg: str,
    attempt: int,
    include_rule: bool,
) -> str:
    """使用 DAIL prompt 修复失败 SQL。

    参数：
    - question：用户自然语言问题。
    - schema_sql：当前数据库 schema。
    - examples：已选 few-shot 示例。
    - bad_sql：失败 SQL。
    - error_msg：SQLite 错误信息。
    - attempt：第几次修复。
    - include_rule：是否包含 SQL 生成规则。
    """
    prompt = (
        build_dail_prompt(question=question, schema_sql=schema_sql, examples=examples, include_rule=include_rule)
        + "\n\n"
        + "/* The SQL above failed on SQLite. Fix it and return only corrected SQL wrapped in <sql> tags. */\n"
        + f"/* Attempt: {attempt} */\n"
        + f"/* Failed SQL: {bad_sql} */\n"
        + f"/* SQLite error: {error_msg} */"
    )
    return _extract_sql(call_llm(prompt))


def run_sql_pipeline_dail(
    question: str,
    db_path: str | None = None,
    *,
    selector_type: str = "masked_question",
    example_num: int = 9,
    use_retrieval: bool = True,
    use_fix: bool = False,
    include_rule: bool = True,
    force_bow_selector: bool = False,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    max_ans_len: int = DEFAULT_MAX_ANS_LEN,
    scope_factor: int = DEFAULT_SCOPE_FACTOR,
    tokenizer_model: str = DEFAULT_TOKENIZER_MODEL,
) -> dict[str, Any]:
    """旧版/独立 DAIL SQL pipeline 入口。

    参数：
    - question：用户自然语言问题。
    - db_path：可选指定 SQLite 数据库路径。
    - selector_type：DAIL 示例选择策略。
    - example_num：示例数量。
    - use_retrieval：是否自动检索数据库。
    - use_fix：是否启用失败修复。
    - include_rule：是否加入生成规则。
    - force_bow_selector：是否强制 BOW selector。
    - max_seq_len：prompt 长度预算。
    - max_ans_len：答案长度预算。
    - scope_factor：示例召回倍数。
    - tokenizer_model：token 估算模型名。
    """
    record: dict[str, Any] = {
        "question": question,
        "success": False,
        "result": "",
        "sql": "",
        "error": "",
        "retries": 0,
        "retrieved_db": "",
        "used_db_path": "",
        "schema_full": "",
        "examples": [],
        "selector_type": selector_type,
        "example_num": example_num,
        "question_masked": "",
        "prompt_tokens": 0,
        "force_bow_selector": force_bow_selector,
    }

    if db_path is None:
        if use_retrieval:
            try:
                from .schema_retriever import retrieve_db

                retrieval = retrieve_db(question)
                db_path = retrieval["db_path"]
                record["retrieved_db"] = retrieval["db_id"]
            except Exception:
                db_path = SQLITE_DB_PATH
        else:
            db_path = SQLITE_DB_PATH

    record["used_db_path"] = db_path
    db_id = _find_db_id_by_path(db_path)
    target_info = _build_target_info(question, db_id)
    record["question_masked"] = target_info["question_masked"]

    conn = connect_sqlite(db_path)
    try:
        schema_sql = extract_schema_create_table(conn)
        record["schema_full"] = schema_sql

        recall_size = max(example_num, example_num * scope_factor)

        ranked_examples = _rank_examples(
            selector_type=selector_type,
            question=question,
            question_masked=target_info["question_masked"],
            db_id=db_id,
            force_bow=force_bow_selector,
        )
        prompt, examples, prompt_tokens = _build_prompt_from_ranked_examples(
            question=question,
            schema_sql=schema_sql,
            ranked_examples=ranked_examples[:recall_size],
            include_rule=include_rule,
            example_num=example_num,
            max_seq_len=max_seq_len,
            max_ans_len=max_ans_len,
            tokenizer_model=tokenizer_model,
        )

        record["examples"] = [
            {"db_id": item.db_id, "question": item.question, "query": item.query}
            for item in examples
        ]
        record["prompt_tokens"] = prompt_tokens
        sql = _extract_sql(call_llm(prompt))
        record["sql"] = sql

        attempts = MAX_RETRIES if use_fix else 0
        for attempt in range(attempts + 1):
            ok, output = execute_sql(conn, sql)
            if ok:
                record["success"] = True
                record["result"] = output
                record["retries"] = attempt
                return record

            record["error"] = output
            if attempt >= attempts:
                record["result"] = f"执行失败（已重试 {attempts} 次）：{output}"
                return record

            sql = fix_sql_dail(
                question=question,
                schema_sql=schema_sql,
                examples=examples,
                bad_sql=sql,
                error_msg=output,
                attempt=attempt + 1,
                include_rule=include_rule,
            )
            record["sql"] = sql

        return record
    finally:
        conn.close()
