import re
from typing import Any

from openai import OpenAI

from .config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
)
from .llm_utils import call_chat_completion
from .sql_prompt_dail import build_dail_prompt

llm = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

# 中文函数说明补充索引：
# - call_llm(prompt, trace, label)：修复阶段 LLM 调用入口。
# - _extract_sql/_extract_cypher：从修复模型输出中提取查询。
# - fix_sql_query(...)：根据 SQL 错误信息、schema 和可选 DAIL prompt 生成修复 SQL。
# - fix_cypher_query(...)：根据 Cypher 错误信息和 schema 生成修复 Cypher。


def call_llm(prompt: str, trace: list[dict[str, str]] | None = None, label: str = "") -> str:
    """调用修复阶段使用的 LLM。

    参数：
    - prompt：修复提示词。
    - trace：可选 LLM trace，用于记录修复调用。
    - label：本次修复调用标签。
    """
    return call_chat_completion(llm, prompt, trace=trace, label=label or "llm_call")


def _extract_sql(raw: str) -> str:
    """从修复模型输出中提取 SQL。

    参数：
    - raw：模型返回的原始修复文本。
    """
    tagged = re.search(r"<sql>(.*?)</sql>", raw, re.DOTALL | re.IGNORECASE)
    if tagged:
        return tagged.group(1).strip()
    code_block = re.search(r"```sql\s*(.*?)```", raw, re.DOTALL | re.IGNORECASE)
    if code_block:
        return code_block.group(1).strip()
    return raw.strip()


def _extract_cypher(raw: str) -> str:
    """从修复模型输出中提取 Cypher。

    参数：
    - raw：模型返回的原始修复文本。
    """
    tagged = re.search(r"<cypher>(.*?)</cypher>", raw, re.DOTALL | re.IGNORECASE)
    if tagged:
        return tagged.group(1).strip()
    code_block = re.search(r"```cypher\s*(.*?)```", raw, re.DOTALL | re.IGNORECASE)
    if code_block:
        return code_block.group(1).strip()
    fallback = re.search(r"(MATCH[\s\S]+)", raw, re.IGNORECASE)
    return fallback.group(1).strip() if fallback else raw.strip()


def fix_sql_query(
    *,
    question: str,
    schema_sql: str,
    examples: list[Any],
    bad_sql: str,
    error_msg: str,
    attempt: int,
    include_rule: bool,
    trace: list[dict[str, str]] | None = None,
) -> tuple[str, str]:
    """修复执行失败的 SQL 查询。

    参数：
    - question：原始自然语言问题。
    - schema_sql：当前数据库 schema 文本。
    - examples：DAIL few-shot 示例列表。
    - bad_sql：执行失败的 SQL。
    - error_msg：SQLite 返回的错误信息。
    - attempt：第几次修复尝试。
    - include_rule：是否在 DAIL prompt 中包含生成规则。
    - trace：可选 LLM trace。

    返回：
    - `(fixed_sql, raw_response)`，即提取后的 SQL 和模型原始输出。
    """
    prompt = (
        build_dail_prompt(
            question=question,
            schema_sql=schema_sql,
            examples=examples,
            include_rule=include_rule,
        )
        + "\n\n"
        + "/* The SQL above failed on SQLite. Fix it and return only corrected SQL wrapped in <sql> tags. */\n"
        + f"/* Attempt: {attempt} */\n"
        + f"/* Failed SQL: {bad_sql} */\n"
        + f"/* SQLite error: {error_msg} */"
    )
    raw = call_llm(
        prompt,
        trace=trace,
        label=f"sql_fix_attempt_{attempt}",
    )
    return _extract_sql(raw), raw


def fix_cypher_query(
    *,
    question: str,
    filtered_schema: str,
    bad_cypher: str,
    error_msg: str,
    attempt: int,
    trace: list[dict[str, str]] | None = None,
) -> tuple[str, str]:
    """修复执行失败的 Cypher 查询。

    参数：
    - question：原始自然语言问题。
    - filtered_schema：当前图库中过滤后的相关 schema。
    - bad_cypher：执行失败的 Cypher。
    - error_msg：Neo4j 或 schema validation 返回的错误信息。
    - attempt：第几次修复尝试。
    - trace：可选 LLM trace。

    返回：
    - `(fixed_cypher, raw_response)`，即提取后的 Cypher 和模型原始输出。
    """
    prompt = f"""You previously generated Cypher that failed.

Attempt: {attempt}
Question:
{question}

Relevant schema:
{filtered_schema}

Failed query:
{bad_cypher}

Database error:
{error_msg}

Return only the corrected Cypher wrapped in <cypher> tags.
"""
    raw = call_llm(
        prompt,
        trace=trace,
        label=f"cypher_fix_attempt_{attempt}",
    )
    return _extract_cypher(raw), raw
