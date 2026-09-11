import re
import sqlite3

from nested_logic_hints import build_nested_logic_guidance
from sql_example_pool_dail import ExamplePoolItem

# 中文函数说明补充索引：
# - extract_schema_create_table(conn)：从 SQLite 中抽取 CREATE TABLE schema，SQL DAIL 正式链路使用完整 schema。
# - count_tokens(text, model)：估算 prompt token 数，用于 DAIL 示例预算。
# - build_dail_prompt_with_budget(...)：按 token 预算拼接 instruction、schema、示例、question、nested guidance。
# - build_dail_prompt(...)：兼容旧调用的 DAIL prompt 构造入口。

try:
    import tiktoken
except Exception:  # pragma: no cover - optional dependency
    tiktoken = None


def extract_schema_create_table(conn: sqlite3.Connection) -> str:
    """从 SQLite 连接中抽取 CREATE TABLE 形式的 schema。

    参数：
    - conn：SQLite 连接。
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT sql
        FROM sqlite_master
        WHERE type='table' AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        """
    )
    sqls = [row[0].strip() for row in cur.fetchall() if row[0]]
    cur.close()
    return "\n\n".join(sqls)


def format_dail_examples(examples: list[ExamplePoolItem], include_rule: bool = False) -> str:
    """把 DAIL 示例格式化为 prompt 片段。

    参数：
    - examples：选中的 few-shot 示例。
    - include_rule：是否在每个示例中使用“无解释回答”的规则语气。
    """
    if not examples:
        return ""
    lines = ["/* Some SQL examples are provided based on similar problems: */"]
    for example in examples:
        if include_rule:
            lines.append(f"/* Answer the following with no explanation: {example.question} */")
        else:
            lines.append(f"/* Answer the following: {example.question} */")
        lines.append(example.query.strip())
        lines.append("")
    return "\n".join(lines).strip()


def count_prompt_tokens(text: str, tokenizer_model: str = "gpt-3.5-turbo") -> int:
    """估算 prompt token 数。

    参数：
    - text：待估算文本。
    - tokenizer_model：tiktoken 使用的模型名。
    """
    if tiktoken is not None:
        try:
            encoding = tiktoken.encoding_for_model(tokenizer_model)
        except Exception:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    return len(re.findall(r"\S+", text))


def build_dail_prompt_with_budget(
    *,
    question: str,
    schema_sql: str,
    value_hints: str = "",
    candidate_examples: list[ExamplePoolItem],
    include_rule: bool = True,
    target_example_num: int = 9,
    max_seq_len: int = 4096,
    max_ans_len: int = 200,
    tokenizer_model: str = "gpt-3.5-turbo",
) -> tuple[str, list[ExamplePoolItem], int]:
    """在 token 预算内构造 DAIL prompt。

    参数：
    - question：用户自然语言问题。
    - schema_sql：当前数据库 schema。
    - value_hints：当前数据库真实值提示。
    - candidate_examples：排序后的候选示例。
    - include_rule：是否加入 SQL 生成规则。
    - target_example_num：目标示例数量。
    - max_seq_len：总 prompt 长度预算。
    - max_ans_len：预留给答案的长度。
    - tokenizer_model：token 估算模型名。
    """
    prompt_target = build_dail_prompt(
        question=question,
        schema_sql=schema_sql,
        value_hints=value_hints,
        examples=[],
        include_rule=include_rule,
    )
    total_tokens = count_prompt_tokens(prompt_target, tokenizer_model=tokenizer_model)

    if target_example_num <= 0 or not candidate_examples:
        return prompt_target, [], total_tokens

    example_prefix = "/* Some SQL examples are provided based on similar problems: */"
    selected_examples: list[ExamplePoolItem] = []
    selected_blocks: list[str] = []

    for example in candidate_examples:
        example_block = format_dail_examples([example], include_rule=include_rule)
        if example_block.startswith(example_prefix):
            example_block = example_block[len(example_prefix):].strip()

        candidate_sections = [example_prefix]
        if selected_blocks:
            candidate_sections.append("\n\n".join(selected_blocks + [example_block]))
        else:
            candidate_sections.append(example_block)
        candidate_sections.append(prompt_target)
        candidate_prompt = "\n\n".join(section.strip() for section in candidate_sections if section.strip())
        candidate_tokens = count_prompt_tokens(candidate_prompt, tokenizer_model=tokenizer_model)

        if candidate_tokens + max_ans_len <= max_seq_len:
            selected_examples.append(example)
            selected_blocks.append(example_block)
            total_tokens = candidate_tokens
            if len(selected_examples) >= target_example_num:
                break

    final_prompt = build_dail_prompt(
        question=question,
        schema_sql=schema_sql,
        value_hints=value_hints,
        examples=selected_examples,
        include_rule=include_rule,
    )
    return final_prompt, selected_examples, total_tokens


def build_dail_prompt(
    *,
    question: str,
    schema_sql: str,
    value_hints: str = "",
    examples: list[ExamplePoolItem] | None = None,
    include_rule: bool = True,
) -> str:
    """构造 SQL DAIL prompt。

    参数：
    - question：用户自然语言问题。
    - schema_sql：当前数据库 schema。
    - value_hints：可选真实值提示。
    - examples：可选 few-shot 示例。
    - include_rule：是否加入 SQL 生成规则。
    """
    sections: list[str] = []
    if examples:
        sections.append(format_dail_examples(examples, include_rule=include_rule))

    sections.append("/* Given the following database schema: */\n" + schema_sql)
    if value_hints.strip():
        sections.append(
            "/* Value grounding hints from the target database.\n"
            "Use these values with the exact original spelling and case when they match the question.\n"
            "When filtering by an entity value, prefer the column where the value appears in these hints. */\n"
            + value_hints.strip()
        )
    if include_rule:
        nested_guidance = build_nested_logic_guidance(question, "sql")
        if nested_guidance["guidance"]:
            sections.append("/* " + nested_guidance["guidance"].replace("*/", "") + " */")
        sections.append(
            "/* Answer the following with no explanation.\n"
            "Use exact table and column names from the schema.\n"
            "Do not invent a different casing for string constants when a matching database value is shown above.\n"
            "For grouped aggregate questions, put conditions on aggregate values in HAVING, not WHERE.\n"
            "For superlative entity questions such as 'which model has the most horsepower', do not GROUP BY the returned entity unless the question asks for each/per/grouped results; filter rows, ORDER BY the measure, and LIMIT 1.\n"
            "When the question asks 'which/list/show/find <entities>' and does not ask for full records, codes, IDs, countries, cities, or details, return the entity's canonical name/title column when the schema provides one.\n"
            "When returning attributes or metrics for each entity from a previous step, include that entity's id and canonical name when the current schema provides them, even if the question mainly lists other attributes; otherwise rows become unattributable. If the question says each player's, each student's, each singer's, or each entity's attributes, do not return only the attributes without the entity id/name.\n"
            "For a list of related entities qualified by another entity, return the upstream entity name/id together with the related entity fields. For concerts, matches, events, visits, appearances, or other dated/contextual records, include the record's year/date and venue/context fields when the schema provides them.\n"
            "If a filter uses a context table such as stadium, country, category, tournament, venue, or event and the question asks for results qualified by that context, return the relevant context fields when the schema provides them.\n"
            "If the question says known entities, exclude rows whose primary name/title/label value is NULL, empty, or a literal placeholder such as 'unknown'.\n"
            "Return only the columns requested by the question; do not include helper columns used only for filtering, grouping, or ordering.\n"
            "Do not use SELECT * unless the question explicitly asks for all columns or full records.\n"
            "Do not add DISTINCT unless the question asks for unique, distinct, different, or non-duplicated results.\n"
            "If COUNT/SUM/AVG/MIN/MAX is only needed for ORDER BY or HAVING, do not return that aggregate unless the question asks for the value.\n"
            f"Question: {question} */\nSELECT "
        )
    else:
        sections.append(f"/* Answer the following: {question} */\nSELECT ")
    return "\n\n".join(section.strip() for section in sections if section and section.strip())
