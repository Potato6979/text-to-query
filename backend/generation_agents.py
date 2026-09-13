import re
import time
from typing import Any

from openai import OpenAI

from .config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
)
from .llm_utils import call_chat_completion
from .generation_feedback import format_generation_feedback
from .nested_logic_hints import build_nested_logic_guidance

llm = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

# 中文函数说明补充索引：
# - call_llm(prompt, trace, label)：SQL/Cypher 生成阶段 LLM 调用入口。
# - _extract_sql/_extract_cypher：从模型原始输出中抽取可执行查询。
# - generate_sql_query(...)：根据 DAIL prompt、schema context、示例和 retry feedback 生成 SQL。
# - generate_cypher_query(...)：根据图 schema、问题和 retry feedback 生成 Cypher。


def _now_ms() -> float:
    """返回当前高精度时间，单位毫秒。"""
    return time.perf_counter() * 1000


def call_llm(prompt: str, trace: list[dict[str, str]] | None = None, label: str = "") -> str:
    """调用生成模型，并把调用过程写入 trace。

    参数：
    - prompt：发送给模型的提示词。
    - trace：可选 LLM trace 列表，用于调试和预算估算。
    - label：本次调用标签。
    """
    return call_chat_completion(llm, prompt, trace=trace, label=label or "llm_call")


def _extract_sql(raw: str) -> str:
    """从模型原始输出中提取 SQL。

    参数：
    - raw：模型返回的原始文本，可能包含 `<sql>` 标签、Markdown 代码块或普通文本。
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


def _extract_cypher(raw: str) -> str:
    """从模型原始输出中提取 Cypher。

    参数：
    - raw：模型返回的原始文本，可能包含 `<cypher>` 标签、Markdown 代码块或普通文本。
    """
    tagged = re.search(r"<cypher>(.*?)</cypher>", raw, re.DOTALL | re.IGNORECASE)
    if tagged:
        return tagged.group(1).strip()
    code_block = re.search(r"```cypher\s*(.*?)```", raw, re.DOTALL | re.IGNORECASE)
    if code_block:
        return code_block.group(1).strip()
    fallback = re.search(r"(MATCH[\s\S]+)", raw, re.IGNORECASE)
    return fallback.group(1).strip() if fallback else raw.strip()


def generate_sql_query(prompt: str, trace: list[dict[str, str]]) -> dict[str, Any]:
    """根据已构造好的 SQL prompt 生成 SQL 查询。

    参数：
    - prompt：包含 schema、示例、规则和用户问题的 DAIL prompt。
    - trace：LLM 调用记录列表。
    """
    started_ms = _now_ms()
    generation_raw = call_llm(prompt, trace=trace, label="sql_generation")
    query = _extract_sql(generation_raw)
    return {
        "query": query,
        "query_type": "sql",
        "sql": query,
        "cot_raw": generation_raw,
        "generation_ms": round(_now_ms() - started_ms, 2),
    }


def generate_cypher_query(
    question: str,
    filtered_schema: str,
    trace: list[dict[str, str]],
    generation_feedback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """根据用户问题和过滤后的图 schema 生成 Cypher 查询。

    参数：
    - question：用户自然语言问题。
    - filtered_schema：与问题相关的图 schema 子集。
    - trace：LLM 调用记录列表。
    """
    nested_guidance = build_nested_logic_guidance(question, "cypher")
    nested_guidance_text = f"\nNested logic guidance:\n{nested_guidance['guidance']}\n" if nested_guidance["guidance"] else ""
    feedback_text = format_generation_feedback(generation_feedback, "cypher")
    feedback_block = f"\n{feedback_text}\n" if feedback_text else ""
    prompt = f"""You are an expert Neo4j engineer. Think step by step and generate the best Cypher query.

Relevant graph schema:
{filtered_schema}

Question:
{question}
{nested_guidance_text}
{feedback_block}

Rules:
- Follow the relationship directions shown in the schema.
- Keep property names exactly as defined.
- Relationship properties listed in the schema are property names, not pattern literals. Do not write [:REL {{prop}}] unless assigning or comparing a concrete value; bind the relationship as [r:REL] and use r.prop.
- Avoid accidental Cartesian products: when the question says two entities are related, connect them through explicit relationship paths instead of independent MATCH clauses.
- Prefer the direct relationship whose type or property matches the wording of the question. Do not add intermediate nodes or extra constraints unless the question mentions them or the direct relationship is unavailable in the schema.
- If the question explicitly says matches are in a tournament, category, date range, or other context, connect the Match variable back to that bound context when the schema provides such a relationship.
- Do not add a tournament, date, category, or other context filter to separate actions or events unless the question explicitly says those actions or events occurred in that context.
- If the question asks about a relationship property, aggregate that property directly. Do not replace a relationship property aggregation with an event count unless the question explicitly asks for event records.
- For scoreline or match score questions, if the schema exposes a score property on a team-match relationship, use that score property.
- For team goal totals in a tournament or match context, prefer counting goal-event relationships from players/persons who represent that team when the schema exposes both REPRESENTS and SCORED_GOAL. Use the score property only when the question explicitly asks for score, scoreline, or the score property, or when no goal-event path is available.
- Treat words that share a clear stem with a schema property as property mentions, such as "score" and "scored".
- When counting events represented by relationships, bind the relationship to a variable and count the relationship variable, not just the endpoint node.
- For averages, make the aggregation grain explicit. Only group by match, team, squad, or another entity when the question says "per", "for each", "each", or otherwise asks for grouped results.
- For age questions in an event or tournament context, do not use the current date unless the question asks for current age. If the schema has an event/tournament year, compute age from that contextual year and the person's birth date.
- If a birth date property is a date value, use property.year directly. Do not convert dates to strings and do not use substring(toString(dateValue), 0, 4) for age calculations.
- Do not use AVG(DISTINCT scalarValue) as a shortcut for de-duplicating entities. If each person/team/match should count once, first use WITH DISTINCT entity, scalarValue, then aggregate AVG(scalarValue).
- Do not de-duplicate players before an average unless the question explicitly asks for unique/distinct players. If the question says players who played in matches, preserve the player-match participation grain.
- Do not use collect(DISTINCT value) unless the question explicitly asks for unique/distinct values. For repeated visits, appearances, interactions, or other event-level relationships, collect the event-level values without DISTINCT.
- When collecting two different relationship-value lists from the same anchor node, collect the first list in a WITH clause before matching the second relationship; otherwise the two expansions multiply each other.
- If the schema contains both a base relationship type such as INTERACTS and numbered variants such as INTERACTS_1, use the base relationship unless the question explicitly asks to include all variants.
- Do not use OPTIONAL MATCH for a relationship the question says must exist. For average number of required related items, use MATCH so entities without that required relationship do not enter the denominator.
- Do not use COUNT {{ pattern }} for averaging required relationship counts, because it counts zero for entities without the relationship. Use MATCH relationship first, then COUNT the matched relationship per entity.
- Neo4j does not support duration.totalSeconds(x). For duration values, either use the requested component directly, such as rel.duration.seconds, or expand to days * 86400 + hours * 3600 + minutes * 60 + seconds when the question explicitly asks for total seconds.
- When counting logical node entities after adding another relationship expansion in the same scope, use COUNT(DISTINCT nodeVariable) so joins do not multiply the count. Count a relationship variable without DISTINCT only when the question asks for event/relationship occurrences.
- When averaging counts per event over all matched events, bind the event set first and use OPTIONAL MATCH only for the counted related events so zero-count events remain in the denominator. Do not use OPTIONAL MATCH for core entities that the question says must exist, such as players who played in matches.
- Use DISTINCT when a graph pattern can reach the same logical entity through multiple paths and the question asks for unique entities or counts.
- Return one row per requested result item unless the question explicitly asks for a grouped list. Do not use collect(...) just because multiple related values exist.
- Do not add DISTINCT to RETURN or WITH unless the question asks for unique/distinct results or a later aggregate would otherwise double-count the same logical entity.
- When returning attributes or metrics for each entity from a previous step, include that entity's id and canonical name when the current schema provides them, even if the question mainly lists other attributes; otherwise rows become unattributable.
- For a list of related entities qualified by another entity, such as "their concerts", return the upstream entity name/id together with the related entity fields. For concerts, matches, events, visits, appearances, or other dated/contextual records, include the record's year/date and venue/context fields when the schema provides them.
- If a filter uses a node such as a stadium, country, category, tournament, venue, or event and the question asks for results qualified by that context, return the relevant context fields when the schema provides them, such as stadium name and capacity for stadium-capacity filters.
- If the question says "known" entities, exclude rows whose primary name/title/label value is null, empty, or a literal placeholder such as "unknown".
- Return only the fields requested by the question. Do not return helper counts, IDs, paths, dates, or grouping keys unless they are part of the requested answer.
- Return the final query wrapped in <cypher> tags.
"""
    started_ms = _now_ms()
    raw = call_llm(prompt, trace=trace, label="cypher_generation")
    query = _extract_cypher(raw)
    return {
        "query": query,
        "query_type": "cypher",
        "cypher": query,
        "cot_raw": raw,
        "task_analysis": nested_guidance["analysis"],
        "nested_logic_guidance": nested_guidance["guidance"],
        "generation_feedback": generation_feedback or {},
        "generation_ms": round(_now_ms() - started_ms, 2),
    }
