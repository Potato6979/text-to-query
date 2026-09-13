from typing import Any

from .task_mode_analyzer import analyze_task_mode

# 中文函数说明索引：
# - build_nested_logic_guidance(question, query_type)：当 task_mode_analyzer 判断为 single_query_with_nested_logic 时，为 SQL/Cypher prompt 生成嵌套查询指导；普通单步则返回空 guidance。


def build_nested_logic_guidance(question: str, query_type: str) -> dict[str, Any]:
    analysis = analyze_task_mode(question)
    strategy = analysis.get("query_strategy", "single_query")
    if strategy != "single_query_with_nested_logic":
        return {"analysis": analysis, "guidance": ""}

    signals = analysis.get("signals", [])
    signal_names = [item.get("signal", "") for item in signals]
    signal_text = "; ".join(item.get("text", "") for item in signals[:3] if item.get("text"))

    if query_type == "sql":
        guidance_lines = [
            "Nested query guidance:",
            "- The question contains an implicit dependency that should still be answered with one SQL query.",
            "- Use a subquery, CTE, derived table, HAVING clause, or window function when needed.",
            "- Do not solve the inner dependency mentally or invent a constant value.",
            "- Keep the final SELECT focused on the entity or value requested by the user.",
        ]
        if any(name in signal_names for name in ["modifier_superlative_dependency", "relative_superlative_dependency", "one_with_superlative_dependency"]):
            guidance_lines.append("- For superlative dependencies, compute the max/min target in SQL and join/filter against it.")
        if any(name in signal_names for name in ["average_comparison_dependency", "aggregate_reference_dependency"]):
            guidance_lines.append("- For comparisons to aggregate values, compute the aggregate in SQL and compare against that result.")
    elif query_type == "cypher":
        guidance_lines = [
            "Nested graph query guidance:",
            "- The question contains an implicit dependency that should still be answered with one Cypher query.",
            "- Use WITH scopes, ORDER BY/LIMIT, aggregation, collect/unwind, or pattern comprehensions when needed.",
            "- Do not solve the inner dependency mentally or invent a constant value.",
            "- Keep the final RETURN focused on the entity or value requested by the user.",
        ]
        if any(name in signal_names for name in ["modifier_superlative_dependency", "relative_superlative_dependency", "one_with_superlative_dependency"]):
            guidance_lines.append("- For superlative dependencies, compute the ranked entity in Cypher, then continue matching from that entity in the same query.")
        if any(name in signal_names for name in ["average_comparison_dependency", "aggregate_reference_dependency"]):
            guidance_lines.append("- For comparisons to aggregate values, compute the aggregate in a WITH scope and filter with WHERE/HAVING-equivalent logic.")
    else:
        guidance_lines = []

    if signal_text:
        guidance_lines.append(f"- Detected dependency phrase: {signal_text}")

    return {"analysis": analysis, "guidance": "\n".join(guidance_lines)}
