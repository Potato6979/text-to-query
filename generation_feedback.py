from typing import Any

# 中文函数说明补充索引：
# - _safe_prompt_text(text)：把 verification reason 等反馈压缩成安全长度，避免 retry prompt 过长。
# - build_generation_retry_feedback(verification_result)：把 nested contract / Cypher semantic failure 转成下一轮 generation prompt 的结构化反馈。
# - apply_generation_feedback_to_prompt(prompt, feedback)：把 active feedback 插入 SQL/Cypher 生成 prompt。


NESTED_LOGIC_CONTRACT_UNMET = "nested_logic_contract_unmet"
CYPHER_GOAL_EVENT_FAILURES = {
    "cypher_goal_total_uses_score_property",
    "cypher_goal_total_counts_distinct_scorers",
    "cypher_generation_failure_under_strong_route_evidence",
}


def build_generation_retry_feedback(verification_result: dict[str, Any]) -> dict[str, Any]:
    """Build deterministic feedback for the next generation attempt."""
    failure_type = verification_result.get("failure_type", "")
    if failure_type in CYPHER_GOAL_EVENT_FAILURES:
        return {
            "active": True,
            "failure_type": failure_type,
            "reason": verification_result.get("reason", ""),
            "instruction": (
                "Regenerate the Cypher in the graph route. For team goal-total questions, "
                "count SCORED_GOAL relationship events from Person nodes that REPRESENT the team, "
                "do not sum PLAYED_IN.score unless the question asks for scoreline/score property, "
                "and do not count distinct scorer nodes as the goal total."
            ),
            "required_markers": ["SCORED_GOAL relationship count", "Person REPRESENTS Team", "Team PLAYED_IN Match"],
            "detected_markers": [],
        }
    if failure_type != NESTED_LOGIC_CONTRACT_UNMET:
        return {"active": False}

    contract_check = verification_result.get("generation_contract_check", {}) or {}
    return {
        "active": True,
        "failure_type": failure_type,
        "reason": verification_result.get("reason", ""),
        "query_strategy": contract_check.get("query_strategy", ""),
        "recommended_generation_strategy": contract_check.get("recommended_generation_strategy", ""),
        "required_markers": contract_check.get("required_markers", []),
        "detected_markers": contract_check.get("detected_markers", []),
        "instruction": (
            "The previous query was rejected because it looked like a flat query while "
            "the question requires nested logic inside one query."
        ),
    }


def _safe_prompt_text(value: Any) -> str:
    text = str(value or "").replace("*/", "").strip()
    return " ".join(text.split())


def format_generation_feedback(feedback: dict[str, Any] | None, query_type: str) -> str:
    """Render generation feedback as a short prompt block."""
    if not feedback or not feedback.get("active"):
        return ""

    required = ", ".join(_safe_prompt_text(item) for item in feedback.get("required_markers", []) if item)
    detected = ", ".join(_safe_prompt_text(item) for item in feedback.get("detected_markers", []) if item)
    lines = [
        "Regeneration feedback from Verification:",
        f"- Failure type: {_safe_prompt_text(feedback.get('failure_type'))}",
        f"- Reason: {_safe_prompt_text(feedback.get('reason'))}",
        f"- Instruction: {_safe_prompt_text(feedback.get('instruction'))}",
    ]
    if required:
        lines.append(f"- Expected query-shape markers: {required}.")
    if detected:
        lines.append(f"- Previous detected markers: {detected}.")
    if query_type == "sql":
        lines.append("- Revise the SQL to compute the dependency in SQL, using a CTE, subquery, derived table, HAVING clause, or window function as appropriate.")
    elif query_type == "cypher":
        lines.append("- Revise the Cypher to compute the dependency in the graph query, using WITH scopes, aggregation, ORDER BY/LIMIT, collect/unwind, or pattern comprehension as appropriate.")
    lines.append("- Do not replace the inner dependency with a guessed constant.")
    return "\n".join(lines)
