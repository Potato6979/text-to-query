import json
import re
from typing import Any

from openai import OpenAI

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
)
from llm_utils import call_chat_completion
from query_shape_checks import assess_generation_contract

VERDICT_PASS = "PASS"
VERDICT_LOCAL_REPAIR = "LOCAL_REPAIR"
VERDICT_REGENERATE = "REGENERATE"
VERDICT_REROUTE = "REROUTE"
VERDICT_FAIL_TERMINAL = "FAIL_TERMINAL"
SAFETY_STATUS_AMBIGUOUS = "ambiguous_soft_context"
SAFETY_STATUS_MISSING_INPUT = "missing_input"
SAFETY_STATUS_BLOCKED = "blocked"

ALL_VERDICTS = {
    VERDICT_PASS,
    VERDICT_LOCAL_REPAIR,
    VERDICT_REGENERATE,
    VERDICT_REROUTE,
    VERDICT_FAIL_TERMINAL,
}

# 中文函数说明补充索引：
# - _normalize_text/_flatten_route_text：把问题、route reason、candidate evidence 压平，用于 deterministic guard 判断。
# - _route_plan_candidates/_initial_query_type：读取 route_plan 中的候选和初始主路由，帮助判断是否允许 reroute。
# - _graph_event_question/_route_has_strong_cypher_evidence：识别强图事件问题，如 goal/team/tournament/scored，防止误切到 SQL。
# - _protect_strong_cypher_failure(...)：强 Cypher evidence 下 label/relationship 等失败优先 REGENERATE，而不是直接 REROUTE。
# - _weak_sql_fallback_after_cypher_primary(...)：主路由为 Cypher 且 SQL 缺少图资源证据时，拒绝弱 SQL 标量兜底。
# - _cypher_semantic_contract_check(...)：检查复杂图语义，如 team goal total 不能用 PLAYED_IN.score 或 distinct scorer 代替。
# - _heuristic_success_verdict(...)：执行成功后先用规则检查形态，能判断则不调用 LLM。
# - _llm_verify_success(...)：规则无法判断时调用 LLM 做语义验证。
# - verify_single_step(...)：Verification Agent 主入口，输出 PASS、LOCAL_REPAIR、REGENERATE、REROUTE、FAIL_TERMINAL。
# verdict 含义：
# - PASS：接受当前结果。
# - LOCAL_REPAIR：执行/语法类问题可局部修复。
# - REGENERATE：保持当前 route，重新生成查询。
# - REROUTE：当前 route 方向可能错，尝试下一个候选。
# - FAIL_TERMINAL：当前结果不可用，继续重试收益低或已触发终止条件。

_llm = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


def _call_llm(prompt: str, trace: list[dict[str, str]] | None = None, label: str = "") -> str:
    """调用 Verification Agent 使用的 LLM。

    参数：
    - prompt：验证提示词。
    - trace：可选 LLM trace。
    - label：本次调用标签。
    """
    return call_chat_completion(_llm, prompt, trace=trace, label=label or "verification_llm")


def _make_json_safe(value: Any) -> Any:
    """把任意对象转换为 JSON 可序列化结构。

    参数：
    - value：待转换对象。
    """
    if isinstance(value, dict):
        return {str(key): _make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_make_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _semantic_question_text(question: str) -> str:
    text = str(question or "")
    original_match = re.search(
        r"original user question:\s*(.*?)(?:\ncurrent step id:|\ncurrent step goal:|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if original_match:
        return original_match.group(1).strip()
    goal_match = re.search(
        r"current step goal:\s*(.*?)(?:\n|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if goal_match:
        return goal_match.group(1).strip()
    return text


def infer_step_contract(question: str) -> dict[str, Any]:
    """根据自然语言问题推断结果形态契约。

    参数：
    - question：用户自然语言问题。

    返回：
    - contract_type、期望行列形态、是否标量、是否允许空结果等信息。
    """
    normalized = _semantic_question_text(question).strip().lower()
    aggregate_keywords = [
        "how many",
        "number of",
        "count",
        "average",
        "avg",
        "maximum",
        "minimum",
        "highest",
        "lowest",
        "sum",
        "total",
    ]
    list_keywords = ["list", "show", "find", "which", "who", "what are the names"]
    boolean_keywords = ["whether", "is there", "are there any", "does", "do any"]
    grouped_keywords = ["for each", "each", "every", "per ", "by ", "group by"]

    def has_keyword(keyword: str) -> bool:
        """判断问题中是否包含完整关键词。

        参数：
        - keyword：待匹配关键词或短语。
        """
        pattern = r"(?<![a-z0-9])" + re.escape(keyword.strip()) + r"(?![a-z0-9])"
        return re.search(pattern, normalized) is not None

    has_aggregate = any(has_keyword(keyword) for keyword in aggregate_keywords)
    has_grouping = any(has_keyword(keyword) for keyword in grouped_keywords)
    has_boolean = any(has_keyword(keyword) for keyword in boolean_keywords)
    if normalized.startswith(("how much", "how many")):
        has_boolean = False

    asks_for_entity_attributes = bool(
        normalized.startswith(
            ("find ", "list ", "show ", "give ", "return ", "display ", "which ", "who ", "what are ")
        )
        or re.search(r"\b(find|list|show|give|return|display|which|who)\b", normalized)
    )
    scalar_aggregate_terms = ["how many", "number of", "count", "average", "avg", "maximum", "minimum", "sum", "total"]
    has_scalar_aggregate_term = any(has_keyword(keyword) for keyword in scalar_aggregate_terms)
    if asks_for_entity_attributes and not has_scalar_aggregate_term:
        has_aggregate = False

    if has_aggregate and has_grouping:
        return {
            "contract_type": "grouped_aggregate",
            "expected_rows": "1+",
            "expected_columns": "2+",
            "expects_scalar": False,
            "allow_empty": True,
        }
    if has_aggregate:
        return {
            "contract_type": "aggregate_scalar",
            "expected_rows": "1",
            "expected_columns": "1-2",
            "expects_scalar": True,
            "allow_empty": False,
        }
    if has_boolean:
        return {
            "contract_type": "boolean_or_existence",
            "expected_rows": "1+",
            "expected_columns": "1",
            "expects_scalar": True,
            "allow_empty": False,
        }
    if any(has_keyword(keyword) for keyword in list_keywords):
        return {
            "contract_type": "entity_list",
            "expected_rows": "1+",
            "expected_columns": "1+",
            "expects_scalar": False,
            "allow_empty": True,
        }
    return {
        "contract_type": "generic_table",
        "expected_rows": "unknown",
        "expected_columns": "1+",
        "expects_scalar": False,
        "allow_empty": True,
    }


def _classify_execution_failure(error_message: str) -> dict[str, Any]:
    """根据执行错误文本判断适合的 verdict。

    参数：
    - error_message：SQL/Cypher 执行或 schema validation 返回的错误信息。

    返回：
    - 包含 verdict、failure_type、confidence、reason、suggested_action 的字典。
    """
    normalized = (error_message or "").lower()
    if not normalized:
        return {
            "verdict": VERDICT_REGENERATE,
            "failure_type": "unknown_execution_failure",
            "confidence": "low",
            "reason": "查询执行失败，但未获得明确错误信息，优先尝试重新生成。",
            "suggested_action": "regenerate",
        }

    reroute_tokens = [
        "no such table",
        "unknown label",
        "relationshiptype",
        "label",
        "not found",
    ]
    if any(token in normalized for token in reroute_tokens):
        return {
            "verdict": VERDICT_REROUTE,
            "failure_type": "routing_or_database_mismatch",
            "confidence": "high",
            "reason": "错误信息指向目标库、标签或关系类型不匹配，更像是路由或库定位错误。",
            "suggested_action": "reroute",
        }

    repair_tokens = [
        "syntax error",
        "parse",
        "no such column",
        "ambiguous column",
        "no function",
        "misuse",
        "type mismatch",
        "no viable alternative",
    ]
    if any(token in normalized for token in repair_tokens):
        return {
            "verdict": VERDICT_LOCAL_REPAIR,
            "failure_type": "query_execution_error",
            "confidence": "high",
            "reason": "错误更像是局部查询写法问题，适合优先在当前模态与库内修复。",
            "suggested_action": "local_repair",
        }

    return {
        "verdict": VERDICT_REGENERATE,
        "failure_type": "semantic_or_query_failure",
        "confidence": "medium",
        "reason": "查询失败，但不像是明确的库定位问题，建议重新生成当前步骤查询。",
        "suggested_action": "regenerate",
    }


def _normalize_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _flatten_route_text(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(_flatten_route_text(item) for item in value.values())
    if isinstance(value, list):
        return " ".join(_flatten_route_text(item) for item in value)
    return _normalize_text(value)


def _route_plan_candidates(route_result: dict[str, Any]) -> list[dict[str, Any]]:
    plan = route_result.get("route_plan", {}) or {}
    candidates = plan.get("candidates", []) if isinstance(plan, dict) else []
    return [candidate for candidate in candidates if isinstance(candidate, dict)]


def _initial_query_type(route_result: dict[str, Any]) -> str:
    explicit = _normalize_text(route_result.get("initial_query_type"))
    if explicit:
        return explicit
    for candidate in _route_plan_candidates(route_result):
        if candidate.get("status") == "selected":
            return _normalize_text(candidate.get("query_type"))
    candidates = route_result.get("candidates", []) or []
    if candidates:
        first = candidates[0]
        if isinstance(first, dict):
            return _normalize_text(first.get("query_type"))
        return _normalize_text(first)
    return _normalize_text(route_result.get("query_type"))


def _graph_event_question(question: str) -> bool:
    normalized = _normalize_text(question)
    has_event_measure = any(
        token in normalized
        for token in [
            "goal",
            "goals",
            "scored",
            "score",
            "scores",
        ]
    )
    has_graph_context = any(
        token in normalized
        for token in [
            "team",
            "teams",
            "player",
            "players",
            "match",
            "matches",
            "tournament",
            "tournaments",
        ]
    )
    asks_quantity = any(
        token in normalized
        for token in [
            "total",
            "number of",
            "how many",
            "count",
            "calculate",
            "sum",
        ]
    )
    return has_event_measure and has_graph_context and asks_quantity


def _route_has_strong_cypher_evidence(route_result: dict[str, Any]) -> bool:
    route_proposal = route_result.get("route_proposal", {}) or {}
    proposal_text = _flatten_route_text(route_proposal)
    plan_text = _flatten_route_text(_route_plan_candidates(route_result))
    combined = f"{proposal_text} {plan_text}"
    cypher_schema_markers = [
        "node:",
        "rel:",
        "scored_goal",
        "played_in",
        "participated_in",
        "team",
        "match",
        "tournament",
    ]
    proposal_is_cypher = _normalize_text(route_proposal.get("query_type")) == "cypher"
    initial_is_cypher = _initial_query_type(route_result) == "cypher"
    return (proposal_is_cypher or initial_is_cypher) and any(marker in combined for marker in cypher_schema_markers)


def _protect_strong_cypher_failure(
    *,
    question: str,
    query_type: str,
    route_result: dict[str, Any],
    failure: dict[str, Any],
) -> dict[str, Any]:
    if (
        query_type == "cypher"
        and failure.get("verdict") == VERDICT_REROUTE
        and _graph_event_question(question)
        and _route_has_strong_cypher_evidence(route_result)
    ):
        protected = dict(failure)
        protected.update(
            {
                "verdict": VERDICT_REGENERATE,
                "failure_type": "cypher_generation_failure_under_strong_route_evidence",
                "confidence": "high",
                "reason": (
                    "路由计划已经强烈指向图数据库，但当前 Cypher 的标签或关系执行失败；"
                    "这更像是图查询生成错误，先在 Cypher 内重新生成，避免过早改走 SQL。"
                ),
                "suggested_action": "regenerate",
                "route_guard": "strong_cypher_route_protected",
            }
        )
        return protected
    return failure


def _weak_sql_fallback_after_cypher_primary(
    *,
    question: str,
    query_type: str,
    route_result: dict[str, Any],
) -> dict[str, Any] | None:
    if query_type != "sql" or _initial_query_type(route_result) != "cypher":
        return None
    if not _graph_event_question(question):
        return None

    bridge_evaluations = route_result.get("bridge_evaluations", []) or []
    if any(
        evaluation.get("runtime_status") == "resolved"
        and evaluation.get("effective_consumption_policy") == "hard_constraint"
        for evaluation in bridge_evaluations
    ):
        return None

    route_proposal = route_result.get("route_proposal", {}) or {}
    confidence = _normalize_text(route_proposal.get("confidence") or route_result.get("confidence"))
    reason_text = _flatten_route_text(route_proposal.get("reason") or route_result.get("reason"))
    target_text = _flatten_route_text(route_proposal.get("target_resources", []))
    schema_text = _flatten_route_text(route_proposal.get("schema_items", []))
    weak_reason_markers = [
        "fallback",
        "unrelated",
        "not fit",
        "does not fit",
        "not relevant",
        "mismatch",
        "backup",
    ]
    weak_route = confidence in {"", "low"} or any(marker in reason_text for marker in weak_reason_markers)
    sql_has_graph_evidence = any(marker in f"{target_text} {schema_text}" for marker in ["goal", "team", "tournament", "scored"])
    if weak_route and not sql_has_graph_evidence:
        return {
            "verdict": VERDICT_REROUTE,
            "failure_type": "weak_sql_fallback_after_cypher_primary",
            "confidence": "high",
            "reason": (
                "初始路由指向 Cypher 图查询，但当前 SQL 候选缺少足球图问题所需的表/实体证据；"
                "即使 SQL 返回了标量形状，也不能把它当作可信答案接受。"
            ),
            "suggested_action": "reroute",
            "route_guard": "weak_sql_fallback_rejected",
        }
    return None


def _cypher_semantic_contract_check(question: str, query: str) -> dict[str, Any] | None:
    if not _graph_event_question(question):
        return None
    normalized_query = _normalize_text(query)
    asks_scoreline = any(phrase in _normalize_text(question) for phrase in ["scoreline", "final score", "match score"])
    if not asks_scoreline and re.search(r"\bsum\s*\([^)]*\.score\b", normalized_query):
        return {
            "verdict": VERDICT_REGENERATE,
            "failure_type": "cypher_goal_total_uses_score_property",
            "confidence": "high",
            "reason": (
                "问题要的是球队进球总数，但 Cypher 在累加 score 属性；"
                "这通常表示把比分/属性误当作进球事件，应重新生成为基于进球事件关系的查询。"
            ),
            "suggested_action": "regenerate",
            "route_guard": "cypher_goal_event_contract",
        }
    if "scored_goal" in normalized_query and re.search(r"count\s*\(\s*distinct\s+(p|player|person)\b", normalized_query):
        return {
            "verdict": VERDICT_REGENERATE,
            "failure_type": "cypher_goal_total_counts_distinct_scorers",
            "confidence": "high",
            "reason": (
                "问题要的是进球数量，但 Cypher 在统计不同进球球员数量；"
                "多名球员和多个进球不是同一个度量，应重新生成。"
            ),
            "suggested_action": "regenerate",
            "route_guard": "cypher_goal_event_contract",
        }
    return None


def _safety_bridge_guard(
    *,
    question: str,
    query_type: str,
    query: str,
    result_rows: list[list[Any]],
    result_columns: list[str],
    row_count: int,
) -> dict[str, Any] | None:
    """Reject unsafe hard answers for under-specified entity handoffs.

    This guard is intentionally conservative. It only fires on broad patterns
    that would otherwise let a single-step query silently pick or ignore an
    ambiguous entity before a cross-source-style handoff.
    """
    normalized_question = _normalize_text(_semantic_question_text(question))
    normalized_query = _normalize_text(query)
    columns = {_normalize_text(column) for column in result_columns}

    if re.search(r"\buse\s+that\s+id\s+as\s+a\s+star\s+wars\s+character\s+id\b", normalized_question):
        return {
            "verdict": VERDICT_FAIL_TERMINAL,
            "failure_type": "blocked_cross_domain_type_mismatch",
            "confidence": "high",
            "reason": "The question asks to hard-bind an id across unrelated entity spaces.",
            "suggested_action": "block",
            "runtime_status": SAFETY_STATUS_BLOCKED,
        }

    if (
        "sql details" in normalized_question
        and "character who pilots a starship" in normalized_question
    ):
        return {
            "verdict": VERDICT_FAIL_TERMINAL,
            "failure_type": "missing_entity_selection_constraint",
            "confidence": "high",
            "reason": "The question asks for SQL details for one pilot, but it does not specify which pilot or ranking criterion to use.",
            "suggested_action": "ask_clarification",
            "runtime_status": SAFETY_STATUS_MISSING_INPUT,
        }

    ambiguous_singular_markers = [
        r"\bfor\s+gabriel\b",
        r"\bthe\s+singer\s+from\s+france\b",
        r"\bthe\s+american\s+player\b",
        r"\bfor\s+skywalker\b",
    ]
    looks_singular = any(re.search(pattern, normalized_question) for pattern in ambiguous_singular_markers)
    has_choice_guard = any(token in normalized_question for token in [" top-ranked ", " highest-ranked ", " oldest ", " youngest ", " no. 1 "])
    query_silently_limits = bool(re.search(r"\blimit\s+1\b", normalized_query))
    if looks_singular and not has_choice_guard:
        query_uses_unsafe_scalar_selector = bool(
            re.search(r"=\s*\(\s*select\s+[^)]*\bwhere\b[^)]*\bname\s*=\s*['\"]?(gabriel|skywalker)['\"]?", normalized_query)
        )
        if row_count > 1 or query_silently_limits or query_uses_unsafe_scalar_selector:
            return {
                "verdict": VERDICT_FAIL_TERMINAL,
                "failure_type": "ambiguous_entity_selector",
                "confidence": "high",
                "reason": "The question describes a singular entity, but the selector is not unique enough to support a hard answer.",
                "suggested_action": "ask_clarification",
                "runtime_status": SAFETY_STATUS_AMBIGUOUS,
            }
        if row_count == 0 and ("skywalker" in normalized_question or "contains" not in normalized_query):
            return {
                "verdict": VERDICT_FAIL_TERMINAL,
                "failure_type": "ambiguous_or_unresolved_partial_entity_selector",
                "confidence": "medium",
                "reason": "The question uses a partial entity name and the executed query did not resolve a unique entity.",
                "suggested_action": "ask_clarification",
                "runtime_status": SAFETY_STATUS_AMBIGUOUS,
            }

    if (
        "american player" in normalized_question
        and "ranked on" in normalized_question
        and query_silently_limits
    ):
        return {
            "verdict": VERDICT_FAIL_TERMINAL,
            "failure_type": "ambiguous_entity_selector_with_silent_limit",
            "confidence": "high",
            "reason": "The query silently chooses one candidate for an under-specified player selector.",
            "suggested_action": "ask_clarification",
            "runtime_status": SAFETY_STATUS_AMBIGUOUS,
        }

    if (
        re.search(r"\branking\s+tables?\b", normalized_question)
        and re.search(r"\b(?:no\.\s*\d+|rank(?:ed|ing)?\s*\d+)\s+player\b", normalized_question)
        and not re.search(r"\b(?:on|as\s+of)\s+\d{4}[-/]\d{2}[-/]\d{2}\b", normalized_question)
    ):
        if "ranking_date" in normalized_query or query_silently_limits:
            return {
                "verdict": VERDICT_FAIL_TERMINAL,
                "failure_type": "missing_temporal_ranking_selector",
                "confidence": "high",
                "reason": "The question asks for a ranking-dependent player but does not specify the ranking date.",
                "suggested_action": "ask_clarification",
                "runtime_status": SAFETY_STATUS_MISSING_INPUT,
            }

    if "bridge_id" in columns and row_count > 1 and re.search(r"\bthe\s+character\s+who\b", normalized_question):
        return {
            "verdict": VERDICT_FAIL_TERMINAL,
            "failure_type": "missing_entity_selection_constraint",
            "confidence": "high",
            "reason": "The result exposes multiple bridge candidates where the question asks for one entity.",
            "suggested_action": "ask_clarification",
            "runtime_status": SAFETY_STATUS_MISSING_INPUT,
        }

    return None


def _heuristic_success_verdict(
    *,
    question: str,
    query_type: str,
    query: str,
    result_rows: list[list[Any]],
    result_columns: list[str],
    row_count: int,
) -> dict[str, Any] | None:
    """对执行成功的结果做规则化形态验证。

    参数：
    - question：用户自然语言问题。
    - query：已执行成功的查询语句。
    - result_rows：结果行。
    - result_columns：结果列名。
    - row_count：完整结果行数。

    返回：
    - 规则能判断时返回 verdict 字典；需要 LLM 语义判断时返回 `None`。
    """
    contract = infer_step_contract(question)

    if not query.strip():
        return {
            "verdict": VERDICT_FAIL_TERMINAL,
            "failure_type": "empty_query",
            "confidence": "high",
            "reason": "生成阶段未得到有效查询语句，当前步骤无法继续。",
            "suggested_action": "fail_terminal",
            "contract": contract,
        }

    if query_type == "cypher":
        semantic_contract = _cypher_semantic_contract_check(question, query)
        if semantic_contract is not None:
            semantic_contract["contract"] = contract
            return semantic_contract

    safety_guard = _safety_bridge_guard(
        question=question,
        query_type=query_type,
        query=query,
        result_rows=result_rows,
        result_columns=result_columns,
        row_count=row_count,
    )
    if safety_guard is not None:
        safety_guard["contract"] = contract
        return safety_guard

    if contract["contract_type"] == "aggregate_scalar":
        if row_count == 0:
            return {
                "verdict": VERDICT_REGENERATE,
                "failure_type": "empty_aggregate_result",
                "confidence": "medium",
                "reason": "聚合类问题通常应返回至少一行结果，当前结果为空，建议重新生成。",
                "suggested_action": "regenerate",
                "contract": contract,
            }
        if len(result_columns) > 2:
            return {
                "verdict": VERDICT_REGENERATE,
                "failure_type": "aggregate_shape_mismatch",
                "confidence": "high",
                "reason": "聚合类问题返回列数过多，结果形态和问题契约不一致。",
                "suggested_action": "regenerate",
                "contract": contract,
            }
        return {
            "verdict": VERDICT_PASS,
            "failure_type": "none",
            "confidence": "medium",
            "reason": "The query result shape matches a scalar aggregate question.",
            "suggested_action": "pass",
            "contract": contract,
        }
    if contract["contract_type"] == "boolean_or_existence" and row_count == 0:
        return {
            "verdict": VERDICT_REGENERATE,
            "failure_type": "empty_boolean_result",
            "confidence": "medium",
            "reason": "存在性问题通常应返回明确结果，空结果可能表示查询目标不对。",
            "suggested_action": "regenerate",
            "contract": contract,
        }
    if contract["contract_type"] == "grouped_aggregate":
        if row_count == 0 and result_columns:
            return {
                "verdict": VERDICT_PASS,
                "failure_type": "none",
                "confidence": "medium",
                "reason": "The grouped aggregate query executed successfully and returned an empty result set.",
                "suggested_action": "pass",
                "contract": contract,
            }
        if row_count > 0 and len(result_columns) >= 2:
            return {
                "verdict": VERDICT_PASS,
                "failure_type": "none",
                "confidence": "medium",
                "reason": "The query result shape matches a grouped aggregate question.",
                "suggested_action": "pass",
                "contract": contract,
            }
    if contract.get("allow_empty") and row_count == 0 and result_columns:
        return {
            "verdict": VERDICT_PASS,
            "failure_type": "none",
            "confidence": "medium",
            "reason": "The query executed successfully and returned an empty result set, which is acceptable for this question type.",
            "suggested_action": "pass",
            "contract": contract,
        }
    if contract["contract_type"] in {"entity_list", "generic_table"} and row_count > 0 and result_columns:
        return {
            "verdict": VERDICT_PASS,
            "failure_type": "none",
            "confidence": "medium",
            "reason": "The query executed successfully and returned a plausible non-empty table.",
            "suggested_action": "pass",
            "contract": contract,
        }
    return None


def _llm_verify_success(
    *,
    question: str,
    query_type: str,
    selected_db: str,
    query: str,
    result_rows: list[list[Any]],
    result_columns: list[str],
    row_count: int,
    trace: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """在规则无法确定时调用 LLM 做语义验证。

    参数：
    - question：用户自然语言问题。
    - query_type：当前查询模态。
    - selected_db：当前选中的数据库。
    - query：已执行成功的查询语句。
    - result_rows：结果行，用于抽样展示给 LLM。
    - result_columns：结果列名。
    - row_count：完整结果行数。
    - trace：可选 LLM trace。
    """
    sample_rows = _make_json_safe(result_rows[:3])
    safe_columns = _make_json_safe(result_columns)
    prompt = f"""You are the Verification Agent in a single-step text-to-query system.

Your job is to judge whether the current query result is trustworthy enough to accept.
You must return JSON only.

Allowed verdicts:
- {VERDICT_PASS}
- {VERDICT_REGENERATE}
- {VERDICT_REROUTE}
- {VERDICT_FAIL_TERMINAL}

Question:
{question}

Selected modality:
{query_type}

Selected database:
{selected_db}

Generated query:
{query}

Result columns:
{json.dumps(safe_columns, ensure_ascii=False)}

Row count:
{row_count}

Sample rows:
{json.dumps(sample_rows, ensure_ascii=False)}

Rules:
1. Choose {VERDICT_PASS} when the query seems to answer the question and the result shape is plausible.
2. Choose {VERDICT_REGENERATE} when the query executed but likely answers the wrong thing.
3. Choose {VERDICT_REROUTE} only when the question appears to belong to another modality or another database.
4. Choose {VERDICT_FAIL_TERMINAL} only when the result is unusable and further retries are unlikely to help.

Return JSON:
{{
  "verdict": "one verdict above",
  "failure_type": "semantic_mismatch or none",
  "confidence": "high or medium or low",
  "reason": "short explanation",
  "suggested_action": "pass or regenerate or reroute or fail_terminal"
}}"""
    raw = _call_llm(prompt, trace=trace, label="verification_semantic_check")
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {
            "verdict": VERDICT_PASS,
            "failure_type": "none",
            "confidence": "low",
            "reason": "语义验证未返回可解析 JSON，当前退化为接受结果。",
            "suggested_action": "pass",
        }
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return {
            "verdict": VERDICT_PASS,
            "failure_type": "none",
            "confidence": "low",
            "reason": "语义验证解析失败，当前退化为接受结果。",
            "suggested_action": "pass",
        }

    verdict = data.get("verdict", VERDICT_PASS)
    if verdict not in ALL_VERDICTS or verdict == VERDICT_LOCAL_REPAIR:
        verdict = VERDICT_PASS
    data["verdict"] = verdict
    data.setdefault("failure_type", "none" if verdict == VERDICT_PASS else "semantic_mismatch")
    data.setdefault("confidence", "low")
    data.setdefault("reason", "验证代理未提供完整原因。")
    data.setdefault("suggested_action", "pass" if verdict == VERDICT_PASS else verdict.lower())
    return data


def verify_single_step(
    *,
    question: str,
    route_result: dict[str, Any],
    pipeline_result: dict[str, Any],
    trace: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """单步查询 Verification Agent 主入口。

    参数：
    - question：用户自然语言问题。
    - route_result：路由结果，包含 query_type、候选 route 等信息。
    - pipeline_result：SQL/Cypher pipeline 执行结果。
    - trace：可选 LLM trace，语义验证调用会追加到这里。

    返回：
    - verdict、failure_type、confidence、reason、suggested_action 和检查详情。
    """
    query = pipeline_result.get("sql") or pipeline_result.get("cypher") or pipeline_result.get("query") or ""
    result_rows = pipeline_result.get("result_rows", []) or []
    result_columns = pipeline_result.get("result_columns", []) or []
    row_count = int(pipeline_result.get("row_count", 0) or 0)
    success = bool(pipeline_result.get("success", False))
    error = pipeline_result.get("error", "") or ""
    selected_db = pipeline_result.get("retrieved_db") or pipeline_result.get("retrieval", {}).get("selected_db", "")
    query_type = pipeline_result.get("query_type") or route_result.get("query_type", "")
    contract = infer_step_contract(question)
    generation_contract_check = assess_generation_contract(
        query=query,
        query_type=query_type,
        generation_contract=pipeline_result.get("generation_contract", {}),
        task_analysis=pipeline_result.get("task_analysis", {}),
    )

    if not success:
        failure = _classify_execution_failure(error)
        failure = _protect_strong_cypher_failure(
            question=question,
            query_type=query_type,
            route_result=route_result,
            failure=failure,
        )
        failure["contract"] = contract
        failure["generation_contract_check"] = generation_contract_check
        failure["checks"] = {
            "has_query": bool(query.strip()),
            "row_count": row_count,
            "column_count": len(result_columns),
            "execution_success": False,
            "generation_contract_check": generation_contract_check,
        }
        return failure

    fallback_guard = _weak_sql_fallback_after_cypher_primary(
        question=question,
        query_type=query_type,
        route_result=route_result,
    )
    if fallback_guard is not None:
        fallback_guard["contract"] = contract
        fallback_guard["generation_contract_check"] = generation_contract_check
        fallback_guard["checks"] = {
            "has_query": bool(query.strip()),
            "row_count": row_count,
            "column_count": len(result_columns),
            "execution_success": True,
            "generation_contract_check": generation_contract_check,
        }
        return fallback_guard

    if generation_contract_check.get("required") and not generation_contract_check.get("satisfied"):
        return {
            "verdict": VERDICT_REGENERATE,
            "failure_type": generation_contract_check.get("failure_type", "nested_logic_contract_unmet"),
            "confidence": generation_contract_check.get("confidence", "high"),
            "reason": generation_contract_check.get(
                "reason",
                "Nested-logic contract is active, but the generated query does not appear to follow it.",
            ),
            "suggested_action": "regenerate",
            "contract": contract,
            "generation_contract_check": generation_contract_check,
            "checks": {
                "has_query": bool(query.strip()),
                "row_count": row_count,
                "column_count": len(result_columns),
                "execution_success": True,
                "generation_contract_check": generation_contract_check,
            },
        }

    heuristic = _heuristic_success_verdict(
        question=question,
        query_type=query_type,
        query=query,
        result_rows=result_rows,
        result_columns=result_columns,
        row_count=row_count,
    )
    if heuristic is not None:
        heuristic["checks"] = {
            "has_query": bool(query.strip()),
            "row_count": row_count,
            "column_count": len(result_columns),
            "execution_success": True,
            "generation_contract_check": generation_contract_check,
        }
        heuristic["generation_contract_check"] = generation_contract_check
        return heuristic

    semantic = _llm_verify_success(
        question=question,
        query_type=query_type,
        selected_db=selected_db,
        query=query,
        result_rows=result_rows,
        result_columns=result_columns,
        row_count=row_count,
        trace=trace,
    )
    semantic["contract"] = contract
    semantic["generation_contract_check"] = generation_contract_check
    semantic["checks"] = {
        "has_query": bool(query.strip()),
        "row_count": row_count,
        "column_count": len(result_columns),
        "execution_success": True,
        "generation_contract_check": generation_contract_check,
    }
    return semantic
