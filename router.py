# router.py -- Routing Agent public entrypoint.
# The implementation is split into evidence, scoring, judge, and guard modules.

from typing import Any

from routing_common import (
    ALL_QUERY_TYPES,
    GENERIC_ENTITY_TOKENS,
    GRAPH_SURFACE_HINTS,
    INTENT_KEYWORDS,
    MQL_SURFACE_HINTS,
    QUERY_TYPES,
    ROUTING_STOPWORDS,
    SQL_SURFACE_HINTS,
    TOP_K_CANDIDATES,
    VECTOR_SURFACE_HINTS,
    contains_any as _contains_any,
    load_schema_indexes as _load_schema_indexes,
    meaningful_tokens as _meaningful_tokens,
    normalize_text as _normalize_text,
    parse_schema_summary_terms as _parse_schema_summary_terms,
    tokenize as _tokenize,
)
from routing_evidence import build_route_evidence, extract_entity_hints, extract_intent_signals
from routing_guard import fallback_route as _fallback_route
from routing_guard import guard_route, keyword_fallback as _keyword_fallback
from routing_judge import call_llm, judge_route_with_llm
from routing_plan import build_keyword_route_plan, build_route_plan
from routing_scorer import confidence_from_scores as _confidence_from_scores
from routing_scorer import score_modalities
from task_mode_analyzer import analyze_task_mode


def route_query_v2(question: str, schema_summary: str = "", context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Evidence-based routing flow: evidence extraction, scoring, LLM judge, guard calibration."""
    signals = extract_intent_signals(question)
    entity_hints = extract_entity_hints(question, schema_summary=schema_summary)
    scored = score_modalities(question, signals, entity_hints)
    evidence = build_route_evidence(question, signals, entity_hints, scored)
    try:
        judge_result = judge_route_with_llm(question, evidence)
    except Exception:
        return _fallback_route(
            question,
            evidence,
            reason="LLM judge failed, fallback to rule-based modality scoring.",
            uncertainty_source="judge_parse_failure",
        )
    return guard_route(judge_result, evidence, context=context)


def route_query(question: str, schema_summary: str = "", context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Current public routing entrypoint used by Coordinator."""
    result = route_query_v2(question, schema_summary=schema_summary, context=context)
    result.setdefault("query_type", "sql")
    result.setdefault("confidence", "low")
    result.setdefault("reason", "路由结果缺失，已回退。")
    return result


def evaluate_router(test_cases: list[dict[str, str]]) -> dict[str, Any]:
    total = len(test_cases)
    correct = 0
    results = []

    for case in test_cases:
        question = case["question"]
        expected = case["expected"]

        result = route_query(question)
        predicted = result["query_type"]
        is_correct = predicted == expected
        if is_correct:
            correct += 1

        results.append(
            {
                "question": question,
                "expected": expected,
                "predicted": predicted,
                "correct": is_correct,
                "confidence": result.get("confidence"),
                "reason": result.get("reason"),
                "task_mode": result.get("task_mode"),
                "candidates": [candidate.get("query_type") for candidate in result.get("candidates", [])],
            }
        )

        status = "✓" if is_correct else "✗"
        print(f"  {status} [{expected}→{predicted}] {question[:60]}")

    accuracy = correct / total * 100 if total else 0.0
    print(f"\n路由准确率：{correct}/{total} ({accuracy:.1f}%)")

    return {
        "accuracy": round(accuracy, 1),
        "correct": correct,
        "total": total,
        "results": results,
    }


if __name__ == "__main__":
    test_cases = [
        {"question": "How many singers are there?", "expected": "sql"},
        {"question": "What is the average age of all singers from France?", "expected": "sql"},
        {"question": "Find all players who scored goals in the France 2019 tournament.", "expected": "cypher"},
        {"question": "Who are the coaches of squads that participated in Sweden 1995?", "expected": "cypher"},
        {"question": "Find all orders where the items array contains a product with price over 100.", "expected": "mql"},
        {"question": "Find articles semantically similar to this paper about machine learning.", "expected": "vector"},
    ]
    print("=" * 60)
    print("路由层评估（V1 分层混合路由）")
    print("=" * 60)
    evaluate_router(test_cases)
