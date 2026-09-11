import time
from typing import Any, Callable

# 中文函数说明索引：
# - _now_ms()：返回毫秒时间，用于统计 execution 和 repair 耗时。
# - run_execution_with_repair(...)：统一 SQL/Cypher 执行修复循环；失败时调用 fix_fn，记录 retry_history，直到成功、换候选或重试耗尽。


def _now_ms() -> float:
    """返回当前高精度时间，单位毫秒。"""
    return time.perf_counter() * 1000


def run_execution_with_repair(
    *,
    initial_query: str,
    max_retries: int,
    execute_fn: Callable[[str], dict[str, Any]],
    fix_fn: Callable[[str, str, int], tuple[str, str]],
    should_try_next_candidate_fn: Callable[[str, int], bool],
    query_key: str,
    bad_query_key: str,
    fixed_query_key: str,
    terminal_failure_message: Callable[[str, int], str],
) -> dict[str, Any]:
    """执行查询，并在失败时调用修复函数重试。

    参数：
    - initial_query：第一次要执行的 SQL/Cypher 查询。
    - max_retries：最多允许调用修复 Agent 的次数。
    - execute_fn：真正执行查询的函数，输入查询字符串，返回统一执行结果。
    - fix_fn：修复函数，输入坏查询、错误信息和第几次修复，返回修复后的查询和原始修复输出。
    - should_try_next_candidate_fn：判断当前错误是否应放弃当前数据库候选、尝试下一个候选。
    - query_key：成功或失败结果中保存最终查询的字段名，例如 `sql` 或 `cypher`。
    - bad_query_key：retry_history 中保存坏查询的字段名。
    - fixed_query_key：retry_history 中保存修复后查询的字段名。
    - terminal_failure_message：重试耗尽后生成最终失败说明的函数。
    """
    execution_total = 0.0
    retry_total = 0.0
    retry_history: list[dict[str, Any]] = []
    current_query = initial_query
    last_error = ""

    for attempt in range(max_retries + 1):
        execution_started = _now_ms()
        execution = execute_fn(current_query)
        execution_total += _now_ms() - execution_started

        if execution["success"]:
            return {
                "success": True,
                "result": execution["result_text"],
                "result_rows": execution["rows"],
                "result_columns": execution["columns"],
                "row_count": execution["row_count"],
                "error": "",
                "retries": attempt,
                "retry_history": retry_history,
                query_key: current_query,
                "should_try_next_candidate": False,
                "execution_ms": round(execution_total, 2),
                "retry_ms": round(retry_total, 2),
            }

        error_message = execution["error"] or "Unknown execution error"
        last_error = error_message
        if should_try_next_candidate_fn(error_message, attempt):
            return {
                "success": False,
                "result": "",
                "result_rows": [],
                "result_columns": [],
                "row_count": 0,
                "error": error_message,
                "retries": attempt,
                "retry_history": retry_history,
                query_key: current_query,
                "should_try_next_candidate": True,
                "execution_ms": round(execution_total, 2),
                "retry_ms": round(retry_total, 2),
            }

        if attempt >= max_retries:
            return {
                "success": False,
                "result": terminal_failure_message(error_message, max_retries),
                "result_rows": [],
                "result_columns": [],
                "row_count": 0,
                "error": error_message,
                "retries": attempt,
                "retry_history": retry_history,
                query_key: current_query,
                "should_try_next_candidate": False,
                "execution_ms": round(execution_total, 2),
                "retry_ms": round(retry_total, 2),
            }

        retry_started = _now_ms()
        fixed_query, fix_raw = fix_fn(current_query, error_message, attempt + 1)
        retry_total += _now_ms() - retry_started
        retry_history.append(
            {
                "attempt": attempt + 1,
                bad_query_key: current_query,
                "error": error_message,
                fixed_query_key: fixed_query,
                "cot_raw": fix_raw,
            }
        )
        current_query = fixed_query

    return {
        "success": False,
        "result": terminal_failure_message(last_error or "Unknown execution error", max_retries),
        "result_rows": [],
        "result_columns": [],
        "row_count": 0,
        "error": last_error,
        "retries": max_retries,
        "retry_history": retry_history,
        query_key: current_query,
        "should_try_next_candidate": False,
        "execution_ms": round(execution_total, 2),
        "retry_ms": round(retry_total, 2),
    }
