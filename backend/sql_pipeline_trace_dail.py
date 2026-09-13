import time
from typing import Any

from .generation_feedback import format_generation_feedback
from .pipeline_compat import apply_sql_legacy_aliases
from .sqlite_utils import connect_sqlite
from .sql_pipeline_roles import (
    extract_schema,
    execute_sql_with_repair,
    generate_sql_query,
    prepare_sql_schema_context,
    retrieve_sql_candidates,
)

DEFAULT_SELECTOR_TYPE = "masked_question"
DEFAULT_EXAMPLE_NUM = 5
DEFAULT_MAX_SEQ_LEN = 4096
DEFAULT_MAX_ANS_LEN = 200
DEFAULT_SCOPE_FACTOR = 100
DEFAULT_TOKENIZER_MODEL = "gpt-3.5-turbo"

# 中文函数说明补充索引：
# - _now_ms()：返回毫秒时间，用于 pipeline trace 分阶段耗时。
# - _add_sql_generation_feedback(prompt, feedback_text)：把 Coordinator retry feedback 注入 SQL prompt 的 SELECT 前。
# - run_sql_pipeline(question, db_path, generation_feedback, route_proposal)：正式 SQL DAIL 单步链路入口，包含 retrieval、schema context、generation、execution、repair 和 trace 字段。


def _now_ms() -> float:
    """返回当前高精度时间，单位毫秒。"""
    return time.perf_counter() * 1000


def _add_sql_generation_feedback(prompt: str, feedback_text: str) -> str:
    feedback_block = "/* " + feedback_text.replace("*/", "") + " */"
    marker = "\nSELECT "
    if marker in prompt:
        head, tail = prompt.rsplit(marker, 1)
        return head + "\n\n" + feedback_block + marker + tail
    return prompt + "\n\n" + feedback_block


def run_sql_pipeline(
    question: str,
    db_path: str | None = None,
    *,
    selector_type: str = DEFAULT_SELECTOR_TYPE,
    example_num: int = DEFAULT_EXAMPLE_NUM,
    use_retrieval: bool = True,
    use_fix: bool = True,
    include_rule: bool = True,
    use_value_hints: bool = True,
    force_bow_selector: bool = False,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    max_ans_len: int = DEFAULT_MAX_ANS_LEN,
    scope_factor: int = DEFAULT_SCOPE_FACTOR,
    tokenizer_model: str = DEFAULT_TOKENIZER_MODEL,
    generation_feedback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """SQL DAIL 单步主链路入口。

    参数：
    - question：用户自然语言问题。
    - db_path：可选，指定 SQLite 数据库路径；为空时使用检索模块自动定位候选库。
    - selector_type：DAIL 示例选择策略。
    - example_num：放入 prompt 的示例数量。
    - use_retrieval：是否启用数据库检索；为 False 时通常配合 `db_path` 使用。
    - use_fix：是否启用执行失败后的修复。
    - include_rule：是否在 DAIL prompt 中加入生成规则。
    - force_bow_selector：是否强制使用 BOW 示例选择器。
    - max_seq_len：prompt 最大序列长度预算。
    - max_ans_len：答案最大长度预算。
    - scope_factor：DAIL 示例召回范围因子；SQL schema 本身不做 table/column 级裁剪。
    - tokenizer_model：用于估算 prompt token 的 tokenizer 名称。

    返回：
    - 统一 SQL pipeline record，包含检索、完整 schema、生成、执行、修复、耗时和兼容字段。

    重要约定：
    - SQL 链路保留完整 CREATE TABLE schema。此前实验表明 schema 裁剪会降低最终 EX，因此不要在此链路加入表/列级 schema pruning。
    """
    started_ms = _now_ms()
    record: dict[str, Any] = {
        "question": question,
        "success": False,
        "result": "",
        "result_rows": [],
        "result_columns": [],
        "row_count": 0,
        "query": "",
        "query_type": "sql",
        "sql": "",
        "error": "",
        "retries": 0,
        "selected_db": "",
        "retrieved_db": "",
        "used_db_path": "",
        "schema_full": "",
        "working_schema": "",
        "schema_linked": "",
        "value_hints": "",
        "cot_raw": "",
        "retry_history": [],
        "llm_trace": [],
        "retrieval": {
            "method": "specified" if db_path else "vector",
            "selected_db": "",
            "selected_path": "",
            "score": None,
            "top_candidates": [],
        },
        "timing": {
            "retrieval_ms": 0,
            "schema_extraction_ms": 0,
            "schema_linking_ms": 0,
            "generation_ms": 0,
            "execution_ms": 0,
            "retry_ms": 0,
            "total_ms": 0,
        },
        "selector_type": selector_type,
        "example_num": example_num,
        "question_masked": "",
        "examples": [],
        "prompt_tokens": 0,
        "task_analysis": {},
        "nested_logic_guidance": "",
        "generation_contract": {},
        "generation_feedback": generation_feedback or {},
        "use_value_hints": use_value_hints,
    }

    retrieval_result = retrieve_sql_candidates(
        question,
        db_path=db_path,
        use_retrieval=use_retrieval,
    )
    db_candidates = retrieval_result["db_candidates"]
    record["retrieval"].update(retrieval_result["retrieval"])
    record["timing"]["retrieval_ms"] = retrieval_result["retrieval_ms"]

    for candidate in db_candidates:
        current_db_id = candidate["db_id"]
        current_db_path = candidate["db_path"]
        conn = connect_sqlite(current_db_path)
        try:
            schema_context = prepare_sql_schema_context(
                conn=conn,
                question=question,
                current_db_path=current_db_path,
                selector_type=selector_type,
                example_num=example_num,
                include_rule=include_rule,
                force_bow_selector=force_bow_selector,
                max_seq_len=max_seq_len,
                max_ans_len=max_ans_len,
                scope_factor=scope_factor,
                tokenizer_model=tokenizer_model,
                trace=record["llm_trace"],
                use_value_hints=use_value_hints,
            )
            record.update(
                {
                    "schema_full": schema_context["schema_full"],
                    "working_schema": schema_context["working_schema"],
                    "schema_linked": schema_context["schema_linked"],
                    "value_hints": schema_context.get("value_hints", ""),
                    "question_masked": schema_context["question_masked"],
                    "examples": schema_context["examples_serialized"],
                    "prompt_tokens": schema_context["prompt_tokens"],
                    "task_analysis": schema_context.get("task_analysis", {}),
                    "nested_logic_guidance": schema_context.get("nested_logic_guidance", ""),
                    "generation_contract": {
                        "query_strategy": schema_context.get("task_analysis", {}).get("query_strategy", "single_query"),
                        "recommended_generation_strategy": schema_context.get("task_analysis", {}).get("recommended_generation_strategy", "direct_query"),
                        "nested_logic_guidance": schema_context.get("nested_logic_guidance", ""),
                    },
                }
            )
            record["timing"]["schema_extraction_ms"] = schema_context["schema_extraction_ms"]
            record["timing"]["schema_linking_ms"] = schema_context["schema_linking_ms"]

            feedback_text = format_generation_feedback(generation_feedback, "sql")
            generation_prompt = schema_context["prompt"]
            if feedback_text:
                generation_prompt = _add_sql_generation_feedback(generation_prompt, feedback_text)
                record["generation_feedback"] = generation_feedback or {}

            generation_result = generate_sql_query(
                generation_prompt,
                trace=record["llm_trace"],
            )
            record["query"] = generation_result["query"]
            record["query_type"] = generation_result["query_type"]
            record["sql"] = generation_result["sql"]
            record["cot_raw"] = generation_result["cot_raw"]
            record["timing"]["generation_ms"] = generation_result["generation_ms"]

            execution_result = execute_sql_with_repair(
                conn=conn,
                question=question,
                schema_sql=schema_context["schema_sql"],
                examples=schema_context["examples"],
                sql=generation_result["sql"],
                trace=record["llm_trace"],
                include_rule=include_rule,
                use_fix=use_fix,
            )
            record["sql"] = execution_result["sql"]
            record["query"] = execution_result["sql"]
            record["error"] = execution_result["error"]

            if execution_result["success"]:
                record.update(
                    {
                        "success": True,
                        "result": execution_result["result"],
                        "result_rows": execution_result["result_rows"],
                        "result_columns": execution_result["result_columns"],
                        "row_count": execution_result["row_count"],
                        "retries": execution_result["retries"],
                        "retry_history": execution_result["retry_history"],
                        "selected_db": current_db_id,
                        "retrieved_db": current_db_id,
                        "used_db_path": current_db_path,
                    }
                )
                record["retrieval"]["selected_db"] = current_db_id
                record["retrieval"]["selected_path"] = current_db_path
                record["timing"]["execution_ms"] = execution_result["execution_ms"]
                record["timing"]["retry_ms"] = execution_result["retry_ms"]
                record["timing"]["total_ms"] = round(_now_ms() - started_ms, 2)
                return apply_sql_legacy_aliases(record)

            if execution_result["should_try_next_candidate"]:
                continue

            record.update(
                {
                    "result": execution_result["result"],
                    "retries": execution_result["retries"],
                    "retry_history": execution_result["retry_history"],
                    "selected_db": current_db_id,
                    "retrieved_db": current_db_id,
                    "used_db_path": current_db_path,
                }
            )
            record["retrieval"]["selected_db"] = current_db_id
            record["retrieval"]["selected_path"] = current_db_path
            record["timing"]["execution_ms"] = execution_result["execution_ms"]
            record["timing"]["retry_ms"] = execution_result["retry_ms"]
            record["timing"]["total_ms"] = round(_now_ms() - started_ms, 2)
            return apply_sql_legacy_aliases(record)
        finally:
            conn.close()

    record["result"] = "Execution failed: no candidate database produced a valid answer."
    record["timing"]["total_ms"] = round(_now_ms() - started_ms, 2)
    return apply_sql_legacy_aliases(record)
