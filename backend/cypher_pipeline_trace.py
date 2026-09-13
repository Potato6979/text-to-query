import time
from typing import Any

from neo4j import GraphDatabase

from .config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER
from .pipeline_compat import apply_cypher_legacy_aliases
from .cypher_pipeline_roles import (
    execute_cypher,
    execute_cypher_with_repair,
    execute_cypher_without_repair,
    generate_cypher_query,
    normalize_schema_name,
    prepare_cypher_schema_context,
    retrieve_cypher_candidates,
)
from .schema_agents import extract_graph_schema

# 中文函数说明补充索引：
# - _now_ms()：返回毫秒时间，用于 Cypher pipeline trace 分阶段耗时。
# - run_cypher_pipeline(question, db_name, schema_format, generation_feedback)：正式 Cypher 单步链路入口，包含 graph retrieval、schema context、generation、execution/repair 和 trace 字段。


def _now_ms() -> float:
    """返回当前高精度时间，单位毫秒。"""
    return time.perf_counter() * 1000


def run_cypher_pipeline(
    question: str,
    db_name: str | None = None,
    schema_format: str = "cypher_create",
    generation_feedback: dict[str, Any] | None = None,
    schema_grounding_mode: str = "llm_filter",
    use_repair: bool = True,
) -> dict[str, Any]:
    """Cypher 单步主链路入口。

    参数：
    - question：用户自然语言问题。
    - db_name：可选，指定 Neo4j 数据库名；为空时自动检索候选图库。
    - schema_format：图 schema 格式，目前常用 `cypher_create`。
    - schema_grounding_mode：schema grounding 策略，默认 `llm_filter` 与正式系统一致。
    - use_repair：是否启用执行反馈修复，默认开启。

    返回：
    - 统一 Cypher pipeline record，包含检索、schema、生成、执行、修复、耗时和兼容字段。
    """
    started_ms = _now_ms()
    normalized_format = normalize_schema_name(schema_format)
    record: dict[str, Any] = {
        "question": question,
        "schema_format": normalized_format,
        "schema_grounding_mode": schema_grounding_mode,
        "use_repair": use_repair,
        "success": False,
        "result": "",
        "result_rows": [],
        "result_columns": [],
        "row_count": 0,
        "query": "",
        "query_type": "cypher",
        "cypher": "",
        "error": "",
        "retries": 0,
        "selected_db": "",
        "retrieved_db": "",
        "schema_full": "",
        "working_schema": "",
        "schema_filtered": "",
        "cot_raw": "",
        "task_analysis": {},
        "nested_logic_guidance": "",
        "generation_contract": {},
        "generation_feedback": generation_feedback or {},
        "retry_history": [],
        "llm_trace": [],
        "retrieval": {
            "method": "specified" if db_name else "vector",
            "selected_db": "",
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
    }

    retrieval_result = retrieve_cypher_candidates(question, db_name=db_name)
    db_candidates = retrieval_result["db_candidates"]
    record["retrieval"].update(retrieval_result["retrieval"])
    record["timing"]["retrieval_ms"] = retrieval_result["retrieval_ms"]

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        for candidate in db_candidates:
            current_db = candidate["db_name"]

            schema_context = prepare_cypher_schema_context(
                driver=driver,
                question=question,
                current_db=current_db,
                schema_format=normalized_format,
                trace=record["llm_trace"],
                schema_grounding_mode=schema_grounding_mode,
            )
            record["schema_full"] = schema_context["schema_full"]
            record["working_schema"] = schema_context["working_schema"]
            record["schema_filtered"] = schema_context["schema_filtered"]
            relationships_data = schema_context.get("relationships_data", [])
            record["timing"]["schema_extraction_ms"] = schema_context["schema_extraction_ms"]
            record["timing"]["schema_linking_ms"] = schema_context["schema_linking_ms"]

            generation_result = generate_cypher_query(
                question,
                schema_context["schema_filtered"],
                trace=record["llm_trace"],
                generation_feedback=generation_feedback,
            )
            record["query"] = generation_result["query"]
            record["query_type"] = generation_result["query_type"]
            record["cypher"] = generation_result["cypher"]
            record["cot_raw"] = generation_result["cot_raw"]
            record["task_analysis"] = generation_result.get("task_analysis", {})
            record["nested_logic_guidance"] = generation_result.get("nested_logic_guidance", "")
            record["generation_feedback"] = generation_result.get("generation_feedback", generation_feedback or {})
            record["generation_contract"] = {
                "query_strategy": record["task_analysis"].get("query_strategy", "single_query"),
                "recommended_generation_strategy": record["task_analysis"].get("recommended_generation_strategy", "direct_query"),
                "nested_logic_guidance": record["nested_logic_guidance"],
            }
            record["timing"]["generation_ms"] = generation_result["generation_ms"]

            if use_repair:
                execution_result = execute_cypher_with_repair(
                    driver=driver,
                    question=question,
                    filtered_schema=schema_context["schema_filtered"],
                    cypher=generation_result["cypher"],
                    current_db=current_db,
                    relationships_data=relationships_data,
                    trace=record["llm_trace"],
                )
            else:
                execution_result = execute_cypher_without_repair(
                    driver=driver,
                    cypher=generation_result["cypher"],
                    current_db=current_db,
                )
            record["cypher"] = execution_result["cypher"]
            record["query"] = execution_result["cypher"]
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
                        "selected_db": current_db,
                        "retrieved_db": current_db,
                    }
                )
                record["retrieval"]["selected_db"] = current_db
                record["timing"]["execution_ms"] = execution_result["execution_ms"]
                record["timing"]["retry_ms"] = execution_result["retry_ms"]
                record["timing"]["total_ms"] = round(_now_ms() - started_ms, 2)
                return apply_cypher_legacy_aliases(record)

            if execution_result["should_try_next_candidate"]:
                continue

            record.update(
                {
                    "result": execution_result["result"],
                    "retries": execution_result["retries"],
                    "retry_history": execution_result["retry_history"],
                    "selected_db": current_db,
                    "retrieved_db": current_db,
                }
            )
            record["retrieval"]["selected_db"] = current_db
            record["timing"]["execution_ms"] = execution_result["execution_ms"]
            record["timing"]["retry_ms"] = execution_result["retry_ms"]
            record["timing"]["total_ms"] = round(_now_ms() - started_ms, 2)
            return apply_cypher_legacy_aliases(record)
    finally:
        driver.close()

    record["result"] = "Execution failed: no candidate graph database produced a valid answer."
    record["timing"]["total_ms"] = round(_now_ms() - started_ms, 2)
    return apply_cypher_legacy_aliases(record)
