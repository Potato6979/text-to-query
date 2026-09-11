import pickle
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pipeline_compat import (
    get_pipeline_query,
    get_pipeline_selected_db,
    get_pipeline_working_schema,
)
from pydantic import BaseModel, Field

from coordinator import run_coordinated_query
from config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER
from cypher_pipeline_trace import extract_graph_schema, run_cypher_pipeline
from paired_benchmark_resources import (
    paired_cypher_metadata_records,
    paired_sql_metadata_records,
    resolve_actual_cypher_database,
)
from sqlite_utils import connect_sqlite
from sql_pipeline_trace_dail import extract_schema, run_sql_pipeline

# 中文函数说明补充索引：
# - _pipeline_for(query_type)：按路由模态选择 SQL/Cypher pipeline。
# - _query_from_pipeline(result)：从兼容字段中读取最终 query。
# - _build_dev_response(question, coordinator_result)：构造开发模式完整 trace，包含 route/retrieval/schema/generation/execution/verification/coordinator。
# - _build_user_response(question, coordinator_result)：构造用户模式精简响应，保留最终答案、query、结果表和 runtime_decision。
# - query(req)：/api/query 主入口，调用 run_coordinated_query 并按 mode 返回结果。
# - health()：健康检查 API，检查 schema index 和 Neo4j 连接。
# - databases()：列出当前可检索 SQL/Cypher 资源，供前端或调试查看。

app = FastAPI(title="Text-to-Query API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


SAFE_TERMINAL_STATUSES = {
    "blocked",
    "missing_input",
    "ambiguous_soft_context",
    "unresolved_soft_context",
}


def _load_pickle_list(path: str) -> list[dict[str, Any]]:
    """Load a pickled metadata list if it exists."""
    meta_path = Path(path)
    if not meta_path.exists():
        return []
    with meta_path.open("rb") as handle:
        raw = pickle.load(handle)
    return raw if isinstance(raw, list) else []


def _sql_metadata_records() -> list[dict[str, Any]]:
    return _load_pickle_list("schema_index/metadata.pkl") + paired_sql_metadata_records()


def _cypher_metadata_records() -> list[dict[str, Any]]:
    return _load_pickle_list("schema_index/cypher_metadata.pkl") + paired_cypher_metadata_records()


def _normalize_resource_rows(rows: Any) -> list[list[Any]]:
    """Convert row payloads from SQL/Cypher pipelines into table-friendly arrays."""
    if not isinstance(rows, list):
        return []
    normalized: list[list[Any]] = []
    for row in rows:
        if isinstance(row, dict):
            normalized.append(list(row.values()))
        elif isinstance(row, (list, tuple)):
            normalized.append(list(row))
        else:
            normalized.append([row])
    return normalized


def _sql_cypher_schema_summary() -> str:
    """Build the API-facing schema summary from the current SQL/Cypher indexes.

    The old demo summary only mentioned a legacy SQL database and one graph
    database. The current system routes over indexed Spider SQL databases,
    Mind-the-Query graph databases, and paired benchmark split resources, so
    the API should expose the same scope to the Routing Agent.
    """
    sql_items: dict[str, set[str]] = {}
    for item in _sql_metadata_records():
        db_id = str(item.get("db_id", "")).strip()
        table = str(item.get("table", "")).strip()
        if db_id and table:
            sql_items.setdefault(db_id, set()).add(table)

    cypher_nodes: dict[str, set[str]] = {}
    cypher_rels: dict[str, set[str]] = {}
    for item in _cypher_metadata_records():
        db_name = str(item.get("db_name", "")).strip()
        label = str(item.get("label", "")).strip()
        item_type = str(item.get("type", "")).strip()
        if not db_name or not label:
            continue
        if item_type == "rel":
            cypher_rels.setdefault(db_name, set()).add(label)
        else:
            cypher_nodes.setdefault(db_name, set()).add(label)

    lines = [
        "当前演示系统范围：关系数据库 SQL 与图数据库 Cypher。",
        "系统包含自动路由、任务模式分析、模式感知候选生成、SQL/Cypher 单源查询、多步执行、桥接解析器（Bridge Resolver）、执行验证（Verification）和最终答案合成。",
        "",
        "【SQL 关系数据库资源】",
    ]
    for db_id in sorted(sql_items):
        tables = ", ".join(sorted(sql_items[db_id])[:12])
        lines.append(f"- {db_id}: 表 {tables}")

    lines.append("")
    lines.append("【Cypher 图数据库资源】")
    for db_name in sorted(set(cypher_nodes) | set(cypher_rels)):
        nodes = ", ".join(sorted(cypher_nodes.get(db_name, set()))[:12]) or "无节点标签记录"
        rels = ", ".join(sorted(cypher_rels.get(db_name, set()))[:10]) or "无关系类型记录"
        lines.append(f"- {db_name}: 节点 {nodes}; 关系 {rels}")
    return "\n".join(lines)


API_SCHEMA_SUMMARY = _sql_cypher_schema_summary()


SYSTEM_COMPONENTS = [
    "Routing Agent",
    "Task Mode Analyzer",
    "Schema-Aware Proposal",
    "SQL Agent",
    "Cypher Agent",
    "Multi-step Planner",
    "Multi-step Runtime",
    "Bridge Resolver",
    "Verification Agent",
    "Final Answer Synthesizer",
]


class QueryRequest(BaseModel):
    """API 查询请求体。

    参数：
    - question：用户输入的自然语言问题。
    - mode：返回模式，`dev` 返回完整调试 trace，`user` 返回面向用户的精简结果。
    - time_budget_ms：可选，覆盖本次 Coordinator 的时间预算，单位毫秒。
    - token_budget：可选，覆盖本次 Coordinator 的 token 估算预算。
    """

    question: str = Field(..., min_length=1)
    mode: str = "dev"
    time_budget_ms: int | None = None
    token_budget: int | None = None
    enable_multi_step_runtime: bool | None = None
    enable_bridge_resolver: bool | None = None


def _pipeline_for(query_type: str):
    """根据查询模态返回对应的 pipeline 函数。

    参数：
    - query_type：路由层选择的查询类型，例如 `sql`、`cypher`。
    """
    return {
        "sql": run_sql_pipeline,
        "cypher": run_cypher_pipeline,
    }.get(query_type, run_sql_pipeline)


def _query_from_pipeline(result: dict[str, Any]) -> str:
    """从 pipeline 结果中读取最终查询语句。

    参数：
    - result：SQL/Cypher pipeline 返回的字典，可能同时包含新旧字段名。
    """
    return get_pipeline_query(result)


def _final_status(coordinator_result: dict[str, Any]) -> str:
    final_answer = coordinator_result.get("final_answer", {})
    coordinator = coordinator_result.get("coordinator", {})
    verification = coordinator_result.get("verification", {})
    return (
        str(final_answer.get("status") or "")
        or str(final_answer.get("runtime_status") or "")
        or str(verification.get("runtime_status") or "")
        or str(coordinator.get("status") or "")
    )


def _status_label(status: str, success: bool = False) -> str:
    labels = {
        "succeeded": "执行完成",
        "answered": "已回答",
        "failed": "执行失败",
        "blocked": "安全终止：已阻断不可靠依赖",
        "missing_input": "安全终止：缺少必要输入",
        "ambiguous_soft_context": "安全终止：映射存在歧义",
        "unresolved_soft_context": "安全终止：未找到可靠映射",
        "budget_exceeded": "预算超限",
        "not_started": "未开始执行",
    }
    if status in labels:
        return labels[status]
    if success:
        return "执行完成"
    return status or "未返回状态"


def _is_safe_terminal(status: str) -> bool:
    return status.lower() in SAFE_TERMINAL_STATUSES


def _result_rows_and_columns(
    coordinator_result: dict[str, Any],
    pipeline_result: dict[str, Any],
) -> tuple[list[list[Any]], list[str]]:
    final_answer = coordinator_result.get("final_answer", {})
    rows = pipeline_result.get("result_rows", [])
    columns = pipeline_result.get("result_columns", [])
    if not rows and isinstance(final_answer.get("synthesis"), dict):
        raw_text = str(final_answer.get("raw_result_text") or "")
        if raw_text and not _is_safe_terminal(_final_status(coordinator_result)):
            return [[raw_text]], ["answer"]
    return _normalize_resource_rows(rows), [str(column) for column in columns]


def _build_dev_response(
    question: str,
    coordinator_result: dict[str, Any],
) -> dict[str, Any]:
    """构造开发调试模式响应。

    参数：
    - question：原始自然语言问题。
    - coordinator_result：`run_coordinated_query(...)` 返回的完整状态结果。

    返回：
    - 包含路由、检索、schema、生成、执行、验证、Coordinator 和耗时信息的调试响应。
    """
    route_result = coordinator_result.get("route", {})
    pipeline_result = coordinator_result.get("pipeline", {})
    verification = coordinator_result.get("verification", {})
    coordinator = coordinator_result.get("coordinator", {})
    timing = coordinator_result.get("timing", {})
    retrieval = pipeline_result.get("retrieval", {})
    final_status = _final_status(coordinator_result)
    result_rows, result_columns = _result_rows_and_columns(coordinator_result, pipeline_result)
    return {
        "question": question,
        "system": {
            "scope": "SQL + Cypher",
            "components": SYSTEM_COMPONENTS,
            "safe_terminal_statuses": sorted(SAFE_TERMINAL_STATUSES),
        },
        "pipeline": {
            "route": route_result,
            "retrieval": {
                "method": retrieval.get("method", "schema_retrieval"),
                "top3_candidates": [
                    candidate.get("db_id") or candidate.get("db_name") or ""
                    for candidate in retrieval.get("top_candidates", [])[:3]
                ],
                "candidates": retrieval.get("top_candidates", [])[:3],
                "selected_db": get_pipeline_selected_db(pipeline_result) or retrieval.get("selected_db") or "",
                "score": retrieval.get("score"),
            },
            "schema_linking": {
                "full_schema_preview": pipeline_result.get("schema_full", "")[:1200],
                "full_schema": pipeline_result.get("schema_full", ""),
                "linked_schema": get_pipeline_working_schema(pipeline_result),
            },
            "generation": {
                "query": _query_from_pipeline(pipeline_result),
                "query_type": pipeline_result.get("query_type", route_result["query_type"]),
                "cot_reasoning": pipeline_result.get("cot_raw", ""),
                "task_analysis": pipeline_result.get("task_analysis", {}),
                "nested_logic_guidance": pipeline_result.get("nested_logic_guidance", ""),
                "generation_contract": pipeline_result.get("generation_contract", {}),
                "generation_feedback": pipeline_result.get("generation_feedback", {}),
                "route_proposal": pipeline_result.get("route_proposal", {}),
                "proposal_target_resource": pipeline_result.get("proposal_target_resource", {}),
                "llm_trace": pipeline_result.get("llm_trace", []),
            },
            "execution": {
                "success": pipeline_result.get("success", False),
                "result_text": pipeline_result.get("result", ""),
                "result_rows": result_rows,
                "result_columns": result_columns,
                "error": pipeline_result.get("error", ""),
                "retries": pipeline_result.get("retries", 0),
                "retry_history": pipeline_result.get("retry_history", []),
                "row_count": pipeline_result.get("row_count", 0),
            },
            "verification": verification,
            "coordinator": coordinator,
        },
        "final_answer": {
            "status": final_status,
            "status_label": _status_label(final_status, coordinator_result.get("final_answer", {}).get("success", False)),
            "safe_terminal": _is_safe_terminal(final_status),
            "success": coordinator_result.get("final_answer", {}).get("success", False),
            "query": coordinator_result.get("final_answer", {}).get("query", _query_from_pipeline(pipeline_result)),
            "result_text": coordinator_result.get("final_answer", {}).get("result_text", pipeline_result.get("result", "")),
        },
        "runtime_decision": route_result.get("runtime_decision", {}),
        "timing": {
            "total_ms": round(timing.get("total_ms", 0)),
            "route_ms": round(timing.get("route_ms", 0)),
            "retrieval_ms": round(pipeline_result.get("timing", {}).get("retrieval_ms", 0)),
            "schema_extraction_ms": round(pipeline_result.get("timing", {}).get("schema_extraction_ms", 0)),
            "schema_linking_ms": round(pipeline_result.get("timing", {}).get("schema_linking_ms", 0)),
            "generation_ms": round(pipeline_result.get("timing", {}).get("generation_ms", 0)),
            "execution_ms": round(pipeline_result.get("timing", {}).get("execution_ms", 0)),
            "retry_ms": round(pipeline_result.get("timing", {}).get("retry_ms", 0)),
            "verification_ms": round(timing.get("verification_ms", 0)),
            "coordinator_ms": round(timing.get("coordinator_ms", 0)),
        },
    }


def _build_user_response(
    question: str,
    coordinator_result: dict[str, Any],
) -> dict[str, Any]:
    """构造面向普通用户的精简响应。

    参数：
    - question：原始自然语言问题。
    - coordinator_result：Coordinator 的完整执行结果。

    返回：
    - 只保留状态、失败类型、最终查询和结果表格等用户需要的信息。
    """
    route_result = coordinator_result.get("route", {})
    pipeline_result = coordinator_result.get("pipeline", {})
    final_status = _final_status(coordinator_result)
    result_rows, result_columns = _result_rows_and_columns(coordinator_result, pipeline_result)
    return {
        "question": question,
        "system": {
            "scope": "SQL + Cypher",
            "components": SYSTEM_COMPONENTS,
        },
        "status": final_status,
        "status_label": _status_label(final_status, coordinator_result.get("final_answer", {}).get("success", False)),
        "safe_terminal": _is_safe_terminal(final_status),
        "success": coordinator_result.get("final_answer", {}).get("success", False),
        "failure_type": coordinator_result.get("verification", {}).get("failure_type", ""),
        "query_type": route_result["query_type"],
        "task_mode": route_result.get("task_mode") or coordinator_result.get("coordinator", {}).get("mode", ""),
        "runtime_decision": route_result.get("runtime_decision", {}),
        "result_text": coordinator_result.get("final_answer", {}).get("result_text", pipeline_result.get("result", "")),
        "query": coordinator_result.get("final_answer", {}).get("query", _query_from_pipeline(pipeline_result)),
        "result_rows": result_rows,
        "result_columns": result_columns,
    }


@app.post("/api/query")
def query(req: QueryRequest):
    """自然语言查询主接口。

    参数：
    - req：请求体，包含用户问题、返回模式和可选预算覆盖值。

    流程：
    - 调用 Coordinator 完成路由、生成、执行、验证和回退。
    - 根据 `mode` 返回开发调试响应或用户精简响应。
    """
    coordinator_result = run_coordinated_query(
        req.question,
        schema_summary=API_SCHEMA_SUMMARY,
        time_budget_ms=req.time_budget_ms,
        token_budget=req.token_budget,
        enable_multi_step_runtime=req.enable_multi_step_runtime,
        enable_bridge_resolver=req.enable_bridge_resolver,
    )

    if req.mode == "user":
        return _build_user_response(req.question, coordinator_result)
    return _build_dev_response(req.question, coordinator_result)


@app.get("/api/health")
def health():
    """健康检查接口。

    返回：
    - API 是否可用、schema index 是否存在、Neo4j 是否可连接等状态。
    """
    checks: dict[str, Any] = {
        "api": "ok",
        "scope": "sql_cypher",
        "components": SYSTEM_COMPONENTS,
        "router_schema_summary": bool(API_SCHEMA_SUMMARY.strip()),
        "sql_index": Path("schema_index/faiss.index").exists(),
        "cypher_index": Path("schema_index/cypher_faiss.index").exists(),
    }
    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as session:
            session.run("RETURN 1").single()
        driver.close()
        checks["neo4j"] = "ok"
    except Exception as exc:
        checks["neo4j"] = f"error: {exc}"
    return checks


@app.get("/api/databases")
def databases():
    """列出当前可被系统检索到的数据库资源。

    返回：
    - SQL 数据库列表和 Cypher 图数据库列表。
    """
    sql_items: dict[str, dict[str, Any]] = {}
    cypher_items: dict[str, dict[str, Any]] = {}

    sql_meta = Path("schema_index/metadata.pkl")
    metadata = _sql_metadata_records()
    for item in metadata:
        entry = sql_items.setdefault(
            item["db_id"],
            {"id": item["db_id"], "type": "sql", "path": item["db_path"], "tables": set()},
        )
        entry["tables"].add(item["table"])

    cypher_meta = Path("schema_index/cypher_metadata.pkl")
    metadata = _cypher_metadata_records()
    for item in metadata:
        entry = cypher_items.setdefault(
            item["db_name"],
            {
                "id": item["db_name"],
                "type": "cypher",
                "dataset_dir": item["dataset_dir"],
                "actual_db": item.get("actual_db_name", item["db_name"]),
                "labels": set(),
            },
        )
        entry["labels"].add(f"{item['type']}:{item['label']}")

    return {
        "scope": "sql_cypher",
        "note": "The current project scope covers SQL relational databases and Cypher graph databases.",
        "sql": [
            {**item, "tables": sorted(item["tables"])}
            for item in sorted(sql_items.values(), key=lambda row: row["id"])
        ],
        "cypher": [
            {**item, "labels": sorted(item["labels"])}
            for item in sorted(cypher_items.values(), key=lambda row: row["id"])
        ],
    }


@app.get("/api/schema/{db_id}")
def schema(db_id: str):
    """读取指定数据库的 schema。

    参数：
    - db_id：SQL 数据库 id 或 Cypher 图数据库名称。

    返回：
    - SQL schema 文本或图数据库 schema 文本；找不到时抛出 404。
    """
    sql_meta = Path("schema_index/metadata.pkl")
    metadata = _sql_metadata_records()
    sql_item = next((item for item in metadata if item["db_id"] == db_id), None)
    if sql_item:
        conn = connect_sqlite(sql_item["db_path"])
        try:
            return {"id": db_id, "type": "sql", "schema": extract_schema(conn)}
        finally:
            conn.close()

    cypher_meta = Path("schema_index/cypher_metadata.pkl")
    metadata = _cypher_metadata_records()
    cypher_item = next((item for item in metadata if item["db_name"] == db_id), None)
    if cypher_item:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        try:
            actual_db = resolve_actual_cypher_database(db_id)
            return {
                "id": db_id,
                "actual_db": actual_db,
                "type": "cypher",
                "schema": extract_graph_schema(driver, actual_db),
            }
        finally:
            driver.close()

    raise HTTPException(status_code=404, detail=f"Schema for '{db_id}' was not found.")
