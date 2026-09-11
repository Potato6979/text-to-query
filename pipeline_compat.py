from typing import Any


def get_pipeline_query(result: dict[str, Any]) -> str:
    """读取 pipeline 结果中的查询语句。

    参数：
    - result：SQL/Cypher pipeline 输出，可能包含 `query`、`sql` 或 `cypher`。
    """
    return result.get("query") or result.get("sql") or result.get("cypher") or ""


def get_pipeline_selected_db(result: dict[str, Any]) -> str:
    """读取 pipeline 最终选中的数据库标识。

    参数：
    - result：pipeline 输出，兼容 `selected_db`、`retrieved_db` 和 retrieval 子字段。
    """
    return (
        result.get("selected_db")
        or result.get("retrieved_db")
        or result.get("retrieval", {}).get("selected_db", "")
    )


def get_pipeline_selected_path(result: dict[str, Any]) -> str:
    """读取 SQL pipeline 使用的数据库文件路径。

    参数：
    - result：pipeline 输出；Cypher 通常没有本地文件路径。
    """
    return result.get("used_db_path") or result.get("retrieval", {}).get("selected_path", "")


def get_pipeline_working_schema(result: dict[str, Any]) -> str:
    """读取 pipeline 实际用于生成的 schema 上下文。

    参数：
    - result：pipeline 输出，兼容 SQL 的 linked schema 和 Cypher 的 filtered schema。
    """
    return (
        result.get("working_schema")
        or result.get("schema_linked")
        or result.get("schema_filtered")
        or ""
    )


def apply_sql_legacy_aliases(record: dict[str, Any]) -> dict[str, Any]:
    """为 SQL pipeline 输出补齐旧字段别名。

    参数：
    - record：SQL pipeline 的标准输出字典，会被原地补充兼容字段。
    """
    # 在统一字段迁移期间保持历史 SQL 字段同步，避免旧调用方失效。
    record["query_type"] = record.get("query_type") or "sql"
    record["query"] = get_pipeline_query(record)
    record["sql"] = record.get("sql") or record["query"]
    record["selected_db"] = get_pipeline_selected_db(record)
    record["retrieved_db"] = record.get("retrieved_db") or record["selected_db"]
    record["working_schema"] = get_pipeline_working_schema(record)
    record["schema_linked"] = record.get("schema_linked") or record["working_schema"]
    record["used_db_path"] = get_pipeline_selected_path(record)
    return record


def apply_cypher_legacy_aliases(record: dict[str, Any]) -> dict[str, Any]:
    """为 Cypher pipeline 输出补齐旧字段别名。

    参数：
    - record：Cypher pipeline 的标准输出字典，会被原地补充兼容字段。
    """
    # 在统一字段迁移期间保持历史 Cypher 字段同步，避免旧调用方失效。
    record["query_type"] = record.get("query_type") or "cypher"
    record["query"] = get_pipeline_query(record)
    record["cypher"] = record.get("cypher") or record["query"]
    record["selected_db"] = get_pipeline_selected_db(record)
    record["retrieved_db"] = record.get("retrieved_db") or record["selected_db"]
    record["working_schema"] = get_pipeline_working_schema(record)
    record["schema_filtered"] = record.get("schema_filtered") or record["working_schema"]
    return record
