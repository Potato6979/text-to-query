# main.py — 系统主入口
# 完整流程：自然语言 → 意图路由 → 对应链路 → 执行结果
# 四层架构：接入层、意图路由层、专项生成层和执行反馈层

from .router import route_query
from .sql_pipeline import run_sql_pipeline
from .cypher_pipeline_trace import run_cypher_pipeline

# 中文函数说明补充索引：
# - run_mql_pipeline(question)：MongoDB/MQL 占位链路，保持多模态接口完整。
# - run_vector_pipeline(question)：向量检索占位链路，保持多模态接口完整。
# - main(question)：旧命令行/脚本主入口，根据 route_query 结果调用对应 pipeline；当前 API/Coordinator 是正式入口。

# ─────────────────────────────────────────────────────
# 各数据库的 Schema 概要（统一元数据描述框架）
# 告知路由层每个数据库里有哪些核心实体，
# 辅助判断问题应该路由到哪个查询引擎
# 后续引入向量化检索后，这部分由向量索引动态提供
# ─────────────────────────────────────────────────────
SCHEMA_SUMMARY = """
【SQL 数据库 - concert_singer】
  表：singer（歌手信息：Singer_ID, Name, Country, Age, Song_Name）
  表：stadium（体育场信息：Stadium_ID, Name, Location, Capacity）
  表：concert（演唱会信息：concert_ID, concert_Name, Theme, Year）
  表：singer_in_concert（歌手-演唱会关联：concert_ID, Singer_ID）
  适用问题：统计歌手数量、查询演唱会信息、体育场容量排名等

【图数据库 - wwc2019（女足世界杯）】
  节点：Person（球员/教练）、Team（球队）、Match（比赛）、Squad（阵容）、Tournament（锦标赛）
  关系：PLAYED_IN（参加比赛）、SCORED_GOAL（进球）、REPRESENTS（代表球队）、
        PARTICIPATED_IN（球队参加锦标赛）、COACH_FOR（执教阵容）、IN_SQUAD（在阵容中）
  适用问题：查询球员参加比赛的路径、进球关系、球队与锦标赛的连接等

【文档数据库 - MongoDB】（待接入）
  适用问题：嵌套文档查询、数组字段操作等

【向量数据库 - Milvus】（待接入）
  适用问题：语义相似度搜索、模糊匹配等
"""


# ─────────────────────────────────────────────────────
# MQL 和 Vector 扩展链路占位
# 后续接入 MongoDB 和 Milvus 时替换
# ─────────────────────────────────────────────────────
def run_mql_pipeline(question: str) -> dict:
    """MQL 占位 pipeline。

    参数：
    - question：用户自然语言问题。

    说明：
    - 当前项目重点是 SQL/Cypher，该函数只用于保持多模态接口完整。
    """
    return {
        "question": question,
        "query":    "MQL 链路尚未实现",
        "success":  False,
        "result":   "待实现",
    }


def run_vector_pipeline(question: str) -> dict:
    """Vector 检索占位 pipeline。

    参数：
    - question：用户自然语言问题。

    说明：
    - 当前尚未接入真实向量数据库查询，只返回占位结果。
    """
    return {
        "question": question,
        "query":    "Vector 链路尚未实现",
        "success":  False,
        "result":   "待实现",
    }


# ─────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────
PIPELINE_MAP = {
    "sql":    run_sql_pipeline,
    "cypher": run_cypher_pipeline,
    "mql":    run_mql_pipeline,
    "vector": run_vector_pipeline,
}


def run(question: str, verbose: bool = True) -> dict:
    """
    端到端处理一个自然语言问题：
    1. 意图路由：判断查询类型
    2. 分发到对应链路
    3. 返回统一格式的结果
    """
    if verbose:
        print("\n" + "=" * 60)
        print(f"输入问题：{question}")
        print("─" * 60)

    # Step 1：意图路由
    route_result = route_query(question, schema_summary=SCHEMA_SUMMARY)
    query_type   = route_result["query_type"]
    confidence   = route_result["confidence"]
    reason       = route_result["reason"]

    if verbose:
        print(f"[路由] 类型={query_type}  置信度={confidence}")
        print(f"       理由：{reason}")
        print("─" * 60)

    # Step 2：分发到对应链路
    pipeline = PIPELINE_MAP.get(query_type, run_sql_pipeline)
    result   = pipeline(question)

    # Step 3：统一输出格式
    output = {
        "question":   question,
        "route":      route_result,
        "query_type": query_type,
        "query":      result.get("sql") or result.get("cypher") or result.get("query", ""),
        "success":    result.get("success", False),
        "result":     result.get("result", ""),
        "retries":    result.get("retries", 0),
    }

    if verbose:
        print(f"\n[结果]")
        print(f"  成功：{output['success']}")
        print(f"  查询语句：{output['query'][:100]}...")
        print(f"  执行结果：\n{output['result']}")

    return output


# ─────────────────────────────────────────────────────
# 批量演示
# ─────────────────────────────────────────────────────
def demo():
    """
    演示系统处理不同类型问题的能力，
    每种路由类型各一个示例。
    """
    questions = [
        # SQL 类
        "How many singers are there?",
        "What is the average age of all singers from France?",

        # Cypher 类
        "How many teams are there?",
        "Find all players who scored goals in the France 2019 tournament.",

        # MQL 类（占位）
        "Find all orders where the items array contains a product with price over 100.",

        # Vector 类（占位）
        "Find articles similar to this paper about machine learning.",
    ]

    print("=" * 60)
    print("系统端到端演示")
    print("=" * 60)

    results = []
    for q in questions:
        r = run(q, verbose=True)
        results.append(r)

    # 汇总
    print("\n" + "=" * 60)
    print("演示汇总")
    print("=" * 60)
    for r in results:
        status = "✓" if r["success"] else "○"
        print(f"  {status} [{r['query_type']}] {r['question'][:50]}")

    success_count = sum(1 for r in results if r["success"])
    print(f"\n成功执行：{success_count}/{len(results)}")


if __name__ == "__main__":
    demo()
