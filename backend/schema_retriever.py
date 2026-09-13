# schema_retriever.py — 在线 Schema 向量化检索
# 功能：给定自然语言问题，从 FAISS 索引中检索最相关的数据库
# 使用基于 RAG 的动态模式筛选
# 参考：CHESS (arxiv 2405.16755) § Schema Selector

import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from pathlib import Path

# 配置
ROOT_DIR           = Path(__file__).resolve().parent.parent
INDEX_SAVE_PATH    = str(ROOT_DIR / "schema_index" / "faiss.index")
METADATA_SAVE_PATH = str(ROOT_DIR / "schema_index" / "metadata.pkl")
MODEL_NAME         = "paraphrase-multilingual-MiniLM-L12-v2"
TOP_K_TABLES       = 10

# 模块级单例：只加载一次
_model    = None
_index    = None
_metadata = None

# 中文函数说明补充索引：
# - load_resources()：加载 SQL FAISS index、metadata 和 embedding 模型。
# - retrieve_schema(question)：根据自然语言问题检索 SQL schema/table 候选。
# - get_db_path(db_id)：把 Spider db_id 转为 SQLite 文件路径。
# - format_schema_context(...)：把检索结果组织成下游可读 schema context。


def _load_resources():
    """懒加载 SQL schema 检索所需的 embedding 模型、FAISS 索引和元数据。"""
    global _model, _index, _metadata
    if _model is None:
        print("[Retriever] 加载 Embedding 模型...")
        _model = SentenceTransformer(MODEL_NAME)
    if _index is None:
        print("[Retriever] 加载 FAISS 索引...")
        _index = faiss.read_index(INDEX_SAVE_PATH)
        with open(METADATA_SAVE_PATH, "rb") as f:
            _metadata = pickle.load(f)
        print(f"[Retriever] 索引就绪，共 {_index.ntotal} 条向量")


def retrieve_db(question: str, top_k_tables: int = TOP_K_TABLES) -> dict:
    """
    给定自然语言问题，检索最相关的数据库。

    原理：
      1. 将问题向量化
      2. 在 FAISS 索引中检索 top-k 个最相似的表
      3. 按数据库聚合分数，分数最高的库即为目标库

    返回：
      {
        "db_id":      最相关的数据库ID,
        "db_path":    对应的 sqlite 文件路径,
        "score":      聚合相似度得分,
        "top_tables": 命中的表名列表,
        "candidates": 所有候选数据库的排名列表
      }
    """
    _load_resources()

    # Step 1：问题向量化
    query_vec = _model.encode(
        [question],
        normalize_embeddings=True,
    ).astype(np.float32)

    # Step 2：FAISS 检索 top-k 个最相似的表
    scores, indices = _index.search(query_vec, top_k_tables)
    scores  = scores[0].tolist()
    indices = indices[0].tolist()

    # Step 3：按数据库聚合分数（命中表分数求和）
    db_scores = defaultdict(float)
    db_tables = defaultdict(list)
    db_paths  = {}

    for score, idx in zip(scores, indices):
        if idx < 0:
            continue
        record = _metadata[idx]
        db_id  = record["db_id"]
        db_scores[db_id] += score
        db_tables[db_id].append(record["table"])
        db_paths[db_id]   = record["db_path"]

    # Step 4：按分数排序，取最高分的库
    ranked     = sorted(db_scores.items(), key=lambda x: -x[1])
    best_db_id = ranked[0][0]

    return {
        "db_id":      best_db_id,
        "db_path":    db_paths[best_db_id],
        "score":      round(db_scores[best_db_id], 4),
        "top_tables": db_tables[best_db_id],
        "candidates": [
            {
                "db_id":   db_id,
                "db_path": db_paths[db_id],
                "score":   round(score, 4),
                "tables":  db_tables[db_id],
            }
            for db_id, score in ranked
        ],
    }


def evaluate_retrieval_accuracy():
    """
    评估检索准确率：用 Spider dev.json 里的问题测试，
    判断检索到的数据库是否和标准答案一致。
    """
    import json

    DEV_JSON_PATH = str(ROOT_DIR / "data" / "spider_data" / "dev.json")
    DB_IDS = {
        "world_1", "car_1", "cre_Doc_Template_Mgt", "dog_kennels",
        "flight_2", "student_transcripts_tracking", "wta_1", "tvshow",
        "network_1", "concert_singer", "pets_1", "orchestra",
        "employee_hire_evaluation",
    }

    with open(DEV_JSON_PATH, "r", encoding="utf-8") as f:
        dev_data = json.load(f)

    items = [d for d in dev_data if d["db_id"] in DB_IDS]
    print(f"\n评估检索准确率，共 {len(items)} 道题")
    print("=" * 60)

    total        = len(items)
    correct      = 0
    top3_correct = 0
    errors       = []

    for i, item in enumerate(items, 1):
        question   = item["question"]
        gold_db_id = item["db_id"]

        result     = retrieve_db(question)
        pred_db_id = result["db_id"]
        candidates = [c["db_id"] for c in result["candidates"]]

        if pred_db_id == gold_db_id:
            correct += 1
        if gold_db_id in candidates[:3]:
            top3_correct += 1
        else:
            errors.append({
                "question":   question,
                "gold":       gold_db_id,
                "pred":       pred_db_id,
                "candidates": candidates[:3],
            })

        if i % 50 == 0:
            print(f"  [{i}/{total}] Top-1 准确率: {correct/i*100:.1f}%")

    print(f"\n{'='*60}")
    print(f"检索准确率评估结果")
    print(f"{'='*60}")
    print(f"总题目数：     {total}")
    print(f"Top-1 正确：   {correct}  ({correct/total*100:.1f}%)")
    print(f"Top-3 正确：   {top3_correct}  ({top3_correct/total*100:.1f}%)")

    if errors:
        print(f"\n── 错误案例（前10个）──")
        for e in errors[:10]:
            print(f"  问题：{e['question'][:60]}")
            print(f"    正确库: {e['gold']}  预测库: {e['pred']}  候选: {e['candidates']}")

    return {
        "total":         total,
        "top1_correct":  correct,
        "top3_correct":  top3_correct,
        "top1_accuracy": round(correct / total * 100, 1),
        "top3_accuracy": round(top3_correct / total * 100, 1),
    }


if __name__ == "__main__":
    # 功能测试
    test_questions = [
        "How many singers are there?",
        "What is the maximum capacity of all stadiums?",
        "How many dogs are there in the kennel?",
        "Which country has the most airports?",
        "How many students are enrolled in each department?",
        "What is the average age of tennis players?",
        "List all TV shows that aired in 2010.",
        "How many flights depart from each airport?",
    ]

    print("=" * 60)
    print("Schema 向量化检索功能测试")
    print("=" * 60)

    for q in test_questions:
        result = retrieve_db(q)
        print(f"\n问题：{q}")
        print(f"  -> 检索到：{result['db_id']}  (得分: {result['score']})")
        print(f"  -> 命中表：{result['top_tables']}")
        print(f"  -> 候选库：{[c['db_id'] for c in result['candidates'][:3]]}")

    # 准确率评估
    print("\n" + "=" * 60)
    print("开始评估检索准确率（基于 Spider dev.json）")
    evaluate_retrieval_accuracy()
