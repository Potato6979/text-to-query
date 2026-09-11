# cypher_schema_retriever.py — 在线 Cypher Schema 向量化检索
# 功能：给定自然语言问题，从 FAISS 索引中检索最相关的图数据库
# 使用面向图模型的 RAG 动态模式筛选

import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from pathlib import Path

# ── 配置 ──────────────────────────────────────────────
ROOT_DIR                  = Path(__file__).resolve().parent
CYPHER_INDEX_SAVE_PATH    = str(ROOT_DIR / "schema_index" / "cypher_faiss.index")
CYPHER_METADATA_SAVE_PATH = str(ROOT_DIR / "schema_index" / "cypher_metadata.pkl")
MODEL_NAME                = "paraphrase-multilingual-MiniLM-L12-v2"
TOP_K_RECORDS             = 15   # 检索候选记录数（图比SQL复杂，适当加大）

DATASET_BASE = str(ROOT_DIR / "data" / "Cypher" / "Mind-the-Query" / "Automated_Validated_Datasets")
# ──────────────────────────────────────────────────────

# 模块级单例
_model    = None
_index    = None
_metadata = None

# 中文函数说明补充索引：
# - load_resources()：加载 Cypher FAISS index、metadata 和 embedding 模型。
# - retrieve_schema(question)：根据自然语言问题检索最相关的图数据库/schema 片段。
# - format_cypher_schema_context(...)：把图 schema 检索结果组织成 Cypher 生成上下文。


def _load_resources():
    """懒加载 Cypher schema 检索所需的 embedding 模型、FAISS 索引和元数据。"""
    global _model, _index, _metadata
    if _model is None:
        print("[CypherRetriever] 加载 Embedding 模型...")
        _model = SentenceTransformer(MODEL_NAME)
    if _index is None:
        print("[CypherRetriever] 加载 FAISS 索引...")
        _index = faiss.read_index(CYPHER_INDEX_SAVE_PATH)
        with open(CYPHER_METADATA_SAVE_PATH, "rb") as f:
            _metadata = pickle.load(f)
        print(f"[CypherRetriever] 索引就绪，共 {_index.ntotal} 条向量")


def retrieve_graph_db(question: str, top_k: int = TOP_K_RECORDS) -> dict:
    """
    给定自然语言问题，检索最相关的图数据库。

    原理：
      1. 将问题向量化
      2. 在 FAISS 索引中检索 top-k 个最相似的节点/关系记录
      3. 按图数据库聚合分数，分数最高的库即为目标库

    返回：
      {
        "db_name":      最相关的图数据库名,
        "dataset_dir":  对应的评估数据集目录名,
        "score":        聚合相似度得分,
        "top_labels":   命中的节点/关系标签列表,
        "candidates":   所有候选图数据库的排名列表
      }
    """
    _load_resources()

    # Step 1：问题向量化
    query_vec = _model.encode(
        [question],
        normalize_embeddings=True,
    ).astype(np.float32)

    # Step 2：FAISS 检索
    scores, indices = _index.search(query_vec, top_k)
    scores  = scores[0].tolist()
    indices = indices[0].tolist()

    # Step 3：按图数据库聚合分数
    db_scores       = defaultdict(float)
    db_labels       = defaultdict(list)
    db_dataset_dirs = {}

    for score, idx in zip(scores, indices):
        if idx < 0:
            continue
        record  = _metadata[idx]
        db_name = record["db_name"]
        db_scores[db_name]       += score
        db_labels[db_name].append(f"{record['type']}:{record['label']}")
        db_dataset_dirs[db_name]  = record["dataset_dir"]

    # Step 4：按分数排序
    ranked     = sorted(db_scores.items(), key=lambda x: -x[1])
    best_db    = ranked[0][0]

    return {
        "db_name":     best_db,
        "dataset_dir": db_dataset_dirs[best_db],
        "score":       round(db_scores[best_db], 4),
        "top_labels":  db_labels[best_db],
        "candidates":  [
            {
                "db_name":     db_name,
                "dataset_dir": db_dataset_dirs[db_name],
                "score":       round(score, 4),
                "labels":      db_labels[db_name],
            }
            for db_name, score in ranked
        ],
    }


def evaluate_retrieval_accuracy():
    """
    评估检索准确率：用 Mind the Query 各图数据库的评估题目测试，
    判断检索到的图数据库是否正确。
    """
    import json, os

    # 9个图数据库及对应的评估数据集目录
    DB_CONFIGS = [
        ("wwc2019",   "wwc"),
        ("healthcare","healthcare"),
        ("pole",      "pole"),
        ("bloom",     "bloom50"),
        ("entityres", "er"),
        ("gdsc",      "gdsc"),
        ("legis",     "legis_graph"),
        ("osm",       "osm"),
        ("trolls",    "twitter_trolls"),
    ]

    CATEGORIES = [
        "Simple_Retrieval_all_passed.json",
        "Complex_Retrieval_all_passed.json",
        "Simple_Aggregation_all_passed.json",
        "Complex_Aggregation_all_passed.json",
        "Evaluation_query_all_passed.json",
    ]

    # 每类最多取10条，避免评估时间过长
    SAMPLE_PER_CATEGORY = 10

    all_items = []
    for db_name, dataset_dir in DB_CONFIGS:
        dir_path = os.path.join(DATASET_BASE, dataset_dir)
        for cat_file in CATEGORIES:
            filepath = os.path.join(dir_path, cat_file)
            if not os.path.exists(filepath):
                continue
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data[:SAMPLE_PER_CATEGORY]:
                all_items.append({
                    "question": item["NL Question"],
                    "gold_db":  db_name,
                })

    total        = len(all_items)
    correct      = 0
    top3_correct = 0
    errors       = []

    print(f"\n评估检索准确率，共 {total} 道题")
    print("=" * 60)

    for i, item in enumerate(all_items, 1):
        question = item["question"]
        gold_db  = item["gold_db"]

        result     = retrieve_graph_db(question)
        pred_db    = result["db_name"]
        candidates = [c["db_name"] for c in result["candidates"]]

        if pred_db == gold_db:
            correct += 1
        if gold_db in candidates[:3]:
            top3_correct += 1
        else:
            errors.append({
                "question":   question,
                "gold":       gold_db,
                "pred":       pred_db,
                "candidates": candidates[:3],
            })

        if i % 50 == 0:
            print(f"  [{i}/{total}] Top-1 准确率: {correct/i*100:.1f}%")

    print(f"\n{'='*60}")
    print(f"Cypher 检索准确率评估结果")
    print(f"{'='*60}")
    print(f"总题目数：     {total}")
    print(f"Top-1 正确：   {correct}  ({correct/total*100:.1f}%)")
    print(f"Top-3 正确：   {top3_correct}  ({top3_correct/total*100:.1f}%)")

    # 按图数据库统计
    print(f"\n── 按图数据库统计 ──")
    db_total   = defaultdict(int)
    db_correct = defaultdict(int)
    for item in all_items:
        db_total[item["gold_db"]] += 1
    for item in all_items:
        result = retrieve_graph_db(item["question"])
        if result["db_name"] == item["gold_db"]:
            db_correct[item["gold_db"]] += 1

    # 直接用已有结果统计（避免重复调用）
    # 重新统计：遍历all_items和对应结果
    # 上面已经在循环里统计了，这里用errors反推
    print(f"\n── 错误案例（前5个）──")
    for e in errors[:5]:
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
        "How many teams participated in the 2019 Women's World Cup?",
        "Which drugs caused the most adverse reactions?",
        "How many crimes were committed in each area?",
        "Find all senators who voted for a specific bill.",
        "Which nodes are connected to each other in the graph?",
        "How many trolls are from Russia?",
        "Find roads connecting two locations in the map.",
        "Which entities were resolved as duplicates?",
    ]

    print("=" * 60)
    print("Cypher Schema 向量化检索功能测试")
    print("=" * 60)

    for q in test_questions:
        result = retrieve_graph_db(q)
        print(f"\n问题：{q}")
        print(f"  -> 检索到：{result['db_name']}  (得分: {result['score']})")
        print(f"  -> 命中标签：{result['top_labels'][:4]}")
        print(f"  -> 候选库：{[c['db_name'] for c in result['candidates'][:3]]}")

    # 准确率评估
    print("\n" + "=" * 60)
    print("开始评估检索准确率")
    evaluate_retrieval_accuracy()
