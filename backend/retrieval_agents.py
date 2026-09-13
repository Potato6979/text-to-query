import pickle
import re
import time
from collections import Counter
from functools import lru_cache
from typing import Any

from .config import SCHEMA_INDEX_DIR, SQLITE_DB_PATH
from .paired_benchmark_resources import (
    paired_cypher_metadata_records,
    paired_sql_metadata_records,
    rank_paired_cypher_resources,
)

# 中文函数说明补充索引：
# - _now_ms()：返回毫秒时间，记录检索耗时。
# - _tokenize(text)：检索用轻量分词，统一 schema/question token。
# - _split_schema_and_sample(text)：区分 schema 正文和样例值，降低 value hint 对库选择的干扰。
# - _lexical_score(question, schema_text)：计算问题与 schema 的词面相关度。
# - retrieve_sql_targets(question)：从 SQL FAISS/metadata 中检索候选 SQLite 库。
# - retrieve_cypher_targets(question)：从 Cypher schema index 中检索候选 Neo4j 图。


def _now_ms() -> float:
    """返回当前高精度时间，单位毫秒。"""
    return time.perf_counter() * 1000


LEXICAL_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "with", "by", "from", "what", "which",
    "who", "how", "many", "much", "is", "are", "was", "were", "be", "do", "does", "did", "all", "each",
    "find", "list", "show", "give", "name", "names",
}


def _tokenize(text: str) -> set[str]:
    """把问题或 schema 文本切分为检索用 token 集合。

    参数：
    - text：待分词文本，可能是问题、表名、列名、label 或 relationship。
    """
    expanded = re.sub(r"([a-z])([A-Z])", r"\1 \2", text.replace("_", " "))
    tokens: set[str] = set()
    for token in re.findall(r"[a-zA-Z][a-zA-Z0-9]*", expanded.lower()):
        if len(token) <= 1 or token in LEXICAL_STOPWORDS:
            continue
        if token.endswith("ies") and len(token) > 4:
            token = token[:-3] + "y"
        elif token.endswith("s") and len(token) > 3:
            token = token[:-1]
        tokens.add(token)
    return tokens


def _split_schema_and_sample(text: str) -> tuple[str, str]:
    """把 schema 文本和样例值文本分开。

    参数：
    - text：schema index 中保存的原始文本。
    """
    for marker in ("样例", "Сщ±ѕ"):
        if marker in text:
            schema_text, sample_text = text.split(marker, 1)
            return schema_text, sample_text
    return text, ""


@lru_cache(maxsize=1)
def _load_sql_db_lexical_index() -> dict[str, dict[str, Any]]:
    """加载并缓存 SQL 数据库级词面索引。

    返回：
    - db_id 到 token、schema token、sample token、表名和路径的映射。
    """
    metadata_path = SCHEMA_INDEX_DIR / "metadata.pkl"
    with open(metadata_path, "rb") as handle:
        metadata = pickle.load(handle)

    buckets: dict[str, dict[str, Any]] = {}
    for item in metadata + paired_sql_metadata_records():
        db_id = item["db_id"]
        bucket = buckets.setdefault(
            db_id,
            {
                "db_path": item["db_path"],
                "tokens": set(),
                "schema_tokens": set(),
                "sample_tokens": set(),
                "tables": set(),
            },
        )
        bucket["tables"].add(item["table"])
        schema_text, sample_text = _split_schema_and_sample(item["text"])
        schema_tokens = _tokenize(schema_text)
        sample_tokens = _tokenize(sample_text)
        bucket["schema_tokens"].update(schema_tokens)
        bucket["sample_tokens"].update(sample_tokens)
        bucket["tokens"].update(schema_tokens | sample_tokens)

    return buckets


@lru_cache(maxsize=1)
def _load_sql_db_token_df() -> dict[str, int]:
    """统计每个 SQL schema token 出现在多少个数据库中。

    用途：
    - 给较少见、更有区分度的 schema token 更高权重。
    """
    lexical_index = _load_sql_db_lexical_index()
    df: Counter[str] = Counter()
    for bucket in lexical_index.values():
        for token in bucket["schema_tokens"] | bucket["sample_tokens"]:
            df[token] += 1
    return dict(df)


def _weighted_lexical_score(question_tokens: set[str], bucket: dict[str, Any], df: dict[str, int], db_count: int) -> float:
    """计算问题 token 与某个 SQL 数据库 token 桶的加权词面分。

    参数：
    - question_tokens：用户问题 token 集合。
    - bucket：某个数据库的 schema/sample token 桶。
    - df：token 的数据库文档频次。
    - db_count：总数据库数量。
    """
    schema_overlap = question_tokens & bucket["schema_tokens"]
    sample_overlap = question_tokens & bucket["sample_tokens"]

    def token_weight(token: str) -> float:
        """计算单个 token 的 IDF 风格权重。

        参数：
        - token：问题与 schema 重叠的 token。
        """
        # 罕见 schema 词比 student 这类常见领域名词更能区分数据库。
        return 1.0 + (db_count / max(df.get(token, 1), 1)) ** 0.5

    schema_score = sum(token_weight(token) for token in schema_overlap)
    sample_only_score = sum(0.35 * token_weight(token) for token in sample_overlap - schema_overlap)
    return schema_score + sample_only_score


def _rerank_sql_candidates(question: str, raw_candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """结合向量检索顺序和 schema 词面证据重排 SQL 候选库。

    参数：
    - question：用户自然语言问题。
    - raw_candidates：向量检索返回的原始候选数据库列表。
    """
    lexical_index = _load_sql_db_lexical_index()
    token_df = _load_sql_db_token_df()
    question_tokens = _tokenize(question)
    if not question_tokens:
        return raw_candidates

    question_text = str(question or "").lower()
    paired_sql_ids = {
        candidate["db_id"]
        for candidate in raw_candidates
        if str(candidate.get("db_id", "")).endswith("_sql")
    }
    if not paired_sql_ids:
        paired_sql_ids = {
            db_id
            for db_id in _load_sql_db_lexical_index()
            if str(db_id).endswith("_sql")
        }

    def paired_stem(resource_id: str) -> str:
        return resource_id[:-4] if resource_id.endswith("_sql") else resource_id

    explicit_paired_sql = "sql table" in question_text or "sql tables" in question_text
    explicit_paired_hit = ""
    if explicit_paired_sql:
        for db_id in sorted(paired_sql_ids):
            stem = paired_stem(str(db_id))
            stem_tokens = _tokenize(stem)
            if stem in question_text or (stem_tokens and stem_tokens <= question_tokens):
                explicit_paired_hit = str(db_id)
                break

    candidate_map = {candidate["db_id"]: dict(candidate) for candidate in raw_candidates}
    vector_rank_score = {
        candidate["db_id"]: max(0.0, 1.0 - 0.2 * rank)
        for rank, candidate in enumerate(raw_candidates)
    }

    lexical_scores: dict[str, float] = {}
    raw_lexical_scores: dict[str, float] = {}
    max_raw_score = 0.0
    for db_id, bucket in lexical_index.items():
        raw_score = _weighted_lexical_score(question_tokens, bucket, token_df, len(lexical_index))
        if raw_score > 0:
            raw_lexical_scores[db_id] = raw_score
            max_raw_score = max(max_raw_score, raw_score)

    if max_raw_score > 0:
        lexical_scores = {db_id: score / max_raw_score for db_id, score in raw_lexical_scores.items()}

    all_db_ids = set(candidate_map) | set(lexical_scores)
    ranked: list[tuple[float, str]] = []
    for db_id in all_db_ids:
        lexical_score = lexical_scores.get(db_id, 0.0)
        vector_score = vector_rank_score.get(db_id, 0.0)
        combined = 0.45 * vector_score + 0.55 * lexical_score
        if explicit_paired_hit and db_id == explicit_paired_hit:
            combined += 2.0
        elif explicit_paired_sql and str(db_id).endswith("_sql"):
            combined += 0.35
        elif explicit_paired_hit and db_id == paired_stem(explicit_paired_hit):
            combined -= 0.5
        ranked.append((combined, db_id))

    ranked.sort(key=lambda item: (-item[0], item[1]))
    if ranked:
        lexical_best_db = max(all_db_ids, key=lambda db_id: lexical_scores.get(db_id, 0.0))
        lexical_best_score = lexical_scores.get(lexical_best_db, 0.0)
        current_best_db = ranked[0][1]
        current_best_lexical = lexical_scores.get(current_best_db, 0.0)
        if lexical_best_score >= 0.15 and current_best_lexical == 0.0 and lexical_best_db != current_best_db and lexical_best_db in candidate_map:
            ranked = [(score, db_id) for score, db_id in ranked if db_id != lexical_best_db]
            ranked.insert(0, (lexical_best_score, lexical_best_db))
        elif lexical_best_score >= 0.45 and lexical_best_score - current_best_lexical >= 0.15 and lexical_best_db != current_best_db:
            ranked = [(score, db_id) for score, db_id in ranked if db_id != lexical_best_db]
            ranked.insert(0, (lexical_best_score, lexical_best_db))

    reranked: list[dict[str, Any]] = []
    for combined_score, db_id in ranked[:3]:
        if db_id in candidate_map:
            candidate = candidate_map[db_id]
        else:
            bucket = lexical_index[db_id]
            candidate = {
                "db_id": db_id,
                "db_path": bucket["db_path"],
                "score": None,
                "tables": sorted(bucket["tables"]),
            }
        candidate = dict(candidate)
        candidate["hybrid_score"] = round(combined_score, 4)
        candidate["lexical_score"] = round(lexical_scores.get(db_id, 0.0), 4)
        reranked.append(candidate)

    return reranked if reranked else raw_candidates


def _normalize_sql_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    """把 SQL 候选库转换为统一 retrieval candidate 格式。

    参数：
    - candidate：SQL 检索阶段的原始候选。
    """
    return {
        "resource_id": candidate.get("db_id", ""),
        "resource_path": candidate.get("db_path", ""),
        "score": candidate.get("score"),
        "metadata": {
            "tables": candidate.get("tables", []),
        },
        "raw": candidate,
    }


def _normalize_cypher_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    """把 Cypher 候选图库转换为统一 retrieval candidate 格式。

    参数：
    - candidate：Cypher 检索阶段的原始候选。
    """
    return {
        "resource_id": candidate.get("db_name", ""),
        "resource_path": "",
        "score": candidate.get("score"),
        "metadata": {
            "labels": candidate.get("labels", []),
        },
        "raw": candidate,
    }


def _rerank_cypher_candidates(question: str, raw_candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """结合向量检索顺序和 label/relationship 词面证据重排 Cypher 候选图库。

    参数：
    - question：用户自然语言问题。
    - raw_candidates：图 schema 检索返回的候选图库列表。
    """
    question_tokens = _tokenize(question)
    if not question_tokens or not raw_candidates:
        return raw_candidates

    ranked: list[tuple[float, dict[str, Any]]] = []
    for rank, candidate in enumerate(raw_candidates):
        label_tokens: set[str] = set()
        for label in candidate.get("labels", []):
            label_tokens.update(_tokenize(str(label)))
        lexical_score = len(question_tokens & label_tokens) / max(len(question_tokens), 1)
        vector_score = max(0.0, 1.0 - 0.15 * rank)
        combined = 0.45 * vector_score + 0.55 * lexical_score
        enriched = dict(candidate)
        enriched["lexical_score"] = round(lexical_score, 4)
        enriched["hybrid_score"] = round(combined, 4)
        ranked.append((combined, enriched))

    ranked.sort(key=lambda item: (-item[0], item[1].get("db_name", "")))
    return [candidate for _, candidate in ranked]


def retrieve_sql_targets(
    question: str,
    *,
    db_path: str | None = None,
    use_retrieval: bool = True,
) -> dict[str, Any]:
    """检索 SQL 查询的目标数据库。

    参数：
    - question：用户自然语言问题。
    - db_path：可选指定 SQLite 文件路径；指定后跳过检索。
    - use_retrieval：是否启用 schema retriever。
    """
    if db_path is not None:
        raw_candidates = [{"db_id": "specified", "db_path": db_path, "score": None, "tables": []}]
        return {
            "query_type": "sql",
            "selected_db": "specified",
            "selected_path": db_path,
            "score": None,
            "top_candidates": [_normalize_sql_candidate(candidate) for candidate in raw_candidates],
            "raw_candidates": raw_candidates,
            "retrieval_ms": 0.0,
            "method": "specified",
        }

    started_ms = _now_ms()
    selected_db = ""
    selected_path = ""
    score = None
    method = "vector"
    try:
        if use_retrieval:
            from .schema_retriever import retrieve_db

            result = retrieve_db(question)
            raw_candidates = _rerank_sql_candidates(question, result["candidates"][:3])
            selected_db = raw_candidates[0]["db_id"]
            selected_path = raw_candidates[0]["db_path"]
            score = result.get("score")
        else:
            raw_candidates = [{"db_id": "default", "db_path": SQLITE_DB_PATH, "score": None, "tables": []}]
            selected_db = "default"
            selected_path = SQLITE_DB_PATH
            method = "default"
    except Exception as exc:
        raw_candidates = [
            {
                "db_id": "default",
                "db_path": SQLITE_DB_PATH,
                "score": None,
                "tables": [],
                "error": str(exc),
            }
        ]
        selected_db = "default"
        selected_path = SQLITE_DB_PATH
        method = "fallback"

    return {
        "query_type": "sql",
        "selected_db": selected_db,
        "selected_path": selected_path,
        "score": score,
        "top_candidates": [_normalize_sql_candidate(candidate) for candidate in raw_candidates],
        "raw_candidates": raw_candidates,
        "retrieval_ms": round(_now_ms() - started_ms, 2),
        "method": method,
    }


def retrieve_cypher_targets(
    question: str,
    *,
    db_name: str | None = None,
) -> dict[str, Any]:
    """检索 Cypher 查询的目标图数据库。

    参数：
    - question：用户自然语言问题。
    - db_name：可选指定 Neo4j 数据库名；指定后跳过检索。
    """
    if db_name is not None:
        raw_candidates = [{"db_name": db_name, "score": None, "labels": []}]
        return {
            "query_type": "cypher",
            "selected_db": db_name,
            "selected_path": "",
            "score": None,
            "top_candidates": [_normalize_cypher_candidate(candidate) for candidate in raw_candidates],
            "raw_candidates": raw_candidates,
            "retrieval_ms": 0.0,
            "method": "specified",
        }

    started_ms = _now_ms()
    selected_db = ""
    score = None
    method = "vector"
    try:
        from .cypher_schema_retriever import retrieve_graph_db

        result = retrieve_graph_db(question)
        raw_candidates = _rerank_cypher_candidates(question, result["candidates"][:3])
        paired_candidates = rank_paired_cypher_resources(question)
        seen = {candidate.get("db_name") for candidate in raw_candidates}
        for candidate in paired_candidates:
            if candidate["db_name"] in seen:
                continue
            raw_candidates.append(candidate)
            seen.add(candidate["db_name"])
        raw_candidates = _rerank_cypher_candidates(question, raw_candidates)
        selected_db = raw_candidates[0]["db_name"]
        score = result.get("score")
    except Exception as exc:
        raw_candidates = rank_paired_cypher_resources(question)
        if not raw_candidates:
            raw_candidates = [{"db_name": "wwc2019", "score": None, "labels": [], "error": str(exc)}]
            selected_db = "wwc2019"
        else:
            selected_db = raw_candidates[0]["db_name"]
        method = "fallback"

    return {
        "query_type": "cypher",
        "selected_db": selected_db,
        "selected_path": "",
        "score": score,
        "top_candidates": [_normalize_cypher_candidate(candidate) for candidate in raw_candidates],
        "raw_candidates": raw_candidates,
        "retrieval_ms": round(_now_ms() - started_ms, 2),
        "method": method,
    }


def retrieve_targets(
    question: str,
    *,
    query_type: str,
    specified_target: str | None = None,
    use_retrieval: bool = True,
) -> dict[str, Any]:
    """按 query type 分发到 SQL 或 Cypher 目标检索函数。

    参数：
    - question：用户自然语言问题。
    - query_type：查询模态，目前支持 `sql` 和 `cypher`。
    - specified_target：可选指定数据库路径或图库名。
    - use_retrieval：SQL 场景下是否启用检索。
    """
    if query_type == "sql":
        return retrieve_sql_targets(
            question,
            db_path=specified_target,
            use_retrieval=use_retrieval,
        )
    if query_type == "cypher":
        return retrieve_cypher_targets(
            question,
            db_name=specified_target,
        )
    raise ValueError(f"Unsupported query_type for retrieval: {query_type}")
