import pickle
import re
from functools import lru_cache
from pathlib import Path

# 中文函数说明索引：
# - normalize_text(text)：把自然语言和 schema 文本统一小写、压缩空白，供路由关键词和 token 匹配使用。
# - tokenize(text)：抽取英文/数字 token，是 evidence、scorer、planner 共用的轻量分词函数。
# - meaningful_tokens(text)：去掉停用词后的 token 集合，用于 schema/proposal 与问题的词面重合度判断。
# - contains_any(text, keywords)：判断规范化文本是否包含任一触发词，用于路由信号和 guard 规则。
# - load_schema_indexes()：读取 SQL/Cypher schema index metadata，给路由层提供可检索资源清单。
# - parse_schema_summary_terms(schema_summary)：从拼接后的 schema summary 中抽取 SQL/图相关术语，作为无索引时的 fallback evidence。


QUERY_TYPES = {
    "sql": "关系型数据库查询（PostgreSQL/SQLite）。适用于结构化表格数据的统计、聚合、多表连接、条件筛选等操作。",
    "cypher": "图数据库查询（Neo4j Cypher）。适用于实体间关系的路径遍历、多跳连接、图模式匹配等操作。",
    "mql": "文档数据库查询（MongoDB MQL）。适用于嵌套文档结构、数组字段、灵活 Schema 的查询操作。",
    "vector": "向量相似度搜索。适用于语义相似度匹配、模糊检索、基于内容的推荐等操作。",
}

ALL_QUERY_TYPES = list(QUERY_TYPES.keys())
TOP_K_CANDIDATES = 2

INTENT_KEYWORDS = {
    "aggregation": ["how many", "count", "average", "avg", "sum", "total", "maximum", "minimum", "max", "min"],
    "sorting": ["top", "highest", "lowest", "most", "least", "rank", "latest", "earliest"],
    "filtering": ["with", "where", "from", "whose", "in year", "over", "under", "before", "after"],
    "graph_relation": [
        "relationship",
        "relationships",
        "relation",
        "relations",
        "path",
        "paths",
        "through",
        "via",
        "between",
        "connected",
        "related",
        "linked",
        "associated",
    ],
    "similarity": ["similar", "semantic", "relevant", "like this", "look similar", "closest"],
    "document": ["nested", "array", "document", "field", "json", "embedded"],
    "cross_source": [
        "first",
        "then",
        "after that",
        "based on",
        "according to the result",
        "across databases",
        "in sql",
        "in graph",
    ],
}

SQL_SURFACE_HINTS = [
    "table",
    "tables",
    "column",
    "columns",
    "field",
    "fields",
    "rows",
    "row",
    "records",
    "record",
]
GRAPH_SURFACE_HINTS = [
    "relationship",
    "relationships",
    "path",
    "paths",
    "connected",
    "related",
    "linked",
    "between",
    "through",
    "via",
]
MQL_SURFACE_HINTS = ["nested", "array", "json", "document", "field path"]
VECTOR_SURFACE_HINTS = ["similar", "semantic", "relevant", "embedding", "look like"]

GENERIC_ENTITY_TOKENS = {
    "id",
    "ids",
    "name",
    "names",
    "type",
    "types",
    "code",
    "codes",
    "description",
    "descriptions",
    "date",
    "dates",
    "year",
    "years",
    "value",
    "values",
    "data",
    "info",
    "information",
    "region",
    "regions",
    "area",
    "areas",
    "location",
    "locations",
    "language",
    "languages",
    "has",
    "have",
    "count",
    "return",
    "during",
    "their",
    "order",
    "average",
    "number",
    "total",
    "per",
}

ROUTING_STOPWORDS = {
    "the",
    "a",
    "an",
    "of",
    "in",
    "on",
    "for",
    "to",
    "and",
    "or",
    "with",
    "by",
    "from",
    "what",
    "which",
    "who",
    "how",
    "many",
    "much",
    "is",
    "are",
    "was",
    "were",
    "be",
    "do",
    "does",
    "did",
    "all",
    "each",
    "that",
    "this",
    "these",
    "those",
    "it",
    "its",
    "as",
    "at",
    "into",
    "than",
    "then",
    "has",
    "have",
}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def tokenize(text: str) -> list[str]:
    expanded = re.sub(r"([a-z])([A-Z])", r"\1 \2", text.replace("_", " "))
    normalized = normalize_text(expanded)
    tokens: list[str] = []
    for token in re.findall(r"[a-zA-Z][a-zA-Z0-9]*", normalized):
        tokens.append(token)
        if token.endswith("ies") and len(token) > 4:
            tokens.append(token[:-3] + "y")
        elif token.endswith("s") and len(token) > 3:
            tokens.append(token[:-1])
    return tokens


def meaningful_tokens(text: str) -> set[str]:
    return {
        token
        for token in tokenize(text)
        if len(token) > 1 and token not in GENERIC_ENTITY_TOKENS and token not in ROUTING_STOPWORDS
    }


def contains_any(normalized_question: str, phrases: list[str]) -> tuple[bool, list[str]]:
    matched = []
    for phrase in phrases:
        stripped = phrase.strip()
        pattern = r"(?<![a-z0-9])" + re.escape(stripped) + r"(?![a-z0-9])"
        if re.search(pattern, normalized_question):
            matched.append(phrase)
    return bool(matched), matched


@lru_cache(maxsize=1)
def load_schema_indexes() -> dict[str, dict[str, list[str]]]:
    sql_terms: dict[str, set[str]] = {}
    cypher_terms: dict[str, set[str]] = {}

    sql_meta_path = Path("schema_index/metadata.pkl")
    if sql_meta_path.exists():
        with sql_meta_path.open("rb") as handle:
            sql_metadata = pickle.load(handle)
        for item in sql_metadata:
            db_id = item["db_id"]
            bucket = sql_terms.setdefault(db_id, set())
            bucket.add(item["db_id"].lower())
            bucket.add(item["table"].lower())
            bucket.update(tokenize(item.get("text", "")))

    cypher_meta_path = Path("schema_index/cypher_metadata.pkl")
    if cypher_meta_path.exists():
        with cypher_meta_path.open("rb") as handle:
            cypher_metadata = pickle.load(handle)
        for item in cypher_metadata:
            db_name = item["db_name"]
            bucket = cypher_terms.setdefault(db_name, set())
            bucket.add(item["db_name"].lower())
            bucket.add(item["label"].lower())
            bucket.add(item["type"].lower())
            bucket.update(tokenize(item.get("text", "")))

    return {
        "sql": {db_id: sorted(terms) for db_id, terms in sql_terms.items()},
        "cypher": {db_name: sorted(terms) for db_name, terms in cypher_terms.items()},
    }


def parse_schema_summary_terms(schema_summary: str) -> dict[str, list[str]]:
    if not schema_summary.strip():
        return {"sql": [], "cypher": [], "mql": [], "vector": []}

    sql_terms: set[str] = set()
    cypher_terms: set[str] = set()
    mql_terms: set[str] = set()
    vector_terms: set[str] = set()

    current_mode = ""
    for line in schema_summary.splitlines():
        stripped = line.strip()
        lowered = stripped.lower()
        if "sql" in lowered:
            current_mode = "sql"
        elif "图数据库" in stripped or "cypher" in lowered:
            current_mode = "cypher"
        elif "mongodb" in lowered or "mql" in lowered:
            current_mode = "mql"
        elif "milvus" in lowered or "vector" in lowered:
            current_mode = "vector"

        tokens = tokenize(stripped)
        if current_mode == "sql":
            sql_terms.update(tokens)
        elif current_mode == "cypher":
            cypher_terms.update(tokens)
        elif current_mode == "mql":
            mql_terms.update(tokens)
        elif current_mode == "vector":
            vector_terms.update(tokens)

    return {
        "sql": sorted(sql_terms),
        "cypher": sorted(cypher_terms),
        "mql": sorted(mql_terms),
        "vector": sorted(vector_terms),
    }
