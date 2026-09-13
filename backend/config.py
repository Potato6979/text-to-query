# config.py — 所有配置集中在这里
# 中文配置说明：
# - DEEPSEEK_*：所有 OpenAI-compatible LLM 调用共享的模型配置，当前统一为 deepseek-v4-flash。
# - SQLITE_DB_PATH：旧单库 SQL pipeline 的默认 SQLite 路径；正式 Coordinator 会通过 retrieval/proposal 指定目标库。
# - NEO4J_*：Cypher pipeline / Bridge Resolver 连接 Neo4j 的配置。
# - MAX_RETRIES/MAX_RESULT_ROWS：执行修复次数和默认结果行上限。
# - LLM_*：LLM 温度、随机种子和请求超时时间。

# ── DeepSeek API ──────────────────────────────────────
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
SCHEMA_INDEX_DIR = ROOT_DIR / "schema_index"

DEEPSEEK_API_KEY  = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL    = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

# 硅基流动API
# DEEPSEEK_BASE_URL = "https://api.siliconflow.cn/v1"
# DEEPSEEK_MODEL    = "deepseek-ai/DeepSeek-V3"

# ── SQLite 数据库路径（Spider 官方格式）─────────────────
SQLITE_DB_PATH = str(DATA_DIR / "spider_data" / "database" / "concert_singer" / "concert_singer.sqlite")

# ── Neo4j 连接 ────────────────────────────────────────
NEO4J_URI      = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
PAIRED_BENCHMARK_NEO4J_DATABASE = os.getenv("PAIRED_BENCHMARK_NEO4J_DATABASE", "pairedbenchmark")

# ── 链路行为参数 ──────────────────────────────────────
MAX_RETRIES     = 2
MAX_RESULT_ROWS = 20
LLM_TEMPERATURE = 0
LLM_SEED        = 42
LLM_REQUEST_TIMEOUT = 120
