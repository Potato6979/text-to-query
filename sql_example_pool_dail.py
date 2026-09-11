import hashlib
import json
import os
import pickle
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from sql_schema_masking_dail import mask_question


SPIDER_BASE = Path(__file__).resolve().parent / "data" / "spider_data"
SPIDER_TABLES_PATH = SPIDER_BASE / "tables.json"
SPIDER_TEST_TABLES_PATH = SPIDER_BASE / "test_tables.json"
SPIDER_DEV_PATH = SPIDER_BASE / "dev.json"
SPIDER_TRAIN_MERGED_PATH = SPIDER_BASE / "train_spider_and_others.json"
SPIDER_TRAIN_LINK_PATH = SPIDER_BASE / "enc" / "train_schema-linking.jsonl"
SPIDER_TEST_LINK_PATH = SPIDER_BASE / "enc" / "test_schema-linking.jsonl"
SPIDER_TRAIN_PATHS = [
    SPIDER_BASE / "train_spider.json",
    SPIDER_BASE / "train_others.json",
]
EXAMPLE_POOL_CACHE_VERSION = "sql_dail_example_pool_v2"
DEFAULT_EXAMPLE_POOL_CACHE_PATH = (
    Path(__file__).resolve().parent / "schema_index" / "sql_dail_example_pool_cache" / "example_pool.pkl"
)


@dataclass(frozen=True)
class ExamplePoolItem:
    """DAIL 示例池中的单条示例。

    字段：
    - id：示例编号。
    - db_id：示例所属 Spider 数据库。
    - question：示例自然语言问题。
    - query：示例 SQL。
    - question_masked / question_pattern：schema/value mask 后的问题形式。
    - path_db：示例数据库路径。
    """

    id: int
    db_id: str
    question: str
    query: str
    question_masked: str
    question_pattern: str
    path_db: str


@lru_cache(maxsize=1)
def load_spider_tables() -> dict[str, dict[str, Any]]:
    """加载 Spider `tables.json`，并按 db_id 建立映射。"""
    tables: list[dict[str, Any]] = []
    for path in [SPIDER_TABLES_PATH, SPIDER_TEST_TABLES_PATH]:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            tables.extend(json.load(f))
    return {table["db_id"]: table for table in tables}


def get_spider_db_path(db_id: str) -> str:
    """根据 Spider db_id 构造本地 SQLite 路径。

    参数：
    - db_id：Spider 数据库 id。
    """
    for root_name in ["database", "test_database"]:
        path = SPIDER_BASE / root_name / db_id / f"{db_id}.sqlite"
        if path.exists():
            return str(path)
    return str(SPIDER_BASE / "database" / db_id / f"{db_id}.sqlite")


def _example_pool_cache_enabled() -> bool:
    return os.getenv("SQL_DAIL_DISABLE_EXAMPLE_POOL_CACHE", "").strip().lower() not in {"1", "true", "yes", "on"}


def _example_pool_cache_path() -> Path:
    configured = os.getenv("SQL_DAIL_EXAMPLE_POOL_CACHE_PATH", "").strip()
    return Path(configured) if configured else DEFAULT_EXAMPLE_POOL_CACHE_PATH


def _hash_file(path: Path) -> str:
    if not path.exists():
        return "missing"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _example_pool_source_paths() -> list[Path]:
    paths = [SPIDER_TABLES_PATH]
    if SPIDER_TRAIN_MERGED_PATH.exists() and SPIDER_TRAIN_LINK_PATH.exists():
        paths.extend([SPIDER_TRAIN_MERGED_PATH, SPIDER_TRAIN_LINK_PATH])
    else:
        paths.extend(SPIDER_TRAIN_PATHS)
    return paths


def _example_pool_cache_metadata() -> dict[str, Any]:
    return {
        "version": EXAMPLE_POOL_CACHE_VERSION,
        "sources": [
            {
                "path": str(path),
                "sha256": _hash_file(path),
            }
            for path in _example_pool_source_paths()
        ],
    }


def _load_cached_example_pool(metadata: dict[str, Any]) -> list[ExamplePoolItem] | None:
    path = _example_pool_cache_path()
    if not _example_pool_cache_enabled() or not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception:
        return None
    if payload.get("metadata") != metadata:
        return None
    items = payload.get("items")
    return items if isinstance(items, list) else None


def _write_cached_example_pool(metadata: dict[str, Any], items: list[ExamplePoolItem]) -> None:
    if not _example_pool_cache_enabled():
        return
    path = _example_pool_cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(
                {
                    "metadata": metadata,
                    "items": items,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
    except Exception:
        return


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL 文件。

    参数：
    - path：JSONL 文件路径。
    """
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _mask_question_with_schema_linking(linking_info: dict[str, Any], mask_tag: str = "<mask>", value_tag: str = "<unk>") -> str:
    """根据 schema linking 结果生成 DAIL masked question。

    参数：
    - linking_info：Spider schema-linking 记录。
    - mask_tag：schema 命中 token 的替换标签。
    - value_tag：值命中 token 的替换标签。
    """
    sc_link = linking_info["sc_link"]
    cv_link = linking_info["cv_link"]
    q_col_match = sc_link["q_col_match"]
    q_tab_match = sc_link["q_tab_match"]
    num_date_match = cv_link["num_date_match"]
    cell_match = cv_link["cell_match"]
    question_for_copying = list(linking_info["question_for_copying"])

    def apply_mask(tokens: list[str], match_ids: list[int], tag: str) -> list[str]:
        """把指定 token 下标替换为标签。

        参数：
        - tokens：问题 token 列表。
        - match_ids：需要替换的 token 下标。
        - tag：替换标签。
        """
        return [tag if idx in match_ids else tok for idx, tok in enumerate(tokens)]

    value_match_q_ids = [int(match.split(",")[0]) for match in list(num_date_match) + list(cell_match)]
    question_toks = apply_mask(question_for_copying, value_match_q_ids, value_tag)

    schema_match_q_ids = [int(match.split(",")[0]) for match in list(q_col_match) + list(q_tab_match)]
    question_toks = apply_mask(question_toks, schema_match_q_ids, mask_tag)
    return " ".join(question_toks)


def _question_pattern_with_schema_linking(linking_info: dict[str, Any]) -> str:
    """根据 schema linking 结果生成 question pattern。

    参数：
    - linking_info：Spider schema-linking 记录。
    """
    sc_link = linking_info["sc_link"]
    cv_link = linking_info["cv_link"]
    q_col_match = sc_link["q_col_match"]
    q_tab_match = sc_link["q_tab_match"]
    num_date_match = cv_link["num_date_match"]
    cell_match = cv_link["cell_match"]
    question_for_copying = list(linking_info["question_for_copying"])

    def apply_mask(tokens: list[str], match_ids: list[int], tag: str) -> list[str]:
        """把指定 token 下标替换为 pattern 标签。

        参数：
        - tokens：问题 token 列表。
        - match_ids：需要替换的 token 下标。
        - tag：替换标签。
        """
        return [tag if idx in match_ids else tok for idx, tok in enumerate(tokens)]

    value_match_q_ids = [int(match.split(",")[0]) for match in list(num_date_match) + list(cell_match)]
    question_toks = apply_mask(question_for_copying, value_match_q_ids, "_")

    schema_match_q_ids = [int(match.split(",")[0]) for match in list(q_col_match) + list(q_tab_match)]
    question_toks = apply_mask(question_toks, schema_match_q_ids, "_")
    return " ".join(question_toks)


@lru_cache(maxsize=1)
def load_spider_dev_targets() -> dict[tuple[str, str], dict[str, Any]]:
    """加载 Spider dev 集的 schema-linking 目标信息。

    返回：
    - `(db_id, question)` 到 masked question、pattern 和 linking 信息的映射。
    """
    if not SPIDER_TEST_LINK_PATH.exists():
        return {}
    with SPIDER_DEV_PATH.open("r", encoding="utf-8") as f:
        dev_rows = json.load(f)
    linking_rows = _load_jsonl(SPIDER_TEST_LINK_PATH)
    targets: dict[tuple[str, str], dict[str, Any]] = {}
    for row, linking in zip(dev_rows, linking_rows):
        key = (row["db_id"], row["question"].strip())
        targets[key] = {
            "question_masked": _mask_question_with_schema_linking(linking),
            "question_pattern": _question_pattern_with_schema_linking(linking),
            "question_for_copying": linking["question_for_copying"],
            "sc_link": linking["sc_link"],
            "cv_link": linking["cv_link"],
            "column_to_table": linking["column_to_table"],
        }
    return targets


def _build_spider_example_pool() -> list[ExamplePoolItem]:
    """加载 DAIL 示例池。

    返回：
    - 由 Spider train 数据构造的 `ExamplePoolItem` 列表。
    """
    table_map = load_spider_tables()
    items: list[ExamplePoolItem] = []
    if SPIDER_TRAIN_MERGED_PATH.exists() and SPIDER_TRAIN_LINK_PATH.exists():
        with SPIDER_TRAIN_MERGED_PATH.open("r", encoding="utf-8") as f:
            rows = json.load(f)
        linking_rows = _load_jsonl(SPIDER_TRAIN_LINK_PATH)
        for next_id, (row, linking) in enumerate(zip(rows, linking_rows)):
            db_id = row["db_id"]
            schema = table_map[db_id]
            items.append(
                ExamplePoolItem(
                    id=next_id,
                    db_id=db_id,
                    question=row["question"].strip(),
                    query=row["query"].strip(),
                    question_masked=_mask_question_with_schema_linking(linking),
                    question_pattern=_question_pattern_with_schema_linking(linking),
                    path_db=get_spider_db_path(db_id),
                )
            )
        return items

    next_id = 0
    for path in SPIDER_TRAIN_PATHS:
        with open(path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        for row in rows:
            db_id = row["db_id"]
            schema = table_map[db_id]
            table_names = schema.get("table_names_original", [])
            column_names = [column_name for _, column_name in schema.get("column_names_original", [])]
            items.append(
                ExamplePoolItem(
                    id=next_id,
                    db_id=db_id,
                    question=row["question"].strip(),
                    query=row["query"].strip(),
                    question_masked=mask_question(row["question"], table_names, column_names),
                    question_pattern=mask_question(row["question"], table_names, column_names, mask_tag="_", value_tag="_"),
                    path_db=get_spider_db_path(db_id),
                )
            )
            next_id += 1
    return items


@lru_cache(maxsize=1)
def load_spider_example_pool() -> list[ExamplePoolItem]:
    """Load the DAIL example pool with a disk cache for deterministic parsed items."""
    metadata = _example_pool_cache_metadata()
    cached = _load_cached_example_pool(metadata)
    if cached is not None:
        return cached
    items = _build_spider_example_pool()
    _write_cached_example_pool(metadata, items)
    return items
