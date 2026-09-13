import hashlib
import importlib.metadata
import json
import math
import os
import pickle
import platform
import random
import re
import sys
import warnings
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from .sql_example_pool_dail import ExamplePoolItem

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None

warnings.filterwarnings(
    "ignore",
    message="`resume_download` is deprecated",
    category=FutureWarning,
)

EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_MODEL_DEVICE = "cpu"
EMBEDDING_ENCODE_KWARGS = {
    "batch_size": 32,
    "show_progress_bar": False,
    "convert_to_numpy": True,
    "normalize_embeddings": False,
}
EMBEDDING_CACHE_VERSION = "sql_dail_embeddings_v2"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EMBEDDING_CACHE_DIR = PROJECT_ROOT / "schema_index" / "sql_dail_embedding_cache"
MODEL_FINGERPRINT_FILES = [
    "config.json",
    "config_sentence_transformers.json",
    "model.safetensors",
    "modules.json",
    "sentence_bert_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "1_Pooling/config.json",
]

# 中文函数说明补充索引：
# - jaccard_similarity(text_a, text_b)：计算 skeleton/token 的 Jaccard 相似度。
# - select_examples(...)：按 masked question、skeleton、embedding 等策略选择 DAIL few-shot 示例。
# - load_embedding_model(...)：按需加载 sentence-transformers 模型。
# - embedding_similarity(...)：计算问题与示例的语义相似度。


def jaccard_similarity(text_a: str, text_b: str) -> float:
    """计算两个 skeleton 文本的 Jaccard 相似度。

    参数：
    - text_a：第一个文本。
    - text_b：第二个文本。
    """
    counts_a = Counter(text_a.split())
    counts_b = Counter(text_b.split())
    if not counts_a and not counts_b:
        return 1.0
    intersection = sum(min(counts_a[token], counts_b[token]) for token in counts_a if token in counts_b)
    union = sum(counts_a.values()) + sum(counts_b.values()) - intersection
    return float(intersection) / union if union else 0.0


def _tokenize_for_fallback(text: str) -> Counter[str]:
    """在 sentence-transformer 不可用时使用的简单词袋分词。

    参数：
    - text：待编码文本。
    """
    return Counter(re.findall(r"[a-zA-Z_<>]+", text.lower()))


def _embedding_cache_enabled() -> bool:
    return os.getenv("SQL_DAIL_DISABLE_EMBEDDING_CACHE", "").strip().lower() not in {"1", "true", "yes", "on"}


def _embedding_cache_dir() -> Path:
    configured = os.getenv("SQL_DAIL_EMBEDDING_CACHE_DIR", "").strip()
    return Path(configured) if configured else DEFAULT_EMBEDDING_CACHE_DIR


def _hash_texts(texts: list[str]) -> str:
    payload = json.dumps(texts, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _json_hash(payload: object) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _package_version(package_name: str) -> str:
    try:
        return importlib.metadata.version(package_name)
    except Exception:
        return "unavailable"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=4)
def _model_snapshot_path(model_name: str) -> str:
    try:
        from huggingface_hub import snapshot_download

        return str(Path(snapshot_download(model_name, local_files_only=True)).resolve())
    except Exception:
        return ""


@lru_cache(maxsize=4)
def _model_file_fingerprint(model_name: str) -> dict[str, object]:
    snapshot_path = _model_snapshot_path(model_name)
    if not snapshot_path:
        return {
            "snapshot_path": "",
            "files": {},
            "status": "snapshot_unavailable",
        }
    root = Path(snapshot_path)
    files: dict[str, dict[str, object]] = {}
    for relative in MODEL_FINGERPRINT_FILES:
        path = root / relative
        if not path.exists() or not path.is_file():
            files[relative] = {
                "exists": False,
                "size": 0,
                "sha256": "missing",
            }
            continue
        files[relative] = {
            "exists": True,
            "size": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
    return {
        "snapshot_path": snapshot_path,
        "files": files,
        "status": "ok",
    }


def _runtime_fingerprint(*, backend: str, model_name: str, force_bow: bool) -> dict[str, object]:
    packages = {
        "sentence-transformers": _package_version("sentence-transformers"),
        "transformers": _package_version("transformers"),
        "torch": _package_version("torch"),
        "huggingface-hub": _package_version("huggingface-hub"),
        "numpy": _package_version("numpy"),
    }
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "backend": backend,
        "model_name": model_name,
        "model_device": EMBEDDING_MODEL_DEVICE if backend == "sentence_transformer" else "none",
        "force_bow": force_bow,
        "encode_kwargs": EMBEDDING_ENCODE_KWARGS if backend == "sentence_transformer" else {},
        "model_files": _model_file_fingerprint(model_name) if backend == "sentence_transformer" else {},
    }


def _cache_path(*, metadata: dict[str, object]) -> Path:
    key_payload = json.dumps(
        metadata,
        sort_keys=True,
        separators=(",", ":"),
    )
    cache_key = hashlib.sha256(key_payload.encode("utf-8")).hexdigest()
    return _embedding_cache_dir() / f"{cache_key}.pkl"


def _load_cached_embeddings(path: Path, metadata: dict[str, object]):
    if not _embedding_cache_enabled() or not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception:
        return None
    if payload.get("metadata") != metadata:
        return None
    return payload.get("question_embeddings"), payload.get("masked_embeddings")


def _write_cached_embeddings(path: Path, metadata: dict[str, object], question_embeddings, masked_embeddings) -> None:
    if not _embedding_cache_enabled():
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(
                {
                    "metadata": metadata,
                    "question_embeddings": question_embeddings,
                    "masked_embeddings": masked_embeddings,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
    except Exception:
        return


class _QuestionEmbedder:
    """DAIL 示例选择用的问题编码器。

    参数：
    - force_bow：是否强制使用词袋编码，而不加载 sentence-transformer。
    """

    def __init__(self, force_bow: bool = False) -> None:
        """初始化编码器。

        参数：
        - force_bow：是否强制词袋模式。
        """
        self._model = None
        self.force_bow = force_bow
        if not self.force_bow and SentenceTransformer is not None:
            try:
                self._model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=EMBEDDING_MODEL_DEVICE)
            except Exception:
                self._model = None
        self.backend = "sentence_transformer" if self._model is not None else "bow"
        self.model_name = EMBEDDING_MODEL_NAME if self._model is not None else "bow"

    def encode_many(self, texts: Iterable[str]):
        """批量编码文本。

        参数：
        - texts：待编码文本序列。
        """
        texts = list(texts)
        if self._model is not None:
            return self._model.encode(texts, **EMBEDDING_ENCODE_KWARGS)
        return [_tokenize_for_fallback(text) for text in texts]

    def distance(self, encoded_a, encoded_b) -> float:
        """计算两个编码向量或词袋之间的距离。

        参数：
        - encoded_a：第一个编码结果。
        - encoded_b：第二个编码结果。
        """
        if self._model is not None:
            total = 0.0
            for a, b in zip(encoded_a, encoded_b):
                total += (float(a) - float(b)) ** 2
            return math.sqrt(total)

        keys = set(encoded_a) | set(encoded_b)
        total = 0.0
        for key in keys:
            total += float(encoded_a.get(key, 0) - encoded_b.get(key, 0)) ** 2
        return math.sqrt(total)


class DAILExampleSelector:
    """DAIL few-shot 示例选择器。

    参数：
    - example_pool：可选示例池。
    - random_seed：random selector 使用的随机种子。
    - force_bow：是否强制使用词袋编码。
    """

    def __init__(
        self,
        example_pool: list[ExamplePoolItem],
        random_seed: int = 42,
        force_bow: bool | None = None,
    ) -> None:
        """初始化示例选择器并预编码问题文本。

        参数：
        - example_pool：示例池。
        - random_seed：随机种子。
        - force_bow：是否强制词袋模式；为空时读取环境变量。
        """
        self.example_pool = example_pool
        self.random = random.Random(random_seed)
        if force_bow is None:
            force_bow = os.getenv("SQL_DAIL_FORCE_BOW", "").strip().lower() in {"1", "true", "yes", "on"}
        self.force_bow = force_bow
        self.embedder = _QuestionEmbedder(force_bow=force_bow)
        self._question_texts = [item.question for item in example_pool]
        self._masked_texts = [item.question_masked for item in example_pool]
        question_hash = _hash_texts(self._question_texts)
        masked_hash = _hash_texts(self._masked_texts)
        runtime_fingerprint = _runtime_fingerprint(
            backend=self.embedder.backend,
            model_name=self.embedder.model_name,
            force_bow=self.force_bow,
        )
        cache_metadata = {
            "version": EMBEDDING_CACHE_VERSION,
            "count": len(example_pool),
            "question_hash": question_hash,
            "masked_hash": masked_hash,
            "runtime_fingerprint": runtime_fingerprint,
        }
        cache_path = _cache_path(metadata=cache_metadata)
        cached = _load_cached_embeddings(cache_path, cache_metadata)
        if cached is not None:
            self._question_embeddings, self._masked_embeddings = cached
            self.embedding_cache = {
                "enabled": _embedding_cache_enabled(),
                "hit": True,
                "path": str(cache_path),
                "backend": self.embedder.backend,
                "model_name": self.embedder.model_name,
                "metadata_hash": _json_hash(cache_metadata),
            }
        else:
            self._question_embeddings = self.embedder.encode_many(self._question_texts)
            self._masked_embeddings = self.embedder.encode_many(self._masked_texts)
            _write_cached_embeddings(
                cache_path,
                cache_metadata,
                self._question_embeddings,
                self._masked_embeddings,
            )
            self.embedding_cache = {
                "enabled": _embedding_cache_enabled(),
                "hit": False,
                "path": str(cache_path),
                "backend": self.embedder.backend,
                "model_name": self.embedder.model_name,
                "metadata_hash": _json_hash(cache_metadata),
            }
        self.embedding_cache_metadata = cache_metadata

    def rank(
        self,
        *,
        selector_type: str,
        question: str,
        question_masked: str,
        db_id: str | None,
        cross_domain_only: bool = True,
    ) -> list[ExamplePoolItem]:
        """按指定策略对示例池排序。

        参数：
        - selector_type：选择策略，支持 random、question、masked_question。
        - question：原始自然语言问题。
        - question_masked：schema/value mask 后的问题。
        - db_id：当前目标数据库 id。
        - cross_domain_only：是否只选非当前数据库示例。
        """
        candidate_indexes = self._iter_candidates(db_id, cross_domain_only)
        if selector_type == "random":
            shuffled = candidate_indexes[:]
            seed_material = f"{question}\n{question_masked}\n{db_id or ''}"
            stable_seed = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:16], 16)
            random.Random(stable_seed).shuffle(shuffled)
            return [self.example_pool[idx] for idx in shuffled]

        if selector_type == "question":
            target_embedding = self.embedder.encode_many([question])[0]
            score_source = self._question_embeddings
        else:
            target_embedding = self.embedder.encode_many([question_masked])[0]
            score_source = self._masked_embeddings

        ranked: list[tuple[float, int]] = []
        for idx in candidate_indexes:
            item = self.example_pool[idx]
            if item.question == question:
                continue
            ranked.append((self.embedder.distance(target_embedding, score_source[idx]), idx))
        ranked.sort(key=lambda pair: pair[0])

        if selector_type in {"masked_question", "question"}:
            return [self.example_pool[idx] for _, idx in ranked]

        raise ValueError(f"Unsupported selector_type: {selector_type}")

    def _iter_candidates(self, db_id: str | None, cross_domain_only: bool) -> list[int]:
        """生成候选示例下标。

        参数：
        - db_id：当前目标数据库 id。
        - cross_domain_only：是否排除同库示例。
        """
        indexes = list(range(len(self.example_pool)))
        if cross_domain_only and db_id is not None:
            indexes = [idx for idx in indexes if self.example_pool[idx].db_id != db_id]
        return indexes

    def select(
        self,
        *,
        selector_type: str,
        question: str,
        question_masked: str,
        db_id: str | None,
        num_examples: int,
        cross_domain_only: bool = True,
    ) -> list[ExamplePoolItem]:
        """选择指定数量的 DAIL 示例。

        参数：
        - selector_type：选择策略。
        - question：原始自然语言问题。
        - question_masked：mask 后的问题。
        - db_id：当前目标数据库 id。
        - num_examples：需要返回的示例数量。
        - cross_domain_only：是否排除同库示例。
        """
        ranked = self.rank(
            selector_type=selector_type,
            question=question,
            question_masked=question_masked,
            db_id=db_id,
            cross_domain_only=cross_domain_only,
        )
        return ranked[:num_examples]
