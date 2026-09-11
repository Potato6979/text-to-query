import sqlite3
from pathlib import Path
from typing import Any

# 中文函数说明索引：
# - decode_sqlite_text(data)：容错解码 SQLite TEXT bytes，避免脏数据中断执行。
# - connect_sqlite(path, **kwargs)：创建统一 text_factory 的 SQLite 连接。


def decode_sqlite_text(data: bytes) -> str:
    """把 SQLite TEXT 原始字节安全解码为字符串。

    参数：
    - data：SQLite 返回的原始 bytes。

    说明：
    - 使用 `errors="replace"` 容忍历史数据集中的坏编码，避免执行层因为单个字符失败。
    """
    return data.decode("utf-8", errors="replace")


def connect_sqlite(path: str | Path, **kwargs: Any) -> sqlite3.Connection:
    """创建带统一 text_factory 的 SQLite 连接。

    参数：
    - path：SQLite 数据库文件路径。
    - kwargs：传给 `sqlite3.connect(...)` 的其他连接参数。
    """
    conn = sqlite3.connect(path, **kwargs)
    conn.text_factory = decode_sqlite_text
    return conn
