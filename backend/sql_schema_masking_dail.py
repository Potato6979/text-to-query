import re
from typing import Iterable


_WORD_RE = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b")
_QUOTED_RE = re.compile(r"'[^']*'|\"[^\"]*\"")
_NUMBER_RE = re.compile(r"\b-?\d+(?:\.\d+)?\b")

# 中文函数说明补充索引：
# - normalize_identifier(text)：规范化表名、列名和问题 token。
# - iter_schema_terms(table_names, column_names)：从 schema 标识符生成可匹配 term 集合。
# - mask_question_with_schema_terms(question, table_names, column_names)：把问题中的 schema term 标记出来，供 DAIL 示例选择和 prompt 使用。


def normalize_identifier(text: str) -> str:
    """规范化 schema 标识符。

    参数：
    - text：表名、列名或问题 token。
    """
    return text.strip().lower()


def iter_schema_terms(table_names: Iterable[str], column_names: Iterable[str]) -> set[str]:
    """从表名和列名中生成可匹配的 schema term 集合。

    参数：
    - table_names：表名列表。
    - column_names：列名列表。
    """
    terms: set[str] = set()
    for name in table_names:
        norm = normalize_identifier(name)
        if norm:
            terms.add(norm)
            terms.update(part for part in re.split(r"[_\s]+", norm) if part)
    for name in column_names:
        norm = normalize_identifier(name)
        if norm and norm != "*":
            terms.add(norm)
            terms.update(part for part in re.split(r"[_\s]+", norm) if part)
    return terms


def mask_question(
    question: str,
    table_names: Iterable[str],
    column_names: Iterable[str],
    *,
    mask_tag: str = "<mask>",
    value_tag: str = "<unk>",
) -> str:
    """把问题中的 schema 词和值替换为 DAIL mask 标签。

    参数：
    - question：原始自然语言问题。
    - table_names：当前数据库表名。
    - column_names：当前数据库列名。
    - mask_tag：schema token 替换标签。
    - value_tag：数值或引号值替换标签。
    """
    schema_terms = iter_schema_terms(table_names, column_names)

    masked = _QUOTED_RE.sub(value_tag, question)
    masked = _NUMBER_RE.sub(value_tag, masked)

    def replace_word(match: re.Match[str]) -> str:
        """替换单个单词 token。

        参数：
        - match：正则匹配到的问题单词。
        """
        word = match.group(0)
        return mask_tag if normalize_identifier(word) in schema_terms else word.lower()

    masked = _WORD_RE.sub(replace_word, masked)
    masked = re.sub(r"\s+", " ", masked).strip()
    return masked
