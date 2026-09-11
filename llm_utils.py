import time
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError

from config import (
    DEEPSEEK_MODEL,
    LLM_REQUEST_TIMEOUT,
    LLM_SEED,
    LLM_TEMPERATURE,
)


RETRYABLE_LLM_ERRORS = (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    APIStatusError,
)


def call_chat_completion(
    client: OpenAI,
    prompt: str,
    *,
    trace: list[dict[str, Any]] | None = None,
    label: str = "",
    max_attempts: int = 3,
    base_delay_seconds: float = 1.5,
) -> str:
    """调用聊天模型并对临时 API / 网络错误做轻量重试。

    参数：
    - client：OpenAI 兼容客户端。
    - prompt：发送给模型的完整提示词。
    - trace：可选，记录 prompt、response 和重试错误，供 dev 面板和预算估算使用。
    - label：本次 LLM 调用的标签，例如 `sql_generation`。
    - max_attempts：最大尝试次数。
    - base_delay_seconds：重试退避的基础等待秒数。
    """
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=LLM_TEMPERATURE,
                seed=LLM_SEED,
                timeout=LLM_REQUEST_TIMEOUT,
            )
            choices = getattr(response, "choices", None) or []
            if not choices:
                if trace is not None:
                    trace.append(
                        {
                            "label": f"{label or 'llm_call'}_malformed_response",
                            "error": "LLM response did not contain choices.",
                        }
                    )
                raise RuntimeError("LLM response did not contain choices.")
            message = getattr(choices[0], "message", None)
            content = (getattr(message, "content", None) or "").strip()
            if trace is not None:
                trace.append({"label": label or "llm_call", "prompt": prompt, "response": content})
            return content
        except RETRYABLE_LLM_ERRORS as exc:
            last_error = exc
            if trace is not None:
                trace.append(
                    {
                        "label": f"{label or 'llm_call'}_retry",
                        "attempt": attempt,
                        "error": str(exc),
                    }
                )
            if attempt >= max_attempts:
                break
            time.sleep(base_delay_seconds * attempt)

    assert last_error is not None
    raise last_error
