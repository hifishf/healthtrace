"""统一 LLM 调用层：工厂、超时/重试封装与失败原因分类。业务代码只经 invoke_llm 调用，失败统一抛 LLMUnavailable 并交由调用方降级到规则/模板。"""
from typing import Any

from langchain_openai import ChatOpenAI

from app.config import settings


class LLMUnavailable(Exception):
    """LLM 调用失败（超时/5xx/网络等）且重试耗尽后的统一异常。"""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"LLM 调用失败：{reason}")


def classify_llm_reason(exc: Exception) -> str:
    """把底层异常归类为 fallback_reason：超时 / 其它不可用。"""
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    if "timeout" in name or "timed out" in text or "read timed" in text:
        return "llm_timeout"
    return "llm_unavailable"


def get_llm(temperature: float = 0.0) -> ChatOpenAI:
    """返回 OpenAI 兼容的 ChatOpenAI 实例（默认 DeepSeek）。

    request_timeout / max_tokens 做单次调用与单响应成本限制；SDK 层重试置 0，
    重试由 invoke_llm 统一控制，便于观测与测试。
    """
    return ChatOpenAI(
        model_name=settings.llm_model,
        openai_api_key=settings.llm_api_key,
        openai_api_base=settings.llm_base_url,
        temperature=temperature,
        request_timeout=settings.llm_timeout_seconds,
        max_retries=0,
        max_tokens=settings.llm_max_tokens,
    )


async def invoke_llm(runnable, input, retries: int | None = None) -> Any:
    """带重试的 LLM 调用；重试耗尽后抛 LLMUnavailable。

    runnable 为 ChatOpenAI 或 with_structured_output 生成的可运行对象，
    input 为消息列表；重试次数默认取配置 llm_max_retries。
    """
    attempts = (settings.llm_max_retries if retries is None else retries) + 1
    last_exc: Exception | None = None
    for _ in range(attempts):
        try:
            return await runnable.ainvoke(input)
        except Exception as exc:  # noqa: BLE001 - 统一兜底后转 LLMUnavailable
            last_exc = exc
    raise LLMUnavailable(classify_llm_reason(last_exc)) from last_exc
