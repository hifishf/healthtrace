"""LLM 工厂：按 .env 配置创建 OpenAI 兼容的 ChatOpenAI（默认 DeepSeek）。"""
from langchain_openai import ChatOpenAI

from app.config import settings


def get_llm(temperature: float = 0.0) -> ChatOpenAI:
    """返回 OpenAI 兼容的 ChatOpenAI 实例。

    通过 openai_api_base 指向 DeepSeek 等兼容服务，切换供应商只需改 .env，
    业务代码零改动。路由/结构化输出用 temperature=0 追求确定性。
    """
    return ChatOpenAI(
        model_name=settings.llm_model,
        openai_api_key=settings.llm_api_key,
        openai_api_base=settings.llm_base_url,
        temperature=temperature,
    )
