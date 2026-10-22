"""语义评估的 judge：用 DeepSeek 当评委，temperature 固定为 0。

为什么 temperature=0：judge 的职责是「按同一套标准打分」，不是创作；
0 温度让同一份卷子打两次分结果一致，分数才能复现、可对比、可进 CI。
"""
from deepeval.models import DeepSeekModel

from app.config import settings


def build_judge() -> DeepSeekModel:
    """构造语义评估评委（DeepSeek，temperature=0，密钥来自 .env 的 LLM_API_KEY）。"""
    return DeepSeekModel(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        temperature=0.0,
    )
