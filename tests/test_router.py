"""Router 结构化输出与阈值回落的确定性断言。"""
import pytest
from pydantic import ValidationError

from app.agent.router import SYSTEM_PROMPT, apply_confidence_fallback, route
from app.schemas import RouteDecision


class _FakeStructuredRunnable:
    """返回固定 RouteDecision 的可运行对象（替代真实 LLM 的结构化输出）。"""

    def __init__(self, decision: RouteDecision) -> None:
        self.decision = decision
        self.last_messages: list = []

    async def ainvoke(self, messages: list) -> RouteDecision:
        self.last_messages = messages
        return self.decision


class _FakeLLM:
    """只实现 route() 用到的 with_structured_output 的假模型。"""

    def __init__(self, decision: RouteDecision) -> None:
        self.runnable = _FakeStructuredRunnable(decision)

    def with_structured_output(self, schema, method: str) -> _FakeStructuredRunnable:
        return self.runnable


def test_route_decision_rejects_unknown_intent() -> None:
    """intent 只能是 4 个枚举值。"""
    with pytest.raises(ValidationError):
        RouteDecision(intent="unknown", confidence=0.9, extracted_slots={})


def test_apply_fallback_low_confidence_to_chat() -> None:
    """低置信度 -> 强制 chat。"""
    decision = RouteDecision(intent="triage", confidence=0.3, extracted_slots={})
    result = apply_confidence_fallback(decision, 0.7)
    assert result.intent == "chat"
    assert result.confidence == 0.3


def test_apply_fallback_keeps_high_confidence() -> None:
    """高置信度 -> 保持原意图。"""
    decision = RouteDecision(intent="triage", confidence=0.9, extracted_slots={})
    result = apply_confidence_fallback(decision, 0.7)
    assert result.intent == "triage"


async def test_route_injects_system_prompt_and_returns_decision() -> None:
    """route() 注入系统提示词并返回分类结果。"""
    llm = _FakeLLM(RouteDecision(intent="chat", confidence=0.9, extracted_slots={}))
    result = await route("昨晚睡了 6 小时", llm)

    assert result.intent == "chat"
    # 系统提示词确实被注入（防止误删路由口径）
    assert any(
        getattr(m, "content", "") == SYSTEM_PROMPT for m in llm.runnable.last_messages
    )


async def test_route_low_confidence_falls_back_to_chat() -> None:
    """route() 全链路：低置信度经阈值回落为 chat。"""
    llm = _FakeLLM(RouteDecision(intent="triage", confidence=0.3, extracted_slots={}))
    result = await route("随便一句话", llm)
    assert result.intent == "chat"
