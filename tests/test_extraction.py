"""症状抽取（LLM 结构化输出）+ 主体边界兜底的确定性断言。"""
from app.agent.extraction import (
    SYSTEM_PROMPT,
    detect_subject,
    extract_symptoms,
    resolve_subject,
)
from app.schemas import SymptomExtraction


class _FakeStructuredRunnable:
    """返回固定 SymptomExtraction 的可运行对象。"""

    def __init__(self, result: SymptomExtraction) -> None:
        self.result = result
        self.last_messages: list = []

    async def ainvoke(self, messages: list) -> SymptomExtraction:
        self.last_messages = messages
        return self.result


class _FakeLLM:
    """只实现 extract_symptoms 用到的 with_structured_output 的假模型。"""

    def __init__(self, result: SymptomExtraction) -> None:
        self.runnable = _FakeStructuredRunnable(result)

    def with_structured_output(self, schema, method) -> _FakeStructuredRunnable:
        return self.runnable


async def test_extract_symptoms_injects_prompt_and_returns_result() -> None:
    """抽取注入系统提示词并返回结构化结果。"""
    llm = _FakeLLM(SymptomExtraction(symptoms=["乏力"], subject="self"))
    result = await extract_symptoms("最近总是累", llm)

    assert result.symptoms == ["乏力"]
    assert result.subject == "self"
    assert any(
        getattr(message, "content", "") == SYSTEM_PROMPT
        for message in llm.runnable.last_messages
    )


async def test_extract_symptoms_deterministic_subject_overrides_llm() -> None:
    """LLM 说 self，但文本含“我爸爸”时确定性兜底改判 other。"""
    llm = _FakeLLM(SymptomExtraction(symptoms=["胸痛"], subject="self"))
    result = await extract_symptoms("我爸爸胸痛", llm)
    assert result.subject == "other"


def test_detect_subject_other_priority() -> None:
    """他人指代词 -> other（“我爸爸”同时含“我”，也要判 other）。"""
    assert detect_subject("我爸爸胸痛") == "other"
    assert detect_subject("我妈头晕") == "other"
    assert detect_subject("孩子发烧") == "other"


def test_detect_subject_self() -> None:
    """本人指代词 -> self。"""
    assert detect_subject("我最近胸口疼") == "self"
    assert detect_subject("自己有点累") == "self"


def test_detect_subject_none_for_bare_third_party() -> None:
    """裸“他/她”不在确定性词表，返回 None 交给 LLM。"""
    assert detect_subject("他最近头晕") is None
    assert detect_subject("胸口疼") is None


def test_resolve_subject_prefers_deterministic() -> None:
    """确定性判定优先，无法判定时回退 LLM subject。"""
    assert resolve_subject("我爸爸胸痛", "self") == "other"
    assert resolve_subject("我最近累", "other") == "self"
    assert resolve_subject("他最近头晕", "other") == "other"
