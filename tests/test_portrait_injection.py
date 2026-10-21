"""画像注入：portrait.summarize 摘要 + 注入症状抽取上下文。"""
from app.agent.extraction import build_system_prompt, extract_symptoms
from app.portrait import summarize
from app.schemas import SymptomExtraction


class _FakeStructuredRunnable:
    """返回固定 SymptomExtraction，并记录收到的消息（用于断言画像注入）。"""

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


def test_summarize_full_portrait() -> None:
    """有画像字段时拼成一行中文摘要。"""
    text = summarize(
        {"sex": "女", "birth_year": 1989, "height_cm": 165.0, "weight_kg": 58.0, "bmi": 21.3}
    )
    assert text == "女，1989年生，身高165.0cm，体重58.0kg，BMI 21.3"


def test_summarize_empty_portrait() -> None:
    """无有效字段时返回「暂无画像信息」。"""
    assert summarize({}) == "暂无画像信息"
    assert summarize({"birth_year": None, "sex": None}) == "暂无画像信息"


def test_build_system_prompt_without_portrait_is_base() -> None:
    """无画像摘要时提示词保持原样。"""
    from app.agent.extraction import SYSTEM_PROMPT

    assert build_system_prompt(None) == SYSTEM_PROMPT
    assert build_system_prompt("") == SYSTEM_PROMPT


async def test_extract_symptoms_injects_portrait_summary() -> None:
    """有画像摘要时，系统提示词里包含该摘要。"""
    llm = _FakeLLM(SymptomExtraction(symptoms=["乏力"], subject="self"))
    await extract_symptoms(
        "最近总是累", llm, portrait_summary="女，1989年生，BMI 21.3"
    )

    system_message = llm.runnable.last_messages[0]
    assert "用户健康画像" in system_message.content
    assert "女，1989年生，BMI 21.3" in system_message.content


async def test_extract_symptoms_without_portrait_is_base_prompt() -> None:
    """不传画像摘要时，系统提示词不含画像段落。"""
    from app.agent.extraction import SYSTEM_PROMPT

    llm = _FakeLLM(SymptomExtraction(symptoms=["乏力"], subject="self"))
    await extract_symptoms("最近总是累", llm)

    assert llm.runnable.last_messages[0].content == SYSTEM_PROMPT
