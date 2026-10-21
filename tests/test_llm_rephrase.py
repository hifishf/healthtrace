"""LLM 转写层：追问话术与建议措辞交给 LLM，失败回退确定性模板。"""
import uuid

from langchain_core.messages import AIMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command

from app.agent.triage import _SUGGESTIONS, build_triage_graph, initial_triage_state
from app.config import settings
from app.schemas import SymptomExtraction


class _FakeRunnable:
    """抽症状用的结构化输出假对象。"""

    def __init__(self, extract_fn) -> None:
        self.extract_fn = extract_fn

    async def ainvoke(self, messages: list) -> SymptomExtraction:
        return self.extract_fn(messages[-1].content)


class _FakeLLM:
    """同时支持抽症状（with_structured_output）与转写（ainvoke）的假模型。"""

    def __init__(self, extract_fn, rephrase_fn=None, advice_fn=None) -> None:
        self._runnable = _FakeRunnable(extract_fn)
        self._rephrase_fn = rephrase_fn
        self._advice_fn = advice_fn

    def with_structured_output(self, schema, method) -> _FakeRunnable:
        return self._runnable

    async def ainvoke(self, messages: list) -> AIMessage:
        joined = "\n".join(getattr(message, "content", "") for message in messages)
        if "待确认症状" in joined:
            if self._rephrase_fn is None:
                raise RuntimeError("LLM 转写不可用")
            return AIMessage(content=self._rephrase_fn())
        if "建议级别" in joined:
            if self._advice_fn is None:
                raise RuntimeError("LLM 转写不可用")
            return AIMessage(content=self._advice_fn())
        raise RuntimeError("未知提示词")


def _fatigue_then_empty(text: str) -> SymptomExtraction:
    """首轮「累」抽到乏力，后续回答不抽到新症状。"""
    if "累" in text:
        return SymptomExtraction(symptoms=["乏力"], subject="self")
    return SymptomExtraction(symptoms=[], subject="self")


async def _drive(graph, config, initial_text: str, answers: list[str]) -> dict:
    """跑一遍：首次 invoke，然后按 interrupt 逐条 resume 直到结束。"""
    result = await graph.ainvoke(initial_triage_state(initial_text), config=config)
    for answer in answers:
        if "__interrupt__" not in result:
            break
        result = await graph.ainvoke(Command(resume=answer), config=config)
    return result


async def test_question_rephrased_by_llm() -> None:
    """追问话术由 LLM 转写（不用模板）。"""
    llm = _FakeLLM(_fatigue_then_empty, rephrase_fn=lambda: "最近会不会觉得特别没力气呀？")
    async with AsyncPostgresSaver.from_conn_string(
        settings.checkpoint_database_url
    ) as saver:
        await saver.setup()
        graph = build_triage_graph(llm=llm, checkpointer=saver)
        config = {"configurable": {"thread_id": f"q-{uuid.uuid4().hex[:8]}"}}
        result = await graph.ainvoke(initial_triage_state("最近总是累"), config=config)

    assert result["__interrupt__"][0].value.get("question") == "最近会不会觉得特别没力气呀？"


async def test_question_falls_back_to_template_when_llm_fails() -> None:
    """LLM 转写失败 → 追问回退确定性模板。"""
    llm = _FakeLLM(_fatigue_then_empty, rephrase_fn=None)
    async with AsyncPostgresSaver.from_conn_string(
        settings.checkpoint_database_url
    ) as saver:
        await saver.setup()
        graph = build_triage_graph(llm=llm, checkpointer=saver)
        config = {"configurable": {"thread_id": f"qf-{uuid.uuid4().hex[:8]}"}}
        result = await graph.ainvoke(initial_triage_state("最近总是累"), config=config)

    question = result["__interrupt__"][0].value.get("question")
    assert "请问你最近有没有" in question
    assert "的情况？" in question


async def test_advice_written_by_llm() -> None:
    """最终建议措辞由 LLM 转写（级别仍由规则算）。"""
    llm = _FakeLLM(
        _fatigue_then_empty,
        rephrase_fn=lambda: "再问一句？",
        advice_fn=lambda: "建议你先好好休息，观察几天变化。",
    )
    async with AsyncPostgresSaver.from_conn_string(
        settings.checkpoint_database_url
    ) as saver:
        await saver.setup()
        graph = build_triage_graph(llm=llm, checkpointer=saver)
        config = {"configurable": {"thread_id": f"a-{uuid.uuid4().hex[:8]}"}}
        result = await _drive(graph, config, "最近总是累", ["有", "有", "有"])

    assert result["final_advice"]["suggestion"] == "建议你先好好休息，观察几天变化。"


async def test_advice_falls_back_to_template_when_llm_fails() -> None:
    """LLM 转写失败 → 建议回退该级别的模板，级别仍由规则算。"""
    llm = _FakeLLM(_fatigue_then_empty, rephrase_fn=None, advice_fn=None)
    async with AsyncPostgresSaver.from_conn_string(
        settings.checkpoint_database_url
    ) as saver:
        await saver.setup()
        graph = build_triage_graph(llm=llm, checkpointer=saver)
        config = {"configurable": {"thread_id": f"af-{uuid.uuid4().hex[:8]}"}}
        result = await _drive(graph, config, "最近总是累", ["有", "有", "有"])

    level = result["final_advice"]["level"]
    assert result["final_advice"]["suggestion"] == _SUGGESTIONS[level]
