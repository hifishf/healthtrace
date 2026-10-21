"""追问分诊状态机：interrupt/Command 挂起恢复 + 硬上限 + 红旗 + 主体边界。"""
import uuid

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command

from app.agent.triage import (
    MAX_FOLLOWUP_ROUNDS,
    build_triage_graph,
    initial_triage_state,
)
from app.config import settings
from app.schemas import SymptomExtraction


class _FakeStructuredRunnable:
    """按最后一条用户消息内容返回 SymptomExtraction 的可运行对象。"""

    def __init__(self, extract_fn) -> None:
        self.extract_fn = extract_fn

    async def ainvoke(self, messages: list) -> SymptomExtraction:
        text = messages[-1].content
        return self.extract_fn(text)


class _FakeLLM:
    """只实现 extract_symptoms 用到的 with_structured_output 的假模型。"""

    def __init__(self, extract_fn) -> None:
        self.runnable = _FakeStructuredRunnable(extract_fn)

    def with_structured_output(self, schema, method) -> _FakeStructuredRunnable:
        return self.runnable


def _fatigue_then_empty(text: str) -> SymptomExtraction:
    """首轮「累」抽到乏力，后续回答不抽到新症状（用于驱动到硬上限）。"""
    if "累" in text:
        return SymptomExtraction(symptoms=["乏力"], subject="self")
    return SymptomExtraction(symptoms=[], subject="self")


async def _drive(graph, config, initial_text: str, answers: list[str]) -> dict:
    """跑一遍：首次 invoke，然后按 __interrupt__ 逐条 resume 直到结束。"""
    result = await graph.ainvoke(initial_triage_state(initial_text), config=config)
    for answer in answers:
        if "__interrupt__" not in result:
            break
        result = await graph.ainvoke(Command(resume=answer), config=config)
    return result


async def test_hard_cap_followup_round_not_exceed() -> None:
    """「最近总是累」最多 3 轮追问，到达上限后给出 final_advice。"""
    async with AsyncPostgresSaver.from_conn_string(
        settings.checkpoint_database_url
    ) as saver:
        await saver.setup()
        graph = build_triage_graph(llm=_FakeLLM(_fatigue_then_empty), checkpointer=saver)
        config = {"configurable": {"thread_id": f"hardcap-{uuid.uuid4().hex[:8]}"}}
        result = await _drive(graph, config, "最近总是累", ["有", "有", "有"])

    assert result["followup_round"] == MAX_FOLLOWUP_ROUNDS
    assert result["followup_round"] <= MAX_FOLLOWUP_ROUNDS
    assert result["final_advice"] is not None
    assert result["final_advice"]["level"] in {"observe", "adjust", "online_consult", "seek_care"}
    assert "__interrupt__" not in result


async def test_red_flag_triggers_immediately() -> None:
    """「胸口疼得喘不上气」触发红旗，立即就医并终止。"""
    async with AsyncPostgresSaver.from_conn_string(
        settings.checkpoint_database_url
    ) as saver:
        await saver.setup()
        graph = build_triage_graph(
            llm=_FakeLLM(lambda text: SymptomExtraction(symptoms=[], subject="self")),
            checkpointer=saver,
        )
        config = {"configurable": {"thread_id": f"redflag-{uuid.uuid4().hex[:8]}"}}
        result = await graph.ainvoke(
            initial_triage_state("胸口疼得喘不上气"), config=config
        )

    assert result["red_flag_triggered"] is True
    assert result["final_advice"]["level"] == "emergency"
    assert "__interrupt__" not in result


async def test_other_subject_accompany_not_red_flag() -> None:
    """「我爸爸胸痛」不触发红旗，输出陪同就医。"""
    async with AsyncPostgresSaver.from_conn_string(
        settings.checkpoint_database_url
    ) as saver:
        await saver.setup()
        graph = build_triage_graph(
            llm=_FakeLLM(lambda text: SymptomExtraction(symptoms=["胸痛"], subject="self")),
            checkpointer=saver,
        )
        config = {"configurable": {"thread_id": f"other-{uuid.uuid4().hex[:8]}"}}
        result = await graph.ainvoke(initial_triage_state("我爸爸胸痛"), config=config)

    assert result["red_flag_triggered"] is False
    assert result["final_advice"]["level"] == "accompany"
    assert "陪同就医" in result["final_advice"]["suggestion"]
    assert "__interrupt__" not in result
