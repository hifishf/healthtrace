"""不重复追问：回答极性判定 + 肯定回填 + 否定排除（确定性，不依赖 LLM）。"""
import uuid

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command

from app.agent.triage import (
    answer_polarity,
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


def _fatigue_only(text: str) -> SymptomExtraction:
    """首轮「累」抽到乏力；「有/没有」这类回答不抽到新症状（模拟 LLM 抽空）。"""
    if "累" in text:
        return SymptomExtraction(symptoms=["乏力"], subject="self")
    return SymptomExtraction(symptoms=[], subject="self")


async def _drive_interrupts(
    graph,
    config,
    initial_text: str,
    answers: list[str],
) -> tuple[list[str], dict]:
    """跑一遍，收集每轮 interrupt 的问题文本，返回 (问题列表, 最终状态)。"""
    questions: list[str] = []
    result = await graph.ainvoke(initial_triage_state(initial_text), config=config)
    if "__interrupt__" in result:
        questions.append(result["__interrupt__"][0].value.get("question"))
    for answer in answers:
        if "__interrupt__" not in result:
            break
        result = await graph.ainvoke(Command(resume=answer), config=config)
        if "__interrupt__" in result:
            questions.append(result["__interrupt__"][0].value.get("question"))
    return questions, result


def test_answer_polarity_affirm_deny_unknown() -> None:
    """极性判定：肯定/否定/无法判定（否定优先，避免「没有」被「有」误吞）。"""
    assert answer_polarity("有") == "affirm"
    assert answer_polarity("嗯，有的") == "affirm"
    assert answer_polarity("没有") == "deny"
    assert answer_polarity("没这情况") == "deny"
    assert answer_polarity("不是") == "deny"
    assert answer_polarity("随便说说") == "unknown"


async def test_affirmative_answer_backfills_and_not_reasked() -> None:
    """答「有」→ 上一轮症状回填进已采集，下一问不再是同一个问题。"""
    async with AsyncPostgresSaver.from_conn_string(
        settings.checkpoint_database_url
    ) as saver:
        await saver.setup()
        graph = build_triage_graph(llm=_FakeLLM(_fatigue_only), checkpointer=saver)
        config = {"configurable": {"thread_id": f"affirm-{uuid.uuid4().hex[:8]}"}}
        questions, result = await _drive_interrupts(graph, config, "最近总是累", ["有"])

    assert len(questions) >= 2
    assert questions[0] != questions[1]  # 不再重复问同一个症状
    assert len(result["collected_symptoms"]) == 2  # 乏力 + 回填的症状


async def test_negative_answer_ruled_out_and_not_reasked() -> None:
    """答「没有」→ 上一轮症状记入排除集，下一问不再是同一个问题。"""
    async with AsyncPostgresSaver.from_conn_string(
        settings.checkpoint_database_url
    ) as saver:
        await saver.setup()
        graph = build_triage_graph(llm=_FakeLLM(_fatigue_only), checkpointer=saver)
        config = {"configurable": {"thread_id": f"deny-{uuid.uuid4().hex[:8]}"}}
        questions, result = await _drive_interrupts(graph, config, "最近总是累", ["没有"])

    assert len(questions) >= 2
    assert questions[0] != questions[1]  # 不再重复问同一个症状
    assert len(result["ruled_out_symptoms"]) == 1  # 已排除的症状
    assert len(result["collected_symptoms"]) == 1  # 只有乏力，没被回填
