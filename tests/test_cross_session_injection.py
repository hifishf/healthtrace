"""第 7 周 Step 4：跨会话注入（画像 + 最近情景记忆）的确定性断言。"""
import uuid

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from sqlalchemy import delete

from app import crud, memory
from app.agent.chat import build_chat_system_prompt
from app.agent.extraction import SYSTEM_PROMPT, build_system_prompt
from app.agent.triage import build_triage_graph, initial_triage_state
from app.config import settings
from app.db import async_session
from app.memory import summarize_episodic
from app.models import Medication
from app.schemas import SymptomExtraction

UID = settings.default_user_id


def test_summarize_episodic_joins_recent_memories() -> None:
    """近期情景记忆压成带子图标签的摘要。"""
    text = summarize_episodic(
        [
            {"subgraph": "triage", "summary": "已确认乏力", "symptom_text": "乏力"},
            {"subgraph": "chat", "summary": "已记录睡眠"},
        ]
    )
    assert "[triage]" in text
    assert "[chat]" in text
    assert "乏力" in text


def test_build_system_prompt_includes_recent_memory() -> None:
    """抽取提示词带近期记忆，并提示不要重复追问。"""
    prompt = build_system_prompt(recent_memory="[triage] 已确认乏力")
    assert prompt != SYSTEM_PROMPT
    assert "近期会话记忆" in prompt
    assert "不要重复追问" in prompt


def test_chat_system_prompt_includes_portrait() -> None:
    """chat 提示词带画像，并提示已知用药不要重复问。"""
    prompt = build_chat_system_prompt(portrait_summary="在服：二甲双胍")
    assert "二甲双胍" in prompt
    assert "不要重复询问" in prompt


class _CaptureStructuredRunnable:
    """返回空症状，并记录收到的消息（用于断言注入内容）。"""

    def __init__(self) -> None:
        self.last_messages: list = []

    async def ainvoke(self, messages) -> SymptomExtraction:
        self.last_messages = messages
        return SymptomExtraction(symptoms=[], subject="self")


class _CaptureLLM:
    def with_structured_output(self, schema, method):
        return _CaptureStructuredRunnable()


async def test_triage_injects_portrait_and_recent_memory_across_sessions() -> None:
    """上一会话的用药与症状，在新会话的抽取上下文里都能看到。"""
    med_id = await crud.add_medication(
        UID, "二甲双胍", dosage="500mg", frequency="每日两次"
    )
    ep_id = await memory.record_episodic(
        UID,
        "triage",
        "最近总是累",
        summary="已确认乏力",
        symptom_text="乏力",
    )
    capture = _CaptureLLM()
    try:
        async with AsyncPostgresSaver.from_conn_string(
            settings.checkpoint_database_url
        ) as saver:
            await saver.setup()
            graph = build_triage_graph(llm=capture, checkpointer=saver)
            tid = f"inj-{uuid.uuid4().hex[:8]}"
            config = {"configurable": {"thread_id": tid}}
            await graph.ainvoke(initial_triage_state("最近总是累"), config=config)

        system = capture.runnable.last_messages[0].content
        assert "二甲双胍" in system
        assert "乏力" in system
    finally:
        async with async_session() as session:
            await session.execute(delete(Medication).where(Medication.id == med_id))
            await session.commit()
        await memory.delete_episodic(UID, ep_id)
