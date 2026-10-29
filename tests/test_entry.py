"""统一入口 dispatch 的集成断言：一句话路由 chat / triage / stub。"""
import uuid

from langchain_core.messages import AIMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.agent.chat import build_chat_graph
from app.agent.entry import dispatch
from app.agent.triage import build_triage_graph
from app.config import settings
from app.schemas import RouteDecision, SymptomExtraction


class _FakeRouterRunnable:
    def __init__(self, decision: RouteDecision) -> None:
        self.decision = decision

    async def ainvoke(self, messages) -> RouteDecision:
        return self.decision


class _FakeRouterLLM:
    def __init__(self, decision: RouteDecision) -> None:
        self._runnable = _FakeRouterRunnable(decision)

    def with_structured_output(self, schema, method):
        return self._runnable


class _ChatModel:
    async def ainvoke(self, messages):
        return AIMessage(content="已记录：昨晚睡了 6 小时")


class _TriageExtract:
    async def ainvoke(self, messages):
        text = messages[-1].content
        if "累" in text:
            return SymptomExtraction(symptoms=["乏力"], subject="self")
        return SymptomExtraction(symptoms=[], subject="self")


class _TriageLLM:
    def with_structured_output(self, schema, method):
        return _TriageExtract()


class _FakeInsightGraph:
    """返回固定 final_insight 的假 insight 子图。"""

    async def ainvoke(self, state):
        return {
            "final_insight": {
                "intent": "insight",
                "kind": "insight",
                "synthetic": True,
                "n": 120,
                "grade": "强相关",
                "text": "咖啡因摄入与入睡延迟呈强正相关。",
            }
        }


def _decision(intent: str) -> RouteDecision:
    return RouteDecision(intent=intent, confidence=0.9, extracted_slots={})


async def test_entry_routes_chat_to_reply() -> None:
    """一句话记录指标 -> chat 子图 -> 返回最终回复。"""
    chat = build_chat_graph(model=_ChatModel())
    result = await dispatch(
        "昨晚睡了 6 小时",
        _FakeRouterLLM(_decision("chat")),
        chat,
        triage_graph=None,
    )

    assert result["intent"] == "chat"
    assert result["kind"] == "reply"
    assert result["reply"] == "已记录：昨晚睡了 6 小时"


async def test_entry_routes_triage_to_question() -> None:
    """描述症状 -> triage 子图 -> 返回追问问题 + thread_id。"""
    async with AsyncPostgresSaver.from_conn_string(
        settings.checkpoint_database_url
    ) as saver:
        await saver.setup()
        triage = build_triage_graph(llm=_TriageLLM(), checkpointer=saver)
        result = await dispatch(
            "最近总是累",
            _FakeRouterLLM(_decision("triage")),
            chat_graph=None,
            triage_graph=triage,
        )

    assert result["intent"] == "triage"
    assert result["kind"] == "question"
    assert result["thread_id"]
    assert result["question"]


async def test_entry_triage_resume_with_thread_id() -> None:
    """同一 thread_id 续跑追问分诊，状态跨轮保留。"""
    async with AsyncPostgresSaver.from_conn_string(
        settings.checkpoint_database_url
    ) as saver:
        await saver.setup()
        triage = build_triage_graph(llm=_TriageLLM(), checkpointer=saver)
        router = _FakeRouterLLM(_decision("triage"))

        first = await dispatch("最近总是累", router, chat_graph=None, triage_graph=triage)
        tid = first["thread_id"]

        second = await dispatch(
            "有",
            router,
            chat_graph=None,
            triage_graph=triage,
            thread_id=tid,
        )

    assert first["kind"] == "question"
    assert second["thread_id"] == tid
    assert second["kind"] in {"question", "advice"}


async def test_entry_routes_insight_without_graph_and_report_to_stub() -> None:
    """未提供 insight 子图时 insight 回 stub；report 始终回 stub。"""
    for intent in ("insight", "report"):
        result = await dispatch(
            "随便一句话",
            _FakeRouterLLM(_decision(intent)),
            chat_graph=None,
            triage_graph=None,
        )
        assert result["intent"] == intent
        assert result["kind"] == "stub"
        assert "后续周" in result["reply"]


async def test_entry_routes_insight_to_graph() -> None:
    """提供 insight 子图时，insight 意图 -> 返回 final_insight 相关线索。"""
    result = await dispatch(
        "为什么我喝咖啡就睡不好",
        _FakeRouterLLM(_decision("insight")),
        chat_graph=None,
        triage_graph=None,
        insight_graph=_FakeInsightGraph(),
    )

    assert result["intent"] == "insight"
    assert result["kind"] == "insight"
    assert result["final_insight"]["text"]
