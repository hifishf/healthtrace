"""父图（Router -> chat/stub）的集成断言（fake router + fake chat 模型）。"""
from datetime import date
from decimal import Decimal

from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy import delete, select

from app.agent.chat import build_chat_graph
from app.agent.graph import build_agent_graph
from app.config import settings
from app.db import async_session
from app.models import MetricRecord
from app.schemas import RouteDecision

TEST_METRIC = "sleep_duration"
TEST_DATE = "2020-01-01"


class _FakeRouterRunnable:
    """返回固定 RouteDecision 的可运行对象。"""

    def __init__(self, decision: RouteDecision) -> None:
        self.decision = decision

    async def ainvoke(self, messages) -> RouteDecision:
        return self.decision


class _FakeRouterLLM:
    """只实现 route() 用到的 with_structured_output 的假 Router。"""

    def __init__(self, decision: RouteDecision) -> None:
        self.decision = decision

    def with_structured_output(self, schema, method) -> _FakeRouterRunnable:
        return _FakeRouterRunnable(self.decision)


class _FakeChatModel:
    """先返回 record_metric tool_call，再返回最终文本。"""

    def __init__(self) -> None:
        self._it = iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "record_metric",
                            "args": {
                                "metric_code": TEST_METRIC,
                                "value": 6,
                                "unit": "h",
                                "date": TEST_DATE,
                            },
                            "id": "call_1",
                        }
                    ],
                ),
                AIMessage(content="已记录：昨晚睡了 6 小时"),
            ]
        )

    async def ainvoke(self, messages):
        return next(self._it)


async def _cleanup() -> None:
    async with async_session() as session:
        await session.execute(
            delete(MetricRecord).where(
                MetricRecord.user_id == settings.default_user_id,
                MetricRecord.metric_code == TEST_METRIC,
                MetricRecord.recorded_date == date.fromisoformat(TEST_DATE),
            )
        )
        await session.commit()


async def test_chat_intent_routes_to_chat_and_records() -> None:
    """chat 意图 -> chat 子图 -> record_metric 落库。"""
    await _cleanup()
    try:
        router = _FakeRouterLLM(
            RouteDecision(intent="chat", confidence=0.9, extracted_slots={})
        )
        chat_graph = build_chat_graph(model=_FakeChatModel())
        graph = build_agent_graph(router_llm=router, chat_graph=chat_graph)

        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="昨晚睡了 6 小时")]}
        )

        assert result["intent"] == "chat"
        assert result["messages"][-1].content == "已记录：昨晚睡了 6 小时"

        async with async_session() as session:
            rows = (
                await session.execute(
                    select(MetricRecord).where(
                        MetricRecord.user_id == settings.default_user_id,
                        MetricRecord.metric_code == TEST_METRIC,
                        MetricRecord.recorded_date == date.fromisoformat(TEST_DATE),
                    )
                )
            ).scalars().all()
        assert len(rows) == 1
        assert rows[0].value == Decimal("6")
    finally:
        await _cleanup()


async def test_non_chat_intent_goes_to_stub() -> None:
    """非 chat 意图 -> stub 占位，不崩溃。"""
    router = _FakeRouterLLM(
        RouteDecision(intent="triage", confidence=0.9, extracted_slots={})
    )
    chat_graph = build_chat_graph(model=_FakeChatModel())
    graph = build_agent_graph(router_llm=router, chat_graph=chat_graph)

    result = await graph.ainvoke({"messages": [HumanMessage(content="最近总是累")]})

    assert result["intent"] == "triage"
    assert "后续周实现" in result["messages"][-1].content
