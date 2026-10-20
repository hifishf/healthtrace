"""chat 子图 + record_metric 工具的闭环断言（fake LLM，不打真模型）。"""
from datetime import date
from decimal import Decimal

from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy import delete, select

from app.agent.chat import build_chat_graph
from app.config import settings
from app.db import async_session
from app.models import MetricRecord

TEST_METRIC = "sleep_duration"
TEST_DATE = "2020-01-01"


class _FakeBoundModel:
    """先返回一个 record_metric 的 tool_call，再返回最终文本。"""

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


async def test_chat_subgraph_records_metric_and_replies() -> None:
    """chat 子图：LLM 决定调 record_metric -> 工具落库 -> 返回最终回复。"""
    await _cleanup()
    try:
        graph = build_chat_graph(model=_FakeBoundModel())
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="昨晚睡了 6 小时")]}
        )

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
