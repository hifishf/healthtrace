"""chat 子图 + record_metric 工具的闭环断言（fake LLM，不打真模型）。"""
from datetime import date
from decimal import Decimal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from sqlalchemy import delete, select

from app.agent.chat import build_chat_graph, build_chat_system_prompt
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


class _CaptureModel:
    """记录模型入参，用于验证日期上下文注入。"""

    def __init__(self) -> None:
        self.messages = []

    async def ainvoke(self, messages):
        self.messages = messages
        return AIMessage(content="好的")


async def test_chat_agent_injects_current_date_context() -> None:
    """chat 模型每次调用都能看到当前日期，且不自行换算相对日期。"""
    model = _CaptureModel()
    graph = build_chat_graph(model=model)

    await graph.ainvoke(
        {"messages": [HumanMessage(content="昨天的昨天的昨天睡了 12 小时")]}
    )

    assert isinstance(model.messages[0], SystemMessage)
    assert "当前日期是" in model.messages[0].content
    assert "不要在模型侧把相对日期换算" in model.messages[0].content
    assert "昨天的昨天的昨天" in model.messages[0].content
    assert model.messages[1].content == "昨天的昨天的昨天睡了 12 小时"


def test_build_chat_system_prompt_uses_fixed_date() -> None:
    """提示词可使用固定日期测试，避免依赖机器当前时间。"""
    prompt = build_chat_system_prompt(date(2026, 10, 20))
    assert "当前日期是 2026-10-20" in prompt
