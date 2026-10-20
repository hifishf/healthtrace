"""record_metric 工具的确定性断言。"""
from datetime import date
from decimal import Decimal

import pytest
from sqlalchemy import delete, select

from app.agent.tools import resolve_metric_date, record_metric
from app.config import settings
from app.db import async_session
from app.models import MetricRecord

TEST_METRIC = "sleep_duration"
TEST_DATE = "2020-01-01"


async def _cleanup() -> None:
    """清理测试写入的指标行，保证测试幂等。"""
    async with async_session() as session:
        await session.execute(
            delete(MetricRecord).where(
                MetricRecord.user_id == settings.default_user_id,
                MetricRecord.metric_code == TEST_METRIC,
                MetricRecord.recorded_date == date.fromisoformat(TEST_DATE),
            )
        )
        await session.commit()


async def test_record_metric_ok_and_overwrite() -> None:
    """同一天重复录睡眠 -> 仍只有 1 行，值被覆盖。"""
    await _cleanup()
    try:
        first = await record_metric.ainvoke(
            {"metric_code": TEST_METRIC, "value": 6, "unit": "h", "date": TEST_DATE}
        )
        second = await record_metric.ainvoke(
            {"metric_code": TEST_METRIC, "value": 7, "unit": "h", "date": TEST_DATE}
        )
        assert first["ok"] is True
        assert second["ok"] is True
        assert first["id"] == second["id"]

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
        assert rows[0].value == Decimal("7")
    finally:
        await _cleanup()


async def test_record_metric_invalid_metric_code() -> None:
    """空 metric_code -> 返回错误码，不落库。"""
    result = await record_metric.ainvoke(
        {"metric_code": "  ", "value": 6, "unit": "h", "date": TEST_DATE}
    )
    assert result["ok"] is False
    assert result["error_code"] == "INVALID_METRIC_CODE"


async def test_record_metric_invalid_date() -> None:
    """非法日期 -> 返回错误码。"""
    result = await record_metric.ainvoke(
        {"metric_code": TEST_METRIC, "value": 6, "unit": "h", "date": "not-a-date"}
    )
    assert result["ok"] is False
    assert result["error_code"] == "INVALID_DATE"


@pytest.mark.parametrize(
    ("date_text", "expected"),
    [
        ("2026-02-11", date(2026, 2, 11)),
        ("今天", date(2026, 10, 20)),
        ("昨晚", date(2026, 10, 19)),
        ("前天", date(2026, 10, 18)),
        ("三天前", date(2026, 10, 17)),
        ("昨天的昨天的昨天", date(2026, 10, 17)),
    ],
)
def test_resolve_metric_date(date_text: str, expected: date) -> None:
    """绝对日期和相对日期都按固定基准日确定性换算。"""
    assert resolve_metric_date(date_text, today=date(2026, 10, 20)) == expected
