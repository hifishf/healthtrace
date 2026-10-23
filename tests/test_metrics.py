"""metric_records 覆盖语义的确定性断言。"""
from datetime import date
from decimal import Decimal
from uuid import UUID

from sqlalchemy import delete, select

from app import crud
from app.db import async_session
from app.models import MetricRecord

# 测试专用 user_id + metric_code，测完清理，不污染默认用户数据
TEST_USER_ID = UUID("00000000-0000-0000-0000-00000000ffff")
TEST_METRIC = "sleep_duration"
TEST_DATE = date(2026, 10, 20)


async def test_upsert_same_day_overwrites_single_row() -> None:
    """同一天重复提交同一指标 -> 只有 1 行，值被覆盖。"""
    async with async_session() as session:
        # 清理可能的历史残留，保证测试幂等
        await session.execute(
            delete(MetricRecord).where(
                MetricRecord.user_id == TEST_USER_ID,
                MetricRecord.metric_code == TEST_METRIC,
                MetricRecord.recorded_date == TEST_DATE,
            )
        )
        await session.commit()

        try:
            # 第一次写入 6 小时
            id1 = await crud.upsert_metric(
                TEST_USER_ID, TEST_METRIC, Decimal("6"), "h", TEST_DATE
            )
            # 第二次写入 7 小时（同一天同一指标，应覆盖）
            id2 = await crud.upsert_metric(
                TEST_USER_ID, TEST_METRIC, Decimal("7"), "h", TEST_DATE
            )

            # 两次返回同一个 id（同一行）
            assert id1 == id2

            # 表里只有 1 行，且值被覆盖为 7
            rows = (
                await session.execute(
                    select(MetricRecord).where(
                        MetricRecord.user_id == TEST_USER_ID,
                        MetricRecord.metric_code == TEST_METRIC,
                        MetricRecord.recorded_date == TEST_DATE,
                    )
                )
            ).scalars().all()

            assert len(rows) == 1
            assert rows[0].value == Decimal("7")
        finally:
            await session.execute(
                delete(MetricRecord).where(
                    MetricRecord.user_id == TEST_USER_ID,
                    MetricRecord.metric_code == TEST_METRIC,
                    MetricRecord.recorded_date == TEST_DATE,
                )
            )
            await session.commit()
