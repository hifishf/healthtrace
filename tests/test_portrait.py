"""健康画像读写：读画像、record 后刷新 updated_at 的确定性断言。"""
import datetime

from sqlalchemy import delete

from app import portrait
from app.agent.tools import query_profile, record_metric
from app.config import settings
from app.db import async_session
from app.models import HealthProfile, MetricRecord

TEST_METRIC = "sleep_duration"
TEST_DATE = "2020-01-01"


async def _cleanup_metric() -> None:
    async with async_session() as session:
        await session.execute(
            delete(MetricRecord).where(
                MetricRecord.user_id == settings.default_user_id,
                MetricRecord.metric_code == TEST_METRIC,
                MetricRecord.recorded_date == datetime.date.fromisoformat(TEST_DATE),
            )
        )
        await session.commit()


async def test_get_health_portrait_returns_seed_fields() -> None:
    """读画像返回种子用户的完整结构：基础字段 + 五个子表列表。"""
    result = await portrait.get_health_portrait(settings.default_user_id)
    assert result["user_id"] == str(settings.default_user_id)
    assert set(result.keys()) == {
        "user_id",
        "birth_year",
        "sex",
        "height_cm",
        "weight_kg",
        "bmi",
        "updated_at",
        "conditions",
        "medications",
        "allergies",
        "preferences",
        "followups",
    }


async def test_query_profile_tool_returns_portrait() -> None:
    """query_profile 工具返回默认用户画像。"""
    result = await query_profile.ainvoke({})
    assert result["user_id"] == str(settings.default_user_id)


async def test_record_metric_updates_portrait_updated_at() -> None:
    """record_metric 落库后刷新画像 updated_at。"""
    await _cleanup_metric()
    old_ts = datetime.datetime(2000, 1, 1, tzinfo=datetime.timezone.utc)

    async with async_session() as session:
        profile = await session.get(HealthProfile, settings.default_user_id)
        original_ts = profile.updated_at
        profile.updated_at = old_ts
        await session.commit()

    try:
        await record_metric.ainvoke(
            {"metric_code": TEST_METRIC, "value": 6, "unit": "h", "date": TEST_DATE}
        )

        async with async_session() as session:
            profile = await session.get(HealthProfile, settings.default_user_id)
        assert profile.updated_at > old_ts
    finally:
        # 恢复画像 updated_at，并清理测试指标行
        async with async_session() as session:
            profile = await session.get(HealthProfile, settings.default_user_id)
            profile.updated_at = original_ts
            await session.commit()
        await _cleanup_metric()
