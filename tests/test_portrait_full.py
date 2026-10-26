"""第 7 周 Step 1：画像扩到 30+ 字段 + record_medication 写路径的确定性断言。"""
from sqlalchemy import delete

from app import crud, portrait
from app.agent.tools import record_medication
from app.config import settings
from app.db import async_session
from app.models import Allergy, Condition, Followup, Medication, Preference

UID = settings.default_user_id


async def _delete_medication(name: str) -> None:
    async with async_session() as session:
        await session.execute(
            delete(Medication).where(Medication.user_id == UID, Medication.name == name)
        )
        await session.commit()


async def _delete_condition(name: str) -> None:
    async with async_session() as session:
        await session.execute(
            delete(Condition).where(
                Condition.user_id == UID, Condition.condition_name == name
            )
        )
        await session.commit()


async def test_get_health_portrait_has_extended_keys() -> None:
    """完整画像返回 12 个顶层键：基础 7 键 + 五个子表列表。"""
    result = await portrait.get_health_portrait(UID)
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


async def test_portrait_roundtrip_flattens_to_30_plus_fields() -> None:
    """五类各写一条后，portrait dict 展开去重语义键 ≥ 30（表族合计 38 列）。"""
    await _delete_medication("二甲双胍")
    await _delete_condition("高血压")
    try:
        await crud.add_condition(UID, "高血压", status="active")
        await crud.add_medication(UID, "二甲双胍", dosage="500mg", frequency="每日两次")
        await crud.add_allergy(UID, "青霉素", severity="severe")
        await crud.add_preference(UID, "沟通风格", "简洁")
        await crud.add_followup(UID, "三个月后复查血糖", status="pending")

        result = await portrait.get_health_portrait(UID)
        flattened = set(result.keys())
        for key in ("conditions", "medications", "allergies", "preferences", "followups"):
            assert result[key], f"{key} 应至少有一条记录"
            for item in result[key]:
                flattened.update(item.keys())
        assert len(flattened) >= 30
    finally:
        async with async_session() as session:
            await session.execute(delete(Followup).where(Followup.user_id == UID))
            await session.execute(delete(Preference).where(Preference.user_id == UID))
            await session.execute(delete(Allergy).where(Allergy.user_id == UID))
        await _delete_medication("二甲双胍")
        await _delete_condition("高血压")


async def test_record_medication_then_summarize_includes_name() -> None:
    """「在吃二甲双胍」落库后，画像与摘要都包含该药名（验收 #1 的写路径）。"""
    await _delete_medication("二甲双胍")
    try:
        await record_medication.ainvoke(
            {"name": "二甲双胍", "dosage": "500mg", "frequency": "每日两次"}
        )

        result = await portrait.get_health_portrait(UID)
        assert any(m["name"] == "二甲双胍" for m in result["medications"])
        summary = portrait.summarize(result)
        assert "二甲双胍" in summary
        assert "在服" in summary
    finally:
        await _delete_medication("二甲双胍")


async def test_record_condition_then_summarize_includes_name() -> None:
    """写入病情字段后，画像摘要包含病情名（验收 #2 的写路径）。"""
    await _delete_condition("高血压")
    try:
        await crud.add_condition(UID, "高血压", status="active")

        result = await portrait.get_health_portrait(UID)
        summary = portrait.summarize(result)
        assert "高血压" in summary
        assert "病史" in summary
    finally:
        await _delete_condition("高血压")
