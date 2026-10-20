"""知识库症状归一化匹配的确定性断言（依赖已灌种子的真库）。"""
from sqlalchemy import select

from app.db import async_session
from app.kb import _match_one, match_symptoms
from app.models import Symptom


async def _symptom_id(name: str) -> int:
    async with async_session() as session:
        row = (
            await session.execute(select(Symptom).where(Symptom.name == name))
        ).scalar_one()
        return row.id


async def test_match_symptoms_normalizes_colloquial() -> None:
    """口语化文本 -> 标准症状 id（总是累 -> 乏力、胸口疼 -> 胸痛、喘不上气 -> 呼吸困难）。"""
    fatigue = await _symptom_id("乏力")
    chest = await _symptom_id("胸痛")
    dyspnea = await _symptom_id("呼吸困难")

    async with async_session() as session:
        ids = await match_symptoms(session, ["总是累", "胸口疼", "喘不上气"])

    assert ids == [fatigue, chest, dyspnea]


async def test_match_symptoms_dedupes() -> None:
    """同一症状的多种说法只返回一个 id。"""
    fatigue = await _symptom_id("乏力")

    async with async_session() as session:
        ids = await match_symptoms(session, ["累", "乏力", "没力气"])

    assert ids == [fatigue]


async def test_match_symptoms_drops_unmatched() -> None:
    """无法匹配的文本被丢弃，不阻断匹配。"""
    fatigue = await _symptom_id("乏力")

    async with async_session() as session:
        ids = await match_symptoms(session, ["有点不舒服", "最近总是累"])

    assert ids == [fatigue]


async def test_match_one_exact_name_first() -> None:
    """_match_one：name 精确匹配优先。"""
    async with async_session() as session:
        symptoms = (await session.execute(select(Symptom).order_by(Symptom.id))).scalars().all()
    fatigue = await _symptom_id("乏力")
    assert _match_one("乏力", symptoms) == fatigue
