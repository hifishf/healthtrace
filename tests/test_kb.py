"""知识库症状归一化匹配的确定性断言（依赖已灌种子的真库）。"""
from sqlalchemy import select

from app.db import async_session
from app.kb import (
    _match_one,
    classify_advice_level,
    compute_candidates,
    match_symptoms,
    select_followup_questions,
)
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


async def test_compute_candidates_intersection_narrows() -> None:
    """候选集 = 已采集症状的交集：叠加嗜睡后从 乏力 的候选集里收窄。"""
    fatigue = await _symptom_id("乏力")
    sleepy = await _symptom_id("嗜睡")

    async with async_session() as session:
        initial = await compute_candidates(session, [fatigue])
        narrowed = await compute_candidates(session, [fatigue, sleepy])

    initial_names = {disease.name for disease in initial}
    narrowed_names = {disease.name for disease in narrowed}

    assert "慢性疲劳综合征" in initial_names
    assert len(initial) >= 4
    assert narrowed_names == {
        "慢性疲劳综合征",
        "甲状腺功能减退",
        "阻塞性睡眠呼吸暂停",
        "2型糖尿病",
    }
    assert narrowed_names < initial_names  # 真子集，确实收窄


async def test_compute_candidates_empty_for_no_symptoms() -> None:
    """无已采集症状 -> 空候选集。"""
    async with async_session() as session:
        assert await compute_candidates(session, []) == []


async def test_classify_advice_level_max_severity() -> None:
    """乏力+嗜睡 候选里最高级别是 online_consult。"""
    fatigue = await _symptom_id("乏力")
    sleepy = await _symptom_id("嗜睡")

    async with async_session() as session:
        candidates = await compute_candidates(session, [fatigue, sleepy])

    assert classify_advice_level(candidates) == "online_consult"


async def test_classify_advice_level_seek_care() -> None:
    """胸痛候选包含心梗/肺栓塞等，应给出最高级 seek_care。"""
    chest = await _symptom_id("胸痛")

    async with async_session() as session:
        candidates = await compute_candidates(session, [chest])

    assert classify_advice_level(candidates) == "seek_care"


def test_classify_advice_level_defaults_observe() -> None:
    """空候选集默认 observe。"""
    assert classify_advice_level([]) == "observe"


async def test_select_followup_questions_excludes_collected_and_orders() -> None:
    """追问排除已采集症状，并按 prevalence desc 确定性排序。"""
    fatigue = await _symptom_id("乏力")
    sleepy = await _symptom_id("嗜睡")

    async with async_session() as session:
        candidates = await compute_candidates(session, [fatigue, sleepy])
        questions = await select_followup_questions(
            session,
            [disease.id for disease in candidates],
            {fatigue, sleepy},
            k=3,
        )

    assert len(questions) == 3
    returned_ids = {question["id"] for question in questions}
    assert fatigue not in returned_ids
    assert sleepy not in returned_ids

    prevalences = [question["prevalence"] for question in questions]
    assert prevalences == sorted(prevalences, reverse=True)
