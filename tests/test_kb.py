"""知识库症状归一化匹配的确定性断言（依赖已灌种子的真库）。"""
from sqlalchemy import select

from app.db import async_session
from app.kb import (
    _match_one,
    classify_advice_level_by_hints,
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

    ids = await match_symptoms(["总是累", "胸口疼", "喘不上气"])

    assert ids == [fatigue, chest, dyspnea]


async def test_match_symptoms_dedupes() -> None:
    """同一症状的多种说法只返回一个 id。"""
    fatigue = await _symptom_id("乏力")

    ids = await match_symptoms(["累", "乏力", "没力气"])

    assert ids == [fatigue]


async def test_match_symptoms_drops_unmatched() -> None:
    """无法匹配的文本被丢弃，不阻断匹配。"""
    fatigue = await _symptom_id("乏力")

    ids = await match_symptoms(["有点不舒服", "最近总是累"])

    assert ids == [fatigue]


async def test_match_one_exact_name_first() -> None:
    """_match_one：name 精确匹配优先。"""
    async with async_session() as session:
        symptoms = (await session.execute(select(Symptom).order_by(Symptom.id))).scalars().all()
    fatigue = await _symptom_id("乏力")
    assert _match_one("乏力", symptoms) == fatigue


async def test_compute_candidates_union_orders_by_match_count() -> None:
    """候选集 = 并集，按命中症状数降序：同时命中乏力+嗜睡的 4 个疾病排最前。"""
    fatigue = await _symptom_id("乏力")
    sleepy = await _symptom_id("嗜睡")

    candidates = await compute_candidates([fatigue, sleepy])

    names = [candidate.name for candidate in candidates]
    assert set(names[:4]) == {
        "慢性疲劳综合征",
        "甲状腺功能减退",
        "阻塞性睡眠呼吸暂停",
        "2型糖尿病",
    }
    assert len(candidates) == 8  # 乏力关联 8 个疾病，嗜睡是其中 4 个的子集


async def test_compute_candidates_no_overlap_still_non_empty() -> None:
    """乏力 + 腹泻 无交集时，仍返回并集（不塌成空）。"""
    fatigue = await _symptom_id("乏力")
    diarrhea = await _symptom_id("腹泻")

    candidates = await compute_candidates([fatigue, diarrhea])

    names = {candidate.name for candidate in candidates}
    assert len(candidates) > 0
    assert "慢性疲劳综合征" in names  # 乏力侧
    assert "肠易激综合征" in names     # 腹泻侧


async def test_compute_candidates_empty_for_no_symptoms() -> None:
    """无已采集症状 -> 空候选集。"""
    assert await compute_candidates([]) == []


async def test_compute_candidates_exposes_hit_count() -> None:
    """候选集暴露 hit_count：乏力+嗜睡 命中数 2 的 4 个疾病排最前。"""
    fatigue = await _symptom_id("乏力")
    sleepy = await _symptom_id("嗜睡")

    candidates = await compute_candidates([fatigue, sleepy])

    assert [candidate.hit_count for candidate in candidates[:4]] == [2, 2, 2, 2]
    assert candidates[0].hit_count == 2
    assert all(candidate.hit_count <= 2 for candidate in candidates)


def test_classify_advice_level_by_hints_max_severity() -> None:
    """多个 urgency_hint 取最高级。"""
    assert classify_advice_level_by_hints(["observe", "adjust", "online_consult"]) == "online_consult"
    assert classify_advice_level_by_hints(["observe", "seek_care", "adjust"]) == "seek_care"


def test_classify_advice_level_by_hints_defaults_observe() -> None:
    """无有效 urgency_hint 默认 observe。"""
    assert classify_advice_level_by_hints([]) == "observe"
    assert classify_advice_level_by_hints(["bogus"]) == "observe"


async def test_select_followup_questions_excludes_collected_and_orders() -> None:
    """追问排除已采集症状，并按 discrimination_weight desc 确定性排序。"""
    fatigue = await _symptom_id("乏力")
    sleepy = await _symptom_id("嗜睡")

    candidates = await compute_candidates([fatigue, sleepy])
    questions = await select_followup_questions(
        [candidate.id for candidate in candidates],
        {fatigue, sleepy},
        k=3,
    )

    assert len(questions) == 3
    returned_ids = {question["id"] for question in questions}
    assert fatigue not in returned_ids
    assert sleepy not in returned_ids

    weights = [question["discrimination_weight"] for question in questions]
    assert weights == sorted(weights, reverse=True)
    assert all(1 <= weight <= 5 for weight in weights)
