"""知识库确定性访问：症状归一化匹配（候选集计算与分级建议在第 4 步加入）。"""
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Symptom


def _match_one(text: str, symptoms: list[Symptom]) -> int | None:
    """把一个抽取出的症状文本映射到 symptoms.id（name/synonyms 确定性匹配）。

    优先级：name 精确 -> 同义词精确 -> name 与文本互为子串 -> 同义词与文本互为子串。
    子串双向匹配用于承接“最近总是累 -> 乏力（同义词 累）”这类口语化输入。
    """
    value = (text or "").strip()
    if not value:
        return None

    for symptom in symptoms:
        if value == symptom.name:
            return symptom.id
    for symptom in symptoms:
        if symptom.synonyms and value in symptom.synonyms:
            return symptom.id
    for symptom in symptoms:
        if value in symptom.name or symptom.name in value:
            return symptom.id
    for symptom in symptoms:
        if symptom.synonyms:
            for synonym in symptom.synonyms:
                if value in synonym or synonym in value:
                    return symptom.id
    return None


async def match_symptoms(
    session: AsyncSession,
    symptom_texts: list[str],
) -> list[int]:
    """把 LLM 抽取的症状文本映射为 symptoms.id（去重、按输入顺序返回）。"""
    symptoms = (
        await session.execute(select(Symptom).order_by(Symptom.id))
    ).scalars().all()

    matched: list[int] = []
    for text in symptom_texts:
        symptom_id = _match_one(text, symptoms)
        if symptom_id is not None and symptom_id not in matched:
            matched.append(symptom_id)
    return matched
