"""知识库确定性访问：症状匹配 / 候选集计算 / 追问选择 / 分级建议。"""
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Disease, DiseaseSymptom, Symptom

# 分级建议的 4 个等级（红旗 emergency 由 red_flags 单独处理，不在此列）
ADVICE_LEVELS: dict[str, str] = {
    "observe": "观察",
    "adjust": "调整",
    "online_consult": "线上问诊",
    "seek_care": "建议就医",
}
# 等级严重度排序（用于取候选集最高级）
_ADVICE_SEVERITY: dict[str, int] = {
    "observe": 1,
    "adjust": 2,
    "online_consult": 3,
    "seek_care": 4,
}


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


async def get_symptom_names(
    session: AsyncSession,
    symptom_ids: list[int],
) -> dict[int, str]:
    """按 id 批量取症状名，返回 {id: name}。"""
    if not symptom_ids:
        return {}
    symptoms = (
        await session.execute(select(Symptom).where(Symptom.id.in_(symptom_ids)))
    ).scalars().all()
    return {symptom.id: symptom.name for symptom in symptoms}


async def compute_candidates(
    session: AsyncSession,
    symptom_ids: list[int],
) -> list[Disease]:
    """候选疾病集 = 同时关联所有已采集症状的疾病（交集，逐轮收窄）。

    首轮只有 1 个症状时返回该症状关联的全部疾病（初始候选集）；
    后续轮叠加新症状后取交集，候选集随之收窄。无已采集症状时返回空列表。
    """
    if not symptom_ids:
        return []

    disease_id_sets: list[set[int]] = []
    for symptom_id in symptom_ids:
        rows = (
            await session.execute(
                select(DiseaseSymptom.disease_id).where(
                    DiseaseSymptom.symptom_id == symptom_id
                )
            )
        ).scalars().all()
        disease_id_sets.append(set(rows))

    common_ids = set.intersection(*disease_id_sets)
    if not common_ids:
        return []

    return (
        await session.execute(
            select(Disease).where(Disease.id.in_(common_ids)).order_by(Disease.id)
        )
    ).scalars().all()


async def select_followup_questions(
    session: AsyncSession,
    candidate_disease_ids: list[int],
    collected_symptom_ids: set[int],
    k: int = 3,
) -> list[dict]:
    """从候选疾病集里选 top-k 个未采集症状作为下一轮追问。

    第 3 周占位：按 prevalence desc 排序（discrimination_weight 排序第 4 周上）。
    返回 [{"id", "name", "prevalence"}]，prevalence 取该症状在候选疾病里的最大值。
    """
    if not candidate_disease_ids:
        return []

    stmt = select(DiseaseSymptom.symptom_id, DiseaseSymptom.prevalence).where(
        DiseaseSymptom.disease_id.in_(candidate_disease_ids)
    )
    if collected_symptom_ids:
        stmt = stmt.where(DiseaseSymptom.symptom_id.notin_(collected_symptom_ids))
    rows = (await session.execute(stmt)).all()

    best_prevalence: dict[int, float] = {}
    for symptom_id, prevalence in rows:
        value = float(prevalence or 0.0)
        if symptom_id not in best_prevalence or value > best_prevalence[symptom_id]:
            best_prevalence[symptom_id] = value

    # prevalence desc，id asc 兜底，保证确定性
    top_ids = sorted(best_prevalence, key=lambda sid: (-best_prevalence[sid], sid))[:k]
    if not top_ids:
        return []

    symptoms = (
        await session.execute(select(Symptom).where(Symptom.id.in_(top_ids)))
    ).scalars().all()
    name_by_id = {symptom.id: symptom.name for symptom in symptoms}

    return [
        {"id": sid, "name": name_by_id[sid], "prevalence": best_prevalence[sid]}
        for sid in top_ids
    ]


def classify_advice_level(diseases: list[Disease]) -> str:
    """分级建议：取候选疾病的最高 urgency_hint；无候选时默认 observe。"""
    return classify_advice_level_by_hints([d.urgency_hint for d in diseases])


def classify_advice_level_by_hints(urgency_hints: list[str]) -> str:
    """按 urgency_hint 列表取最高级；无有效值默认 observe。"""
    valid = [hint for hint in urgency_hints if hint in _ADVICE_SEVERITY]
    if not valid:
        return "observe"
    return max(valid, key=lambda hint: _ADVICE_SEVERITY[hint])
