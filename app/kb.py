"""知识库确定性访问：症状匹配 / 候选集计算 / 追问选择 / 分级建议。"""
from dataclasses import dataclass

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


@dataclass(frozen=True)
class Candidate:
    """候选疾病：疾病基础信息 + 命中已采集症状数。

    hit_count 是「候选集逐轮缩小可量化」的依据：并集基数不收缩，
    真正收敛的是命中数最高的高置信档（hit_count 并列最高者逐轮变少）。
    """

    id: int
    name: str
    department: str | None
    urgency_hint: str | None
    hit_count: int


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
) -> list[Candidate]:
    """候选疾病集 = 匹配任一已采集症状的疾病，按命中症状数降序、id 升序排列。

    不用严格交集：交集一旦为空，会把「确实不舒服但症状不属于同一疾病」的患者
    错误判成无候选。改用并集 + 命中数排序，候选集永不塌空，命中越多越靠前。
    无已采集症状时返回空列表。
    """
    if not symptom_ids:
        return []

    rows = (
        await session.execute(
            select(DiseaseSymptom.disease_id, DiseaseSymptom.symptom_id).where(
                DiseaseSymptom.symptom_id.in_(symptom_ids)
            )
        )
    ).all()

    match_count: dict[int, int] = {}
    for disease_id, _symptom_id in rows:
        match_count[disease_id] = match_count.get(disease_id, 0) + 1

    if not match_count:
        return []

    diseases = (
        await session.execute(
            select(Disease).where(Disease.id.in_(match_count.keys()))
        )
    ).scalars().all()
    disease_by_id = {disease.id: disease for disease in diseases}

    # 命中症状数多的排前，id 升序兜底保证确定性
    candidates = [
        Candidate(
            id=disease_id,
            name=disease_by_id[disease_id].name,
            department=disease_by_id[disease_id].department,
            urgency_hint=disease_by_id[disease_id].urgency_hint,
            hit_count=match_count[disease_id],
        )
        for disease_id in match_count
    ]
    return sorted(candidates, key=lambda c: (-c.hit_count, c.id))


async def select_followup_questions(
    session: AsyncSession,
    candidate_disease_ids: list[int],
    collected_symptom_ids: set[int],
    ruled_out_symptom_ids: set[int] | None = None,
    k: int = 3,
) -> list[dict]:
    """从候选疾病集里选 top-k 个未采集症状作为下一轮追问。

    第 4 周：按 discrimination_weight desc 排序（第 3 周是 prevalence desc 占位），
    权重相同时按 symptom.id 升序兜底保证确定性；已采集症状与已排除症状都不再追问。
    返回 [{"id", "name", "discrimination_weight"}]，权重取该症状在候选疾病里的最大值。
    """
    if not candidate_disease_ids:
        return []

    excluded_ids = set(collected_symptom_ids)
    if ruled_out_symptom_ids:
        excluded_ids |= set(ruled_out_symptom_ids)

    stmt = select(
        DiseaseSymptom.symptom_id, DiseaseSymptom.discrimination_weight
    ).where(
        DiseaseSymptom.disease_id.in_(candidate_disease_ids)
    )
    if excluded_ids:
        stmt = stmt.where(DiseaseSymptom.symptom_id.notin_(excluded_ids))
    rows = (await session.execute(stmt)).all()

    best_weight: dict[int, int] = {}
    for symptom_id, weight in rows:
        value = int(weight or 0)
        if symptom_id not in best_weight or value > best_weight[symptom_id]:
            best_weight[symptom_id] = value

    # discrimination_weight desc，id asc 兜底，保证确定性
    top_ids = sorted(best_weight, key=lambda sid: (-best_weight[sid], sid))[:k]
    if not top_ids:
        return []

    symptoms = (
        await session.execute(select(Symptom).where(Symptom.id.in_(top_ids)))
    ).scalars().all()
    name_by_id = {symptom.id: symptom.name for symptom in symptoms}

    return [
        {"id": sid, "name": name_by_id[sid], "discrimination_weight": best_weight[sid]}
        for sid in top_ids
    ]


def classify_advice_level_by_hints(urgency_hints: list[str]) -> str:
    """按 urgency_hint 列表取最高级；无有效值默认 observe。"""
    valid = [hint for hint in urgency_hints if hint in _ADVICE_SEVERITY]
    if not valid:
        return "observe"
    return max(valid, key=lambda hint: _ADVICE_SEVERITY[hint])
