"""候选集逐轮缩小可量化：高置信档（命中数并列最高）随追问收敛。"""
from app.kb import compute_candidates, match_symptoms, top_tier_size


async def _symptom_ids(*names: str) -> list[int]:
    return await match_symptoms(list(names))


async def test_candidate_top_tier_shrinks_golden_chain() -> None:
    """黄金链 乏力→嗜睡→怕冷：高置信档 8→4→1（首轮 ≤20，末轮 ≤5）。"""
    ids_round1 = await _symptom_ids("乏力")
    ids_round2 = await _symptom_ids("乏力", "嗜睡")
    ids_round3 = await _symptom_ids("乏力", "嗜睡", "怕冷")

    round1 = await compute_candidates(ids_round1)
    round2 = await compute_candidates(ids_round2)
    round3 = await compute_candidates(ids_round3)

    tier1 = top_tier_size(round1)
    tier2 = top_tier_size(round2)
    tier3 = top_tier_size(round3)

    # 并集基数不缩（都是乏力关联的 8 个疾病），但高置信档逐轮收敛
    assert len(round1) == 8
    assert len(round2) == 8
    assert len(round3) == 8
    assert tier1 == 8
    assert tier2 == 4
    assert tier3 == 1

    # 逐轮单调不增 + 第 4 周阈值标定
    assert tier1 >= tier2 >= tier3
    assert tier1 <= 20
    assert tier3 <= 5


def test_top_tier_size_empty() -> None:
    """空候选集高置信档为 0。"""
    assert top_tier_size([]) == 0
