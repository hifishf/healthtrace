"""药物交互（interactions）的确定性断言：高冲突命中 / 风险分级排序 / 无命中与去重。"""
from sqlalchemy import delete
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.db import async_session
from app.interactions import is_high_risk, check_drug_interactions
from app.models import DrugInteraction

# 华法林+阿司匹林 的规范描述与 seed 脚本一致，测试 upsert 不会改变演示数据语义
WARFARIN_ASPIRIN_DESC = "两者联用显著增加出血风险（来源：Micromedex 演示数据）"


async def _upsert_interaction(drug_a: str, drug_b: str, severity: str, description: str) -> None:
    """按 (drug_a, drug_b) 唯一键幂等写入一条交互规则。"""
    stmt = pg_insert(DrugInteraction).values(
        drug_a=drug_a, drug_b=drug_b, severity=severity, description=description
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=["drug_a", "drug_b"],
        set_={
            "severity": stmt.excluded.severity,
            "description": stmt.excluded.description,
        },
    )
    async with async_session() as session:
        await session.execute(stmt)
        await session.commit()


async def _delete_interaction(drug_a: str, drug_b: str) -> None:
    """删除一条测试专用交互规则（收尾清理用）。"""
    async with async_session() as session:
        await session.execute(
            delete(DrugInteraction).where(
                DrugInteraction.drug_a == drug_a, DrugInteraction.drug_b == drug_b
            )
        )
        await session.commit()


async def test_warfarin_aspirin_high_conflict() -> None:
    """断言 #9：华法林 + 阿司匹林 -> 严重、高冲突告警。"""
    await _upsert_interaction("华法林", "阿司匹林", "严重", WARFARIN_ASPIRIN_DESC)
    result = await check_drug_interactions(["华法林", "阿司匹林"])
    assert result["ok"] is True
    assert len(result["interactions"]) == 1
    hit = result["interactions"][0]
    assert hit["drug_a"] == "华法林"
    assert hit["drug_b"] == "阿司匹林"
    assert hit["severity"] == "严重"
    assert hit["high_risk"] is True


async def test_ordering_and_dedupe() -> None:
    """风险分级排序：禁忌排在中等之前；重复药物名去重后只命中一次。"""
    await _upsert_interaction("测试药A", "测试药B", "中等", "测试交互-中等")
    await _upsert_interaction("测试药C", "测试药D", "禁忌", "测试交互-禁忌")
    try:
        result = await check_drug_interactions(
            ["测试药B", "测试药C", "测试药A", "测试药D", "测试药A"]
        )
        severities = [hit["severity"] for hit in result["interactions"]]
        assert severities == ["禁忌", "中等"]
        assert result["interactions"][0]["high_risk"] is True
        assert result["interactions"][1]["high_risk"] is False
    finally:
        await _delete_interaction("测试药A", "测试药B")
        await _delete_interaction("测试药C", "测试药D")


async def test_no_interaction_and_single_drug() -> None:
    """无命中与单药（不足两种）都返回空 interactions。"""
    await _upsert_interaction("测试药A", "测试药B", "中等", "测试交互-中等")
    try:
        no_hit = await check_drug_interactions(["测试药A", "无交互药物"])
        assert no_hit["interactions"] == []

        single = await check_drug_interactions(["测试药A"])
        assert single["interactions"] == []
    finally:
        await _delete_interaction("测试药A", "测试药B")


def test_is_high_risk_boundaries() -> None:
    """高冲突边界：禁忌/严重为高冲突，中等/轻微不是。"""
    assert is_high_risk("禁忌") is True
    assert is_high_risk("严重") is True
    assert is_high_risk("中等") is False
    assert is_high_risk("轻微") is False
