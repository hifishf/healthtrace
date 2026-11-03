"""报告演示种子：参考范围、药物交互和三年 LDL 合成序列幂等写入。

先决条件：已 `alembic upgrade head`（0006 两张表已建）。
运行：.venv/bin/python scripts/seed_report_demo.py

数据性质：演示用近似参考范围与药物交互规则，非诊疗依据；交互规则来源标注在
description 中（Micromedex / Lexicomp / 药品说明书 / 临床用药指南，演示规模）。
"""
import asyncio
import sys
from datetime import date
from pathlib import Path

import asyncpg

# 允许脚本以任意工作目录运行：把仓库根加入 sys.path，才能 import app 包
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402

# 检验参考范围（演示规模）：metric_code / unit / low / high / age_group / sex
# low 或 high 可空，表示只有单边约束（如 LDL 只有上限 3.4，HDL 只有下限 1.0）
LAB_REFERENCE_RANGES: list[tuple[str, str, float | None, float | None, str | None, str | None]] = [
    ("ldl_cholesterol", "mmol/L", None, 3.4, None, None),
    ("total_cholesterol", "mmol/L", None, 5.2, None, None),
    ("hdl_cholesterol", "mmol/L", 1.0, None, None, None),
    ("triglycerides", "mmol/L", None, 1.7, None, None),
    ("fasting_glucose", "mmol/L", 3.9, 6.1, None, None),
    ("hba1c", "%", None, 6.0, None, None),
]

# 药物交互规则（演示规模）：drug_a / drug_b / severity / description
# severity 固定取 禁忌/严重/中等/轻微 四档；华法林+阿司匹林=严重 是断言 #9 的目标命中
DRUG_INTERACTIONS: list[tuple[str, str, str, str]] = [
    ("华法林", "阿司匹林", "严重", "两者联用显著增加出血风险（来源：Micromedex 演示数据）"),
    ("华法林", "布洛芬", "严重", "NSAID 增加胃肠道出血风险并干扰凝血（来源：临床用药指南 演示）"),
    ("辛伐他汀", "克拉霉素", "禁忌", "CYP3A4 强抑制剂使辛伐他汀血药浓度升高，肌病风险增加（来源：Lexicomp 演示）"),
    ("二甲双胍", "碘造影剂", "禁忌", "联用可能诱发乳酸性酸中毒（来源：药品说明书 演示）"),
    ("阿司匹林", "布洛芬", "中等", "布洛芬可能减弱阿司匹林抗血小板作用（来源：临床用药指南 演示）"),
    ("辛伐他汀", "葡萄柚汁", "中等", "葡萄柚汁抑制 CYP3A4，升高辛伐他汀暴露（来源：Lexicomp 演示）"),
]

# 第 10 周报告子图演示序列：仅覆盖没有被其它来源占用的日期。
REPORT_DEMO_LDL: list[tuple[str, float]] = [
    ("2024-01-15", 3.1),
    ("2025-01-15", 3.4),
    ("2026-01-15", 3.8),
]


async def _seed_lab_reference_ranges(conn: asyncpg.Connection) -> int:
    """按 metric_code 先删后插，保证幂等（该表无唯一键，NULL 又无法参与唯一约束）。"""
    for metric_code, unit, low, high, age_group, sex in LAB_REFERENCE_RANGES:
        # 只清理本脚本负责的指标码，不动其它数据
        await conn.execute(
            "DELETE FROM lab_reference_ranges WHERE metric_code = $1", metric_code
        )
        await conn.execute(
            "INSERT INTO lab_reference_ranges(metric_code, unit, low, high, age_group, sex) "
            "VALUES($1, $2, $3, $4, $5, $6)",
            metric_code,
            unit,
            low,
            high,
            age_group,
            sex,
        )
    return len(LAB_REFERENCE_RANGES)


async def _seed_drug_interactions(conn: asyncpg.Connection) -> int:
    """按 (drug_a, drug_b) 唯一键 upsert，保持描述/等级最新。"""
    for drug_a, drug_b, severity, description in DRUG_INTERACTIONS:
        await conn.execute(
            "INSERT INTO drug_interactions(drug_a, drug_b, severity, description) "
            "VALUES($1, $2, $3, $4) "
            "ON CONFLICT (drug_a, drug_b) DO UPDATE SET "
            "severity=EXCLUDED.severity, description=EXCLUDED.description",
            drug_a,
            drug_b,
            severity,
            description,
        )
    return len(DRUG_INTERACTIONS)


async def _seed_report_metrics(conn: asyncpg.Connection) -> int:
    """写入三年合成 LDL 序列；已有非演示来源的数据不覆盖。"""
    count = 0
    for recorded_date, value in REPORT_DEMO_LDL:
        row = await conn.fetchrow(
            "INSERT INTO metric_records("
            "user_id, metric_code, value, unit, recorded_date, source"
            ") VALUES($1, 'ldl_cholesterol', $2, 'mmol/L', $3, 'synthetic_report_demo') "
            "ON CONFLICT (user_id, metric_code, recorded_date) DO UPDATE SET "
            "value=EXCLUDED.value, unit=EXCLUDED.unit, source=EXCLUDED.source "
            "WHERE metric_records.source = 'synthetic_report_demo' RETURNING id",
            settings.default_user_id,
            value,
            date.fromisoformat(recorded_date),
        )
        if row:
            count += 1
    return count


async def main() -> None:
    """幂等写入参考范围、交互规则和报告演示指标。"""
    dsn = settings.database_url.replace("postgresql+asyncpg://", "postgresql://")
    conn = await asyncpg.connect(dsn)
    try:
        lab_count = await _seed_lab_reference_ranges(conn)
        drug_count = await _seed_drug_interactions(conn)
        metric_count = await _seed_report_metrics(conn)
        print(
            f"lab_reference_ranges={lab_count} drug_interactions={drug_count} "
            f"report_metrics={metric_count} "
            "（幂等可重跑，交互规则来源见 description）"
        )
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
