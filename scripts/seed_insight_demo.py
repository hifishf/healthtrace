"""第 8 周合成演示数据：≥120 天「咖啡因摄入 -> 入睡延迟」写入 metric_records。

数据性质：合成演示数据，source=synthetic，用于展示因果分析子图全链路，非真实健康数据。
关系口径：当天咖啡因越高、当天入睡延迟越长（同向、lag=0），并叠加高斯噪声，
确定性种子保证每次重跑生成相同序列；按 (user_id, metric_code, recorded_date) upsert，幂等。
运行：.venv/bin/python scripts/seed_insight_demo.py
"""
import asyncio
import random
import sys
from datetime import date, timedelta
from pathlib import Path

import asyncpg

# 允许脚本以任意工作目录运行：把仓库根加入 sys.path，才能 import app 包
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.insight import METRIC_CAFFEINE, METRIC_SLEEP_LATENCY  # noqa: E402
from app.config import settings  # noqa: E402

# 生成天数与结束日期（固定，保证幂等与可复现）
DAYS = 150
END_DATE = date(2026, 10, 27)
# 线性关系的斜率/截距/噪声（咖啡因 mg -> 入睡延迟 min）
SLOPE = 0.15
INTERCEPT = 10.0
NOISE_SIGMA = 10.0
SEED = 8


def _generate() -> list[tuple[str, float, date]]:
    """确定性生成 DAYS 天、两条指标的 (metric_code, value, recorded_date) 列表。"""
    rng = random.Random(SEED)
    rows: list[tuple[str, float, date]] = []
    for offset in range(DAYS):
        day = END_DATE - timedelta(days=DAYS - 1 - offset)
        caffeine = float(rng.randint(20, 320))
        latency = INTERCEPT + SLOPE * caffeine + rng.gauss(0.0, NOISE_SIGMA)
        latency = max(5.0, min(90.0, latency))
        rows.append((METRIC_CAFFEINE, caffeine, day))
        rows.append((METRIC_SLEEP_LATENCY, round(latency, 1), day))
    return rows


async def main() -> None:
    """把合成序列按 (user_id, metric_code, recorded_date) upsert 写入 metric_records。"""
    dsn = settings.database_url.replace("postgresql+asyncpg://", "postgresql://")
    rows = _generate()
    conn = await asyncpg.connect(dsn)
    try:
        await conn.executemany(
            "INSERT INTO metric_records(user_id, metric_code, value, unit, recorded_date, source) "
            "VALUES($1, $2, $3, $4, $5, 'synthetic') "
            "ON CONFLICT (user_id, metric_code, recorded_date) "
            "DO UPDATE SET value=EXCLUDED.value, unit=EXCLUDED.unit, source=EXCLUDED.source",
            [
                (
                    settings.default_user_id,
                    metric_code,
                    value,
                    "mg" if metric_code == METRIC_CAFFEINE else "min",
                    day,
                )
                for metric_code, value, day in rows
            ],
        )
        print(f"写入 {DAYS} 天 x 2 指标 = {len(rows)} 行（source=synthetic，幂等可重跑）")
        print(f"指标对：{METRIC_CAFFEINE} -> {METRIC_SLEEP_LATENCY}，当天同向、预期强相关")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
