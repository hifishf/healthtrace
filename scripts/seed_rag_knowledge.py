"""第 10 周 RAG 知识种子：500+ 条原子知识幂等写入 knowledge_entries。

先决条件：已执行 `alembic upgrade head`。
运行：`.venv/bin/python scripts/seed_rag_knowledge.py`

本脚本只写入正文、来源和内容哈希，不生成 embedding；embedding 由第 10 周第 2 步
批量回填。重复执行按 knowledge_key 更新；正文变化时清空旧向量，正文不变则保留向量。
"""
import asyncio
import sys
from pathlib import Path

import asyncpg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from scripts.rag_seed_data import build_knowledge_entries  # noqa: E402


async def seed_knowledge_entries() -> int:
    """幂等写入全部 RAG 知识条目，返回条目总数。"""
    entries = build_knowledge_entries()
    dsn = settings.database_url.replace("postgresql+asyncpg://", "postgresql://")
    conn = await asyncpg.connect(dsn)
    try:
        await conn.executemany(
            """
            INSERT INTO knowledge_entries(
                knowledge_key,
                title,
                content,
                category,
                metric_codes,
                source_name,
                source_url,
                source_version,
                content_hash,
                updated_at
            )
            VALUES($1, $2, $3, $4, $5, $6, $7, $8, $9, now())
            ON CONFLICT (knowledge_key) DO UPDATE SET
                title = EXCLUDED.title,
                content = EXCLUDED.content,
                category = EXCLUDED.category,
                metric_codes = EXCLUDED.metric_codes,
                source_name = EXCLUDED.source_name,
                source_url = EXCLUDED.source_url,
                source_version = EXCLUDED.source_version,
                content_hash = EXCLUDED.content_hash,
                embedding = CASE
                    WHEN knowledge_entries.content_hash = EXCLUDED.content_hash
                    THEN knowledge_entries.embedding
                    ELSE NULL
                END,
                updated_at = now()
            """,
            [
                (
                    entry["knowledge_key"],
                    entry["title"],
                    entry["content"],
                    entry["category"],
                    entry["metric_codes"],
                    entry["source_name"],
                    entry["source_url"],
                    entry["source_version"],
                    entry["content_hash"],
                )
                for entry in entries
            ],
        )
        total = await conn.fetchval("SELECT count(*) FROM knowledge_entries")
        category_rows = await conn.fetch(
            "SELECT category, count(*) AS count "
            "FROM knowledge_entries GROUP BY category ORDER BY category"
        )
    finally:
        await conn.close()

    categories = " ".join(
        f"{row['category']}={row['count']}" for row in category_rows
    )
    print(f"knowledge_entries total={total} {categories}")
    return int(total)


if __name__ == "__main__":
    asyncio.run(seed_knowledge_entries())
