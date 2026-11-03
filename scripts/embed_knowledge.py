"""为尚未生成向量的 RAG 知识批量回填 512 维 bge embedding。

先决条件：
1. 已执行 `alembic upgrade head`
2. 已执行 `.venv/bin/python scripts/seed_rag_knowledge.py`

运行：`.venv/bin/python scripts/embed_knowledge.py`

首次运行会从 ModelScope 下载约 91MB 的 bge-small-zh-v1.5 ONNX 模型，
后续复用本地缓存。正文变更时种子脚本会清空旧向量，本脚本只补缺失向量。
"""
import asyncio
import sys
from pathlib import Path

import asyncpg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.db_types import to_pgvector_literal  # noqa: E402
from app.embeddings import embed_texts  # noqa: E402


async def backfill_embeddings() -> tuple[int, int]:
    """批量生成并回填缺失向量，返回 (本次回填数, 剩余缺失数)。"""
    dsn = settings.database_url.replace("postgresql+asyncpg://", "postgresql://")
    conn = await asyncpg.connect(dsn)
    try:
        rows = await conn.fetch(
            "SELECT id, content, content_hash "
            "FROM knowledge_entries WHERE embedding IS NULL ORDER BY id"
        )
        if not rows:
            return 0, 0

        print(f"待生成向量：{len(rows)} 条")
        vectors = await embed_texts([row["content"] for row in rows])
        updates = [
            (
                row["id"],
                row["content_hash"],
                to_pgvector_literal(vector),
            )
            for row, vector in zip(rows, vectors)
        ]
        async with conn.transaction():
            await conn.executemany(
                "UPDATE knowledge_entries "
                "SET embedding = CAST($3 AS vector), updated_at = now() "
                "WHERE id = $1 AND content_hash = $2",
                updates,
            )
        remaining = await conn.fetchval(
            "SELECT count(*) FROM knowledge_entries WHERE embedding IS NULL"
        )
        return len(updates), int(remaining)
    finally:
        await conn.close()


async def main() -> None:
    """执行回填并打印最终向量覆盖情况。"""
    embedded, remaining = await backfill_embeddings()
    print(f"本次回填={embedded} 剩余未生成={remaining}")


if __name__ == "__main__":
    asyncio.run(main())
