"""RAG 检索确定性断言：向量格式、余弦排序、指标过滤与参数边界。"""
from uuid import uuid4

import pytest
from sqlalchemy import text

from app.db import async_session
from app.db_types import VECTOR_DIMENSIONS, to_pgvector_literal
from app.rag import search_knowledge, search_knowledge_by_vector


def _vector(head: list[float]) -> list[float]:
    """把少量测试权重补齐为冻结的 512 维向量。"""
    assert len(head) <= VECTOR_DIMENSIONS
    return head + [0.0] * (VECTOR_DIMENSIONS - len(head))


def test_pgvector_literal_is_deterministic() -> None:
    """向量转为参数化 SQL 使用的稳定文本格式。"""
    assert to_pgvector_literal([0.1, -1.0, 0.0]) == "[0.1,-1,0]"
    with pytest.raises(ValueError):
        to_pgvector_literal([])


async def test_vector_search_orders_by_cosine_and_filters_metrics() -> None:
    """最近向量排第一，指标过滤保留通用知识和指定指标知识。"""
    prefix = f"test-rag-{uuid4().hex}:"
    rows = [
        (prefix + "general", "通用来源", None, _vector([0.1, 0.9])),
        (prefix + "ldl", "LDL 来源", ["ldl_cholesterol"], _vector([0.9, 0.1])),
        (prefix + "sleep", "睡眠来源", ["sleep_duration"], _vector([0.0, 1.0])),
    ]
    async with async_session() as session:
        try:
            for key, title, metric_codes, vector in rows:
                await session.execute(
                    text(
                        """
                        INSERT INTO knowledge_entries(
                            knowledge_key, title, content, category, metric_codes,
                            source_name, source_url, source_version, content_hash,
                            embedding
                        )
                        VALUES(
                            :key, :title, :content, 'safety', :metric_codes,
                            '测试来源', 'https://example.com/source', 'test',
                            :content_hash, CAST(:embedding AS vector)
                        )
                        """
                    ),
                    {
                        "key": key,
                        "title": title,
                        "content": f"{title}正文",
                        "metric_codes": metric_codes,
                        "content_hash": key[-32:].ljust(32, "0"),
                        "embedding": to_pgvector_literal(vector),
                    },
                )
            await session.commit()

            query = _vector([0.0, 1.0])
            unfiltered = await search_knowledge_by_vector(query, top_k=2)
            assert [hit.knowledge_key for hit in unfiltered[:2]] == [
                prefix + "sleep",
                prefix + "general",
            ]

            filtered = await search_knowledge_by_vector(
                query,
                top_k=5,
                metric_codes=["ldl_cholesterol"],
            )
            filtered_keys = {hit.knowledge_key for hit in filtered}
            assert prefix + "general" in filtered_keys
            assert prefix + "ldl" in filtered_keys
            assert prefix + "sleep" not in filtered_keys
        finally:
            await session.rollback()
            await session.execute(
                text("DELETE FROM knowledge_entries WHERE knowledge_key LIKE :prefix"),
                {"prefix": prefix + "%"},
            )
            await session.commit()


async def test_empty_query_does_not_load_embedding_model() -> None:
    """空查询直接返回空结果。"""
    assert await search_knowledge("   ") == []


async def test_vector_dimension_and_top_k_are_validated() -> None:
    """错误维度和越界 top_k 被确定性拦截。"""
    with pytest.raises(ValueError, match="维度错误"):
        await search_knowledge_by_vector([0.0], top_k=1)
    with pytest.raises(ValueError, match="top_k"):
        await search_knowledge_by_vector(_vector([1.0]), top_k=0)
