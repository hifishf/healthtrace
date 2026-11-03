"""知识 RAG 检索：bge 查询向量 -> pgvector 余弦相似度 -> 带来源知识条目。"""
from dataclasses import dataclass

from app import crud
from app.db_types import VECTOR_DIMENSIONS
from app.embeddings import embed_query

MAX_TOP_K = 20


@dataclass(frozen=True)
class KnowledgeHit:
    """一条可溯源检索结果。"""

    knowledge_key: str
    title: str
    content: str
    category: str
    metric_codes: tuple[str, ...]
    source_name: str
    source_url: str
    source_version: str
    score: float

    @property
    def excerpt(self) -> str:
        """返回最长 240 字的单段摘要，供响应和评估上下文使用。"""
        compact = " ".join(self.content.split())
        return compact if len(compact) <= 240 else compact[:239] + "…"

    def as_dict(self) -> dict:
        """转为 MCP/API 可序列化结构，保留来源与完整上下文。"""
        return {
            "knowledge_key": self.knowledge_key,
            "title": self.title,
            "content": self.content,
            "excerpt": self.excerpt,
            "category": self.category,
            "metric_codes": list(self.metric_codes),
            "source_name": self.source_name,
            "source_url": self.source_url,
            "source_version": self.source_version,
            "score": round(self.score, 6),
        }


def _normalize_metric_codes(metric_codes: list[str] | None) -> list[str] | None:
    """去空、去重并稳定排序指标代码，保持检索条件可复现。"""
    if not metric_codes:
        return None
    cleaned = {str(code).strip() for code in metric_codes if str(code).strip()}
    return sorted(cleaned) or None


def _validate_top_k(top_k: int) -> None:
    """校验检索条数，防止无界查询。"""
    if not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f"top_k 必须在 1 到 {MAX_TOP_K} 之间")


async def search_knowledge_by_vector(
    embedding: list[float],
    top_k: int = 5,
    metric_codes: list[str] | None = None,
) -> list[KnowledgeHit]:
    """使用已生成的查询向量检索知识，便于单元测试和延迟诊断。"""
    _validate_top_k(top_k)
    if len(embedding) != VECTOR_DIMENSIONS:
        raise ValueError(
            f"查询向量维度错误：expected={VECTOR_DIMENSIONS}, actual={len(embedding)}"
        )
    rows = await crud.search_knowledge_entries(
        embedding,
        top_k=top_k,
        metric_codes=_normalize_metric_codes(metric_codes),
    )
    return [
        KnowledgeHit(
            knowledge_key=row["knowledge_key"],
            title=row["title"],
            content=row["content"],
            category=row["category"],
            metric_codes=tuple(row["metric_codes"] or []),
            source_name=row["source_name"],
            source_url=row["source_url"],
            source_version=row["source_version"],
            score=float(row["similarity"]),
        )
        for row in rows
    ]


async def search_knowledge(
    query: str,
    top_k: int = 5,
    metric_codes: list[str] | None = None,
) -> list[KnowledgeHit]:
    """生成查询向量并检索知识；空查询直接返回空列表。"""
    cleaned = (query or "").strip()
    if not cleaned:
        return []
    embedding = await embed_query(cleaned)
    return await search_knowledge_by_vector(
        embedding,
        top_k=top_k,
        metric_codes=metric_codes,
    )
