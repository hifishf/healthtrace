"""RAG 检索冒烟与延迟诊断：真实模型 + 真实 pgvector。"""
import argparse
import asyncio
import math
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.embeddings import embed_query  # noqa: E402
from app.rag import search_knowledge, search_knowledge_by_vector  # noqa: E402

DEFAULT_QUERIES = [
    "血脂连续上升应该注意什么？",
    "服用多种药物时怎么判断交互风险？",
    "头晕时需要补充哪些信息？",
    "体检指标超出参考范围怎么办？",
    "最近总是乏力和睡眠不好",
]


def percentile(values: list[float], ratio: float) -> float:
    """计算最近秩百分位，避免额外统计依赖。"""
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * ratio) - 1)
    return ordered[index]


async def run(queries: list[str], rounds: int) -> None:
    """预热模型和数据库后，逐查询测试来源、命中分数与端到端延迟。"""
    await search_knowledge(queries[0], top_k=1)
    embedding_latencies: list[float] = []
    db_latencies: list[float] = []
    total_latencies: list[float] = []
    for _ in range(rounds):
        for query in queries:
            started = time.perf_counter()
            embedding = await embed_query(query)
            embedded_at = time.perf_counter()
            hits = await search_knowledge_by_vector(embedding, top_k=5)
            finished = time.perf_counter()
            embedding_latencies.append((embedded_at - started) * 1000)
            db_latencies.append((finished - embedded_at) * 1000)
            total_latencies.append((finished - started) * 1000)
            if not hits:
                raise RuntimeError(f"查询无检索结果：{query}")
            top = hits[0]
            print(
                f"query={query} top={top.knowledge_key} "
                f"score={top.score:.4f} source={top.source_url}"
            )

    print(
        f"samples={len(total_latencies)} "
        f"embedding_p95={percentile(embedding_latencies, 0.95):.1f}ms "
        f"db_p95={percentile(db_latencies, 0.95):.1f}ms "
        f"total_p50={statistics.median(total_latencies):.1f}ms "
        f"total_p95={percentile(total_latencies, 0.95):.1f}ms"
    )
    if percentile(db_latencies, 0.95) > 100:
        raise RuntimeError("pgvector 检索 p95 超过 100ms")
    if percentile(total_latencies, 0.95) > 300:
        raise RuntimeError("embedding + pgvector 总耗时 p95 超过 300ms")


def main() -> None:
    """解析参数并执行 RAG 诊断。"""
    parser = argparse.ArgumentParser(description="HealthTrace RAG 检索诊断")
    parser.add_argument("--rounds", type=int, default=5, help="默认查询集的重复轮数")
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds 必须大于 0")
    asyncio.run(run(DEFAULT_QUERIES, args.rounds))


if __name__ == "__main__":
    main()
