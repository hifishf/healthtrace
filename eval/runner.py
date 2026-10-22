"""端到端语义评估运行器：把 golden 用例跑完真实分诊，取出最终建议。

语义评估的目标是「Agent 实际说出的建议质量」，所以这里复用真实的分诊图
（build_triage_graph + 真实 LLM + Postgres checkpointer），而不是另写一套假链路。
追问轮次自动用「没有」续跑，保证每条用例最终都能拿到 final_advice。
"""
from __future__ import annotations

import uuid

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command

from app.agent.llm import get_llm
from app.agent.triage import (
    MAX_FOLLOWUP_ROUNDS,
    build_triage_graph,
    initial_triage_state,
)
from app.config import settings


async def run_triage_batch(texts: list[str]) -> list[dict]:
    """批量跑真实分诊，返回每条输入对应的 final_advice（dict）。"""
    advices: list[dict] = []
    async with AsyncPostgresSaver.from_conn_string(
        settings.checkpoint_database_url
    ) as saver:
        await saver.setup()
        graph = build_triage_graph(
            llm=get_llm(),
            writer_llm=get_llm(temperature=settings.llm_writer_temperature),
            checkpointer=saver,
        )
        for text in texts:
            config = {"configurable": {"thread_id": f"eval-{uuid.uuid4().hex[:8]}"}}
            result = await graph.ainvoke(initial_triage_state(text), config=config)
            # 若停在中途（interrupt），自动用「没有」续跑，直到出最终建议或到硬上限
            for _ in range(MAX_FOLLOWUP_ROUNDS):
                if "__interrupt__" not in result:
                    break
                result = await graph.ainvoke(Command(resume="没有"), config=config)
            advices.append(result.get("final_advice") or {})
    return advices
