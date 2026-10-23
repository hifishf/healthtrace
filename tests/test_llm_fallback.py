"""统一 LLM 调用层的降级验收：超时/5xx 重试、规则模板兜底、红旗不依赖 LLM、token 预算收口。"""
import uuid

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app import observability
from app.agent.llm import LLMUnavailable, invoke_llm
from app.agent.triage import build_triage_graph, initial_triage_state
from app.config import settings
from app.schemas import SymptomExtraction


class _RaiseRunnable:
    """ainvoke 时抛出指定异常的可运行对象。"""

    def __init__(self, exc: Exception) -> None:
        self.exc = exc
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        raise self.exc


class _FailingLLM:
    """结构化输出与转写都失败（模拟 LLM 超时/5xx/宕机）。"""

    def __init__(self, exc: Exception) -> None:
        self._runnable = _RaiseRunnable(exc)

    def with_structured_output(self, schema, method):
        return self._runnable

    async def ainvoke(self, messages):
        self._runnable.calls += 1
        raise self._runnable.exc


class _OkLLM:
    """抽取成功、转写返回固定话术的假模型（token 预算测试用）。"""

    def __init__(self, rephrase: str) -> None:
        self._rephrase = rephrase

    def with_structured_output(self, schema, method):
        return _OkExtract()

    async def ainvoke(self, messages):
        return AIMessage(content=self._rephrase)


class _OkExtract:
    async def ainvoke(self, messages):
        text = messages[-1].content
        if "累" in text:
            return SymptomExtraction(symptoms=["乏力"], subject="self")
        return SymptomExtraction(symptoms=[], subject="self")


async def test_invoke_llm_retries_then_raises() -> None:
    """重试次数按配置：失败重试 N 次后抛 LLMUnavailable。"""
    runnable = _RaiseRunnable(RuntimeError("boom"))
    with pytest.raises(LLMUnavailable) as exc_info:
        await invoke_llm(runnable, ["x"], retries=2)

    assert exc_info.value.reason == "llm_unavailable"
    assert runnable.calls == 3  # 1 次初始 + 2 次重试


@pytest.mark.parametrize(
    ("exc", "expected_reason"),
    [
        (TimeoutError("timed out"), "llm_timeout"),
        (RuntimeError("503 Service Unavailable"), "llm_unavailable"),
    ],
)
async def test_red_flag_survives_llm_down(exc: Exception, expected_reason: str) -> None:
    """停 LLM（超时/5xx）→ 红旗仍执行，紧急建议用固定安全核心，fallback_reason 进 trace。"""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    observability.use_tracer_provider(provider)
    try:
        async with AsyncPostgresSaver.from_conn_string(
            settings.checkpoint_database_url
        ) as saver:
            await saver.setup()
            graph = build_triage_graph(llm=_FailingLLM(exc), checkpointer=saver)
            config = {"configurable": {"thread_id": f"fallback-{uuid.uuid4().hex[:8]}"}}
            result = await graph.ainvoke(
                initial_triage_state("胸口疼得喘不上气"), config=config
            )
    finally:
        observability.use_tracer_provider(None)

    # 红旗照常触发，固定安全核心逐字保留，且不依赖 LLM
    assert result["red_flag_triggered"] is True
    assert result["final_advice"]["level"] == "emergency"
    assert "请立即就医或拨打急救电话" in result["final_advice"]["suggestion"]

    # 降级事件打 fallback_reason 进 trace
    reasons = {
        span.attributes.get("fallback_reason")
        for span in exporter.get_finished_spans()
        if span.attributes.get("fallback_reason")
    }
    assert expected_reason in reasons


async def test_token_budget_exceeded_forces_template(monkeypatch) -> None:
    """token 预算超限 → 追问话术回退确定性模板，不再调用 LLM。"""
    monkeypatch.setattr(settings, "llm_session_token_budget", 1)
    llm = _OkLLM(rephrase="预算没超时的 LLM 转写话术")

    async with AsyncPostgresSaver.from_conn_string(
        settings.checkpoint_database_url
    ) as saver:
        await saver.setup()
        graph = build_triage_graph(llm=llm, checkpointer=saver)
        config = {"configurable": {"thread_id": f"budget-{uuid.uuid4().hex[:8]}"}}
        result = await graph.ainvoke(
            initial_triage_state("最近总是累"), config=config
        )

    question = result["__interrupt__"][0].value.get("question")
    # 抽症状即耗尽预算（limit=1），追问退回模板而非 LLM 转写
    assert "请问你最近有没有" in question
    assert question != "预算没超时的 LLM 转写话术"
