"""triage 子图追踪：节点 span + LLM generation 输入输出。"""
import uuid

from langchain_core.messages import AIMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app import observability
from app.agent.triage import build_triage_graph, initial_triage_state
from app.config import settings
from app.schemas import SymptomExtraction


class _FakeRunnable:
    def __init__(self, extract_fn) -> None:
        self.extract_fn = extract_fn

    async def ainvoke(self, messages: list) -> SymptomExtraction:
        return self.extract_fn(messages[-1].content)


class _FakeLLM:
    def __init__(self, extract_fn) -> None:
        self._runnable = _FakeRunnable(extract_fn)

    def with_structured_output(self, schema, method) -> _FakeRunnable:
        return self._runnable

    async def ainvoke(self, messages: list) -> AIMessage:
        return AIMessage(content="自然语言话术")


def _extract(text: str) -> SymptomExtraction:
    if "失眠" in text:
        return SymptomExtraction(symptoms=["失眠"], subject="self")
    if "不会" in text:
        return SymptomExtraction(symptoms=[], subject="self", answer_verdict="deny")
    return SymptomExtraction(symptoms=[], subject="self", answer_verdict="affirm")


def test_to_json_serializes_pydantic_and_truncates() -> None:
    """结构化结果转 JSON；超长内容截断。"""
    text = observability.to_json(
        SymptomExtraction(symptoms=["失眠"], subject="self", answer_verdict="deny")
    )
    assert "失眠" in text
    assert "answer_verdict" in text

    long_text = observability.to_json("x" * 10000)
    assert long_text.endswith("[truncated]")


async def test_triage_trace_contains_nodes_and_llm_io() -> None:
    """一次追问跑完：节点 span + 三次 LLM 输入输出都在 trace 里。"""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    observability.use_tracer_provider(provider)
    try:
        async with AsyncPostgresSaver.from_conn_string(
            settings.checkpoint_database_url
        ) as saver:
            await saver.setup()
            graph = build_triage_graph(llm=_FakeLLM(_extract), checkpointer=saver)
            config = {"configurable": {"thread_id": f"trace-{uuid.uuid4().hex[:8]}"}}
            result = await graph.ainvoke(
                initial_triage_state("我最近失眠"), config=config
            )
            for answer in ["不会", "有", "有"]:
                if "__interrupt__" not in result:
                    break
                result = await graph.ainvoke(Command(resume=answer), config=config)
    finally:
        observability.use_tracer_provider(None)

    spans = exporter.get_finished_spans()
    by_name: dict[str, list] = {}
    for span in spans:
        by_name.setdefault(span.name, []).append(span)

    assert "graph.triage.extract" in by_name
    assert "graph.triage.advise" in by_name
    assert "llm.triage.extract_symptoms" in by_name
    assert "llm.triage.rephrase_question" in by_name
    assert "llm.triage.write_advice" in by_name

    extract_llm = by_name["llm.triage.extract_symptoms"][0]
    assert extract_llm.attributes["langfuse.observation.type"] == "generation"
    assert extract_llm.attributes["gen_ai.request.model"] == settings.llm_model
    assert "answer_verdict" in extract_llm.attributes["langfuse.observation.output"]

    extract_nodes = by_name["graph.triage.extract"]
    assert any(span.attributes.get("triage.text") == "我最近失眠" for span in extract_nodes)
    assert any(span.attributes.get("triage.answer_verdict") == "deny" for span in extract_nodes)
    assert any(span.attributes.get("triage.candidate_count", 0) > 0 for span in extract_nodes)
