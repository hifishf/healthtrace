"""OpenTelemetry 初始化 + 通用追踪助手（span / LLM generation / 序列化）。"""
import base64
import functools
import json
from contextlib import contextmanager
from typing import Any, Iterator

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from app.config import settings

_TRACER_NAME = "healthtrace"
_MAX_ATTR_LEN = 4000
_provider: TracerProvider | None = None


def _auth_header() -> str:
    """构造 Langfuse OTel 的 Basic Auth 头：username=public key，password=secret key。"""
    raw = f"{settings.langfuse_public_key}:{settings.langfuse_secret_key}"
    token = base64.b64encode(raw.encode()).decode()
    return f"Basic {token}"


def init_observability() -> None:
    """初始化全局 TracerProvider，把 span 上报到 Langfuse 的 OTel 端点。"""
    global _provider
    exporter = OTLPSpanExporter(
        endpoint=f"{settings.langfuse_host}/api/public/otel/v1/traces",
        headers={"Authorization": _auth_header()},
    )
    provider = TracerProvider(
        resource=Resource.create({"service.name": "healthtrace-api"}),
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
    _provider = provider
    trace.set_tracer_provider(provider)


def use_tracer_provider(provider: TracerProvider | None) -> None:
    """测试/嵌入时替换 tracer provider（避免重复触发全局 set 警告）。"""
    global _provider
    _provider = provider


def get_tracer() -> trace.Tracer:
    """返回 tracer（优先使用显式配置的 provider）。"""
    provider = _provider or trace.get_tracer_provider()
    return provider.get_tracer(_TRACER_NAME)


def _truncate(text: str) -> str:
    """截断超长属性，避免把巨大 prompt 全量塞进 trace。"""
    if len(text) <= _MAX_ATTR_LEN:
        return text
    return text[:_MAX_ATTR_LEN] + "...[truncated]"


def to_json(value: Any) -> str:
    """把 Pydantic/字典/列表等转成稳定 JSON 字符串（截断超长内容）。"""
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        text = str(value)
    return _truncate(text)


def _coerce(value: Any) -> str | bool | int | float:
    """OTel 属性只接受标量；复杂值转 JSON 字符串。"""
    if isinstance(value, str):
        return _truncate(value)
    if isinstance(value, (bool, int, float)):
        return value
    return to_json(value)


def set_attr(key: str, value: Any) -> None:
    """给当前 span 设置属性（None 跳过）。"""
    if value is None:
        return
    trace.get_current_span().set_attribute(key, _coerce(value))


def set_output(value: Any) -> None:
    """设置当前 span 的 Langfuse output。"""
    set_attr("langfuse.observation.output", value)


@contextmanager
def llm_span(
    name: str,
    input_data: Any,
    model: str | None = None,
) -> Iterator[trace.Span]:
    """LLM 调用 span：记录输入与模型；输出由调用方 set_output。"""
    with get_tracer().start_as_current_span(name) as span:
        span.set_attribute("langfuse.observation.type", "generation")
        span.set_attribute("gen_ai.system", "openai-compatible")
        span.set_attribute("gen_ai.request.model", model or settings.llm_model)
        span.set_attribute("langfuse.observation.input", to_json(input_data))
        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            span.set_attribute("langfuse.observation.level", "ERROR")
            raise


def traced(name: str):
    """异步节点追踪装饰器：以 span 包裹节点执行。"""

    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            with get_tracer().start_as_current_span(name):
                return await fn(*args, **kwargs)

        return wrapper

    return decorator
