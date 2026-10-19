"""OpenTelemetry 初始化：把 span 通过 OTLP/HTTP 上报到自托管 Langfuse。"""
import base64

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from app.config import settings

_TRACER_NAME = "healthtrace"


def _auth_header() -> str:
    """构造 Langfuse OTel 的 Basic Auth 头：username=public key，password=secret key。"""
    raw = f"{settings.langfuse_public_key}:{settings.langfuse_secret_key}"
    token = base64.b64encode(raw.encode()).decode()
    return f"Basic {token}"


def init_observability() -> None:
    """初始化全局 TracerProvider，把 span 上报到 Langfuse 的 OTel 端点。"""
    exporter = OTLPSpanExporter(
        endpoint=f"{settings.langfuse_host}/api/public/otel/v1/traces",
        headers={"Authorization": _auth_header()},
    )
    provider = TracerProvider(
        resource=Resource.create({"service.name": "healthtrace-api"}),
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)


def get_tracer() -> trace.Tracer:
    """返回全局 tracer。"""
    return trace.get_tracer(_TRACER_NAME)
