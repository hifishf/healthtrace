"""API 请求/响应模型与 Agent 结构化输出。"""
from datetime import date
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field


class MetricCreate(BaseModel):
    """POST /metrics 请求体：一条指标记录（同日期同指标覆盖）。"""

    metric_code: str = Field(min_length=1, max_length=50)
    value: Decimal
    unit: str = Field(min_length=1, max_length=20)
    recorded_date: date


class MetricResponse(BaseModel):
    """POST /metrics 响应：{ok, id}。"""

    ok: bool
    id: int


class RouteDecision(BaseModel):
    """Router 结构化输出：一句话的意图分类结果。

    intent 为严格枚举；confidence 是 LLM 自评的把握（未经校准，非概率）。
    extracted_slots 本周占位，第 3 周抽症状槽位时启用。
    """

    intent: Literal["triage", "insight", "report", "chat"]
    confidence: float
    extracted_slots: dict = Field(default_factory=dict)
