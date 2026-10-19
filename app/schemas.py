"""API 请求/响应模型。"""
from datetime import date
from decimal import Decimal

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
