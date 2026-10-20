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
    """Router 结构化输出：一句话的意图分类结果。"""

    intent: Literal["triage", "insight", "report", "chat"] = Field(
        description=(
            "意图分类：triage=追问分诊 / insight=因果分析 / "
            "report=报告解读 / chat=闲聊或记录指标"
        )
    )
    confidence: float = Field(
        ge=0,
        le=1,
        description="0~1 之间的把握程度，越大越确定；LLM 自评、未经校准，不是概率",
    )
    extracted_slots: dict = Field(
        default_factory=dict,
        description='从用户话里抽出的槽位，如 {"symptom": "疲劳"}；本周占位，第 3 周启用',
    )


class ChatRequest(BaseModel):
    """POST /chat 请求体：用户一句话。"""

    text: str = Field(min_length=1)


class ChatResponse(BaseModel):
    """POST /chat 响应：路由结果 + 最终回复（第 4 步非流式）。"""

    intent: str
    confidence: float
    reply: str
