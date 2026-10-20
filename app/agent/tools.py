"""Agent 工具：把落库等能力包装成 LLM 可调用的函数。"""
import datetime
import re
from decimal import Decimal
from zoneinfo import ZoneInfo

from langchain_core.tools import tool

from app import crud, portrait
from app.config import settings
from app.db import async_session

_METRIC_TIME_ZONE = ZoneInfo("Asia/Shanghai")
_FIXED_RELATIVE_DATES = {
    "今天": 0,
    "今日": 0,
    "昨天": 1,
    "昨日": 1,
    "昨晚": 1,
    "前天": 2,
    "前日": 2,
    "大前天": 3,
}
_CHINESE_DIGITS = {
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}


def metric_today(now: datetime.datetime | None = None) -> datetime.date:
    """返回业务时区（Asia/Shanghai）的今天，避免服务部署在 UTC 时偏一天。"""
    current = now or datetime.datetime.now(_METRIC_TIME_ZONE)
    if current.tzinfo is None:
        current = current.replace(tzinfo=_METRIC_TIME_ZONE)
    return current.astimezone(_METRIC_TIME_ZONE).date()


def _parse_chinese_day_count(value: str) -> int | None:
    """解析 1~99 的中文天数，仅用于“N 天前”这类相对日期。"""
    if value.isdigit():
        return int(value)
    if "十" not in value:
        return _CHINESE_DIGITS.get(value)

    tens_text, ones_text = value.split("十", maxsplit=1)
    tens = _CHINESE_DIGITS.get(tens_text, 1) if tens_text else 1
    if not ones_text:
        return tens * 10
    ones = _CHINESE_DIGITS.get(ones_text)
    return tens * 10 + ones if ones is not None else None


def resolve_metric_date(
    date_text: str | None,
    today: datetime.date | None = None,
) -> datetime.date:
    """把 ISO 日期或常见中文相对日期解析为记录日期。

    date_text 为空时默认今天；“昨晚”归到昨天；连续的“昨天的昨天...”按出现次数
    逐次后退，避免把相对日期换算交给 LLM。无法识别时抛出 ValueError。
    """
    reference_date = today or metric_today()
    if not date_text or not date_text.strip():
        return reference_date

    normalized = re.sub(r"\s+", "", date_text)
    try:
        return datetime.date.fromisoformat(normalized)
    except ValueError:
        pass

    if normalized in _FIXED_RELATIVE_DATES:
        return reference_date - datetime.timedelta(
            days=_FIXED_RELATIVE_DATES[normalized]
        )

    # 支持“昨天的昨天的昨天”这类链式表达，每个“昨天”后退一天。
    chained_yesterday = re.fullmatch(r"昨天(?:的昨天)+", normalized)
    if chained_yesterday:
        return reference_date - datetime.timedelta(
            days=normalized.count("昨天")
        )

    days_ago = re.fullmatch(r"([零一二两三四五六七八九十\d]+)[天日]前", normalized)
    if days_ago:
        day_count = _parse_chinese_day_count(days_ago.group(1))
        if day_count is not None:
            return reference_date - datetime.timedelta(days=day_count)

    raise ValueError(f"无法识别的日期：{date_text}")


@tool
async def record_metric(
    metric_code: str,
    value: float,
    unit: str,
    date: str | None = None,
) -> dict:
    """记录一条健康指标（覆盖语义：同一天同一指标只保留最后一次）。

    参数：
    - metric_code：指标代码，例如 sleep_duration（睡眠时长）。
    - value：指标数值，例如 6。
    - unit：单位，例如 h。
    - date：用户原话中的日期或日期表达，例如“今天”“昨晚”“三天前”；
      也可以直接传 YYYY-MM-DD。不要由模型自行换算相对日期，不传则默认今天。
    示例：用户说“昨晚睡了 6 小时”→
    record_metric(metric_code="sleep_duration", value=6, unit="h", date="昨晚")。
    返回 {ok: true, id, ...} 或 {ok: false, error_code}。
    """
    # 基础校验：metric_code 非空
    if not metric_code or not metric_code.strip():
        return {"ok": False, "error_code": "INVALID_METRIC_CODE"}

    # 日期解析：相对日期由业务代码确定性换算，失败时返回明确错误码
    try:
        record_date = resolve_metric_date(date)
    except ValueError:
        return {"ok": False, "error_code": "INVALID_DATE"}

    async with async_session() as session:
        record_id = await crud.upsert_metric(
            session=session,
            user_id=settings.default_user_id,
            metric_code=metric_code.strip(),
            value=Decimal(str(value)),
            unit=unit,
            recorded_date=record_date,
            source="chat",
        )
        # 记录指标 = 画像数据变过，刷新画像 updated_at（“画像更新”）
        await portrait.touch_portrait(session, settings.default_user_id)

    return {
        "ok": True,
        "id": record_id,
        "metric_code": metric_code.strip(),
        "value": value,
        "unit": unit,
        "date": record_date.isoformat(),
    }


@tool
async def query_profile() -> dict:
    """查询当前用户的健康画像（出生年份、性别、身高、体重、BMI 等）。

    无参数（单用户）。返回画像字段 dict；无画像时返回空 dict。
    """
    async with async_session() as session:
        return await portrait.get_health_portrait(session, settings.default_user_id)
