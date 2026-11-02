"""纵向趋势分析：同指标多次检验序列 -> 确定性趋势结论 + 相邻对比 + 参考范围判定。

第 9 周冻结验收（断言 #10）：LDL [3.1, 3.4, 3.8] -> 「持续上升」。
趋势口径全部确定性，不依赖 LLM；LLM 若介入只负责把结论转写成自然话术（第 9 周未做）。
"""
from datetime import date

# 趋势结论取值（冻结）：持续上升 / 持续下降 / 基本平稳 / 波动 / 数据不足
TREND_RISING = "持续上升"
TREND_FALLING = "持续下降"
TREND_FLAT = "基本平稳"
TREND_MIXED = "波动"
TREND_INSUFFICIENT = "数据不足"

# 相邻变化量的绝对容差：|delta| <= 容差视为持平，用于剔除浮点尾数噪声
_FLAT_DELTA_TOL = 1e-9

# 第 9 周演示指标码 -> 中文展示名（第 8 周起的常量映射方案，未建指标字典表）
METRIC_LABELS = {
    "ldl_cholesterol": "低密度脂蛋白胆固醇(LDL)",
    "total_cholesterol": "总胆固醇",
    "hdl_cholesterol": "高密度脂蛋白胆固醇(HDL)",
    "triglycerides": "甘油三酯",
    "fasting_glucose": "空腹血糖",
    "hba1c": "糖化血红蛋白",
}


def metric_label(metric_code: str) -> str:
    """指标码 -> 中文展示名；未收录时回退到原始指标码。"""
    return METRIC_LABELS.get(metric_code, metric_code)


def _sorted_points(points: list[dict]) -> list[tuple[date, float]]:
    """按记录日期升序返回 (date, value)；同日重复取最后一条（防御，正常唯一）。"""
    by_date: dict[date, float] = {}
    for point in points:
        by_date[date.fromisoformat(point["recorded_date"])] = float(point["value"])
    return [(day, by_date[day]) for day in sorted(by_date)]


def reference_status(value: float, reference: dict | None) -> str | None:
    """按参考范围判定单值状态：偏低 / 偏高 / 正常；无参考范围返回 None。
    口径：value < low 判偏低、value > high 判偏高，其余在区间内判正常（边界并入正常侧）。
    """
    if not reference:
        return None
    low = reference.get("low")
    high = reference.get("high")
    if low is not None and value < low:
        return "偏低"
    if high is not None and value > high:
        return "偏高"
    if low is not None or high is not None:
        return "正常"
    return None


def _direction(delta: float) -> str:
    """相邻变化方向：上升 / 下降 / 持平（用容差剔除浮点噪声）。"""
    if delta > _FLAT_DELTA_TOL:
        return "上升"
    if delta < -_FLAT_DELTA_TOL:
        return "下降"
    return "持平"


def _trend_from_deltas(deltas: list[float]) -> str:
    """由相邻变化量序列判定趋势：全正=持续上升，全负=持续下降，
    全持平=基本平稳，有正有负=波动。
    """
    if all(d > _FLAT_DELTA_TOL for d in deltas):
        return TREND_RISING
    if all(d < -_FLAT_DELTA_TOL for d in deltas):
        return TREND_FALLING
    if all(abs(d) <= _FLAT_DELTA_TOL for d in deltas):
        return TREND_FLAT
    return TREND_MIXED


def _conclusion_text(
    name: str,
    ordered: list[tuple[date, float]],
    trend: str,
    unit: str,
) -> str:
    """把趋势结论渲染成一句中文（确定性模板，不调 LLM）。"""
    n = len(ordered)
    first_value = ordered[0][1]
    last_value = ordered[-1][1]
    suffix = f" {unit}" if unit else ""
    if trend == TREND_RISING:
        return f"{name}连续 {n} 次记录持续上升，从 {first_value}{suffix} 升至 {last_value}{suffix}。"
    if trend == TREND_FALLING:
        return f"{name}连续 {n} 次记录持续下降，从 {first_value}{suffix} 降至 {last_value}{suffix}。"
    if trend == TREND_FLAT:
        return f"{name}近 {n} 次记录基本平稳（{last_value}{suffix}）。"
    return f"{name}近 {n} 次记录有波动，趋势不明显。"


def analyze_metric_trend(
    points: list[dict],
    reference: dict | None = None,
    label: str | None = None,
) -> dict:
    """确定性纵向趋势分析：相邻两次对比（变化量/变化率）+ 参考范围判定 + 趋势结论。

    趋势口径（冻结）：
    - <2 个点 -> 数据不足；
    - 相邻变化量全为正 -> 持续上升；全为负 -> 持续下降；
    - 全为持平（|delta|<=容差）-> 基本平稳；其余 -> 波动。

    :param points: 同一指标的时间序列点，每个至少含 recorded_date（YYYY-MM-DD）与 value
    :param reference: 可选参考范围 {"low": float|None, "high": float|None, "unit": str|None}
    :param label: 指标中文名（用于结论文案），缺省用「该指标」
    :return: {"trend": str, "comparisons": list, "reference": dict|None, "text": str}
    """
    ordered = _sorted_points(points)
    if len(ordered) < 2:
        return {
            "trend": TREND_INSUFFICIENT,
            "comparisons": [],
            "reference": reference,
            "text": "数据不足（至少需要两次记录）才能判断趋势。",
        }

    comparisons: list[dict] = []
    deltas: list[float] = []
    for (prev_date, prev_value), (curr_date, curr_value) in zip(ordered, ordered[1:]):
        delta = curr_value - prev_value
        deltas.append(delta)
        rate = (delta / prev_value) if prev_value != 0 else None
        comparisons.append(
            {
                "from_date": prev_date.isoformat(),
                "to_date": curr_date.isoformat(),
                "from_value": prev_value,
                "to_value": curr_value,
                "delta": delta,
                "rate": rate,
                "direction": _direction(delta),
                "status": reference_status(curr_value, reference),
            }
        )

    trend = _trend_from_deltas(deltas)
    name = label or "该指标"
    unit = (reference or {}).get("unit") or ""
    return {
        "trend": trend,
        "comparisons": comparisons,
        "reference": reference,
        "text": _conclusion_text(name, ordered, trend, unit),
    }
