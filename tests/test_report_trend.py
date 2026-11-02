"""报告纵向趋势（trends）的确定性断言：持续上升 / 持续下降 / 波动 / 平稳 / 参考范围。"""
import math

from app.trends import (
    TREND_FALLING,
    TREND_FLAT,
    TREND_INSUFFICIENT,
    TREND_MIXED,
    TREND_RISING,
    analyze_metric_trend,
    metric_label,
    reference_status,
)


def _point(day: str, value: float) -> dict:
    """构造一个时间序列点（与 query_metrics 返回的字段形状一致）。"""
    return {"recorded_date": day, "value": value}


def _ldl_points(values: list[float], start_year: int = 2024) -> list[dict]:
    """把 LDL 值列表摊到连续年份的固定日期上，构造三年序列。"""
    return [
        _point(f"{start_year + i}-01-15", value)
        for i, value in enumerate(values)
    ]


def test_ldl_three_years_rising_trend() -> None:
    """断言 #10：LDL [3.1, 3.4, 3.8] -> 持续上升，且结论文案含「持续上升」。"""
    result = analyze_metric_trend(
        _ldl_points([3.1, 3.4, 3.8]),
        reference={"low": None, "high": 3.4, "unit": "mmol/L"},
        label=metric_label("ldl_cholesterol"),
    )
    assert result["trend"] == TREND_RISING
    assert "持续上升" in result["text"]
    # 相邻两段都是上升，且最后一段 3.4 -> 3.8 的变化量/变化率正确
    assert [c["direction"] for c in result["comparisons"]] == ["上升", "上升"]
    last = result["comparisons"][-1]
    assert math.isclose(last["delta"], 0.4, abs_tol=1e-9)
    assert math.isclose(last["rate"], 0.4 / 3.4, abs_tol=1e-9)


def test_falling_and_mixed_and_flat_trends() -> None:
    """下降 / 波动 / 基本平稳三种趋势各一条确定性断言。"""
    falling = analyze_metric_trend(_ldl_points([3.8, 3.4, 3.1]))
    assert falling["trend"] == TREND_FALLING

    mixed = analyze_metric_trend(_ldl_points([3.1, 3.8, 3.4]))
    assert mixed["trend"] == TREND_MIXED

    flat = analyze_metric_trend(_ldl_points([3.4, 3.4, 3.4]))
    assert flat["trend"] == TREND_FLAT


def test_insufficient_points() -> None:
    """少于两点 -> 数据不足，且没有对比项。"""
    result = analyze_metric_trend([_point("2026-01-01", 3.1)])
    assert result["trend"] == TREND_INSUFFICIENT
    assert result["comparisons"] == []


def test_reference_status_boundaries() -> None:
    """参考范围判定：偏低 / 正常 / 偏高 / 无范围。"""
    reference = {"low": 3.9, "high": 6.1}
    assert reference_status(3.5, reference) == "偏低"
    assert reference_status(4.5, reference) == "正常"
    assert reference_status(7.0, reference) == "偏高"
    assert reference_status(5.0, None) is None
    # 上限边界并入正常侧（value == high 不算偏高）
    assert reference_status(6.1, reference) == "正常"
