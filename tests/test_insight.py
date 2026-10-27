"""因果分析（insight）的确定性断言：时间对齐 / 滞后相关 / 措辞过滤 / 来源口径。"""
from app.agent.insight import align_daily_series


def _point(day: str, value: float) -> dict:
    """构造一个每日点（与 query_metrics 返回的字段形状一致）。"""
    return {"recorded_date": day, "value": value}


def test_align_full_overlap_keeps_all_points() -> None:
    """两序列日期集合一致 -> 全部保留、升序、不丢点。"""
    result = align_daily_series(
        [_point("2026-01-02", 2.0), _point("2026-01-01", 1.0), _point("2026-01-03", 3.0)],
        [_point("2026-01-03", 30.0), _point("2026-01-01", 10.0), _point("2026-01-02", 20.0)],
    )
    assert result.dates == ["2026-01-01", "2026-01-02", "2026-01-03"]
    assert result.a == [1.0, 2.0, 3.0]
    assert result.b == [10.0, 20.0, 30.0]
    assert result.dropped == 0


def test_align_drops_days_missing_on_either_side() -> None:
    """一边缺日的点被丢弃，dropped 统计两侧缺日总数。"""
    result = align_daily_series(
        [_point("2026-01-01", 1.0), _point("2026-01-02", 2.0), _point("2026-01-03", 3.0)],
        [_point("2026-01-02", 20.0)],
    )
    assert result.dates == ["2026-01-02"]
    assert result.a == [2.0]
    assert result.b == [20.0]
    assert result.dropped == 2


def test_align_duplicate_date_keeps_last_value() -> None:
    """同一天重复出现时取最后一个值（防御性口径）。"""
    result = align_daily_series(
        [_point("2026-01-01", 1.0), _point("2026-01-01", 9.0)],
        [_point("2026-01-01", 10.0)],
    )
    assert result.dates == ["2026-01-01"]
    assert result.a == [9.0]
    assert result.b == [10.0]
    assert result.dropped == 0
