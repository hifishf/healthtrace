"""因果分析（insight）的确定性断言：时间对齐 / 滞后相关 / 措辞过滤 / 来源口径。"""
import math
import random

from app.agent.insight import (
    MIN_PAIRS,
    align_daily_series,
    grade_correlation,
    lagged_correlation,
)


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


def _bivariate_with_correlation(r: float, n: int = 200, seed: int = 0) -> tuple[list[float], list[float]]:
    """构造样本相关系数精确等于 r 的两条序列（中心化 + Gram-Schmidt 正交化 + 方差归一）。"""
    rng = random.Random(seed)
    x_raw = [rng.uniform(-1.0, 1.0) for _ in range(n)]
    mean_x = sum(x_raw) / n
    x = [v - mean_x for v in x_raw]
    z_raw = [rng.uniform(-1.0, 1.0) for _ in range(n)]
    mean_z = sum(z_raw) / n
    zc = [v - mean_z for v in z_raw]
    dot_xx = sum(a * a for a in x)
    dot_xz = sum(a * b for a, b in zip(x, zc))
    z = [zi - (dot_xz / dot_xx) * xi for xi, zi in zip(x, zc)]
    var_x = dot_xx / n
    var_z = sum(v * v for v in z) / n
    scale = math.sqrt(var_x / var_z)
    y = [r * xi + math.sqrt(1.0 - r * r) * scale * zi for xi, zi in zip(x, z)]
    return x, y


def test_lagged_correlation_known_coefficient_tolerance() -> None:
    """断言 #15：构造已知相关系数 0.8 的序列 -> 计算值在容差内、无滞后。"""
    x, y = _bivariate_with_correlation(0.8, n=200, seed=0)
    result = lagged_correlation(x, y, max_lag=7)

    assert result.best_lag == 0
    assert result.n == 200
    assert abs(result.pearson_r - 0.8) <= 0.05
    assert abs(result.spearman_rho - 0.8) <= 0.10


def test_lagged_correlation_detects_shift() -> None:
    """结果序列在 3 天后才随预测指标线性变化 -> 最佳滞后 = 3。"""
    x = [float(i) for i in range(60)]
    y = [0.0, 0.0, 0.0] + [2.0 * x[day - 3] + 1.0 for day in range(3, 60)]
    result = lagged_correlation(x, y, max_lag=7)

    assert result.best_lag == 3
    assert abs(result.pearson_r - 1.0) <= 1e-6


def test_lagged_correlation_negative() -> None:
    """单调相反的两条序列 -> Pearson 与 Spearman 均为 -1。"""
    x = [float(i) for i in range(50)]
    y = [-2.0 * v + 5.0 for v in x]
    result = lagged_correlation(x, y, max_lag=3)

    assert result.best_lag == 0
    assert abs(result.pearson_r + 1.0) <= 1e-6
    assert abs(result.spearman_rho + 1.0) <= 1e-6


def test_grade_correlation_thresholds() -> None:
    """分级阈值：0.7 强相关 / 0.3 弱相关 / 0.29 不足以判断。"""
    assert grade_correlation(0.7, 100) == "强相关"
    assert grade_correlation(-0.7, 100) == "强相关"
    assert grade_correlation(0.5, 100) == "弱相关"
    assert grade_correlation(-0.3, 100) == "弱相关"
    assert grade_correlation(0.29, 100) == "不足以判断"


def test_grade_correlation_insufficient_for_small_n() -> None:
    """样本量不足 MIN_PAIRS 或 r 为 nan -> 不足以判断。"""
    assert grade_correlation(0.9, MIN_PAIRS - 1) == "不足以判断"
    assert grade_correlation(float("nan"), 100) == "不足以判断"
