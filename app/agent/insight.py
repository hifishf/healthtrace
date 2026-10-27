"""因果分析（insight）子图：时间对齐 -> 滞后相关 -> LLM 解释 -> 措辞过滤。"""
from dataclasses import dataclass
from math import isnan

from scipy import stats

# 滞后搜索窗口（天）：在 [-MAX_LAG, MAX_LAG] 内找 |Pearson r| 最大的滞后档
MAX_LAG = 7
# 参与相关计算的最小配对样本数：低于该值不给显著性判断（样本量太小）
MIN_PAIRS = 10
# 相关强度分级阈值（按 |r| 分档）
STRONG_R = 0.7
WEAK_R = 0.3
# 判定 |r| 并列的容差：浮点尾数噪声约 1e-16，取 1e-9 足够把「数值上几乎相等」判为并列，
# 避免线性斜坡这类数据在多个滞后档上因最后一位尾数不同而被误判出不同强度
_R_TIE_TOL = 1e-9


@dataclass(frozen=True)
class AlignedSeries:
    """两条时间序列按日期对齐后的结果（升序、按共有日期配对）。"""

    dates: list[str]  # 两序列共有的日期（YYYY-MM-DD，升序）
    a: list[float]  # 预测指标值（与 dates 一一对齐）
    b: list[float]  # 结果指标值（与 dates 一一对齐）
    dropped: int  # 因另一方缺日而被丢弃的天数


def align_daily_series(points_a: list[dict], points_b: list[dict]) -> AlignedSeries:
    """把两条每日序列按日期对齐，只保留两边都有值的日期（不插值）。
    口径：入参是「每日一点」的序列，每个点至少含 recorded_date（YYYY-MM-DD）与 value；
    以日期为键取交集并按升序配对，同一日期重复出现取最后一个（防御，正常唯一）；
    缺日的点直接丢弃并累计到 dropped，不补值、不插值。
    :param points_a: 预测指标（如咖啡因摄入）的每日点列表
    :param points_b: 结果指标（如入睡延迟）的每日点列表
    :return: 对齐后的共有日期、两列数值与被丢弃天数
    """
    a_by_date: dict[str, float] = {}
    for point in points_a:
        a_by_date[point["recorded_date"]] = float(point["value"])
    b_by_date: dict[str, float] = {}
    for point in points_b:
        b_by_date[point["recorded_date"]] = float(point["value"])
    common_dates = sorted(set(a_by_date) & set(b_by_date))
    return AlignedSeries(
        dates=common_dates,
        a=[a_by_date[day] for day in common_dates],
        b=[b_by_date[day] for day in common_dates],
        dropped=len(a_by_date) + len(b_by_date) - 2 * len(common_dates),
    )


@dataclass(frozen=True)
class LagCorrelation:
    """一条序列相对另一条做滞后相关后，最佳滞后处的统计结果。"""

    best_lag: int  # 最佳滞后（结果指标相对预测指标错开的天数）
    n: int  # 最佳滞后处实际参与计算的配对样本数
    pearson_r: float  # 最佳滞后处的 Pearson 相关系数
    pearson_p: float  # 对应双侧 p 值
    spearman_rho: float  # 最佳滞后处的 Spearman 秩相关系数
    spearman_p: float  # 对应双侧 p 值


def _correlate(xs: list[float], ys: list[float]) -> tuple[float, float, float, float]:
    """算一对等长序列的 Pearson 与 Spearman；任一方无方差时返回 nan（不参与选优）。"""
    if len(set(xs)) < 2 or len(set(ys)) < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    r, p = stats.pearsonr(xs, ys)
    rho, p_rho = stats.spearmanr(xs, ys)
    return float(r), float(p), float(rho), float(p_rho)


def lagged_correlation(
    x: list[float],
    y: list[float],
    max_lag: int = MAX_LAG,
) -> LagCorrelation:
    """对已按日期对齐的两条序列做滞后相关，返回 |Pearson r| 最大的滞后档统计。
    口径：lag=L 表示「预测指标第 t 天 ↔ 结果指标第 t+L 天」；配对窗口随 L 收缩，
    配对不足 3 的滞后档跳过；|r| 并列时取 |lag| 更小的档。计算只负责产出统计量，
    不做因果判断。
    :param x: 预测指标数值（与 y 等长、日期对齐）
    :param y: 结果指标数值（与 x 等长、日期对齐）
    :param max_lag: 滞后搜索窗口（天）
    :return: 最佳滞后处的 Pearson/Spearman 统计
    """
    if len(x) != len(y):
        raise ValueError("两条序列必须等长（先做时间对齐）")
    if len(x) < 3:
        raise ValueError("样本不足 3 天，无法计算滞后相关")
    n = len(x)
    best: LagCorrelation | None = None
    for lag in range(-max_lag, max_lag + 1):
        lo = max(0, -lag)
        hi = n - max(0, lag)
        if hi - lo < 3:
            continue
        xs = x[lo:hi]
        ys = y[lo + lag : hi + lag]
        r, p, rho, p_rho = _correlate(xs, ys)
        if isnan(r):
            continue
        candidate = LagCorrelation(
            best_lag=lag,
            n=hi - lo,
            pearson_r=r,
            pearson_p=p,
            spearman_rho=rho,
            spearman_p=p_rho,
        )
        if best is None:
            best = candidate
            continue
        best_abs = abs(best.pearson_r)
        if abs(r) - best_abs > _R_TIE_TOL:
            best = candidate
        elif abs(abs(r) - best_abs) <= _R_TIE_TOL and abs(lag) < abs(best.best_lag):
            best = candidate
    if best is None:
        return LagCorrelation(
            best_lag=0,
            n=n,
            pearson_r=float("nan"),
            pearson_p=float("nan"),
            spearman_rho=float("nan"),
            spearman_p=float("nan"),
        )
    return best


def grade_correlation(r: float, n: int) -> str:
    """按样本量与 |r| 给相关强度分级：强相关 / 弱相关 / 不足以判断。"""
    if n < MIN_PAIRS or isnan(r):
        return "不足以判断"
    abs_r = abs(r)
    if abs_r >= STRONG_R:
        return "强相关"
    if abs_r >= WEAK_R:
        return "弱相关"
    return "不足以判断"


# 因果断言词的确定性替换（输出前兜底；冻结契约只要求 导致/因为/引起 -> 相关/伴随，
# 确诊/诊断 来自 13.1 断言 #14 的禁用词，一并替换）
CAUSAL_REPLACEMENTS = {
    "导致": "相关",
    "引起": "伴随",
    "因为": "相关",
    "确诊": "相关",
    "诊断": "相关",
}
FORBIDDEN_CAUSAL_WORDS = frozenset(CAUSAL_REPLACEMENTS)


def filter_causal_wording(text: str) -> str:
    """确定性替换因果断言词（输出前兜底），保证结果不含 导致/引起/因为/确诊/诊断。"""
    result = text
    for word, replacement in CAUSAL_REPLACEMENTS.items():
        result = result.replace(word, replacement)
    return result


def build_template_explanation(
    correlation: LagCorrelation,
    label_a: str,
    label_b: str,
    synthetic: bool,
) -> str:
    """把统计结果渲染成一句中文（确定性模板，LLM 失败时的兜底，同样经措辞过滤）。
    口径：synthetic=True 才给出相关强度 + N + p；synthetic=False 只描述趋势、
    不做显著性断言、不写强/弱相关分级。相关方向用正/负，趋势方向用同向/反向。
    :param correlation: 第 3 步算出的最佳滞后统计
    :param label_a: 预测指标中文名（如 咖啡因摄入）
    :param label_b: 结果指标中文名（如 入睡延迟）
    :param synthetic: 是否合成演示数据
    :return: 一行中文解释（不含因果断言词）
    """
    grade = grade_correlation(correlation.pearson_r, correlation.n)
    r = correlation.pearson_r
    n = correlation.n
    lag = correlation.best_lag
    p = correlation.pearson_p

    if synthetic:
        if grade == "不足以判断":
            return f"合成演示数据：样本不足或相关性较弱，不足以判断{label_a}与{label_b}之间的关联。"
        strength = "强" if grade == "强相关" else "弱"
        direction = "正" if r >= 0 else "负"
        p_text = f"{p:.3f}" if not isnan(p) else "—"
        return (
            f"合成演示数据：{label_a}与{label_b}呈{strength}{direction}相关"
            f"（滞后 {lag} 天，N={n}，Pearson r={r:.2f}，p={p_text}）。"
            f"相关不等于因果，仅供参考。"
        )

    if n < MIN_PAIRS or isnan(r):
        return f"你自录的数据天数不足（N={n}），暂不足以观察{label_a}与{label_b}之间的趋势。"
    direction = "同向" if r >= 0 else "反向"
    return (
        f"从你自录的 {n} 天数据看，{label_a}与{label_b}大致呈{direction}变化趋势"
        f"（仅描述趋势，样本量有限，暂不做统计显著性判断）。"
    )
