"""因果分析（insight）的确定性断言：时间对齐 / 滞后相关 / 措辞过滤 / 来源口径。"""
import math
import random

from langchain_core.messages import AIMessage

from app.agent.budget import TokenBudget
from app.agent.llm import LLMUnavailable
from app.agent.insight import (
    FORBIDDEN_CAUSAL_WORDS,
    LagCorrelation,
    MIN_PAIRS,
    align_daily_series,
    build_template_explanation,
    explain_correlation,
    filter_causal_wording,
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


def test_filter_causal_wording_blocks_causal_words() -> None:
    """断言 #14：含 导致/因为/引起/确诊/诊断 的句子过滤后不再包含任何禁用词。"""
    text = "咖啡因导致入睡延迟，因为咖啡因引起神经兴奋，可能被诊断为失眠。"
    filtered = filter_causal_wording(text)

    assert "相关入睡延迟" in filtered
    assert not any(word in filtered for word in FORBIDDEN_CAUSAL_WORDS)


def test_template_synthetic_reports_significance() -> None:
    """合成数据：给出强度 + N + p，且不含因果断言词。"""
    correlation = LagCorrelation(
        best_lag=0,
        n=120,
        pearson_r=0.82,
        pearson_p=0.001,
        spearman_rho=0.80,
        spearman_p=0.002,
    )
    text = build_template_explanation(correlation, "咖啡因摄入", "入睡延迟", synthetic=True)

    assert "合成演示数据" in text
    assert "强正相关" in text
    assert "N=120" in text
    assert "p=0.001" in text
    assert "相关不等于因果" in text
    assert not any(word in text for word in FORBIDDEN_CAUSAL_WORDS)


def test_template_real_data_only_trend_no_significance() -> None:
    """自录数据：只描述趋势，不写强/弱相关分级、不附 p 值。"""
    correlation = LagCorrelation(
        best_lag=1,
        n=30,
        pearson_r=0.60,
        pearson_p=0.02,
        spearman_rho=0.55,
        spearman_p=0.03,
    )
    text = build_template_explanation(correlation, "咖啡因摄入", "入睡延迟", synthetic=False)

    assert "趋势" in text
    assert "强相关" not in text
    assert "弱相关" not in text
    assert "p=" not in text
    assert not any(word in text for word in FORBIDDEN_CAUSAL_WORDS)


def test_template_insufficient_data_declines_judgement() -> None:
    """样本不足或 r 为 nan -> 两种来源都明确「不足以判断」，不给出相关性结论。"""
    insufficient = LagCorrelation(
        best_lag=0,
        n=5,
        pearson_r=0.90,
        pearson_p=0.10,
        spearman_rho=0.90,
        spearman_p=0.10,
    )
    assert "不足以判断" in build_template_explanation(
        insufficient, "咖啡因摄入", "入睡延迟", synthetic=True
    )
    assert "天数不足" in build_template_explanation(
        insufficient, "咖啡因摄入", "入睡延迟", synthetic=False
    )


class _ExplainLLM:
    """记录收到的消息、按预设文本回复的假模型。"""

    def __init__(self, text: str) -> None:
        self.text = text
        self.last_messages: list | None = None

    async def ainvoke(self, messages):
        self.last_messages = messages
        return AIMessage(content=self.text)


class _EmptyExplainLLM:
    """回复空文本的假模型，用于触发空响应兜底。"""

    async def ainvoke(self, messages):
        return AIMessage(content="")


class _FailingExplainLLM:
    """直接抛 LLMUnavailable 的假模型，用于触发失败兜底。"""

    async def ainvoke(self, messages):
        raise LLMUnavailable("llm_unavailable")


async def test_explain_correlation_translates_and_filters() -> None:
    """合成数据走 LLM：拿到 r/N 等统计事实，输出经措辞过滤、不含因果词。"""
    correlation = LagCorrelation(
        best_lag=0,
        n=120,
        pearson_r=0.82,
        pearson_p=0.001,
        spearman_rho=0.80,
        spearman_p=0.002,
    )
    llm = _ExplainLLM("咖啡因摄入导致入睡延迟，二者呈强正相关。")
    budget = TokenBudget(limit=10000)
    text = await explain_correlation(
        correlation, "咖啡因摄入", "入睡延迟", synthetic=True, budget=budget, llm=llm
    )

    assert llm.last_messages is not None
    assert "Pearson 相关系数 r" in llm.last_messages[-1].content
    assert "相关入睡延迟" in text
    assert "导致" not in text


async def test_explain_correlation_empty_falls_back() -> None:
    """LLM 返回空文本 -> 回退确定性模板。"""
    correlation = LagCorrelation(
        best_lag=0,
        n=120,
        pearson_r=0.82,
        pearson_p=0.001,
        spearman_rho=0.80,
        spearman_p=0.002,
    )
    budget = TokenBudget(limit=10000)
    text = await explain_correlation(
        correlation, "咖啡因摄入", "入睡延迟", synthetic=True, budget=budget, llm=_EmptyExplainLLM()
    )

    assert "合成演示数据" in text
    assert not any(word in text for word in FORBIDDEN_CAUSAL_WORDS)


async def test_explain_correlation_llm_failure_falls_back() -> None:
    """LLM 调用失败 -> 回退确定性模板。"""
    correlation = LagCorrelation(
        best_lag=0,
        n=120,
        pearson_r=0.82,
        pearson_p=0.001,
        spearman_rho=0.80,
        spearman_p=0.002,
    )
    budget = TokenBudget(limit=10000)
    text = await explain_correlation(
        correlation, "咖啡因摄入", "入睡延迟", synthetic=True, budget=budget, llm=_FailingExplainLLM()
    )

    assert "合成演示数据" in text


async def test_explain_correlation_real_data_skips_llm() -> None:
    """自录数据不调 LLM，直接返回趋势模板，不做显著性断言。"""
    correlation = LagCorrelation(
        best_lag=1,
        n=30,
        pearson_r=0.60,
        pearson_p=0.02,
        spearman_rho=0.55,
        spearman_p=0.03,
    )
    llm = _ExplainLLM("不应被调用")
    budget = TokenBudget(limit=10000)
    text = await explain_correlation(
        correlation, "咖啡因摄入", "入睡延迟", synthetic=False, budget=budget, llm=llm
    )

    assert llm.last_messages is None
    assert "趋势" in text
    assert "强相关" not in text
    assert "弱相关" not in text
