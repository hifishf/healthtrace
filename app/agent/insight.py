"""因果分析（insight）子图：时间对齐 -> 滞后相关 -> LLM 解释 -> 措辞过滤。"""
from dataclasses import dataclass


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
