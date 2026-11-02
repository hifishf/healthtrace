"""药物交互规则引擎：给定用药清单，确定性配对并查交互表，按风险分级返回命中。

第 9 周冻结验收（断言 #9）：华法林 + 阿司匹林 -> 高冲突告警。
匹配与分级全部确定性，不依赖 LLM；LLM 若介入只负责把命中转写成建议话术（第 9 周未做）。
"""
from app import crud

# 风险等级固定顺序（禁忌 > 严重 > 中等 > 轻微）
SEVERITY_ORDER = {"禁忌": 0, "严重": 1, "中等": 2, "轻微": 3}
# 高冲突等级：禁忌 / 严重 视为高冲突，其余为一般
HIGH_RISK_SEVERITIES = frozenset({"禁忌", "严重"})


def is_high_risk(severity: str) -> bool:
    """风险等级是否属于高冲突（禁忌/严重）。"""
    return severity in HIGH_RISK_SEVERITIES


def _dedupe(names: list[str]) -> list[str]:
    """去空、去首尾空白、保序去重（同名药物只保留第一个）。"""
    seen: set[str] = set()
    result: list[str] = []
    for name in names:
        cleaned = str(name or "").strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result


async def check_drug_interactions(medication_names: list[str]) -> dict:
    """检查用药清单两两之间是否存在已知交互。

    命中按风险等级从高到低排序（禁忌 > 严重 > 中等 > 轻微），每条带 high_risk 标记；
    不足两种药物时无命中，返回空 interactions。
    :param medication_names: 待检查的药物名列表（可为空、可含重复/空白，内部去重）
    :return: {"ok": True, "interactions": [{drug_a, drug_b, severity, description, high_risk}]}
    """
    names = _dedupe(medication_names)
    if len(names) < 2:
        return {"ok": True, "interactions": []}

    hits = await crud.find_drug_interactions(names)
    hits.sort(key=lambda hit: SEVERITY_ORDER.get(hit["severity"], 99))
    for hit in hits:
        hit["high_risk"] = is_high_risk(hit["severity"])
    return {"ok": True, "interactions": hits}
