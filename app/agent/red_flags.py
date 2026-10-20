"""红旗规则：确定性关键词/共现规则表 + 主体边界检查。

红旗是硬安全边界：只对用户本人（subject == "self"）触发；描述他人症状
（subject == "other"）不触发，仅返回“建议陪同就医”。规则是纯关键词/共现匹配，
不依赖 LLM（第 8 周任何降级路径下也要保证执行）。
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class RedFlagResult:
    """红旗检查结果（确定性）。"""

    triggered: bool
    level: str | None = None
    matched: tuple[str, ...] = ()
    subject: str = "self"
    suggestion: str | None = None


# 关键词 -> 口语化 / 同义词 / 错别字表面形式（用于确定性匹配，不依赖 LLM）
_KEYWORD_SURFACE: dict[str, tuple[str, ...]] = {
    "呕血": ("呕血", "吐血", "呕吐带血"),
    "咯血": ("咯血", "咳血", "咳嗽带血", "痰中带血"),
    "意识改变": ("意识改变", "意识模糊", "昏迷", "晕迷", "不省人事", "叫不醒", "神志不清"),
    "抽搐": ("抽搐", "惊厥", "全身抽搐"),
    "胸痛": ("胸痛", "胸口疼", "胸口痛", "胸疼", "心口痛"),
    "呼吸困难": ("呼吸困难", "喘不上气", "喘不过气", "上不来气", "憋气", "气短", "呼吸费力"),
    "出汗": ("出汗", "冷汗", "冒冷汗", "大汗淋漓", "大汗", "出虚汗"),
    "晕厥": ("晕厥", "晕倒", "昏过去", "突然失去意识"),
    "胸闷": ("胸闷", "胸口发闷", "胸口堵", "憋闷", "胸口闷"),
    "突发": ("突发", "突然"),
    "一侧肢体无力": ("一侧肢体无力", "半边身子没劲", "一侧手脚麻木", "手抬不起来", "肢体无力"),
    "口齿不清": ("口齿不清", "说话含糊", "大舌头", "言语不清", "说话不利索"),
    "口角歪斜": ("口角歪斜", "嘴歪", "口眼歪斜", "嘴角歪"),
    "视物模糊": ("视物模糊", "看东西模糊", "眼花", "复视", "视力突然下降"),
    "剧烈头痛": ("剧烈头痛", "剧烈头疼", "炸裂样头痛", "突然剧烈头痛", "要命的头痛"),
    "呕吐": ("呕吐", "吐了", "作呕", "喷射性呕吐"),
    "口唇发紫": ("口唇发紫", "嘴唇发紫", "嘴唇发青", "发绀", "口唇青紫"),
    "发热": ("发热", "发烧", "高烧", "体温高"),
    "放射痛": ("放射痛", "左肩痛", "后背痛", "左臂痛", "左胳膊痛", "左肩背痛"),
    "心悸": ("心悸", "心慌", "心跳快", "心跳乱", "心里突突"),
    "面部麻木": ("面部麻木", "脸麻", "嘴麻", "半边脸麻"),
}


# 约 20 条确定性红旗规则，版本化在 repo 里。
# 单关键词规则：命中任一关键词即高危；共现规则：关键词必须同时出现才高危。
RED_FLAG_RULES: list[dict] = [
    # 单关键词（出现即高危）
    {"keywords": ["呕血"], "cooccur": False, "level": "emergency"},
    {"keywords": ["咯血"], "cooccur": False, "level": "emergency"},
    {"keywords": ["意识改变"], "cooccur": False, "level": "emergency"},
    {"keywords": ["抽搐"], "cooccur": False, "level": "emergency"},
    # 共现（同时出现才高危，单个出现不触发）
    {"keywords": ["胸痛", "呼吸困难"], "cooccur": True, "level": "emergency"},
    {"keywords": ["胸痛", "出汗"], "cooccur": True, "level": "emergency"},
    {"keywords": ["胸痛", "晕厥"], "cooccur": True, "level": "emergency"},
    {"keywords": ["胸痛", "放射痛"], "cooccur": True, "level": "emergency"},
    {"keywords": ["胸闷", "出汗"], "cooccur": True, "level": "emergency"},
    {"keywords": ["突发", "一侧肢体无力"], "cooccur": True, "level": "emergency"},
    {"keywords": ["突发", "口齿不清"], "cooccur": True, "level": "emergency"},
    {"keywords": ["突发", "口角歪斜"], "cooccur": True, "level": "emergency"},
    {"keywords": ["一侧肢体无力", "口齿不清"], "cooccur": True, "level": "emergency"},
    {"keywords": ["剧烈头痛", "呕吐"], "cooccur": True, "level": "emergency"},
    {"keywords": ["突发", "剧烈头痛"], "cooccur": True, "level": "emergency"},
    {"keywords": ["呼吸困难", "口唇发紫"], "cooccur": True, "level": "emergency"},
    {"keywords": ["心悸", "晕厥"], "cooccur": True, "level": "emergency"},
    {"keywords": ["突发", "面部麻木"], "cooccur": True, "level": "emergency"},
    {"keywords": ["突发", "视物模糊"], "cooccur": True, "level": "emergency"},
    {"keywords": ["发热", "呼吸困难"], "cooccur": True, "level": "emergency"},
]


def _rule_label(rule: dict) -> str:
    """规则的展示名（关键词用 + 连接），用于结果回显与测试断言。"""
    return "+".join(rule["keywords"])


def _matches(text: str, keyword: str) -> bool:
    """判断 text 是否命中某关键词的任一表面形式（确定性子串匹配）。"""
    return any(surface in text for surface in _KEYWORD_SURFACE[keyword])


def check_red_flag(text: str, subject: str = "self") -> RedFlagResult:
    """对用户一句话做红旗检查（确定性，不依赖 LLM）。

    主体边界：subject == "other" 时描述的是他人症状，不触发红旗，仅提示陪同就医。
    共现规则要求规则内所有关键词同时命中；单关键词规则命中任一关键词即触发。
    """
    # 主体边界：他人症状不触发红旗
    if subject == "other":
        return RedFlagResult(
            triggered=False,
            subject="other",
            suggestion="建议陪同就医",
        )

    matched: list[str] = []
    for rule in RED_FLAG_RULES:
        hits = [keyword for keyword in rule["keywords"] if _matches(text, keyword)]
        is_cooccur_hit = rule["cooccur"] and len(hits) == len(rule["keywords"])
        is_single_hit = not rule["cooccur"] and bool(hits)
        if is_cooccur_hit or is_single_hit:
            matched.append(_rule_label(rule))

    if matched:
        return RedFlagResult(
            triggered=True,
            level="emergency",
            matched=tuple(matched),
            subject="self",
        )
    return RedFlagResult(triggered=False, subject="self")
