"""红旗规则的确定性断言：20 正例 + 20 负例 + 主体边界 + 关键词覆盖。"""
import pytest

from app.agent.red_flags import RED_FLAG_RULES, _KEYWORD_SURFACE, check_red_flag


# 20 例正例：口语化、同义词、错别字表达都应触发红旗（subject=self）
POSITIVE: list[str] = [
    "我吐血了",  # 呕血
    "咳嗽痰中带血",  # 咯血
    "我晕迷了",  # 意识改变（错别字：昏迷 -> 晕迷）
    "突然全身抽搐",  # 抽搐
    "胸口疼得喘不上气",  # 胸痛 + 呼吸困难
    "胸痛还冒冷汗",  # 胸痛 + 出汗
    "胸痛之后晕倒了",  # 胸痛 + 晕厥
    "胸痛还左肩背痛",  # 胸痛 + 放射痛
    "胸闷还冒冷汗",  # 胸闷 + 出汗
    "突然一侧手脚麻木",  # 突发 + 一侧肢体无力
    "突然说话含糊",  # 突发 + 口齿不清
    "突然嘴歪了",  # 突发 + 口角歪斜
    "半边身子没劲还说话含糊",  # 一侧肢体无力 + 口齿不清
    "剧烈头痛还呕吐",  # 剧烈头痛 + 呕吐
    "突然炸裂样头痛",  # 突发 + 剧烈头痛
    "喘不上气嘴唇发紫",  # 呼吸困难 + 口唇发紫
    "心慌还晕倒了",  # 心悸 + 晕厥
    "突然半边脸麻",  # 突发 + 面部麻木
    "突然看东西模糊",  # 突发 + 视物模糊
    "发高烧还喘不上气",  # 发热 + 呼吸困难
]


# 20 例负例：相似但非红旗（subject=self，不应触发）
NEGATIVE: list[str] = [
    "运动后胸痛",  # 胸痛单独出现，无共现
    "最近总是累",  # 无红旗关键词
    "跑步后喘不上气",  # 呼吸困难单独出现，无共现
    "有点胸闷",  # 胸闷单独出现，无共现
    "昨天吐了一次",  # 呕吐单独出现，无剧烈头痛
    "有点心慌",  # 心悸单独出现，无晕厥
    "今天有点发烧",  # 发热单独出现，无呼吸困难
    "站久了头晕",  # 头晕非红旗
    "睡觉流口水",  # 非红旗
    "坐久了腿麻",  # 非红旗
    "喉咙痛还咳嗽",  # 咽痛咳嗽非红旗
    "吃坏肚子拉稀",  # 腹泻非红旗
    "昨晚失眠",  # 失眠非红旗
    "头痛了一天",  # 头痛非剧烈头痛
    "最近嘴角起泡",  # 非口角歪斜
    "偶尔胸口刺痛一秒",  # 非胸痛关键词
    "吃药后胃里反酸",  # 反酸非红旗
    "走几步就喘",  # 非呼吸困难关键词
    "最近压力大睡不着",  # 非红旗
    "今天头有点晕",  # 头晕非红旗
]


@pytest.mark.parametrize("text", POSITIVE)
def test_red_flag_positive(text: str) -> None:
    """正例：口语化/同义词/错别字表达都应触发红旗。"""
    result = check_red_flag(text, subject="self")
    assert result.triggered is True
    assert result.level == "emergency"


@pytest.mark.parametrize("text", NEGATIVE)
def test_red_flag_negative(text: str) -> None:
    """负例：相似但非红旗的表达不应触发。"""
    result = check_red_flag(text, subject="self")
    assert result.triggered is False


def test_red_flag_reports_matched_rule() -> None:
    """触发时回显命中的规则，便于排查与审计。"""
    result = check_red_flag("胸痛还冒冷汗", subject="self")
    assert result.triggered is True
    assert "胸痛+出汗" in result.matched


def test_single_keyword_without_cooccur_not_triggered() -> None:
    """共现规则：只命中其中一个关键词不触发（胸痛单独出现）。"""
    result = check_red_flag("运动后胸痛", subject="self")
    assert result.triggered is False


def test_other_subject_not_triggered_and_suggests_escort() -> None:
    """主体边界：他人症状不触发红旗，返回陪同就医建议。"""
    result = check_red_flag("我爸爸胸痛", subject="other")
    assert result.triggered is False
    assert result.suggestion == "建议陪同就医"


def test_every_rule_keyword_has_surface() -> None:
    """规则里每个关键词都有表面形式，避免 KeyError。"""
    for rule in RED_FLAG_RULES:
        for keyword in rule["keywords"]:
            assert keyword in _KEYWORD_SURFACE
