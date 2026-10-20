"""知识库种子脚本：4 场景 200-300 条幂等写入 symptoms / diseases / disease_symptoms。

先决条件：`schema/init.sql` 已执行（三张知识库表已建）。
运行：.venv/bin/python scripts/seed_knowledge.py

数据性质：演示用近似医学知识，非诊疗依据。`prevalence` 为演示近似值，
`discrimination_weight` 按「症状在多少疾病中出现」的逆文档频率近似（越特异权重越高），
第 4 周会再做人工校准。
"""
import asyncio
import sys
from pathlib import Path

import asyncpg

# 允许脚本以任意工作目录运行：把仓库根加入 sys.path，才能 import app 包
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402

# 4 场景：疲劳乏力 / 胸痛心慌 / 头晕头痛 / 胃肠消化（另含少量全身症状做候选集交叉）
SYMPTOMS: list[tuple[str, list[str], str]] = [
    ("乏力", ["疲劳", "累", "没力气", "浑身无力", "没劲", "疲软", "没精神"], "疲劳乏力"),
    ("嗜睡", ["犯困", "想睡觉", "睡不醒", "老是困", "打瞌睡"], "疲劳乏力"),
    ("活动后气短", ["走两步就喘", "稍微活动就累", "劳力性气促", "活动后气喘"], "疲劳乏力"),
    ("面色苍白", ["脸色发白", "脸色差", "口唇发白", "面无血色"], "疲劳乏力"),
    ("体重下降", ["瘦了", "体重减轻", "掉秤", "消瘦"], "疲劳乏力"),
    ("情绪低落", ["心情差", "高兴不起来", "没兴趣", "抑郁情绪", "情绪不好"], "疲劳乏力"),
    ("失眠", ["睡不着", "入睡困难", "早醒", "睡眠差"], "疲劳乏力"),
    ("注意力不集中", ["走神", "记性差", "健忘", "难以集中"], "疲劳乏力"),
    ("肌肉酸痛", ["浑身酸", "肌肉疼", "身体酸痛"], "疲劳乏力"),
    ("怕冷", ["畏寒", "手脚冰凉"], "疲劳乏力"),
    ("胸痛", ["胸口疼", "心口痛", "胸疼", "胸口痛"], "胸痛心慌"),
    ("胸闷", ["胸口发闷", "憋闷", "压榨感", "胸口堵"], "胸痛心慌"),
    ("呼吸困难", ["喘不上气", "气短", "憋气", "呼吸费力", "上不来气"], "胸痛心慌"),
    ("心悸", ["心慌", "心跳快", "心跳乱", "心里突突"], "胸痛心慌"),
    ("出汗", ["冒冷汗", "大汗淋漓", "出虚汗"], "胸痛心慌"),
    ("放射痛", ["左肩痛", "后背痛", "左臂痛", "左肩背痛"], "胸痛心慌"),
    ("晕厥", ["晕倒", "昏过去", "眼前发黑"], "胸痛心慌"),
    ("头晕", ["头昏", "眩晕", "天旋地转", "站不稳"], "头晕头痛"),
    ("头痛", ["头疼", "太阳穴疼", "后脑勺疼"], "头晕头痛"),
    ("恶心", ["想吐", "反胃", "干呕"], "头晕头痛"),
    ("呕吐", ["吐了", "作呕"], "头晕头痛"),
    ("视物模糊", ["看东西模糊", "眼花", "复视", "视物不清"], "头晕头痛"),
    ("耳鸣", ["耳朵嗡嗡响", "脑鸣"], "头晕头痛"),
    ("颈部僵硬", ["脖子硬", "落枕", "颈部酸痛", "脖子僵硬"], "头晕头痛"),
    ("口齿不清", ["说话含糊", "大舌头", "言语不清", "说话不利索"], "头晕头痛"),
    ("一侧肢体无力", ["半边身子没劲", "一侧手脚麻木", "手抬不起来", "肢体无力"], "头晕头痛"),
    ("腹痛", ["肚子疼", "胃痛", "腹部不适", "绞痛"], "胃肠消化"),
    ("腹胀", ["肚子胀", "胀气", "嗳气"], "胃肠消化"),
    ("腹泻", ["拉肚子", "大便稀", "水样便"], "胃肠消化"),
    ("便秘", ["大便干结", "排便困难", "大便次数少"], "胃肠消化"),
    ("反酸", ["烧心", "反酸水", "胃酸上涌"], "胃肠消化"),
    ("食欲不振", ["没胃口", "吃不下", "食欲差", "不想吃饭"], "胃肠消化"),
    ("黑便", ["柏油样便", "大便发黑"], "胃肠消化"),
    ("呕血", ["吐血", "呕吐带血"], "胃肠消化"),
    ("便血", ["大便带血", "鲜血便", "便中带血"], "胃肠消化"),
    ("发热", ["发烧", "体温高", "低烧"], "全身症状"),
    ("咳嗽", ["干咳", "咳嗽有痰", "久咳"], "全身症状"),
    ("咽痛", ["嗓子疼", "喉咙痛"], "全身症状"),
    ("水肿", ["浮肿", "脚肿", "眼睑肿"], "全身症状"),
    ("口渴", ["口干", "喝水多", "老渴"], "全身症状"),
    ("尿频", ["频繁上厕所", "夜尿多", "小便次数多"], "全身症状"),
]

DISEASES: list[tuple[str, str, str]] = [
    ("慢性疲劳综合征", "神经内科", "observe"),
    ("缺铁性贫血", "血液科", "adjust"),
    ("甲状腺功能减退", "内分泌科", "online_consult"),
    ("阻塞性睡眠呼吸暂停", "呼吸科", "online_consult"),
    ("抑郁症", "精神科", "seek_care"),
    ("2型糖尿病", "内分泌科", "online_consult"),
    ("稳定型心绞痛", "心内科", "seek_care"),
    ("急性心肌梗死", "心内科", "seek_care"),
    ("胃食管反流病", "消化内科", "observe"),
    ("肋间神经痛", "疼痛科", "observe"),
    ("焦虑症", "精神科", "online_consult"),
    ("肺栓塞", "呼吸科", "seek_care"),
    ("胸膜炎", "呼吸科", "online_consult"),
    ("心房颤动", "心内科", "seek_care"),
    ("偏头痛", "神经内科", "observe"),
    ("紧张性头痛", "神经内科", "observe"),
    ("良性阵发性位置性眩晕", "耳鼻喉科", "online_consult"),
    ("梅尼埃病", "耳鼻喉科", "online_consult"),
    ("高血压病", "心内科", "online_consult"),
    ("颈椎病", "骨科", "adjust"),
    ("短暂性脑缺血发作", "神经内科", "seek_care"),
    ("脑卒中", "神经内科", "seek_care"),
    ("功能性消化不良", "消化内科", "adjust"),
    ("慢性胃炎", "消化内科", "adjust"),
    ("肠易激综合征", "消化内科", "observe"),
    ("消化性溃疡", "消化内科", "online_consult"),
    ("急性胃肠炎", "消化内科", "observe"),
    ("炎症性肠病", "消化内科", "seek_care"),
    ("结直肠癌", "肿瘤科", "seek_care"),
    ("上呼吸道感染", "呼吸科", "observe"),
    ("社区获得性肺炎", "呼吸科", "seek_care"),
]

DISEASE_SYMPTOMS: dict[str, list[str]] = {
    "慢性疲劳综合征": ["乏力", "嗜睡", "失眠", "肌肉酸痛", "注意力不集中", "情绪低落"],
    "缺铁性贫血": ["乏力", "面色苍白", "头晕", "心悸", "活动后气短", "注意力不集中"],
    "甲状腺功能减退": ["乏力", "怕冷", "嗜睡", "便秘", "情绪低落", "水肿"],
    "阻塞性睡眠呼吸暂停": ["嗜睡", "失眠", "乏力", "注意力不集中", "头痛"],
    "抑郁症": ["情绪低落", "失眠", "食欲不振", "乏力", "注意力不集中", "体重下降"],
    "2型糖尿病": ["口渴", "尿频", "体重下降", "乏力", "视物模糊", "嗜睡"],
    "稳定型心绞痛": ["胸痛", "胸闷", "放射痛", "出汗", "心悸", "活动后气短"],
    "急性心肌梗死": ["胸痛", "胸闷", "出汗", "放射痛", "呼吸困难", "晕厥"],
    "胃食管反流病": ["反酸", "胸痛", "胸闷", "咽痛", "咳嗽", "腹胀"],
    "肋间神经痛": ["胸痛", "胸闷", "肌肉酸痛"],
    "焦虑症": ["心悸", "胸闷", "出汗", "呼吸困难", "失眠", "头晕", "情绪低落"],
    "肺栓塞": ["呼吸困难", "胸痛", "晕厥", "心悸", "出汗"],
    "胸膜炎": ["胸痛", "呼吸困难", "咳嗽", "发热"],
    "心房颤动": ["心悸", "晕厥", "头晕", "胸闷", "出汗"],
    "偏头痛": ["头痛", "恶心", "视物模糊", "头晕"],
    "紧张性头痛": ["头痛", "颈部僵硬", "肌肉酸痛"],
    "良性阵发性位置性眩晕": ["头晕", "恶心", "呕吐"],
    "梅尼埃病": ["头晕", "耳鸣", "恶心", "呕吐", "视物模糊"],
    "高血压病": ["头晕", "头痛", "耳鸣", "视物模糊"],
    "颈椎病": ["颈部僵硬", "头痛", "头晕", "一侧肢体无力"],
    "短暂性脑缺血发作": ["一侧肢体无力", "口齿不清", "头晕", "视物模糊", "晕厥"],
    "脑卒中": ["一侧肢体无力", "口齿不清", "头痛", "头晕", "视物模糊", "晕厥"],
    "功能性消化不良": ["腹胀", "腹痛", "反酸", "食欲不振", "恶心"],
    "慢性胃炎": ["腹痛", "腹胀", "反酸", "食欲不振", "恶心"],
    "肠易激综合征": ["腹痛", "腹泻", "便秘", "腹胀"],
    "消化性溃疡": ["腹痛", "反酸", "黑便", "呕血", "食欲不振", "腹胀"],
    "急性胃肠炎": ["腹泻", "腹痛", "恶心", "呕吐", "发热"],
    "炎症性肠病": ["腹泻", "腹痛", "便血", "体重下降", "发热"],
    "结直肠癌": ["便血", "黑便", "便秘", "腹泻", "体重下降", "腹痛"],
    "上呼吸道感染": ["咽痛", "咳嗽", "发热", "头痛", "乏力"],
    "社区获得性肺炎": ["发热", "咳嗽", "呼吸困难", "胸痛", "乏力"],
}

# 常见核心症状给更高的演示出现率，其余默认 0.5（演示近似值）
PREVALENCE_OVERRIDES: dict[str, float] = {
    "乏力": 0.85,
    "嗜睡": 0.70,
    "胸痛": 0.70,
    "头晕": 0.70,
    "头痛": 0.70,
    "腹痛": 0.70,
    "咳嗽": 0.65,
    "发热": 0.65,
    "心悸": 0.60,
    "呼吸困难": 0.60,
    "失眠": 0.60,
    "恶心": 0.55,
    "食欲不振": 0.55,
    "胸闷": 0.55,
    "腹胀": 0.55,
}
DEFAULT_PREVALENCE = 0.5


def _discrimination_weight(disease_count: int) -> int:
    """按症状跨疾病出现次数的逆频率近似鉴别价值（越特异权重越高，1-5）。"""
    if disease_count <= 2:
        return 5
    if disease_count <= 4:
        return 4
    if disease_count <= 6:
        return 3
    if disease_count <= 9:
        return 2
    return 1


def _prevalence(symptom_name: str) -> float:
    """返回症状的演示出现率（默认 0.5）。"""
    return PREVALENCE_OVERRIDES.get(symptom_name, DEFAULT_PREVALENCE)


async def main() -> None:
    """幂等写入 symptoms / diseases / disease_symptoms，打印条数统计。"""
    dsn = settings.database_url.replace("postgresql+asyncpg://", "postgresql://")
    conn = await asyncpg.connect(dsn)
    try:
        # 症状词典：按 name upsert，保持同义词/分类最新
        symptom_ids: dict[str, int] = {}
        for name, synonyms, category in SYMPTOMS:
            row = await conn.fetchrow(
                "INSERT INTO symptoms(name, synonyms, category) VALUES($1, $2, $3) "
                "ON CONFLICT (name) DO UPDATE SET synonyms=EXCLUDED.synonyms, "
                "category=EXCLUDED.category RETURNING id",
                name,
                synonyms,
                category,
            )
            symptom_ids[name] = row["id"]

        # 疾病表：按 name upsert，保持科室/分级最新
        disease_ids: dict[str, int] = {}
        for name, department, urgency_hint in DISEASES:
            row = await conn.fetchrow(
                "INSERT INTO diseases(name, department, urgency_hint) VALUES($1, $2, $3) "
                "ON CONFLICT (name) DO UPDATE SET department=EXCLUDED.department, "
                "urgency_hint=EXCLUDED.urgency_hint RETURNING id",
                name,
                department,
                urgency_hint,
            )
            disease_ids[name] = row["id"]

        # 症状->疾病出现次数：用于逆频率权重
        symptom_disease_count: dict[str, int] = {}
        for disease, symptoms in DISEASE_SYMPTOMS.items():
            for symptom in symptoms:
                symptom_disease_count[symptom] = symptom_disease_count.get(symptom, 0) + 1

        # 关联表：按 (disease_id, symptom_id) upsert
        link_count = 0
        for disease, symptoms in DISEASE_SYMPTOMS.items():
            for symptom in symptoms:
                await conn.execute(
                    "INSERT INTO disease_symptoms(disease_id, symptom_id, prevalence, "
                    "discrimination_weight) VALUES($1, $2, $3, $4) "
                    "ON CONFLICT (disease_id, symptom_id) DO UPDATE SET "
                    "prevalence=EXCLUDED.prevalence, "
                    "discrimination_weight=EXCLUDED.discrimination_weight",
                    disease_ids[disease],
                    symptom_ids[symptom],
                    _prevalence(symptom),
                    _discrimination_weight(symptom_disease_count[symptom]),
                )
                link_count += 1

        total = len(SYMPTOMS) + len(DISEASES) + link_count
        print(f"symptoms={len(SYMPTOMS)} diseases={len(DISEASES)} links={link_count} total={total}")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
