"""第 10 周 RAG 种子数据生成器：把结构化演示知识展开为 500+ 条原子条目。

条目内容是可追溯来源支持下的演示摘要，不替代指南原文，也不作为诊疗依据。
生成过程不访问数据库，便于用确定性测试锁定数量、唯一键和来源完整性。
"""
from hashlib import sha256
from urllib.parse import quote

from scripts.seed_knowledge import (
    DISEASES,
    DISEASE_SYMPTOMS,
    SYMPTOMS,
)

SOURCE_VERSION = "2026-11-03"
SYMPTOM_SOURCE_NAME = "MedlinePlus Symptom Checker（演示摘要）"
SYMPTOM_SOURCE_URL = "https://medlineplus.gov/symptomchecker.html"
DISEASE_SOURCE_NAME = "MedlinePlus 健康主题检索（演示摘要）"
LAB_SOURCE_NAME = "HealthTrace 检验指标参考范围演示知识"
LAB_SOURCE_URL = "https://github.com/hifishf/healthtrace/blob/main/scripts/seed_report_demo.py"
DRUG_SOURCE_NAME = "HealthTrace 药物相互作用演示规则"
DRUG_SOURCE_URL = "https://github.com/hifishf/healthtrace/blob/main/scripts/seed_report_demo.py"
SAFETY_SOURCE_NAME = "MedlinePlus Health Topics（通用安全边界）"
SAFETY_SOURCE_URL = "https://medlineplus.gov/healthtopics.html"

_URGENCY_LABELS = {
    "observe": "建议观察，若持续或加重再评估",
    "adjust": "建议先调整生活方式并关注变化",
    "online_consult": "建议线上咨询专业人员，确认是否需要面诊",
    "seek_care": "建议尽快线下就医评估",
}


def _hash(content: str) -> str:
    """返回正文的 SHA-256，用于内容变更追踪。"""
    return sha256(content.encode("utf-8")).hexdigest()


def _topic_search_url(name: str) -> str:
    """构造按名称检索的可核验来源地址。"""
    return f"https://medlineplus.gov/search/?query={quote(name)}"


def _entry(
    key: str,
    title: str,
    content: str,
    category: str,
    source_name: str,
    source_url: str,
    metric_codes: list[str] | None = None,
) -> dict:
    """组装一条 RAG 知识条目（embedding 由后续步骤回填）。"""
    return {
        "knowledge_key": key,
        "title": title,
        "content": content,
        "category": category,
        "metric_codes": metric_codes,
        "source_name": source_name,
        "source_url": source_url,
        "source_version": SOURCE_VERSION,
        "content_hash": _hash(content),
    }


def _symptom_entries(
    symptom_name: str,
    synonyms: list[str],
    category: str,
    related_diseases: list[str],
) -> list[dict]:
    """为一个症状生成 6 条原子知识：定义、澄清、危险信号、观察、记录、关联方向。"""
    synonyms_text = "、".join(synonyms)
    diseases_text = "、".join(related_diseases) if related_diseases else "暂无结构化关联"
    return [
        _entry(
            f"symptom:{symptom_name}:profile",
            f"{symptom_name}：症状档案",
            f"{symptom_name}属于“{category}”场景。常见口语表达包括{synonyms_text}。"
            "判断意义时需要结合持续时间、严重程度、诱因和伴随症状，"
            "不能仅凭单一症状归因。",
            "symptom",
            SYMPTOM_SOURCE_NAME,
            SYMPTOM_SOURCE_URL,
        ),
        _entry(
            f"symptom:{symptom_name}:clarify",
            f"{symptom_name}：需要补齐的信息",
            f"描述{symptom_name}时，建议补充开始时间、持续时长、诱发或缓解因素、"
            "严重程度变化及同时出现的其他症状，便于缩小健康管理方向。",
            "symptom",
            SYMPTOM_SOURCE_NAME,
            SYMPTOM_SOURCE_URL,
        ),
        _entry(
            f"symptom:{symptom_name}:urgent-signs",
            f"{symptom_name}：需要及时求助的情形",
            f"出现{symptom_name}并伴随意识改变、晕厥、呼吸困难、持续胸痛、"
            "大量出血或快速加重时，不应只做线上观察，应及时联系急救或线下医疗。",
            "symptom",
            SAFETY_SOURCE_NAME,
            SAFETY_SOURCE_URL,
        ),
        _entry(
            f"symptom:{symptom_name}:observation",
            f"{symptom_name}：日常观察边界",
            f"对轻微、短期的{symptom_name}，可先记录频率、时长和相关生活因素；"
            "若症状持续、反复或影响日常活动，应咨询专业人员。",
            "symptom",
            SYMPTOM_SOURCE_NAME,
            SYMPTOM_SOURCE_URL,
        ),
        _entry(
            f"symptom:{symptom_name}:record",
            f"{symptom_name}：建议记录的内容",
            f"建议记录{symptom_name}的日期、时段、严重程度、诱因、缓解方式和伴随症状。"
            "连续记录有助于观察趋势，但不能替代正式检查与诊断。",
            "symptom",
            SYMPTOM_SOURCE_NAME,
            SYMPTOM_SOURCE_URL,
        ),
        _entry(
            f"symptom:{symptom_name}:related-directions",
            f"{symptom_name}：结构化关联方向",
            f"演示知识库中与{symptom_name}存在关联的方向包括：{diseases_text}。"
            "关联仅用于确定追问方向，不代表这些疾病已经发生。",
            "association",
            SYMPTOM_SOURCE_NAME,
            SYMPTOM_SOURCE_URL,
        ),
    ]


def _disease_entries(
    disease_name: str,
    department: str,
    urgency_hint: str,
    symptom_names: list[str],
) -> list[dict]:
    """为一个疾病生成 4 条原子知识：概览、症状线索、分级边界、沟通准备。"""
    symptoms_text = "、".join(symptom_names)
    urgency_text = _URGENCY_LABELS.get(urgency_hint, "应结合实际情况由专业人员评估")
    source_url = _topic_search_url(disease_name)
    return [
        _entry(
            f"disease:{disease_name}:overview",
            f"{disease_name}：知识概览",
            f"{disease_name}通常由{department}相关专业人员评估。"
            f"演示知识库记录的常见线索包括{symptoms_text}；这些线索不构成诊断。",
            "disease",
            DISEASE_SOURCE_NAME,
            source_url,
        ),
        _entry(
            f"disease:{disease_name}:symptom-pattern",
            f"{disease_name}：症状组合线索",
            f"当{disease_name}相关症状同时出现时，"
            "专业人员会结合病史、体征和检查综合判断。"
            f"演示关联症状为：{symptoms_text}。",
            "disease",
            DISEASE_SOURCE_NAME,
            source_url,
        ),
        _entry(
            f"disease:{disease_name}:care-boundary",
            f"{disease_name}：分级与就医边界",
            f"{disease_name}在当前演示规则中的分级提示为：{urgency_text}。"
            "若出现红旗症状或快速恶化，应优先线下就医。",
            "disease",
            SAFETY_SOURCE_NAME,
            SAFETY_SOURCE_URL,
        ),
        _entry(
            f"disease:{disease_name}:consultation",
            f"{disease_name}：就诊沟通准备",
            f"咨询{disease_name}相关问题时，可准备症状时间线、既往检查、"
            "正在使用的药物和过敏史，帮助专业人员获得完整背景。",
            "disease",
            DISEASE_SOURCE_NAME,
            source_url,
        ),
    ]


def _association_entries() -> list[dict]:
    """把每个疾病-症状关联展开为一条带来源的原子条目。"""
    entries: list[dict] = []
    for disease_name, symptom_names in DISEASE_SYMPTOMS.items():
        for symptom_name in symptom_names:
            entries.append(
                _entry(
                    f"association:{disease_name}:{symptom_name}",
                    f"{disease_name}与{symptom_name}的关联线索",
                    f"演示知识库将{symptom_name}列为{disease_name}的可能关联症状之一。"
                    "该关联用于候选方向排序，不能据此确认疾病。",
                    "association",
                    DISEASE_SOURCE_NAME,
                    _topic_search_url(disease_name),
                )
            )
    return entries


def _lab_entries() -> list[dict]:
    """生成报告解读相关检验指标知识，每个指标三类条目。"""
    metrics = [
        ("ldl_cholesterol", "低密度脂蛋白胆固醇（LDL）", "mmol/L", "上限 3.4"),
        ("total_cholesterol", "总胆固醇", "mmol/L", "上限 5.2"),
        ("hdl_cholesterol", "高密度脂蛋白胆固醇（HDL）", "mmol/L", "下限 1.0"),
        ("triglycerides", "甘油三酯", "mmol/L", "上限 1.7"),
        ("fasting_glucose", "空腹血糖", "mmol/L", "参考区间 3.9 至 6.1"),
        ("hba1c", "糖化血红蛋白", "%", "上限 6.0"),
    ]
    entries: list[dict] = []
    for code, name, unit, range_text in metrics:
        source_url = f"https://medlineplus.gov/search/?query={quote(name)}"
        entries.extend(
            [
                _entry(
                    f"lab:{code}:meaning",
                    f"{name}：指标意义",
                    f"{name}的单位为{unit}，当前演示参考范围是{range_text}。"
                    "单次结果需要结合年龄、病史、采样条件和实验室参考区间解释。",
                    "lab",
                    LAB_SOURCE_NAME,
                    LAB_SOURCE_URL,
                    [code],
                ),
                _entry(
                    f"lab:{code}:trend",
                    f"{name}：纵向趋势解读",
                    f"解读{name}时，应同时观察连续多次记录的方向、幅度和参考范围。"
                    "趋势变化用于提示是否需要进一步咨询，不直接等同于疾病进展。",
                    "lab",
                    LAB_SOURCE_NAME,
                    LAB_SOURCE_URL,
                    [code],
                ),
                _entry(
                    f"lab:{code}:followup",
                    f"{name}：异常或持续变化后的行动",
                    f"若{name}超出演示参考范围或连续上升/下降，"
                    "建议携带完整报告咨询专业人员，"
                    "由其结合症状、用药和其他检查决定复查或进一步评估。",
                    "lab",
                    LAB_SOURCE_NAME,
                    source_url,
                    [code],
                ),
            ]
        )
    return entries


def _drug_entries() -> list[dict]:
    """生成药物交互演示知识；只提示风险边界，不提供处方或剂量建议。"""
    interactions = [
        ("华法林", "阿司匹林", "严重", "可能增加出血风险"),
        ("华法林", "布洛芬", "严重", "可能增加胃肠道出血风险"),
        ("辛伐他汀", "克拉霉素", "禁忌", "可能增加肌病风险"),
        ("二甲双胍", "碘造影剂", "禁忌", "存在乳酸性酸中毒风险提示"),
        ("阿司匹林", "布洛芬", "中等", "可能影响抗血小板作用"),
        ("辛伐他汀", "葡萄柚汁", "中等", "可能增加药物暴露"),
    ]
    entries: list[dict] = []
    for drug_a, drug_b, severity, description in interactions:
        entries.append(
            _entry(
                f"drug:{drug_a}:{drug_b}",
                f"{drug_a}与{drug_b}：{severity}交互提示",
                f"演示规则提示{drug_a}与{drug_b}存在{severity}级交互，{description}。"
                "不要自行停药或调整剂量，应让处方医生或药师核对。",
                "drug",
                DRUG_SOURCE_NAME,
                DRUG_SOURCE_URL,
            )
        )
    return entries


def _safety_entries() -> list[dict]:
    """生成通用报告解读安全边界知识。"""
    topics = [
        (
            "single-value",
            "单项异常不能独立诊断",
            "单项检验异常可能受采样、饮食、药物和实验室差异影响，"
            "应结合复查和临床背景。",
        ),
        (
            "reference-range",
            "参考范围不等于个人目标",
            "参考范围用于一般风险判断，个人目标可能需要根据既往病史由专业人员调整。",
        ),
        (
            "trend",
            "趋势需要结合时间跨度",
            "趋势解释需要记录时间跨度和测量方法；短期波动不能直接解释为疾病进展。",
        ),
        (
            "medication",
            "用药冲突需要专业核对",
            "发现潜在药物交互时，应由处方医生或药师核对，不建议自行停药。",
        ),
        (
            "source",
            "医疗建议需要来源",
            "健康建议应能回溯到知识条目和来源；无法溯源的推断不应作为确定性结论。",
        ),
    ]
    return [
        _entry(
            f"safety:{key}",
            title,
            content,
            "safety",
            SAFETY_SOURCE_NAME,
            SAFETY_SOURCE_URL,
        )
        for key, title, content in topics
    ]


def build_knowledge_entries() -> list[dict]:
    """生成全部 RAG 种子条目，并检查冻结的数量与唯一键契约。"""
    symptom_summary: dict[str, list[str]] = {name: [] for name, _, _ in SYMPTOMS}
    for disease_name, symptom_names in DISEASE_SYMPTOMS.items():
        for symptom_name in symptom_names:
            symptom_summary[symptom_name].append(disease_name)

    symptom_names_by_disease = {
        disease_name: list(symptom_names)
        for disease_name, symptom_names in DISEASE_SYMPTOMS.items()
    }

    entries: list[dict] = []
    for symptom_name, synonyms, category in SYMPTOMS:
        entries.extend(
            _symptom_entries(
                symptom_name,
                synonyms,
                category,
                symptom_summary[symptom_name],
            )
        )
    for disease_name, department, urgency_hint in DISEASES:
        entries.extend(
            _disease_entries(
                disease_name,
                department,
                urgency_hint,
                symptom_names_by_disease[disease_name],
            )
        )
    entries.extend(_association_entries())
    entries.extend(_lab_entries())
    entries.extend(_drug_entries())
    entries.extend(_safety_entries())

    keys = [entry["knowledge_key"] for entry in entries]
    if len(entries) < 500:
        raise ValueError(f"RAG 知识条目不足 500 条：当前 {len(entries)}")
    if len(keys) != len(set(keys)):
        raise ValueError("RAG 知识条目存在重复 knowledge_key")
    return entries
