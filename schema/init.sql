-- HealthTrace 数据库初始化脚本
-- 该文件在 db 容器首次初始化时执行（仅在数据卷为空时生效）。
-- 已存在的库可手动幂等重放：
--   docker compose exec db psql -U healthtrace -d healthtrace -f /docker-entrypoint-initdb.d/init.sql

-- 1) 启用 pgvector（第 9 周向量检索用，提前就位）
CREATE EXTENSION IF NOT EXISTS vector;

-- 2) 健康画像主表（单用户场景：种子一条默认用户）
CREATE TABLE IF NOT EXISTS health_profiles (
    user_id UUID PRIMARY KEY,
    birth_year INT,
    sex VARCHAR(10),
    height_cm DECIMAL,
    weight_kg DECIMAL,
    bmi DECIMAL GENERATED ALWAYS AS (weight_kg / ((height_cm / 100) ^ 2)) STORED,
    updated_at TIMESTAMPTZ
);

COMMENT ON TABLE health_profiles IS '健康画像主表（单用户场景，默认种子一条用户）';
COMMENT ON COLUMN health_profiles.user_id IS '用户唯一标识（单用户，固定 UUID）';
COMMENT ON COLUMN health_profiles.birth_year IS '出生年份';
COMMENT ON COLUMN health_profiles.sex IS '性别';
COMMENT ON COLUMN health_profiles.height_cm IS '身高（厘米）';
COMMENT ON COLUMN health_profiles.weight_kg IS '体重（公斤）';
COMMENT ON COLUMN health_profiles.bmi IS 'BMI，生成列，= 体重 / (身高/100)^2，随身高体重自动计算';
COMMENT ON COLUMN health_profiles.updated_at IS '最后更新时间';

-- 3) 指标记录表（睡眠/运动/饮食/检验）
--    幂等约束：同一天同一用户同一指标只有一行（覆盖语义，靠 UNIQUE 兜底）
CREATE TABLE IF NOT EXISTS metric_records (
    id SERIAL PRIMARY KEY,
    user_id UUID,
    metric_code VARCHAR(50),
    value DECIMAL,
    unit VARCHAR(20),
    recorded_date DATE,
    source VARCHAR(30),
    UNIQUE (user_id, metric_code, recorded_date)
);

COMMENT ON TABLE metric_records IS '指标记录表（睡眠/运动/饮食/检验，同日期同指标覆盖语义）';
COMMENT ON COLUMN metric_records.id IS '自增主键';
COMMENT ON COLUMN metric_records.user_id IS '用户唯一标识';
COMMENT ON COLUMN metric_records.metric_code IS '指标代码（如 sleep_duration）';
COMMENT ON COLUMN metric_records.value IS '指标数值';
COMMENT ON COLUMN metric_records.unit IS '单位（如 h/kg）';
COMMENT ON COLUMN metric_records.recorded_date IS '记录日期';
COMMENT ON COLUMN metric_records.source IS '数据来源（如 manual）';

-- 4) 种子：默认单用户（无登录/多用户，固定 UUID，代码侧使用同一常量）
INSERT INTO health_profiles (user_id, birth_year, sex, height_cm, weight_kg, updated_at)
VALUES ('00000000-0000-0000-0000-000000000001', NULL, NULL, NULL, NULL, now())
ON CONFLICT (user_id) DO NOTHING;

-- 5) 知识库：症状词典（追问分诊的症状归一化用）
--    name 唯一，作为 seed 脚本 upsert 的自然键
CREATE TABLE IF NOT EXISTS symptoms (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    synonyms TEXT[],
    category VARCHAR(50)
);

COMMENT ON TABLE symptoms IS '知识库症状词典（追问分诊用，name 唯一，synonyms 支持口语化归一化）';
COMMENT ON COLUMN symptoms.id IS '自增主键';
COMMENT ON COLUMN symptoms.name IS '症状标准名（归一化后的唯一名）';
COMMENT ON COLUMN symptoms.synonyms IS '同义词数组（口语化/错别字归一化匹配用）';
COMMENT ON COLUMN symptoms.category IS '症状分类（疲劳乏力/胸痛心慌/头晕头痛/胃肠消化/全身症状）';

-- 6) 知识库：疾病/病种（候选疾病集）
--    原文档第 9 节叫 conditions，为与第 8 节「病情」表 conditions 消歧义，
--    改名 diseases；name 唯一作为 seed 脚本 upsert 的自然键
CREATE TABLE IF NOT EXISTS diseases (
    id SERIAL PRIMARY KEY,
    name VARCHAR(200) NOT NULL UNIQUE,
    department VARCHAR(100),
    urgency_hint VARCHAR(30)
);

COMMENT ON TABLE diseases IS '知识库疾病表（候选疾病集，原文档 conditions，改名 diseases 避免与病情表冲突）';
COMMENT ON COLUMN diseases.id IS '自增主键';
COMMENT ON COLUMN diseases.name IS '疾病名（唯一）';
COMMENT ON COLUMN diseases.department IS '所属科室';
COMMENT ON COLUMN diseases.urgency_hint IS '分级建议等级：observe观察/adjust调整/online_consult线上问诊/seek_care建议就医';

-- 7) 知识库：疾病-症状关联（候选集反查 + 鉴别价值追问）
--    原文档叫 condition_symptoms，随疾病表改名 disease_symptoms
CREATE TABLE IF NOT EXISTS disease_symptoms (
    id SERIAL PRIMARY KEY,
    disease_id INT NOT NULL REFERENCES diseases(id),
    symptom_id INT NOT NULL REFERENCES symptoms(id),
    prevalence DECIMAL,
    discrimination_weight INT,
    UNIQUE (disease_id, symptom_id)
);

COMMENT ON TABLE disease_symptoms IS '疾病-症状关联表（候选集反查 + 鉴别价值追问，原文档 condition_symptoms）';
COMMENT ON COLUMN disease_symptoms.id IS '自增主键';
COMMENT ON COLUMN disease_symptoms.disease_id IS '疾病 id，引用 diseases.id';
COMMENT ON COLUMN disease_symptoms.symptom_id IS '症状 id，引用 symptoms.id';
COMMENT ON COLUMN disease_symptoms.prevalence IS '该症状在该疾病中的出现率（演示近似值）';
COMMENT ON COLUMN disease_symptoms.discrimination_weight IS '人工标注鉴别价值 1-5，第 4 周启用排序（当前为逆文档频率近似值）';
