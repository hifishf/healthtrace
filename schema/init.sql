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
