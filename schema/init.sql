-- HealthTrace 数据库初始化脚本
-- 该文件在 db 容器首次初始化时执行（仅在数据卷为空时生效）

-- 1) 启用 pgvector（本周围绕基础设施提前就位，向量检索第 9 周再用）
CREATE EXTENSION IF NOT EXISTS vector;
