"""应用配置：从环境变量（.env）读取连接信息。"""
from uuid import UUID

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置项。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 应用数据库（docker-compose 里的 db 服务）
    database_url: str = "postgresql+asyncpg://healthtrace:healthtrace@localhost:5432/healthtrace"
    # Alembic 同步连接串（psycopg 驱动；alembic 用同步引擎，不认 asyncpg 的异步连接串）
    database_url_sync: str = "postgresql+psycopg://healthtrace:healthtrace@localhost:5432/healthtrace"
    # LangGraph Postgres checkpointer（psycopg 驱动；与 SQLAlchemy 的 asyncpg 分开，
    # 因为 AsyncPostgresSaver 走 psycopg，不认 asyncpg 的连接串）
    checkpoint_database_url: str = "postgresql://healthtrace:healthtrace@localhost:5432/healthtrace"
    # Redis（docker-compose 里的 cache 服务，宿主端口 6380）
    redis_url: str = "redis://localhost:6380/0"
    # 会话工作记忆默认 TTL（秒）：到期自动清理，会话结束也会显式清除
    working_memory_ttl_seconds: int = 3600
    # Langfuse 自托管（第 5 步接 OTel 用）
    langfuse_host: str = "http://localhost:3000"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    # LLM（OpenAI 兼容接口，默认 DeepSeek）
    llm_base_url: str = "https://api.deepseek.com"
    llm_api_key: str = ""
    llm_model: str = "deepseek-chat"
    # 转写/话术生成的温度（结构化抽取/路由仍用 temperature=0）
    llm_writer_temperature: float = 0.7
    # LLM 调用超时（秒）与重试次数（重试在统一调用层实现，SDK 层重试已禁用）
    llm_timeout_seconds: float = 30.0
    llm_max_retries: int = 2
    # 单次生成 token 上限（限制单响应成本）
    llm_max_tokens: int = 1024
    # 单会话累计 token 估算上限：超限强制收口，后续降级走规则模板（成本控制）
    llm_session_token_budget: int = 12000
    # Router 置信度阈值：低于该值强制回落 chat 兜底
    router_confidence_threshold: float = 0.7
    # 单用户：固定默认 user_id（与 schema/init.sql 种子一致）
    default_user_id: UUID = UUID("00000000-0000-0000-0000-000000000001")


settings = Settings()
