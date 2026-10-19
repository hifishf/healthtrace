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
    # Redis（docker-compose 里的 cache 服务，宿主端口 6380）
    redis_url: str = "redis://localhost:6380/0"
    # Langfuse 自托管（第 5 步接 OTel 用）
    langfuse_host: str = "http://localhost:3000"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    # 单用户：固定默认 user_id（与 schema/init.sql 种子一致）
    default_user_id: UUID = UUID("00000000-0000-0000-0000-000000000001")


settings = Settings()
