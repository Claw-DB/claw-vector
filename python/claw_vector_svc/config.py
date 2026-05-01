from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="MODEL_NAME",
    )
    device: str = Field(default="cpu", alias="DEVICE")
    grpc_port: int = Field(default=50051, alias="GRPC_PORT")
    http_port: int = Field(default=8080, alias="HTTP_PORT")
    max_batch_size: int = Field(default=64, alias="MAX_BATCH_SIZE")
    cache_size: int = Field(default=10_000, alias="CACHE_SIZE")
    normalize_embeddings: bool = Field(default=True, alias="NORMALIZE_EMBEDDINGS")
    onnx_model_path: Optional[str] = Field(default=None, alias="ONNX_MODEL_PATH")
    grpc_host: str = Field(default="0.0.0.0", alias="GRPC_HOST")
    http_host: str = Field(default="0.0.0.0", alias="HTTP_HOST")
    max_sequence_length: int = Field(default=256, alias="MAX_SEQUENCE_LENGTH")
    claw_api_key: Optional[str] = Field(default=None, alias="CLAW_API_KEY")
    claw_api_keys: Optional[str] = Field(default=None, alias="CLAW_API_KEYS")
    embed_rate_limit_per_minute: int = Field(default=200, alias="EMBED_RATE_LIMIT_PER_MINUTE")

    model_config = SettingsConfigDict(
        populate_by_name=True,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings instance."""
    return Settings()


settings = get_settings()
