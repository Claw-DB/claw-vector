# config.py — Settings loaded from environment variables via pydantic-settings.
from __future__ import annotations

from pathlib import Path  # noqa: F401 (kept for potential future use)

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from the environment."""

    # gRPC server
    grpc_host: str = Field("0.0.0.0", alias="CLAW_GRPC_HOST")
    grpc_port: int = Field(50051, alias="CLAW_GRPC_PORT")

    # Model
    model_name: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2", alias="CLAW_MODEL_NAME"
    )
    device: str = Field("cpu", alias="CLAW_DEVICE")
    normalize_embeddings: bool = Field(True, alias="CLAW_NORMALIZE")

    # Inference
    batch_size: int = Field(64, alias="CLAW_BATCH_SIZE")
    max_sequence_length: int = Field(256, alias="CLAW_MAX_SEQ_LEN")

    # Prometheus
    metrics_port: int = Field(9090, alias="CLAW_METRICS_PORT")

    model_config = {"populate_by_name": True, "env_file": ".env", "env_file_encoding": "utf-8"}


def get_settings() -> Settings:
    """Return a cached Settings singleton."""
    return Settings()
