from __future__ import annotations

from pydantic import BaseModel, Field


class EmbedRequest(BaseModel):
    """REST embedding request payload."""

    texts: list[str] = Field(..., description="Texts to embed.")
    normalize: bool = Field(True, description="L2-normalise output vectors.")


class EmbedVectorSchema(BaseModel):
    """A single embedding vector."""

    values: list[float]
    dimensions: int


class EmbedResponse(BaseModel):
    """REST embedding response payload."""

    vectors: list[EmbedVectorSchema]
    model_name: str
    latency_ms: int


class BatchEmbedRequest(BaseModel):
    """REST batch embedding request payload."""

    texts: list[str] = Field(..., description="Texts to embed.")
    normalize: bool = Field(True, description="L2-normalise output vectors.")


class BatchEmbedResponse(BaseModel):
    """REST batch embedding response payload."""

    vectors: list[list[float]]
    model_name: str
    total_latency_ms: float
    per_batch_latency_ms: list[float]


class HealthResponse(BaseModel):
    """Health check response."""

    ready: bool
    model_name: str
    model_load_time_ms: int


class ReadyResponse(BaseModel):
    """Readiness response."""

    ready: bool


class ModelInfoResponse(BaseModel):
    """Model metadata response."""

    model_name: str
    dimensions: int
    max_sequence_length: int
    device: str
