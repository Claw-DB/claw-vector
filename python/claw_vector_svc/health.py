# health.py — health and readiness endpoint helpers for the FastAPI app.
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from claw_vector_svc.embedder import EmbedderService
from claw_vector_svc.models import HealthResponse, ModelInfoResponse

router = APIRouter(tags=["health"])


def make_health_router(embedder: EmbedderService) -> APIRouter:
    """Return a FastAPI router bound to the provided *embedder* instance."""

    @router.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Liveness probe: returns 200 when the model is loaded."""
        return HealthResponse(
            ready=embedder.is_ready,
            model_name=embedder._settings.model_name,
            model_load_time_ms=embedder.load_time_ms,
        )

    @router.get("/readyz")
    async def readyz() -> JSONResponse:
        """Kubernetes readiness probe."""
        if embedder.is_ready:
            return JSONResponse({"status": "ok"}, status_code=200)
        return JSONResponse({"status": "not_ready"}, status_code=503)

    @router.get("/model-info", response_model=ModelInfoResponse)
    async def model_info() -> ModelInfoResponse:
        """Return metadata about the currently loaded embedding model."""
        return ModelInfoResponse(
            model_name=embedder._settings.model_name,
            dimensions=embedder.dimensions,
            max_sequence_length=embedder.max_sequence_length,
            device=embedder._settings.device,
        )

    return router
