from __future__ import annotations

from collections.abc import Callable

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from claw_vector_svc.embedder import EmbedderService
from claw_vector_svc.models import HealthResponse, ModelInfoResponse, ReadyResponse

def make_health_router(get_embedder: Callable[[], EmbedderService | None]) -> APIRouter:
    """Return a FastAPI router bound to the provided *embedder* instance."""
    router = APIRouter(tags=["health"])

    @router.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Liveness probe: the process is up even before the model is ready."""
        embedder = get_embedder()
        return HealthResponse(
            ready=bool(embedder and embedder.is_ready),
            model_name=embedder.model_name if embedder else "",
            model_load_time_ms=embedder.load_time_ms if embedder else 0,
        )

    @router.get("/ready", response_model=ReadyResponse)
    async def ready() -> JSONResponse:
        """Readiness probe that returns 503 until the model is loaded."""
        embedder = get_embedder()
        if embedder is not None and embedder.is_ready:
            return JSONResponse(ReadyResponse(ready=True).model_dump(), status_code=200)
        return JSONResponse(ReadyResponse(ready=False).model_dump(), status_code=503)

    @router.get("/model-info", response_model=ModelInfoResponse)
    async def model_info() -> ModelInfoResponse:
        """Return metadata about the currently loaded embedding model."""
        embedder = get_embedder()
        return ModelInfoResponse(
            model_name=embedder.model_name if embedder else "",
            dimensions=embedder.dimensions if embedder else 0,
            max_sequence_length=embedder._settings.max_sequence_length if embedder else 0,
            device=embedder._settings.device if embedder else "",
        )

    return router
