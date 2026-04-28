# main.py — FastAPI app factory for the claw-vector embedding microservice.
from __future__ import annotations

import threading
import time

import structlog
import uvicorn
from fastapi import FastAPI

from claw_vector_svc.config import Settings, get_settings
from claw_vector_svc.embedder import EmbedderService
from claw_vector_svc.grpc_server import serve as grpc_serve
from claw_vector_svc.health import make_health_router
from claw_vector_svc.metrics import MODEL_LOAD_TIME_SECONDS, start_metrics_server
from claw_vector_svc.models import EmbedRequest, EmbedResponse, EmbedVectorSchema

log = structlog.get_logger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    if settings is None:
        settings = get_settings()

    embedder = EmbedderService(settings)

    app = FastAPI(
        title="claw-vector-svc",
        description="Embedding microservice for ClawDB semantic memory.",
        version="0.1.0",
    )

    @app.on_event("startup")
    async def startup() -> None:
        with MODEL_LOAD_TIME_SECONDS.time():
            embedder.load()
        grpc_thread = threading.Thread(
            target=grpc_serve,
            args=(embedder, settings),
            daemon=True,
            name="grpc-server",
        )
        grpc_thread.start()
        start_metrics_server(settings.metrics_port)
        log.info("startup complete")

    @app.post("/embed", response_model=EmbedResponse, tags=["embeddings"])
    async def embed(req: EmbedRequest) -> EmbedResponse:
        """Generate embeddings for a list of texts."""
        t0 = time.monotonic()
        model = req.model_name or settings.model_name
        vectors = embedder.embed(req.texts, normalize=req.normalize)
        latency_ms = int((time.monotonic() - t0) * 1000)
        return EmbedResponse(
            vectors=[EmbedVectorSchema(values=v, dimensions=len(v)) for v in vectors],
            model_name=model,
            latency_ms=latency_ms,
        )

    app.include_router(make_health_router(embedder))

    return app


def main() -> None:
    """CLI entry point: start the FastAPI + gRPC server."""
    settings = get_settings()
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ]
    )
    app = create_app(settings)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()
