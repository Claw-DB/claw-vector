from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from claw_vector_svc.config import Settings, get_settings, settings as default_settings
from claw_vector_svc.embedder import EmbedderService
from claw_vector_svc.grpc_server import start_grpc_server
from claw_vector_svc.health import make_health_router
from claw_vector_svc.metrics import MODEL_LOAD_TIME_SECONDS, mark_model_loaded, mark_model_unloaded, record_embed_request
from claw_vector_svc.models import EmbedRequest, EmbedResponse, EmbedVectorSchema, ModelInfoResponse

log = structlog.get_logger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    if settings is None:
        settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.embedder = None
        app.state.grpc_server = None
        mark_model_unloaded()
        with MODEL_LOAD_TIME_SECONDS.time():
            embedder = await asyncio.to_thread(EmbedderService, settings)
        await asyncio.to_thread(embedder.warmup)
        app.state.embedder = embedder
        mark_model_loaded(embedder.dimensions)
        app.state.grpc_server = await start_grpc_server(embedder, settings)
        try:
            yield
        finally:
            mark_model_unloaded()
            server = app.state.grpc_server
            if server is not None:
                await server.stop(grace=5)
                await server.wait_for_termination()
            app.state.embedder = None

    app = FastAPI(
        title="claw-vector-svc",
        description="Embedding microservice for ClawDB semantic memory.",
        version="0.1.0",
        lifespan=lifespan,
    )

    def current_embedder() -> EmbedderService | None:
        return getattr(app.state, "embedder", None)

    @app.post("/embed", response_model=EmbedResponse, tags=["embeddings"])
    async def embed(req: EmbedRequest) -> EmbedResponse:
        """Generate embeddings for a list of texts."""
        embedder = current_embedder()
        if embedder is None or not embedder.is_ready:
            raise HTTPException(status_code=503, detail="embedding model is not ready")
        t0 = time.monotonic()
        try:
            vectors = await asyncio.to_thread(embedder.embed, req.texts, req.normalize)
        except Exception as exc:  # pragma: no cover - defensive error mapping
            record_embed_request("http", "error", len(req.texts), time.monotonic() - t0)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        latency_ms = int((time.monotonic() - t0) * 1000)
        record_embed_request("http", "ok", len(req.texts), time.monotonic() - t0)
        return EmbedResponse(
            vectors=[
                EmbedVectorSchema(values=vector.tolist(), dimensions=len(vector))
                for vector in vectors
            ],
            model_name=embedder.model_name,
            latency_ms=latency_ms,
        )

    @app.get("/model-info", response_model=ModelInfoResponse, tags=["embeddings"])
    async def model_info() -> ModelInfoResponse:
        embedder = current_embedder()
        if embedder is None:
            raise HTTPException(status_code=503, detail="embedding model is not ready")
        return ModelInfoResponse(
            model_name=embedder.model_name,
            dimensions=embedder.dimensions,
            max_sequence_length=settings.max_sequence_length,
            device=settings.device,
        )

    @app.get("/metrics", tags=["metrics"])
    async def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    app.include_router(make_health_router(current_embedder))

    return app


app = create_app(default_settings)


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
    uvicorn.run(
        create_app(settings),
        host=settings.http_host,
        port=settings.http_port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
