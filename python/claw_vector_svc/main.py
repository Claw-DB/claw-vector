from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager

import structlog
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from claw_vector_svc.config import Settings, get_settings, settings as default_settings
from claw_vector_svc.embedder import EmbedderService
from claw_vector_svc.grpc_server import start_grpc_server
from claw_vector_svc.health import make_health_router
from claw_vector_svc.metrics import (
    MODEL_LOAD_TIME_SECONDS,
    mark_model_loaded,
    mark_model_unloaded,
    record_embed_request,
)
from claw_vector_svc.models import (
    BatchEmbedRequest,
    BatchEmbedResponse,
    EmbedRequest,
    EmbedResponse,
    EmbedVectorSchema,
    ModelInfoResponse,
)

log = structlog.get_logger(__name__)


def parse_api_keys(settings: Settings) -> set[str]:
    keys: set[str] = set()
    if settings.claw_api_key:
        keys.add(settings.claw_api_key.strip())
    if settings.claw_api_keys:
        keys.update(
            value.strip()
            for value in settings.claw_api_keys.split(",")
            if value.strip()
        )
    return keys


def _api_key_from_request(request: Request) -> str:
    return request.headers.get("X-Claw-Api-Key", "")


def _api_key_prefix(api_key: str) -> str:
    return api_key[:8] if api_key else ""


def create_app(settings: Settings | None = None) -> FastAPI:
    if settings is None:
        settings = get_settings()

    allowed_keys = parse_api_keys(settings)
    limiter = Limiter(key_func=_api_key_from_request)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.embedder = None
        app.state.grpc_server = None
        app.state.warmup_complete = False
        mark_model_unloaded()
        with MODEL_LOAD_TIME_SECONDS.time():
            embedder = await asyncio.to_thread(EmbedderService, settings)
        app.state.embedder = embedder
        mark_model_loaded(embedder.dimensions)
        app.state.grpc_server = await start_grpc_server(
            embedder,
            settings,
            allowed_keys,
            lambda: bool(getattr(app.state, "warmup_complete", False)),
        )

        async def _warmup() -> None:
            for _ in range(3):
                await asyncio.to_thread(
                    embedder.warmup,
                )
            app.state.warmup_complete = True

        warmup_task = asyncio.create_task(_warmup())
        try:
            yield
        finally:
            warmup_task.cancel()
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
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    def current_embedder() -> EmbedderService | None:
        return getattr(app.state, "embedder", None)

    def warmup_complete() -> bool:
        return bool(getattr(app.state, "warmup_complete", False))

    def require_api_key(x_claw_api_key: str | None) -> str:
        if not allowed_keys:
            return ""
        if not x_claw_api_key or x_claw_api_key not in allowed_keys:
            raise HTTPException(status_code=401, detail="invalid API key")
        return x_claw_api_key

    @app.middleware("http")
    async def request_logging_middleware(request: Request, call_next):
        t0 = time.monotonic()
        status_code = 500
        api_key = request.headers.get("X-Claw-Api-Key", "")
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            latency_ms = int((time.monotonic() - t0) * 1000)
            log.info(
                "http_request",
                method=request.method,
                path=request.url.path,
                api_key_prefix=_api_key_prefix(api_key),
                latency_ms=latency_ms,
                status_code=status_code,
            )

    @app.post("/embed", response_model=EmbedResponse, tags=["embeddings"])
    @limiter.limit(f"{settings.embed_rate_limit_per_minute}/minute")
    async def embed(
        request: Request,
        req: EmbedRequest,
        x_claw_api_key: str | None = Header(default=None),
    ) -> EmbedResponse:
        api_key = require_api_key(x_claw_api_key)
        _ = api_key
        embedder = current_embedder()
        if embedder is None or not embedder.is_ready or not warmup_complete():
            raise HTTPException(status_code=503, detail="embedding model is not ready")
        t0 = time.monotonic()
        try:
            vectors = await asyncio.to_thread(embedder.embed, req.texts, req.normalize)
        except Exception as exc:  # pragma: no cover
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

    @app.post("/batch-embed", response_model=BatchEmbedResponse, tags=["embeddings"])
    async def batch_embed(
        req: BatchEmbedRequest,
        x_claw_api_key: str | None = Header(default=None),
    ) -> BatchEmbedResponse:
        require_api_key(x_claw_api_key)
        embedder = current_embedder()
        if embedder is None or not embedder.is_ready or not warmup_complete():
            raise HTTPException(status_code=503, detail="embedding model is not ready")
        if len(req.texts) > 512:
            raise HTTPException(status_code=400, detail="batch size must be <= 512")

        t0 = time.monotonic()
        vectors: list[EmbedVectorSchema] = []
        per_batch_latency_ms: list[int] = []
        for start in range(0, len(req.texts), settings.max_batch_size):
            texts = req.texts[start : start + settings.max_batch_size]
            batch_t0 = time.monotonic()
            embedded = await asyncio.to_thread(embedder.embed, texts, req.normalize)
            per_batch_latency_ms.append(int((time.monotonic() - batch_t0) * 1000))
            vectors.extend(
                EmbedVectorSchema(values=vector.tolist(), dimensions=len(vector))
                for vector in embedded
            )

        return BatchEmbedResponse(
            vectors=vectors,
            model_name=embedder.model_name,
            total_latency_ms=int((time.monotonic() - t0) * 1000),
            per_batch_latency_ms=per_batch_latency_ms,
        )

    @app.get("/model-info", response_model=ModelInfoResponse, tags=["embeddings"])
    async def model_info(
        x_claw_api_key: str | None = Header(default=None),
    ) -> ModelInfoResponse:
        require_api_key(x_claw_api_key)
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

    app.include_router(make_health_router(current_embedder, warmup_complete))

    return app


app = create_app(default_settings)


def main() -> None:
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
