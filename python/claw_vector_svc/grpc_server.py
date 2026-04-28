# grpc_server.py — gRPC servicer implementation for the EmbeddingService.
from __future__ import annotations

import time
from concurrent import futures

import grpc
import structlog

from claw_vector_svc.config import Settings
from claw_vector_svc.embedder import EmbedderService
from claw_vector_svc.metrics import (
    EMBED_BATCH_SIZE,
    EMBED_LATENCY_SECONDS,
    EMBED_REQUESTS_TOTAL,
)

log = structlog.get_logger(__name__)


def _import_proto():
    """Lazily import the generated gRPC stubs (avoids import errors at test time)."""
    try:
        import proto.vector_pb2 as pb2  # type: ignore[import]
        import proto.vector_pb2_grpc as pb2_grpc  # type: ignore[import]

        return pb2, pb2_grpc
    except ImportError as exc:
        raise ImportError(
            "Generated proto stubs not found. Run:\n"
            "  python -m grpc_tools.protoc -I proto --python_out=. "
            "--grpc_python_out=. proto/vector.proto"
        ) from exc


class EmbeddingServicer:
    """gRPC servicer that wraps an EmbedderService."""

    def __init__(self, embedder: EmbedderService) -> None:
        self._embedder = embedder

    def Embed(self, request, context):
        """Handle a single-shot embedding request."""
        pb2, _ = _import_proto()
        t0 = time.monotonic()
        texts = list(request.texts)
        EMBED_BATCH_SIZE.observe(len(texts))
        try:
            vectors = self._embedder.embed(texts, normalize=request.normalize)
            EMBED_REQUESTS_TOTAL.labels(status="ok").inc()
            latency_ms = int((time.monotonic() - t0) * 1000)
            EMBED_LATENCY_SECONDS.observe(time.monotonic() - t0)
            embed_vectors = [
                pb2.EmbedVector(values=v, dimensions=len(v)) for v in vectors
            ]
            return pb2.EmbedResponse(
                vectors=embed_vectors,
                model_name=self._embedder._settings.model_name,
                latency_ms=latency_ms,
            )
        except Exception as exc:
            log.error("embed failed", error=str(exc))
            EMBED_REQUESTS_TOTAL.labels(status="error").inc()
            context.abort(grpc.StatusCode.INTERNAL, str(exc))

    def Health(self, request, context):
        """Return service readiness and model load time."""
        pb2, _ = _import_proto()
        return pb2.HealthResponse(
            ready=self._embedder.is_ready,
            model_name=self._embedder._settings.model_name,
            model_load_time_ms=self._embedder.load_time_ms,
        )

    def ModelInfo(self, request, context):
        """Return metadata about the currently loaded model."""
        pb2, _ = _import_proto()
        return pb2.ModelInfoResponse(
            model_name=self._embedder._settings.model_name,
            dimensions=self._embedder.dimensions,
            max_sequence_length=self._embedder.max_sequence_length,
            device=self._embedder._settings.device,
        )

    def EmbedStream(self, request_iterator, context):
        """Handle a bidirectional streaming embedding request."""
        pb2, _ = _import_proto()
        for req in request_iterator:
            t0 = time.monotonic()
            texts = list(req.texts)
            vectors = self._embedder.embed(texts, normalize=req.normalize)
            latency_ms = int((time.monotonic() - t0) * 1000)
            embed_vectors = [
                pb2.EmbedVector(values=v, dimensions=len(v)) for v in vectors
            ]
            yield pb2.EmbedResponse(
                vectors=embed_vectors,
                model_name=self._embedder._settings.model_name,
                latency_ms=latency_ms,
            )


def serve(embedder: EmbedderService, settings: Settings) -> None:
    """Start the gRPC server and block until it is stopped."""
    pb2, pb2_grpc = _import_proto()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    servicer = EmbeddingServicer(embedder)
    add_fn = getattr(pb2_grpc, "add_EmbeddingServiceServicer_to_server")
    add_fn(servicer, server)
    addr = f"{settings.grpc_host}:{settings.grpc_port}"
    server.add_insecure_port(addr)
    log.info("gRPC server starting", address=addr)
    server.start()
    server.wait_for_termination()
