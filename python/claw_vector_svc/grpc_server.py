from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import AsyncIterator

import grpc
from grpc_tools import protoc
import structlog

from claw_vector_svc.config import Settings
from claw_vector_svc.embedder import EmbedderService
from claw_vector_svc.metrics import record_embed_request

log = structlog.get_logger(__name__)

PROTO_ROOT = Path(__file__).resolve().parents[1] / "proto"
PROTO_FILE = PROTO_ROOT / "vector.proto"


def _ensure_proto_generated() -> None:
    """Generate Python gRPC stubs on demand when they are missing or stale."""
    vector_pb2 = PROTO_ROOT / "vector_pb2.py"
    vector_pb2_grpc = PROTO_ROOT / "vector_pb2_grpc.py"
    if vector_pb2.exists() and vector_pb2_grpc.exists():
        if vector_pb2.stat().st_mtime >= PROTO_FILE.stat().st_mtime and vector_pb2_grpc.stat().st_mtime >= PROTO_FILE.stat().st_mtime:
            return

    result = protoc.main(
        [
            "grpc_tools.protoc",
            f"-I{PROTO_ROOT}",
            f"--python_out={PROTO_ROOT}",
            f"--grpc_python_out={PROTO_ROOT}",
            str(PROTO_FILE),
        ]
    )
    if result != 0:
        raise RuntimeError("failed to generate Python gRPC stubs from vector.proto")


def _import_proto():
    """Import Python gRPC stubs, generating them first when necessary."""
    _ensure_proto_generated()
    proto_path = str(PROTO_ROOT)
    if proto_path not in sys.path:
        sys.path.insert(0, proto_path)

    import vector_pb2 as pb2  # type: ignore[import]
    import vector_pb2_grpc as pb2_grpc  # type: ignore[import]

    return pb2, pb2_grpc


vector_pb2, vector_pb2_grpc = _import_proto()


class EmbeddingServicer(vector_pb2_grpc.EmbeddingServiceServicer):
    """gRPC servicer that wraps an EmbedderService."""

    def __init__(self, embedder: EmbedderService) -> None:
        self._embedder = embedder

    async def Embed(self, request, context):
        """Handle a single-shot embedding request."""
        t0 = time.monotonic()
        texts = list(request.texts)
        try:
            vectors = await asyncio.to_thread(
                self._embedder.embed,
                texts,
                request.normalize,
            )
            latency_ms = int((time.monotonic() - t0) * 1000)
            record_embed_request("grpc", "ok", len(texts), time.monotonic() - t0)
            embed_vectors = [
                vector_pb2.EmbedVector(values=vector.tolist(), dimensions=len(vector))
                for vector in vectors
            ]
            return vector_pb2.EmbedResponse(
                vectors=embed_vectors,
                model_name=self._embedder.model_name,
                latency_ms=latency_ms,
            )
        except Exception as exc:
            log.error("embed failed", error=str(exc))
            record_embed_request("grpc", "error", len(texts), time.monotonic() - t0)
            await context.abort(grpc.StatusCode.INTERNAL, str(exc))

    async def Health(self, request, context):
        """Return service readiness and model load time."""
        return vector_pb2.HealthResponse(
            ready=self._embedder.is_ready,
            model_name=self._embedder.model_name,
            model_load_time_ms=self._embedder.load_time_ms,
        )

    async def ModelInfo(self, request, context):
        """Return metadata about the currently loaded model."""
        return vector_pb2.ModelInfoResponse(
            model_name=self._embedder.model_name,
            dimensions=self._embedder.dimensions,
            max_sequence_length=self._embedder._settings.max_sequence_length,
            device=self._embedder._settings.device,
        )

    async def EmbedStream(self, request_iterator, context) -> AsyncIterator[object]:
        """Handle a bidirectional streaming embedding request."""
        async for req in request_iterator:
            t0 = time.monotonic()
            texts = list(req.texts)
            vectors = await asyncio.to_thread(self._embedder.embed, texts, req.normalize)
            latency_ms = int((time.monotonic() - t0) * 1000)
            embed_vectors = [
                vector_pb2.EmbedVector(values=vector.tolist(), dimensions=len(vector))
                for vector in vectors
            ]
            record_embed_request("grpc", "ok", len(texts), time.monotonic() - t0)
            yield vector_pb2.EmbedResponse(
                vectors=embed_vectors,
                model_name=self._embedder.model_name,
                latency_ms=latency_ms,
            )


async def start_grpc_server(embedder: EmbedderService, settings: Settings) -> grpc.aio.Server:
    """Start the async gRPC server and return the live server instance."""
    server = grpc.aio.server()
    vector_pb2_grpc.add_EmbeddingServiceServicer_to_server(
        EmbeddingServicer(embedder),
        server,
    )
    address = f"{settings.grpc_host}:{settings.grpc_port}"
    server.add_insecure_port(address)
    await server.start()
    log.info("gRPC server started", address=address)
    return server
