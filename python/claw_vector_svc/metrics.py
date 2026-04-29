from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, make_asgi_app

EMBED_REQUESTS_TOTAL = Counter(
    "claw_embed_requests_total",
    "Total number of embedding requests received.",
    ["transport", "status"],
)

EMBED_LATENCY_SECONDS = Histogram(
    "claw_embed_latency_seconds",
    "End-to-end latency of a single gRPC Embed call.",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

EMBED_BATCH_SIZE = Histogram(
    "claw_embed_batch_size",
    "Number of texts per embed request.",
    buckets=[1, 4, 8, 16, 32, 64, 128, 256],
)

MODEL_LOAD_TIME_SECONDS = Histogram(
    "claw_model_load_time_seconds",
    "Time taken to load the embedding model.",
)

MODEL_READY = Gauge(
    "claw_model_ready",
    "Whether the embedding model has finished loading.",
)

MODEL_DIMENSIONS = Gauge(
    "claw_model_dimensions",
    "Embedding dimension count of the loaded model.",
)


def metrics_app():
    """Return an ASGI app that exposes Prometheus metrics."""
    return make_asgi_app()


def record_embed_request(transport: str, status: str, batch_size: int, latency_s: float) -> None:
    """Record metrics for a single embedding request."""
    EMBED_REQUESTS_TOTAL.labels(transport=transport, status=status).inc()
    EMBED_BATCH_SIZE.observe(batch_size)
    EMBED_LATENCY_SECONDS.observe(latency_s)


def mark_model_loaded(dimensions: int) -> None:
    """Mark the model as ready and publish its output dimensions."""
    MODEL_READY.set(1)
    MODEL_DIMENSIONS.set(dimensions)


def mark_model_unloaded() -> None:
    """Reset the readiness gauge."""
    MODEL_READY.set(0)
