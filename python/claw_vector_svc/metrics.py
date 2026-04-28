# metrics.py — Prometheus instrumentation for the embedding microservice.
from __future__ import annotations

from prometheus_client import Counter, Histogram, start_http_server

EMBED_REQUESTS_TOTAL = Counter(
    "claw_embed_requests_total",
    "Total number of gRPC Embed calls received.",
    ["status"],
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


def start_metrics_server(port: int = 9090) -> None:
    """Start the Prometheus HTTP metrics exposition server on *port*."""
    start_http_server(port)
