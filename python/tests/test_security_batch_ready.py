from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import blake3
import numpy as np
from fastapi.testclient import TestClient

from claw_vector_svc.config import Settings
from claw_vector_svc.main import create_app


def _mock_sentence_transformer(dimensions: int = 16) -> MagicMock:
    model = MagicMock()
    model.get_sentence_embedding_dimension.return_value = dimensions
    model.max_seq_length = 64

    def _encode(texts, **_kwargs):
        return np.ones((len(texts), dimensions), dtype="float32")

    model.encode.side_effect = _encode
    return model


def test_embed_requires_valid_api_key() -> None:
    valid_key = "valid-key"
    settings = Settings(
        MODEL_NAME="test-model",
        DEVICE="cpu",
        MAX_BATCH_SIZE=64,
        CLAW_VECTOR_API_KEYS=blake3.blake3(valid_key.encode()).hexdigest(),
    )
    with patch(
        "claw_vector_svc.embedder.SentenceTransformer",
        return_value=_mock_sentence_transformer(),
    ):
        app = create_app(settings)
        with TestClient(app) as client:
            bad = client.post("/embed", json={"texts": ["x"], "normalize": True})
            ok = client.post(
                "/embed",
                json={"texts": ["x"], "normalize": True},
                headers={"X-Claw-Api-Key": valid_key},
            )

    assert bad.status_code == 401
    assert ok.status_code == 200


def test_batch_embed_300_texts_across_5_minibatches() -> None:
    batch_key = "batch-key"
    settings = Settings(
        MODEL_NAME="test-model",
        DEVICE="cpu",
        MAX_BATCH_SIZE=64,
        CLAW_VECTOR_API_KEYS=blake3.blake3(batch_key.encode()).hexdigest(),
    )
    with patch(
        "claw_vector_svc.embedder.SentenceTransformer",
        return_value=_mock_sentence_transformer(),
    ):
        app = create_app(settings)
        with TestClient(app) as client:
            response = client.post(
                "/embed/batch",
                json={"texts": [f"t-{i}" for i in range(300)], "normalize": True},
                headers={"X-Claw-Api-Key": batch_key},
            )

    body = response.json()
    assert response.status_code == 200
    assert len(body["vectors"]) == 300
    assert len(body["per_batch_latency_ms"]) == 5


def test_ready_is_503_before_warmup_then_200_after() -> None:
    warm_key = "warm-key"
    settings = Settings(
        MODEL_NAME="test-model",
        DEVICE="cpu",
        CLAW_VECTOR_API_KEYS=blake3.blake3(warm_key.encode()).hexdigest(),
    )

    with patch(
        "claw_vector_svc.embedder.SentenceTransformer",
        return_value=_mock_sentence_transformer(),
    ):
        with patch("claw_vector_svc.embedder.EmbedderService.warmup", side_effect=lambda: time.sleep(0.08)):
            app = create_app(settings)
            with TestClient(app) as client:
                early = client.get("/ready")
                time.sleep(0.35)
                late = client.get("/ready")

    assert early.status_code == 503
    assert late.status_code == 200


def test_embed_rate_limit_returns_429() -> None:
    rate_key = "rate-key"
    settings = Settings(
        MODEL_NAME="test-model",
        DEVICE="cpu",
        CLAW_VECTOR_API_KEYS=blake3.blake3(rate_key.encode()).hexdigest(),
    )
    with patch(
        "claw_vector_svc.embedder.SentenceTransformer",
        return_value=_mock_sentence_transformer(),
    ):
        app = create_app(settings)
        with TestClient(app) as client:
            headers = {"X-Claw-Api-Key": rate_key}
            responses = [
                client.post(
                    "/embed",
                    json={"texts": [f"t-{idx}"], "normalize": True},
                    headers=headers,
                )
                for idx in range(201)
            ]

    assert responses[199].status_code == 200
    assert responses[200].status_code == 429
    assert "Retry-After" in responses[200].headers
