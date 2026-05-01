from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

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
    settings = Settings(
        MODEL_NAME="test-model",
        DEVICE="cpu",
        MAX_BATCH_SIZE=64,
        CLAW_API_KEYS="valid-key",
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
                headers={"X-Claw-Api-Key": "valid-key"},
            )

    assert bad.status_code == 401
    assert ok.status_code == 200


def test_batch_embed_300_texts_across_5_minibatches() -> None:
    settings = Settings(
        MODEL_NAME="test-model",
        DEVICE="cpu",
        MAX_BATCH_SIZE=64,
        CLAW_API_KEYS="batch-key",
    )
    with patch(
        "claw_vector_svc.embedder.SentenceTransformer",
        return_value=_mock_sentence_transformer(),
    ):
        app = create_app(settings)
        with TestClient(app) as client:
            response = client.post(
                "/batch-embed",
                json={"texts": [f"t-{i}" for i in range(300)], "normalize": True},
                headers={"X-Claw-Api-Key": "batch-key"},
            )

    body = response.json()
    assert response.status_code == 200
    assert len(body["vectors"]) == 300
    assert len(body["per_batch_latency_ms"]) == 5


def test_ready_is_503_before_warmup_then_200_after() -> None:
    settings = Settings(
        MODEL_NAME="test-model",
        DEVICE="cpu",
        CLAW_API_KEYS="warm-key",
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
    settings = Settings(
        MODEL_NAME="test-model",
        DEVICE="cpu",
        CLAW_API_KEYS="rate-key",
        EMBED_RATE_LIMIT_PER_MINUTE=2,
    )
    with patch(
        "claw_vector_svc.embedder.SentenceTransformer",
        return_value=_mock_sentence_transformer(),
    ):
        app = create_app(settings)
        with TestClient(app) as client:
            headers = {"X-Claw-Api-Key": "rate-key"}
            first = client.post("/embed", json={"texts": ["a"], "normalize": True}, headers=headers)
            second = client.post("/embed", json={"texts": ["b"], "normalize": True}, headers=headers)
            third = client.post("/embed", json={"texts": ["c"], "normalize": True}, headers=headers)

    assert first.status_code == 200
    assert second.status_code == 200
    assert third.status_code == 429
