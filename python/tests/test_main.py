from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
from fastapi.testclient import TestClient

from claw_vector_svc.config import Settings
from claw_vector_svc.main import create_app


def _mock_sentence_transformer() -> MagicMock:
    model = MagicMock()
    model.get_sentence_embedding_dimension.return_value = 16
    model.max_seq_length = 64
    model.encode.return_value = np.ones((1, 16), dtype="float32")
    return model


def test_http_embed_and_health_endpoints() -> None:
    settings = Settings(MODEL_NAME="test-model", DEVICE="cpu", MAX_BATCH_SIZE=4)
    with patch("claw_vector_svc.embedder.SentenceTransformer", return_value=_mock_sentence_transformer()):
        app = create_app(settings)
        with TestClient(app) as client:
            health = client.get("/health")
            ready = client.get("/ready")
            info = client.get("/model-info")
            embed = client.post("/embed", json={"texts": ["hello"], "normalize": True})
            metrics = client.get("/metrics")

    assert health.status_code == 200
    assert health.json()["ready"] is True
    assert ready.status_code == 200
    assert info.status_code == 200
    assert info.json()["model_name"] == "test-model"
    assert embed.status_code == 200
    assert len(embed.json()["vectors"]) == 1
    assert metrics.status_code == 200
    assert "claw_embed_requests_total" in metrics.text
