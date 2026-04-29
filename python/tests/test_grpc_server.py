from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from claw_vector_svc.config import Settings
from claw_vector_svc.embedder import EmbedderService
from claw_vector_svc.grpc_server import EmbeddingServicer, vector_pb2


@pytest.fixture()
def mock_embedder() -> EmbedderService:
    settings = Settings(MODEL_NAME="test-model", DEVICE="cpu")
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = 8
    mock_model.max_seq_length = 64
    mock_model.encode.return_value = np.asarray([[0.1] * 8, [0.2] * 8], dtype="float32")
    with patch("claw_vector_svc.embedder.SentenceTransformer", return_value=mock_model):
        return EmbedderService(settings)


@pytest.fixture()
def servicer(mock_embedder: EmbedderService) -> EmbeddingServicer:
    return EmbeddingServicer(mock_embedder)


@pytest.mark.asyncio
async def test_health_returns_ready(servicer: EmbeddingServicer) -> None:
    response = await servicer.Health(MagicMock(), AsyncMock())
    assert response.ready is True
    assert response.model_name == "test-model"


@pytest.mark.asyncio
async def test_model_info(servicer: EmbeddingServicer) -> None:
    response = await servicer.ModelInfo(MagicMock(), AsyncMock())
    assert response.model_name == "test-model"
    assert response.dimensions == 8


@pytest.mark.asyncio
async def test_embed_returns_vectors(servicer: EmbeddingServicer) -> None:
    request = vector_pb2.EmbedRequest(texts=["hello", "world"], normalize=True)
    response = await servicer.Embed(request, AsyncMock())
    assert len(response.vectors) == 2
    assert response.model_name == "test-model"
