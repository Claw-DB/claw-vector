from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import blake3
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
    return EmbeddingServicer(
        mock_embedder,
        {blake3.blake3(b"valid").hexdigest()},
        lambda: True,
    )


def _context_with_auth(token: str = "valid") -> MagicMock:
    context = MagicMock()
    context.invocation_metadata.return_value = (("authorization", f"Bearer {token}"),)
    context.abort = AsyncMock(side_effect=RuntimeError("aborted"))
    return context


@pytest.mark.asyncio
async def test_health_returns_ready(servicer: EmbeddingServicer) -> None:
    response = await servicer.Health(MagicMock(), _context_with_auth())
    assert response.ready is True
    assert response.model_name == "test-model"


@pytest.mark.asyncio
async def test_model_info(servicer: EmbeddingServicer) -> None:
    response = await servicer.ModelInfo(MagicMock(), _context_with_auth())
    assert response.model_name == "test-model"
    assert response.dimensions == 8


@pytest.mark.asyncio
async def test_embed_returns_vectors(servicer: EmbeddingServicer) -> None:
    request = vector_pb2.EmbedRequest(texts=["hello", "world"], normalize=True)
    response = await servicer.Embed(request, _context_with_auth())
    assert len(response.vectors) == 2
    assert response.model_name == "test-model"


@pytest.mark.asyncio
async def test_embed_rejects_invalid_api_key(mock_embedder: EmbedderService) -> None:
    secured = EmbeddingServicer(
        mock_embedder,
        {blake3.blake3(b"valid").hexdigest()},
        lambda: True,
    )
    context = MagicMock()
    context.invocation_metadata.return_value = (("authorization", "Bearer invalid"),)
    context.abort = AsyncMock(side_effect=RuntimeError("aborted"))
    request = vector_pb2.EmbedRequest(texts=["hello"], normalize=True)

    with pytest.raises(RuntimeError):
        await secured.Embed(request, context)
