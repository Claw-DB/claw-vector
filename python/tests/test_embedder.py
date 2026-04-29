from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from claw_vector_svc.config import Settings
from claw_vector_svc.embedder import EmbedderService


@pytest.fixture()
def settings() -> Settings:
    return Settings(
        MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2",
        DEVICE="cpu",
        MAX_BATCH_SIZE=2,
        MAX_SEQUENCE_LENGTH=64,
        CACHE_SIZE=4,
    )


@pytest.fixture()
def embedder(settings: Settings) -> EmbedderService:
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = 384
    mock_model.max_seq_length = 64
    mock_model.encode.return_value = np.zeros((1, 384), dtype="float32")

    with patch("claw_vector_svc.embedder.SentenceTransformer", return_value=mock_model):
        return EmbedderService(settings)


def test_init_loads_model(embedder: EmbedderService) -> None:
    assert embedder.is_ready
    assert embedder.dimensions == 384
    assert embedder.load_time_ms >= 0


def test_embed_returns_correct_shape(settings: Settings) -> None:
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = 384
    mock_model.max_seq_length = 64
    mock_model.encode.return_value = np.zeros((3, 384), dtype="float32")

    with patch("claw_vector_svc.embedder.SentenceTransformer", return_value=mock_model):
        embedder = EmbedderService(settings)

    result = embedder.embed(["a", "b", "c"])
    assert result.shape == (3, 384)


def test_warmup_calls_embed(settings: Settings) -> None:
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = 384
    mock_model.max_seq_length = 64
    mock_model.encode.return_value = np.zeros((1, 384), dtype="float32")

    with patch("claw_vector_svc.embedder.SentenceTransformer", return_value=mock_model):
        embedder = EmbedderService(settings)

    embedder.warmup()
    assert mock_model.encode.call_count == 1


def test_embed_uses_cache(settings: Settings) -> None:
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = 8
    mock_model.max_seq_length = 64
    mock_model.encode.return_value = np.ones((1, 8), dtype="float32")

    with patch("claw_vector_svc.embedder.SentenceTransformer", return_value=mock_model):
        embedder = EmbedderService(settings)

    first = embedder.embed(["repeat"])
    second = embedder.embed(["repeat"])

    assert np.array_equal(first, second)
    assert mock_model.encode.call_count == 1  # first cache miss; second request is cached
