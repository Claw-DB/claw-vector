# test_embedder.py — unit tests for the EmbedderService.
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from claw_vector_svc.config import Settings
from claw_vector_svc.embedder import EmbedderService


@pytest.fixture()
def settings() -> Settings:
    return Settings(
        CLAW_MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2",
        CLAW_DEVICE="cpu",
        CLAW_BATCH_SIZE=8,
        CLAW_MAX_SEQ_LEN=64,
    )


@pytest.fixture()
def embedder(settings: Settings) -> EmbedderService:
    return EmbedderService(settings)


def test_not_ready_before_load(embedder: EmbedderService) -> None:
    assert not embedder.is_ready
    assert embedder.dimensions == 0


def test_embed_raises_before_load(embedder: EmbedderService) -> None:
    with pytest.raises(RuntimeError, match="not loaded"):
        embedder.embed(["hello"])


def test_load_sets_ready(embedder: EmbedderService) -> None:
    """Patch SentenceTransformer so the test runs without a model download."""
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = 384
    mock_model.max_seq_length = 64
    mock_model.encode.return_value = np.zeros((1, 384), dtype="float32")

    with patch("claw_vector_svc.embedder.SentenceTransformer", return_value=mock_model):
        embedder.load()

    assert embedder.is_ready
    assert embedder.dimensions == 384
    assert embedder.load_time_ms >= 0


def test_embed_returns_correct_shape(embedder: EmbedderService) -> None:
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = 384
    mock_model.max_seq_length = 64
    mock_model.encode.return_value = np.zeros((3, 384), dtype="float32")

    with patch("claw_vector_svc.embedder.SentenceTransformer", return_value=mock_model):
        embedder.load()

    result = embedder.embed(["a", "b", "c"])
    assert len(result) == 3
    assert len(result[0]) == 384
