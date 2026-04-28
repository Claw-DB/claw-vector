# test_grpc_server.py — unit tests for the gRPC servicer.
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from claw_vector_svc.config import Settings
from claw_vector_svc.embedder import EmbedderService
from claw_vector_svc.grpc_server import EmbeddingServicer


@pytest.fixture()
def mock_embedder() -> EmbedderService:
    settings = Settings(CLAW_MODEL_NAME="test-model", CLAW_DEVICE="cpu")
    embedder = EmbedderService(settings)
    embedder._model = MagicMock()
    embedder._model.get_sentence_embedding_dimension.return_value = 8
    embedder._model.max_seq_length = 64
    embedder._model.encode.return_value = [[0.1] * 8, [0.2] * 8]
    return embedder


@pytest.fixture()
def servicer(mock_embedder: EmbedderService) -> EmbeddingServicer:
    return EmbeddingServicer(mock_embedder)


def _patched_import():
    mock_pb2 = MagicMock()
    mock_pb2.HealthResponse.return_value = MagicMock(ready=True)
    mock_pb2.ModelInfoResponse.return_value = MagicMock()
    return mock_pb2, MagicMock()


def test_health_returns_ready(servicer: EmbeddingServicer) -> None:
    req = MagicMock()
    ctx = MagicMock()
    import claw_vector_svc.grpc_server as grpc_mod

    original = grpc_mod._import_proto
    grpc_mod._import_proto = _patched_import
    try:
        servicer.Health(req, ctx)
        # No exception = success
    finally:
        grpc_mod._import_proto = original


def test_model_info(servicer: EmbeddingServicer) -> None:
    req = MagicMock()
    ctx = MagicMock()
    mock_pb2 = MagicMock()
    import claw_vector_svc.grpc_server as grpc_mod

    original = grpc_mod._import_proto

    def fake_import():
        return mock_pb2, MagicMock()

    grpc_mod._import_proto = fake_import
    try:
        servicer.ModelInfo(req, ctx)
        call_kwargs = mock_pb2.ModelInfoResponse.call_args.kwargs
        assert call_kwargs["model_name"] == "test-model"
    finally:
        grpc_mod._import_proto = original
