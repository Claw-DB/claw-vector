# embedder.py — EmbedderService: model loading and batched inference.
from __future__ import annotations

import time

import structlog
from sentence_transformers import SentenceTransformer

from claw_vector_svc.config import Settings

log = structlog.get_logger(__name__)


class EmbedderService:
    """Wraps a SentenceTransformer model and provides batched embedding inference."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: SentenceTransformer | None = None
        self._load_time_ms: int = 0

    def load(self) -> None:
        """Load the model into memory (call once at startup)."""
        t0 = time.monotonic()
        log.info("loading model", model=self._settings.model_name, device=self._settings.device)
        self._model = SentenceTransformer(
            self._settings.model_name,
            device=self._settings.device,
        )
        self._model.max_seq_length = self._settings.max_sequence_length
        self._load_time_ms = int((time.monotonic() - t0) * 1000)
        log.info("model loaded", load_time_ms=self._load_time_ms)

    @property
    def is_ready(self) -> bool:
        """Return True if the model is loaded and ready."""
        return self._model is not None

    @property
    def load_time_ms(self) -> int:
        """Return the wall-clock model load time in milliseconds."""
        return self._load_time_ms

    @property
    def dimensions(self) -> int:
        """Return the output embedding dimensionality."""
        if self._model is None:
            return 0
        return self._model.get_sentence_embedding_dimension() or 0

    @property
    def max_sequence_length(self) -> int:
        """Return the model's configured maximum sequence length."""
        if self._model is None:
            return self._settings.max_sequence_length
        return self._model.max_seq_length

    def embed(self, texts: list[str], normalize: bool = True) -> list[list[float]]:
        """Embed a list of texts and return a list of float vectors.

        Args:
            texts: Input texts to embed.
            normalize: If True, L2-normalise the output vectors.

        Returns:
            A list of embedding vectors (one per input text).
        """
        if self._model is None:
            raise RuntimeError("Model is not loaded. Call EmbedderService.load() first.")

        embeddings = self._model.encode(
            texts,
            batch_size=self._settings.batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
        return [emb.tolist() for emb in embeddings]
