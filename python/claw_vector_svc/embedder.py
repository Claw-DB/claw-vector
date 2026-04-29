from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any

import numpy as np
import structlog

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover - exercised when inference extras are absent
    ort = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - exercised when inference extras are absent
    SentenceTransformer = None

try:
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover - exercised when inference extras are absent
    AutoTokenizer = None

from claw_vector_svc.config import Settings

log = structlog.get_logger(__name__)


class EmbedderService:
    """Sentence-transformers backed embedding service with optional ONNX inference."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._onnx_session: Any | None = None
        self._load_time_ms: int = 0
        self._dimensions: int = 0
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.load()

    def load(self) -> None:
        """Load the embedding model and optional ONNX runtime session."""
        if self._model is not None:
            return

        started = time.monotonic()
        log.info(
            "loading embedding model",
            model_name=self._settings.model_name,
            device=self._settings.device,
            onnx_model_path=self._settings.onnx_model_path,
        )
        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers is not installed; install claw-vector-svc[inference]"
            )
        self._model = SentenceTransformer(
            self._settings.model_name,
            device=self._settings.device,
        )
        self._model.max_seq_length = self._settings.max_sequence_length
        self._dimensions = self._model.get_sentence_embedding_dimension() or 0

        if self._settings.onnx_model_path:
            if ort is None or AutoTokenizer is None:
                raise RuntimeError(
                    "ONNX inference requires claw-vector-svc[inference] dependencies"
                )
            providers = [
                "CUDAExecutionProvider"
                if self._settings.device.lower().startswith("cuda")
                else "CPUExecutionProvider"
            ]
            self._tokenizer = AutoTokenizer.from_pretrained(self._settings.model_name)
            self._onnx_session = ort.InferenceSession(
                self._settings.onnx_model_path,
                providers=providers,
            )

        self._load_time_ms = int((time.monotonic() - started) * 1000)
        log.info(
            "embedding model loaded",
            load_time_ms=self._load_time_ms,
            dimensions=self._dimensions,
        )

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    @property
    def load_time_ms(self) -> int:
        return self._load_time_ms

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        return self._settings.model_name

    def embed(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        """Embed a list of texts and return a float32 array shaped `(N, D)`."""
        if self._model is None:
            raise RuntimeError("Model is not loaded. Call EmbedderService.load() first.")
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)

        effective_normalize = normalize and self._settings.normalize_embeddings
        outputs: list[np.ndarray | None] = [None] * len(texts)
        uncached: list[tuple[int, str]] = []

        for index, text in enumerate(texts):
            cache_key = self._cache_key(text, effective_normalize)
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._cache.move_to_end(cache_key)
                outputs[index] = cached.copy()
            else:
                uncached.append((index, text))

        for start in range(0, len(uncached), self._settings.max_batch_size):
            chunk = uncached[start : start + self._settings.max_batch_size]
            chunk_texts = [text for _, text in chunk]
            if self._onnx_session is not None:
                embedded = self.embed_onnx(chunk_texts, normalize=effective_normalize)
            else:
                embedded = np.asarray(
                    self._model.encode(
                        chunk_texts,
                        batch_size=min(self._settings.max_batch_size, len(chunk_texts)),
                        normalize_embeddings=effective_normalize,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                    ),
                    dtype=np.float32,
                )

            for offset, (index, text) in enumerate(chunk):
                vector = embedded[offset].astype(np.float32, copy=True)
                outputs[index] = vector
                self._cache_insert(self._cache_key(text, effective_normalize), vector)

        return np.vstack([vector for vector in outputs if vector is not None]).astype(np.float32)

    def embed_onnx(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        """Run embeddings through ONNX Runtime when an ONNX model path is configured."""
        if self._onnx_session is None or self._tokenizer is None:
            raise RuntimeError("ONNX runtime is not configured for this service.")
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)

        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self._settings.max_sequence_length,
            return_tensors="np",
        )
        session_inputs = {}
        for input_meta in self._onnx_session.get_inputs():
            if input_meta.name in encoded:
                session_inputs[input_meta.name] = encoded[input_meta.name]
        outputs = self._onnx_session.run(None, session_inputs)
        embeddings = np.asarray(outputs[0], dtype=np.float32)
        if embeddings.ndim == 3:
            attention_mask = encoded["attention_mask"].astype(np.float32)[..., None]
            token_sums = (embeddings * attention_mask).sum(axis=1)
            token_counts = np.clip(attention_mask.sum(axis=1), a_min=1.0, a_max=None)
            embeddings = token_sums / token_counts
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.clip(norms, a_min=1e-12, a_max=None)
        return embeddings.astype(np.float32)

    def warmup(self) -> None:
        """Run a small embedding request to prime the model and optional runtime."""
        self.embed(["warmup"], normalize=self._settings.normalize_embeddings)

    def _cache_key(self, text: str, normalize: bool) -> str:
        return f"{int(normalize)}:{text}"

    def _cache_insert(self, key: str, value: np.ndarray) -> None:
        self._cache[key] = value.astype(np.float32, copy=True)
        self._cache.move_to_end(key)
        while len(self._cache) > self._settings.cache_size:
            self._cache.popitem(last=False)
