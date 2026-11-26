"""Sentence-embedding service with one lazy, thread-safe model instance."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384


class EmbeddingService:
    """Embed document batches and queries in the same normalized vector space."""

    def __init__(
        self, model_name: str = DEFAULT_MODEL_NAME, batch_size: int = 64
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._model_name = model_name
        self._batch_size = batch_size
        self._model: SentenceTransformer | None = None
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    def warm_up(self) -> None:
        """Load the model before serving the first embedding request."""
        self._get_model()

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    self._model = SentenceTransformer(self._model_name, device="cpu")
        return self._model

    def _encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._get_model().encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed many chunks, preserving input order."""
        return self._encode(list(texts))

    def embed_query(self, text: str) -> list[float]:
        """Embed one query with the same model and normalization as documents."""
        vectors = self._encode([text])
        return vectors[0]
