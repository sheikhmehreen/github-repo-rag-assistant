"""Cosine-similarity retriever over an injected vector-search interface.

The threshold is COSINE SIMILARITY, higher is better, in [-1, 1]. Chroma is
configured for cosine space and returns cosine distance, so distance = 1 -
similarity. We convert exactly once in :func:`distance_to_similarity`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

logger = logging.getLogger(__name__)
MAX_TOP_K = 20


@dataclass(frozen=True)
class RawHit:
    """One vector-store result, containing cosine distance."""

    text: str
    path: str
    start_line: int
    end_line: int
    distance: float


@dataclass(frozen=True)
class RetrievedChunk:
    """A retrieved chunk that met the configured similarity threshold."""

    text: str
    path: str
    start_line: int
    end_line: int
    similarity: float


class VectorSearch(Protocol):
    """Interface required by :class:`Retriever`."""

    def query(
        self, repo_id: str, embedding: Sequence[float], top_k: int
    ) -> list[RawHit]: ...


EmbedQuery = Callable[[str], Sequence[float]]


def distance_to_similarity(distance: float) -> float:
    """Convert Chroma cosine distance to cosine similarity."""
    return 1.0 - distance


def _validate(top_k: int, min_similarity: float) -> None:
    if not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}, got {top_k}")
    if not -1.0 <= min_similarity <= 1.0:
        raise ValueError(
            f"min_similarity must be between -1 and 1, got {min_similarity}"
        )


class Retriever:
    """Embed a question, search one repo, and return only strong matches."""

    def __init__(
        self,
        embed_query: EmbedQuery,
        vector_search: VectorSearch,
        *,
        top_k: int = 5,
        min_similarity: float = 0.30,
    ) -> None:
        _validate(top_k, min_similarity)
        self._embed_query = embed_query
        self._search = vector_search
        self._top_k = top_k
        self._min_similarity = min_similarity

    def retrieve(
        self,
        repo_id: str,
        question: str,
        *,
        top_k: int | None = None,
        min_similarity: float | None = None,
    ) -> list[RetrievedChunk]:
        """Return above-threshold chunks in descending similarity order."""
        if not question.strip():
            return []
        k = self._top_k if top_k is None else top_k
        threshold = self._min_similarity if min_similarity is None else min_similarity
        _validate(k, threshold)
        embedding = self._embed_query(question)
        hits = self._search.query(repo_id, embedding, k)
        chunks = [
            RetrievedChunk(
                text=hit.text,
                path=hit.path,
                start_line=hit.start_line,
                end_line=hit.end_line,
                similarity=distance_to_similarity(hit.distance),
            )
            for hit in hits
        ]
        passed = [chunk for chunk in chunks if chunk.similarity >= threshold]
        passed.sort(key=lambda chunk: chunk.similarity, reverse=True)
        logger.info(
            "retrieval repo_id=%s fetched=%d passed=%d threshold=%.2f",
            repo_id,
            len(chunks),
            len(passed),
            threshold,
        )
        return passed[:k]
