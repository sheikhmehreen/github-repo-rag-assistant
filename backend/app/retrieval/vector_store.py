"""Vector-store interface and persistent ChromaDB implementation."""

from __future__ import annotations

import re
import threading
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

_ADD_BATCH_SIZE = 500
_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{1,61}[a-zA-Z0-9]$")


class VectorStoreError(Exception):
    """Base error for vector store operations."""


class CollectionNotFoundError(VectorStoreError):
    """The requested repository collection does not exist."""


def collection_name_for(repo_id: str) -> str:
    """Map a repository UUID to a valid, isolated Chroma collection name."""
    name = f"repo_{repo_id}"
    if not _NAME_RE.fullmatch(name):
        raise ValueError(f"Invalid collection name: {name!r}")
    return name


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any]


class VectorStore(ABC):
    """Interface shared by concrete vector-store implementations."""

    @abstractmethod
    def add_chunks(
        self,
        collection: str,
        ids: Sequence[str],
        texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[dict[str, Any]],
    ) -> None: ...

    @abstractmethod
    def search(
        self,
        collection: str,
        query_embedding: Sequence[float],
        top_k: int = 5,
    ) -> list[SearchHit]: ...

    @abstractmethod
    def delete_collection(self, collection: str) -> None: ...


class _UsageTracker:
    """Allow concurrent readers while making deletes exclusive per collection."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._readers: dict[str, int] = defaultdict(int)
        self._deleting: set[str] = set()

    @contextmanager
    def reading(self, name: str) -> Iterator[None]:
        with self._condition:
            while name in self._deleting:
                self._condition.wait()
            self._readers[name] += 1
        try:
            yield
        finally:
            with self._condition:
                self._readers[name] -= 1
                if self._readers[name] == 0:
                    del self._readers[name]
                self._condition.notify_all()

    @contextmanager
    def deleting(self, name: str) -> Iterator[None]:
        with self._condition:
            while name in self._deleting:
                self._condition.wait()
            self._deleting.add(name)
            while self._readers.get(name, 0) > 0:
                self._condition.wait()
        try:
            yield
        finally:
            with self._condition:
                self._deleting.discard(name)
                self._condition.notify_all()


def _clean_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Keep metadata values within Chroma's supported scalar types."""
    cleaned: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        cleaned[key] = (
            value if isinstance(value, (str, int, float, bool)) else str(value)
        )
    return cleaned


class ChromaVectorStore(VectorStore):
    """Persistent cosine-similarity store with per-collection delete safety."""

    def __init__(self, persist_dir: Path | str) -> None:
        path = Path(persist_dir)
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._usage = _UsageTracker()

    def _existing_names(self) -> set[str]:
        # Chroma versions have returned either collection objects or names.
        return {
            collection if isinstance(collection, str) else collection.name
            for collection in self._client.list_collections()
        }

    def add_chunks(
        self,
        collection: str,
        ids: Sequence[str],
        texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[dict[str, Any]],
    ) -> None:
        if not (len(ids) == len(texts) == len(embeddings) == len(metadatas)):
            raise ValueError(
                "ids, texts, embeddings and metadatas must have equal length"
            )
        if not ids:
            return
        with self._usage.reading(collection):
            col = self._client.get_or_create_collection(
                name=collection,
                metadata={"hnsw:space": "cosine"},
            )
            for start in range(0, len(ids), _ADD_BATCH_SIZE):
                end = start + _ADD_BATCH_SIZE
                col.upsert(
                    ids=list(ids[start:end]),
                    documents=list(texts[start:end]),
                    embeddings=[list(vector) for vector in embeddings[start:end]],
                    metadatas=[
                        _clean_metadata(value) for value in metadatas[start:end]
                    ],
                )

    def search(
        self,
        collection: str,
        query_embedding: Sequence[float],
        top_k: int = 5,
    ) -> list[SearchHit]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not query_embedding:
            raise ValueError("query_embedding must not be empty")
        with self._usage.reading(collection):
            if collection not in self._existing_names():
                raise CollectionNotFoundError(f"Collection not found: {collection}")
            col = self._client.get_collection(name=collection)
            count = col.count()
            if count == 0:
                return []
            result = col.query(
                query_embeddings=[list(query_embedding)],
                n_results=min(top_k, count),
                include=["documents", "metadatas", "distances"],
            )
        return [
            SearchHit(
                chunk_id=chunk_id,
                text=text,
                score=1.0 - float(distance),
                metadata=dict(metadata or {}),
            )
            for chunk_id, text, metadata, distance in zip(
                result["ids"][0],
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
            )
        ]

    def delete_collection(self, collection: str) -> None:
        """Delete idempotently, waiting for searches already using the collection."""
        with self._usage.deleting(collection):
            if collection in self._existing_names():
                self._client.delete_collection(name=collection)
