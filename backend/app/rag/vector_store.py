"""Persistent Chroma vector store with one collection per repository."""

import hashlib
import logging
import re
from collections.abc import Sequence

from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.rag.errors import VectorStoreError
from app.rag.models import RetrievedChunk

logger = logging.getLogger(__name__)


class RepoVectorStore:
    """ChromaDB wrapper with isolated persistent collections per repository."""

    def __init__(self, persist_dir: str, embeddings: Embeddings) -> None:
        self._persist_dir = persist_dir
        self._embeddings = embeddings
        self._client_settings = ChromaSettings(
            is_persistent=True,
            persist_directory=persist_dir,
            anonymized_telemetry=False,
        )

    @staticmethod
    def collection_name(repo_id: str) -> str:
        """Return a stable, Chroma-safe, collision-resistant collection name."""
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", repo_id).lower()[:40].strip("-") or "repo"
        digest = hashlib.sha256(repo_id.encode("utf-8")).hexdigest()[:8]
        return f"repo-{slug}-{digest}"

    def _collection(self, repo_id: str) -> Chroma:
        return Chroma(
            collection_name=self.collection_name(repo_id),
            embedding_function=self._embeddings,
            persist_directory=self._persist_dir,
            client_settings=self._client_settings,
            collection_metadata={"hnsw:space": "cosine"},
        )

    def add_documents(
        self, repo_id: str, documents: Sequence[Document], batch_size: int = 100
    ) -> int:
        if not documents:
            return 0
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        try:
            collection = self._collection(repo_id)
            for start in range(0, len(documents), batch_size):
                batch = documents[start : start + batch_size]
                collection.add_documents(
                    batch, ids=[document.metadata["chunk_id"] for document in batch]
                )
        except Exception as exc:
            raise VectorStoreError(f"Failed to store chunks for {repo_id}") from exc
        logger.info("Stored %d chunks for %s", len(documents), repo_id)
        return len(documents)

    def count(self, repo_id: str) -> int:
        try:
            return self._collection(repo_id)._collection.count()
        except Exception as exc:
            raise VectorStoreError(f"Failed to count chunks for {repo_id}") from exc

    def similarity_search(
        self, repo_id: str, query: str, k: int = 5
    ) -> list[RetrievedChunk]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if k <= 0:
            raise ValueError("k must be positive")
        try:
            results = self._collection(repo_id).similarity_search_with_score(query, k=k)
        except Exception as exc:
            raise VectorStoreError(f"Search failed for {repo_id}") from exc
        return [
            RetrievedChunk(
                content=doc.page_content,
                path=str(doc.metadata.get("path", "")),
                start_line=int(doc.metadata.get("start_line", 1)),
                score=1.0 - float(distance),
                metadata=dict(doc.metadata),
            )
            for doc, distance in results
        ]

    def delete_repo(self, repo_id: str) -> None:
        try:
            self._collection(repo_id).delete_collection()
        except Exception as exc:
            raise VectorStoreError(f"Failed to delete index for {repo_id}") from exc
        logger.info("Deleted index for %s", repo_id)
