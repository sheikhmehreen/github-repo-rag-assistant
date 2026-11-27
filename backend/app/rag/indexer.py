"""Convert repository files into a searchable, replaceable vector index."""

import logging
import time
from collections.abc import Iterable

from langchain_core.documents import Document

from app.rag.chunker import RepoChunker
from app.rag.errors import IndexingError, VectorStoreError
from app.rag.models import IndexResult, SourceFile
from app.rag.vector_store import RepoVectorStore

logger = logging.getLogger(__name__)


class IndexingService:
    """Chunk files then safely replace a repository's existing index."""

    def __init__(
        self,
        chunker: RepoChunker,
        store: RepoVectorStore,
        max_chunks_per_repo: int = 5000,
    ) -> None:
        if max_chunks_per_repo <= 0:
            raise ValueError("max_chunks_per_repo must be positive")
        self._chunker = chunker
        self._store = store
        self._max_chunks = max_chunks_per_repo

    def index_repo(self, repo_id: str, files: Iterable[SourceFile]) -> IndexResult:
        started = time.perf_counter()
        documents: list[Document] = []
        indexed = skipped = 0
        for source in files:
            chunks = self._chunker.chunk_file(repo_id, source)
            if not chunks:
                skipped += 1
                continue
            indexed += 1
            documents.extend(chunks)
            if len(documents) > self._max_chunks:
                raise IndexingError(
                    f"{repo_id} exceeds the limit of {self._max_chunks} chunks"
                )
        if not documents:
            raise IndexingError(f"No indexable content found in {repo_id}")

        # Chunking succeeded before replacing the old collection. Remove partial writes
        # if Chroma fails midway so a later retry starts from a clean state.
        self._store.delete_repo(repo_id)
        try:
            stored = self._store.add_documents(repo_id, documents)
        except VectorStoreError as exc:
            self._store.delete_repo(repo_id)
            raise IndexingError(f"Indexing failed for {repo_id}") from exc
        if stored != len(documents):
            raise IndexingError(f"Only {stored} of {len(documents)} chunks were stored")

        duration = time.perf_counter() - started
        logger.info(
            "Indexed %s: %d files, %d skipped, %d chunks in %.2fs",
            repo_id,
            indexed,
            skipped,
            stored,
            duration,
        )
        return IndexResult(repo_id, indexed, skipped, stored, duration)
