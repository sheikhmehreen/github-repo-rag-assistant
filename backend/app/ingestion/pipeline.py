"""Dependency-injected repository ingestion and vector indexing pipeline."""

from __future__ import annotations

import logging
import os
import shutil
import stat
import sys
import tempfile
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.database.models import RepoRecord
from app.database.repository import RepoRepository, SubmitResult
from app.ingestion.legacy_pipeline import IngestionResult, ingest_repository
from app.ingestion.types import Chunk, RepoInfo, SourceFile
from app.retrieval.vector_store import VectorStore, collection_name_for

logger = logging.getLogger(__name__)

__all__ = [
    "IngestionError",
    "IngestionPipeline",
    "IngestionResult",
    "IngestionTimeoutError",
    "PipelineConfig",
    "PipelineDeps",
    "RepoTooLargeError",
    "ingest_repository",
]


class IngestionError(Exception):
    """An ingestion failure with a named pipeline step."""

    def __init__(self, step: str, message: str) -> None:
        super().__init__(f"[{step}] {message}")
        self.step = step


class RepoTooLargeError(IngestionError):
    """Repository or chunk count exceeds a configured limit."""


class IngestionTimeoutError(IngestionError):
    """The overall ingestion deadline was exceeded."""


class Embedder(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...


@dataclass(frozen=True)
class PipelineConfig:
    timeout_seconds: float = 300.0
    max_repo_size_kb: int = 51_200
    max_chunks: int = 20_000
    embed_batch_size: int = 64
    temp_root: Path | None = None


@dataclass(frozen=True)
class PipelineDeps:
    parse_url: Callable[[str], tuple[str, str]]
    fetch_metadata: Callable[[str, str], RepoInfo]
    download_archive: Callable[[str, str, str, Path, float], Path]
    extract_archive: Callable[[Path, Path], Path]
    list_files: Callable[[Path], Iterable[Path]]
    read_file: Callable[[Path, Path], SourceFile | None]
    chunk_file: Callable[[SourceFile], list[Chunk]]
    embedder: Embedder
    vector_store: VectorStore
    repository: RepoRepository


class _Deadline:
    def __init__(self, seconds: float) -> None:
        self._seconds = seconds
        self._end = time.monotonic() + seconds

    def remaining(self) -> float:
        return max(0.0, self._end - time.monotonic())

    def check(self, step: str) -> None:
        if time.monotonic() >= self._end:
            raise IngestionTimeoutError(
                step, f"overall deadline of {self._seconds:g}s exceeded"
            )


def _remove_tree(path: Path) -> None:
    """Remove temporary trees, retrying read-only files on Windows."""

    def make_writable_and_retry(func, failed_path, _exc) -> None:
        os.chmod(failed_path, stat.S_IWRITE)
        func(failed_path)

    try:
        if sys.version_info >= (3, 12):
            shutil.rmtree(path, onexc=make_writable_and_retry)
        else:
            shutil.rmtree(path, onerror=make_writable_and_retry)
    except OSError:
        logger.warning(
            "Could not fully remove temporary directory %s", path, exc_info=True
        )


class IngestionPipeline:
    """Validate and claim jobs, then safely build and persist repository indexes."""

    def __init__(self, deps: PipelineDeps, config: PipelineConfig) -> None:
        if config.timeout_seconds < 0:
            raise ValueError("timeout_seconds must be non-negative")
        if config.max_repo_size_kb <= 0 or config.max_chunks <= 0:
            raise ValueError("ingestion limits must be positive")
        if config.embed_batch_size <= 0:
            raise ValueError("embed_batch_size must be positive")
        self._deps = deps
        self._config = config

    def submit(self, url: str) -> SubmitResult:
        """Validate, fetch metadata, enforce repository size, and claim a job."""
        owner, name = self._deps.parse_url(url)
        info = self._deps.fetch_metadata(owner, name)
        if info.size_kb > self._config.max_repo_size_kb:
            raise RepoTooLargeError(
                "limits",
                f"repository is {info.size_kb} KB, the limit is "
                f"{self._config.max_repo_size_kb} KB",
            )
        return self._deps.repository.submit(
            url=url,
            owner=info.owner,
            name=info.name,
            commit_sha=info.commit_sha,
        )

    def run(self, record: RepoRecord) -> RepoRecord:
        """Run a claimed job; record failures instead of raising them."""
        deps, config = self._deps, self._config
        repo_id = record.repo_id
        collection = collection_name_for(repo_id)
        deadline = _Deadline(config.timeout_seconds)
        workdir = self._make_workdir()
        step = "start"
        try:
            deps.repository.mark_processing(repo_id)
            deps.vector_store.delete_collection(collection)

            step = "download"
            deadline.check(step)
            archive = deps.download_archive(
                record.owner,
                record.name,
                record.commit_sha,
                workdir,
                deadline.remaining(),
            )

            step = "extract"
            deadline.check(step)
            root = deps.extract_archive(archive, workdir / "src")

            step = "filter"
            deadline.check(step)
            paths = list(deps.list_files(root))
            if not paths:
                raise IngestionError(step, "no indexable files found")

            step = "read"
            sources: list[SourceFile] = []
            for path in paths:
                deadline.check(step)
                source = deps.read_file(root, path)
                if source is not None:
                    sources.append(source)
            if not sources:
                raise IngestionError(step, "no readable text files found")

            step = "chunk"
            chunks: list[Chunk] = []
            for source in sources:
                deadline.check(step)
                chunks.extend(deps.chunk_file(source))
                if len(chunks) > config.max_chunks:
                    raise RepoTooLargeError(
                        step, f"more than {config.max_chunks} chunks"
                    )
            if not chunks:
                raise IngestionError(step, "no chunks produced")

            for start in range(0, len(chunks), config.embed_batch_size):
                batch = chunks[start : start + config.embed_batch_size]
                step = "embed"
                deadline.check(step)
                vectors = deps.embedder.embed_documents([chunk.text for chunk in batch])
                if len(vectors) != len(batch):
                    raise ValueError("embedder returned a different number of vectors")

                step = "store"
                deadline.check(step)
                deps.vector_store.add_chunks(
                    collection,
                    ids=[f"c{start + index:07d}" for index in range(len(batch))],
                    texts=[chunk.text for chunk in batch],
                    embeddings=vectors,
                    metadatas=[
                        {
                            "file_path": chunk.file_path,
                            "start_line": chunk.start_line,
                            "end_line": chunk.end_line,
                        }
                        for chunk in batch
                    ],
                )

            step = "finalize"
            done = deps.repository.mark_ready(
                repo_id, files_count=len(sources), chunks_count=len(chunks)
            )
            logger.info(
                "Ingested %s: %d files, %d chunks", repo_id, len(sources), len(chunks)
            )
            return done
        except Exception as exc:  # noqa: BLE001 - every claimed job must be marked failed
            return self._fail(repo_id, collection, step, exc)
        finally:
            _remove_tree(workdir)

    def ingest(self, url: str) -> RepoRecord:
        """Submit and immediately run a job (convenient for scripts and tests)."""
        submission = self.submit(url)
        if not submission.claimed:
            return submission.record
        return self.run(submission.record)

    def _make_workdir(self) -> Path:
        root = self._config.temp_root
        if root is not None:
            root.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix="repo-rag-", dir=root))

    def _fail(
        self, repo_id: str, collection: str, step: str, exc: Exception
    ) -> RepoRecord:
        message = (
            str(exc)
            if isinstance(exc, IngestionError)
            else f"[{step}] {type(exc).__name__}: {exc}"
        )
        logger.error("Ingestion failed for %s: %s", repo_id, message, exc_info=exc)
        try:
            self._deps.vector_store.delete_collection(collection)
        except Exception:
            logger.warning(
                "Could not remove partial collection %s", collection, exc_info=True
            )
        return self._deps.repository.mark_failed(repo_id, message)
