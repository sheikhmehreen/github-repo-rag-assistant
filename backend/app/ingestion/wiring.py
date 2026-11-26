"""Connect Epic 2 ingestion and Epic 3 chunking to the Epic 4 pipeline."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from app.core.config import Settings, get_settings
from app.database.repository import RepoRepository
from app.ingestion.extractor import extract_archive
from app.ingestion.file_filter import filter_repository_files
from app.ingestion.file_reader import read_repository_files
from app.ingestion.github_client import GitHubClient
from app.ingestion.pipeline import IngestionPipeline, PipelineConfig, PipelineDeps
from app.ingestion.types import Chunk, RepoInfo, SourceFile
from app.ingestion.url_validator import validate_github_url
from app.rag.chunker import RepoChunker
from app.retrieval.embedding_service import EmbeddingService
from app.retrieval.vector_store import ChromaVectorStore


def _parse_url(url: str) -> tuple[str, str]:
    parsed = validate_github_url(url)
    return parsed.owner, parsed.repo


def _fetch_metadata(owner: str, name: str) -> RepoInfo:
    settings = get_settings()
    client_settings = settings.model_copy(
        update={"max_repo_size_kb": settings.ingestion_max_repo_size_kb}
    )
    with GitHubClient(client_settings) as client:
        meta = client.get_repository_info(owner, name)
    return RepoInfo(
        owner=meta.owner,
        name=meta.repo,
        commit_sha=meta.commit_sha,
        size_kb=meta.size_kb,
        default_branch=meta.default_branch,
    )


def _download(owner: str, name: str, sha: str, dest_dir: Path, timeout: float) -> Path:
    settings = get_settings()
    bounded_settings = settings.model_copy(
        update={
            "download_timeout_seconds": min(timeout, settings.download_timeout_seconds)
        }
    )
    with GitHubClient(bounded_settings) as client:
        return client.download_repository_archive(
            owner, name, sha, dest_dir / "repo.zip"
        )


def _extract(archive: Path, destination: Path) -> Path:
    return extract_archive(archive, destination, get_settings()).root


def _list_files(root: Path) -> list[Path]:
    return filter_repository_files(root, get_settings()).files


def _read(root: Path, path: Path) -> SourceFile | None:
    result = read_repository_files(root, [path], get_settings())
    if not result.files:
        return None
    repo_file = result.files[0]
    return SourceFile(path=repo_file.path, content=repo_file.content)


@lru_cache(maxsize=1)
def _chunker() -> RepoChunker:
    settings = get_settings()
    return RepoChunker(settings.chunk_size, settings.chunk_overlap)


def _chunk(source: SourceFile) -> list[Chunk]:
    documents = _chunker().chunk_file("pending", source)
    chunks: list[Chunk] = []
    for document in documents:
        first_line = document.page_content.partition("\n")[2]
        line_count = len(first_line.splitlines())
        start_line = int(document.metadata["start_line"])
        chunks.append(
            Chunk(
                text=document.page_content,
                file_path=source.path,
                start_line=start_line,
                end_line=start_line + max(line_count - 1, 0),
            )
        )
    return chunks


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    settings = get_settings()
    return EmbeddingService(
        model_name=settings.embedding_model_name,
        batch_size=settings.embedding_batch_size,
    )


def warm_up_embeddings() -> None:
    """Load the shared embedding model during application startup."""
    get_embedding_service().warm_up()


@lru_cache(maxsize=1)
def get_vector_store() -> ChromaVectorStore:
    return ChromaVectorStore(get_settings().chroma_persist_dir)


@lru_cache(maxsize=1)
def get_repository() -> RepoRepository:
    settings = get_settings()
    return RepoRepository(
        settings.sqlite_path,
        stale_after_seconds=settings.ingestion_timeout_seconds * 2,
    )


def build_pipeline() -> IngestionPipeline:
    settings: Settings = get_settings()
    deps = PipelineDeps(
        parse_url=_parse_url,
        fetch_metadata=_fetch_metadata,
        download_archive=_download,
        extract_archive=_extract,
        list_files=_list_files,
        read_file=_read,
        chunk_file=_chunk,
        embedder=get_embedding_service(),
        vector_store=get_vector_store(),
        repository=get_repository(),
    )
    config = PipelineConfig(
        timeout_seconds=settings.ingestion_timeout_seconds,
        max_repo_size_kb=settings.ingestion_max_repo_size_kb,
        max_chunks=settings.ingestion_max_chunks,
        embed_batch_size=settings.embedding_batch_size,
        temp_root=settings.ingestion_temp_dir,
    )
    return IngestionPipeline(deps, config)
