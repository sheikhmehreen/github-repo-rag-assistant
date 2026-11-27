"""Adapters from the Epic 2 ingestion models to the RAG source model."""

from collections.abc import Iterable

from app.ingestion.file_reader import RepoFile
from app.rag.models import SourceFile


def to_source_files(repo_files: Iterable[RepoFile]) -> list[SourceFile]:
    """Convert ingested repository files to the RAG layer's input type."""
    return [SourceFile(path=file.path, content=file.content) for file in repo_files]
