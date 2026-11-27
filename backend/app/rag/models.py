"""Shared value objects for repository indexing and retrieval."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class SourceFile:
    """Text file with a path relative to its repository root."""

    path: str
    content: str


@dataclass(frozen=True, slots=True)
class IndexResult:
    repo_id: str
    files_indexed: int
    files_skipped: int
    chunks_indexed: int
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    content: str
    path: str
    start_line: int
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
