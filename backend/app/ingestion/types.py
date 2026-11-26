"""Plain data contracts shared by ingestion pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RepoInfo:
    owner: str
    name: str
    commit_sha: str
    size_kb: int
    default_branch: str = "main"


@dataclass(frozen=True)
class SourceFile:
    path: str
    content: str


@dataclass(frozen=True)
class Chunk:
    text: str
    file_path: str
    start_line: int
    end_line: int
