"""Repository ingestion record and database schema."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RepoStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


@dataclass(frozen=True)
class RepoRecord:
    repo_id: str
    url: str
    owner: str
    name: str
    commit_sha: str
    status: RepoStatus
    files_count: int
    chunks_count: int
    error: str | None
    created_at: str
    updated_at: str


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS repos (
    repo_id TEXT PRIMARY KEY,
    url TEXT NOT NULL,
    owner TEXT NOT NULL,
    name TEXT NOT NULL,
    commit_sha TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'processing', 'ready', 'failed')),
    files_count INTEGER NOT NULL DEFAULT 0,
    chunks_count INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (owner, name, commit_sha)
);
"""
