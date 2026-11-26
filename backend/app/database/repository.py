"""SQLite repository records with atomic job claims and retry handling."""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from app.database.models import SCHEMA_SQL, RepoRecord, RepoStatus

_MAX_ERROR_LENGTH = 1000


class RepoNotFoundError(LookupError):
    """The requested repository record does not exist."""


@dataclass(frozen=True)
class SubmitResult:
    record: RepoRecord
    claimed: bool


def _now() -> datetime:
    return datetime.now(UTC)


def _row_to_record(row: sqlite3.Row) -> RepoRecord:
    return RepoRecord(
        repo_id=row["repo_id"],
        url=row["url"],
        owner=row["owner"],
        name=row["name"],
        commit_sha=row["commit_sha"],
        status=RepoStatus(row["status"]),
        files_count=row["files_count"],
        chunks_count=row["chunks_count"],
        error=row["error"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


class RepoRepository:
    """Persist and deduplicate repository ingestion jobs in SQLite."""

    def __init__(self, db_path: Path | str, stale_after_seconds: int = 1800) -> None:
        if stale_after_seconds < 0:
            raise ValueError("stale_after_seconds must be non-negative")
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._stale_after = timedelta(seconds=stale_after_seconds)
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(SCHEMA_SQL)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self._db_path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except BaseException:
                connection.execute("ROLLBACK")
                raise
            else:
                connection.execute("COMMIT")

    @staticmethod
    def _fetch(connection: sqlite3.Connection, repo_id: str) -> RepoRecord:
        row = connection.execute(
            "SELECT * FROM repos WHERE repo_id = ?", (repo_id,)
        ).fetchone()
        if row is None:
            raise RepoNotFoundError(repo_id)
        return _row_to_record(row)

    def get(self, repo_id: str) -> RepoRecord:
        with self._connection() as connection:
            return self._fetch(connection, repo_id)

    def submit(
        self, *, url: str, owner: str, name: str, commit_sha: str
    ) -> SubmitResult:
        """Atomically claim a new or retryable job; otherwise return its status."""
        owner, name = owner.lower(), name.lower()
        now = _now()
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM repos WHERE owner = ? AND name = ? AND commit_sha = ?",
                (owner, name, commit_sha),
            ).fetchone()
            if row is None:
                repo_id = uuid.uuid4().hex
                stamp = now.isoformat()
                connection.execute(
                    "INSERT INTO repos "
                    "(repo_id, url, owner, name, commit_sha, status, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        repo_id,
                        url,
                        owner,
                        name,
                        commit_sha,
                        RepoStatus.PENDING.value,
                        stamp,
                        stamp,
                    ),
                )
                return SubmitResult(self._fetch(connection, repo_id), claimed=True)

            record = _row_to_record(row)
            if self._can_retry(record, now):
                connection.execute(
                    "UPDATE repos SET status = ?, error = NULL, files_count = 0, "
                    "chunks_count = 0, updated_at = ? WHERE repo_id = ?",
                    (RepoStatus.PENDING.value, now.isoformat(), record.repo_id),
                )
                return SubmitResult(
                    self._fetch(connection, record.repo_id), claimed=True
                )
            return SubmitResult(record, claimed=False)

    def _can_retry(self, record: RepoRecord, now: datetime) -> bool:
        if record.status is RepoStatus.FAILED:
            return True
        if record.status in (RepoStatus.PENDING, RepoStatus.PROCESSING):
            updated = datetime.fromisoformat(record.updated_at)
            return now - updated > self._stale_after
        return False

    def mark_processing(self, repo_id: str) -> RepoRecord:
        return self._set_status(repo_id, RepoStatus.PROCESSING)

    def mark_ready(
        self, repo_id: str, *, files_count: int, chunks_count: int
    ) -> RepoRecord:
        return self._set_status(
            repo_id,
            RepoStatus.READY,
            files_count=files_count,
            chunks_count=chunks_count,
        )

    def mark_failed(self, repo_id: str, error: str) -> RepoRecord:
        return self._set_status(repo_id, RepoStatus.FAILED, error=error)

    def _set_status(
        self,
        repo_id: str,
        status: RepoStatus,
        *,
        files_count: int | None = None,
        chunks_count: int | None = None,
        error: str | None = None,
    ) -> RepoRecord:
        stored_error = error[:_MAX_ERROR_LENGTH] if error else None
        with self._transaction() as connection:
            cursor = connection.execute(
                "UPDATE repos SET status = ?, error = ?, "
                "files_count = COALESCE(?, files_count), "
                "chunks_count = COALESCE(?, chunks_count), updated_at = ? "
                "WHERE repo_id = ?",
                (
                    status.value,
                    stored_error,
                    files_count,
                    chunks_count,
                    _now().isoformat(),
                    repo_id,
                ),
            )
            if cursor.rowcount == 0:
                raise RepoNotFoundError(repo_id)
            return self._fetch(connection, repo_id)
