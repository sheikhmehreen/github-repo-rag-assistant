"""Epic 2 one-shot repository ingestion adapter retained for compatibility."""

from __future__ import annotations

import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from app.core.exceptions import NoUsableFilesError
from app.ingestion.extractor import extract_archive
from app.ingestion.file_filter import filter_repository_files
from app.ingestion.file_reader import RepoFile, read_repository_files
from app.ingestion.github_client import GitHubClient, RepositoryInfo
from app.ingestion.url_validator import validate_github_url

if TYPE_CHECKING:
    from app.core.config import Settings


@dataclass(frozen=True)
class IngestionResult:
    """Repository files and skip counts returned by the Epic 2 ingestion flow."""

    repository: RepositoryInfo
    files: list[RepoFile]
    skipped: dict[str, int]


def ingest_repository(
    url: str, settings: Settings, client: GitHubClient | None = None
) -> IngestionResult:
    """Validate, fetch, download, safely extract, filter, and read a repository."""
    ref = validate_github_url(url)
    owns_client = client is None
    github = client or GitHubClient(settings)
    try:
        info = github.get_repository_info(ref.owner, ref.repo)
        with tempfile.TemporaryDirectory(prefix="repo-ingest-") as work:
            work_dir = Path(work)
            archive = github.download_repository_archive(
                info.owner, info.repo, info.commit_sha, work_dir / "repo.zip"
            )
            extraction = extract_archive(archive, work_dir / "extracted", settings)
            selected = filter_repository_files(extraction.root, settings)
            read = read_repository_files(extraction.root, selected.files, settings)
    finally:
        if owns_client:
            github.close()

    if not read.files:
        raise NoUsableFilesError(
            "The repository contains no files suitable for indexing."
        )
    skipped: Counter[str] = Counter(selected.skipped)
    skipped.update(read.skipped)
    return IngestionResult(repository=info, files=read.files, skipped=dict(skipped))
