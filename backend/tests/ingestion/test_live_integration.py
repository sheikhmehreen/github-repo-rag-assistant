"""Optional end-to-end test against a small public GitHub repository."""

import re

import pytest

from app.core.config import Settings
from app.ingestion.pipeline import ingest_repository


@pytest.mark.integration
def test_ingest_small_public_repository_from_github() -> None:
    """Verify the live GitHub API, archive redirect, extraction, and file reader."""
    settings = Settings(
        _env_file=None,
        app_env="testing",
        debug=False,
        max_repo_size_kb=1_024,
        max_archive_bytes=5 * 1024 * 1024,
        max_extracted_bytes=20 * 1024 * 1024,
        max_files=200,
        max_file_bytes=2 * 1024 * 1024,
        max_compression_ratio=250,
        github_http_timeout_seconds=15,
        download_timeout_seconds=60,
    )

    result = ingest_repository("https://github.com/octocat/Hello-World", settings)

    assert result.repository.owner.lower() == "octocat"
    assert result.repository.repo.lower() == "hello-world"
    assert re.fullmatch(r"[0-9a-f]{40}", result.repository.commit_sha)
    assert result.files
    assert any(
        file.path.rsplit("/", maxsplit=1)[-1].lower().startswith("readme")
        for file in result.files
    )
