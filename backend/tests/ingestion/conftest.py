"""Shared fixtures for ingestion tests.

``make_settings`` builds a lightweight settings object containing only the Epic 2
fields. This keeps the tests independent of any other required Epic 1 settings.
"""

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest

_DEFAULTS: dict[str, Any] = {
    "github_token": None,
    "github_api_base_url": "https://api.github.com",
    "github_http_timeout_seconds": 5.0,
    "max_repo_size_kb": 51_200,
    "max_archive_bytes": 52_428_800,
    "download_timeout_seconds": 60.0,
    "download_chunk_size": 1024,
    "max_redirects": 3,
    "allowed_redirect_hosts": ("api.github.com", "codeload.github.com"),
    "max_extracted_bytes": 209_715_200,
    "max_file_bytes": 1_048_576,
    "max_files": 2_000,
    "max_compression_ratio": 100.0,
}


@pytest.fixture
def make_settings() -> Callable[..., SimpleNamespace]:
    """Return a factory: ``make_settings(max_files=2)`` overrides defaults."""

    def _factory(**overrides: Any) -> SimpleNamespace:
        return SimpleNamespace(**{**_DEFAULTS, **overrides})

    return _factory
