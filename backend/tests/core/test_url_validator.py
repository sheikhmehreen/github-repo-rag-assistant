"""Compatibility coverage for the Epic 2 GitHub URL validator."""

import pytest

from app.core.exceptions import InvalidUrlError
from app.ingestion.url_validator import GitHubRepo, validate_github_url


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            "https://github.com/openai/openai-python",
            GitHubRepo("openai", "openai-python"),
        ),
        (
            "https://github.com/owner-name/repo.major_minor/",
            GitHubRepo("owner-name", "repo.major_minor"),
        ),
        ("https://github.com/a/b.git", GitHubRepo("a", "b")),
    ],
)
def test_validate_github_url_success(url: str, expected: GitHubRepo) -> None:
    assert validate_github_url(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "http://github.com/openai/openai-python",
        "https://gitlab.com/openai/openai-python",
        "https://github.com/openai/openai-python/tree/main",
        "https://github.com/openai/..",
        "https://user:pass@github.com/openai/openai-python",
        "https://github.com/openai/openai-python?tab=readme",
        "https://github.com/\nopenai/openai-python",
        None,
        123,
    ],
)
def test_validate_github_url_rejects_invalid_values(url: object) -> None:
    with pytest.raises(InvalidUrlError):
        validate_github_url(url)  # type: ignore[arg-type]
