"""Strict validation of GitHub repository URLs (Epic 2, T2.1).

Principle: validate, never repair. A URL is either accepted exactly as given
(apart from one optional trailing slash and one optional ``.git`` suffix)
or rejected with ``InvalidUrlError``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlsplit

from app.core.exceptions import InvalidUrlError

MAX_URL_LENGTH = 2048
ALLOWED_HOST = "github.com"

# GitHub usernames/orgs: 1-39 chars, alphanumeric or hyphen, no leading/trailing hyphen.
_OWNER_RE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?")
# Repository names: 1-100 chars from this ASCII allowlist ("." and ".." rejected separately).
_REPO_RE = re.compile(r"[A-Za-z0-9._-]{1,100}")
# Anything that is not printable, non-space ASCII (whitespace, control chars, Unicode).
_DISALLOWED_CHARS_RE = re.compile(r"[^\x21-\x7e]")


@dataclass(frozen=True)
class GitHubRepo:
    """A validated GitHub repository reference."""

    owner: str
    repo: str


def validate_github_url(url: str) -> GitHubRepo:
    """Validate a GitHub repository URL and return its owner and repo name.

    Raises:
        InvalidUrlError: if the URL is not a plain https://github.com/owner/repo URL.
    """
    if not isinstance(url, str) or not url:
        raise InvalidUrlError("URL must be a non-empty string.")
    if len(url) > MAX_URL_LENGTH:
        raise InvalidUrlError("URL is too long.")

    # Check the RAW string before parsing: urlsplit() silently strips tabs/newlines.
    if _DISALLOWED_CHARS_RE.search(url):
        raise InvalidUrlError(
            "URL contains whitespace, control or non-ASCII characters."
        )
    if "\\" in url:
        raise InvalidUrlError("URL must not contain backslashes.")
    if "?" in url or "#" in url:
        raise InvalidUrlError("URL must not contain a query string or fragment.")
    if "%" in url:
        raise InvalidUrlError("URL must not contain percent-encoded characters.")

    try:
        parts = urlsplit(url)
    except ValueError as exc:
        raise InvalidUrlError("URL is malformed.") from exc

    if parts.scheme != "https":
        raise InvalidUrlError("URL must use https.")

    # Exact comparison also rejects credentials (user:pass@), ports and
    # lookalikes such as github.com.attacker.com or www.github.com.
    if parts.netloc != ALLOWED_HOST:
        raise InvalidUrlError("URL host must be exactly github.com.")

    path = parts.path.removeprefix("/")
    path = path.removesuffix("/")  # one optional trailing slash
    segments = path.split("/")
    if len(segments) != 2 or not all(segments):
        raise InvalidUrlError("URL must be https://github.com/<owner>/<repo>.")

    owner, repo = segments
    repo = repo.removesuffix(".git")  # one optional .git suffix

    if not _OWNER_RE.fullmatch(owner):
        raise InvalidUrlError("Invalid repository owner name.")
    if repo in {"", ".", ".."} or not _REPO_RE.fullmatch(repo):
        raise InvalidUrlError("Invalid repository name.")

    return GitHubRepo(owner=owner, repo=repo)
