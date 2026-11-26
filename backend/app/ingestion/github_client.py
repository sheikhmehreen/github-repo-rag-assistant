"""GitHub API client (Epic 2, T2.2 and T2.3).

* ``get_repository_info`` - repository metadata plus the latest commit SHA.
* ``download_repository_archive`` - bounded, streamed ZIP download pinned to a commit.

Security notes: redirects are followed manually against a host allowlist, the token
is sent only to the API host, and the download is bounded by bytes and wall-clock time.
"""

from __future__ import annotations

import logging
import re
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self
from urllib.parse import quote, urljoin, urlsplit

import httpx

from app.core.exceptions import (
    ArchiveLimitExceededError,
    DownloadFailedError,
    GitHubApiError,
    GitHubAuthError,
    GitHubRateLimitError,
    GitHubTimeoutError,
    RepositoryNotFoundError,
    RepositoryTooLargeError,
)

if TYPE_CHECKING:
    from app.core.config import Settings

logger = logging.getLogger(__name__)

_API_VERSION = "2022-11-28"
_USER_AGENT = "github-repo-rag-assistant"
_SHA_RE = re.compile(r"[0-9a-f]{40}")
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_NOT_FOUND_MESSAGE = "Repository not found or not accessible."


@dataclass(frozen=True)
class RepositoryInfo:
    """Repository facts needed by the ingestion pipeline."""

    owner: str
    repo: str
    size_kb: int
    default_branch: str
    commit_sha: str


class GitHubClient:
    """Small synchronous GitHub client. Create once and reuse; call ``close()`` when done."""

    def __init__(self, settings: Settings, client: httpx.Client | None = None) -> None:
        self._settings = settings
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=settings.github_http_timeout_seconds
        )
        self._api_host = urlsplit(settings.github_api_base_url).hostname or ""

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying HTTP client if this instance created it."""
        if self._owns_client:
            self._client.close()

    # ------------------------------------------------------------------ T2.2

    def get_repository_metadata(self, owner: str, repo: str) -> dict[str, Any]:
        """API call 1: ``GET /repos/{owner}/{repo}``."""
        url = self._api_url(f"/repos/{quote(owner, safe='')}/{quote(repo, safe='')}")
        data = self._get_json(url)
        if not isinstance(data, dict):
            raise GitHubApiError("GitHub returned an unexpected metadata response.")
        return data

    def get_latest_commit_sha(self, owner: str, repo: str, branch: str) -> str:
        """API call 2: ``GET /repos/{owner}/{repo}/commits/{branch}``."""
        if not branch or ".." in branch.split("/"):
            raise GitHubApiError("GitHub returned an invalid default branch.")
        url = self._api_url(
            f"/repos/{quote(owner, safe='')}/{quote(repo, safe='')}/commits/{quote(branch, safe='/')}"
        )
        data = self._get_json(url)
        sha = data.get("sha") if isinstance(data, dict) else None
        if not isinstance(sha, str) or not _SHA_RE.fullmatch(sha):
            raise GitHubApiError("GitHub returned an invalid commit SHA.")
        return sha

    def get_repository_info(self, owner: str, repo: str) -> RepositoryInfo:
        """Fetch metadata and latest commit, enforcing public-only and size limits."""
        meta = self.get_repository_metadata(owner, repo)

        if meta.get("private") is True:
            raise RepositoryNotFoundError(
                _NOT_FOUND_MESSAGE
            )  # never reveal private repos

        size_kb = meta.get("size")
        default_branch = meta.get("default_branch")
        name = meta.get("name")
        owner_data = meta.get("owner")
        login = owner_data.get("login") if isinstance(owner_data, dict) else None

        if isinstance(size_kb, bool) or not isinstance(size_kb, int) or size_kb < 0:
            raise GitHubApiError("GitHub response is missing a valid repository size.")
        if not isinstance(default_branch, str) or not default_branch:
            raise GitHubApiError("GitHub response is missing the default branch.")
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(login, str)
            or not login
        ):
            raise GitHubApiError(
                "GitHub response is missing the repository name or owner."
            )

        if size_kb > self._settings.max_repo_size_kb:
            raise RepositoryTooLargeError(
                f"Repository is larger than the allowed {self._settings.max_repo_size_kb} KB."
            )

        sha = self.get_latest_commit_sha(login, name, default_branch)
        return RepositoryInfo(
            owner=login,
            repo=name,
            size_kb=size_kb,
            default_branch=default_branch,
            commit_sha=sha,
        )

    # ------------------------------------------------------------------ T2.3

    def download_repository_archive(
        self, owner: str, repo: str, commit_sha: str, destination: Path
    ) -> Path:
        """Stream the ZIP for ``commit_sha`` to ``destination`` within configured limits.

        Raises:
            ArchiveLimitExceededError: more bytes than ``max_archive_bytes``.
            GitHubTimeoutError: the overall deadline or a request timeout was hit.
            DownloadFailedError: network error, disallowed redirect or non-ZIP body.
            RepositoryNotFoundError / GitHub*Error: mapped from the HTTP status.
        """
        if not _SHA_RE.fullmatch(commit_sha):
            raise GitHubApiError("Invalid commit SHA.")

        destination.parent.mkdir(parents=True, exist_ok=True)
        part_path = destination.with_name(destination.name + ".part")
        deadline = time.monotonic() + self._settings.download_timeout_seconds
        url = self._api_url(
            f"/repos/{quote(owner, safe='')}/{quote(repo, safe='')}/zipball/{commit_sha}"
        )

        try:
            self._stream_to_file(url, part_path, deadline)
            if not zipfile.is_zipfile(part_path):
                raise DownloadFailedError("Downloaded file is not a valid ZIP archive.")
            part_path.replace(destination)
        except BaseException:
            part_path.unlink(missing_ok=True)  # never leave partial downloads behind
            raise
        return destination

    # --------------------------------------------------------------- internals

    def _api_url(self, path: str) -> str:
        return self._settings.github_api_base_url.rstrip("/") + path

    def _headers_for(self, url: str) -> dict[str, str]:
        headers = {"Accept": "application/vnd.github+json", "User-Agent": _USER_AGENT}
        if urlsplit(url).hostname == self._api_host:
            headers["X-GitHub-Api-Version"] = _API_VERSION
            token = self._settings.github_token
            if token is not None:  # token goes to the API host only, never to redirects
                headers["Authorization"] = f"Bearer {token.get_secret_value()}"
        return headers

    def _check_url_allowed(self, url: str) -> None:
        parts = urlsplit(url)
        if (
            parts.scheme != "https"
            or parts.hostname not in self._settings.allowed_redirect_hosts
        ):
            raise DownloadFailedError("Download redirected to a disallowed location.")

    def _get_json(self, url: str) -> Any:
        try:
            response = self._client.get(
                url, headers=self._headers_for(url), follow_redirects=False
            )
        except httpx.TimeoutException as exc:
            raise GitHubTimeoutError("The GitHub API request timed out.") from exc
        except httpx.HTTPError as exc:
            raise GitHubApiError("Could not reach the GitHub API.") from exc

        self._raise_for_status(response)
        try:
            return response.json()
        except ValueError as exc:
            raise GitHubApiError("GitHub returned invalid JSON.") from exc

    def _raise_for_status(self, response: httpx.Response) -> None:
        status = response.status_code
        if status == 200:
            return
        remaining = response.headers.get("x-ratelimit-remaining")
        logger.debug("GitHub HTTP %s (rate limit remaining: %s)", status, remaining)

        if status == 401:
            raise GitHubAuthError("GitHub rejected the configured token.")
        if status in (403, 429):
            if remaining == "0" or "retry-after" in response.headers:
                reset = response.headers.get("x-ratelimit-reset", "")
                raise GitHubRateLimitError(
                    "GitHub API rate limit exceeded.",
                    int(reset) if reset.isdigit() else None,
                )
            raise GitHubApiError("GitHub denied the request.")
        if status == 404:
            raise RepositoryNotFoundError(_NOT_FOUND_MESSAGE)
        if status == 409:
            raise GitHubApiError("Repository is empty or in an unsupported state.")
        if status in _REDIRECT_STATUSES:
            raise GitHubApiError("Repository has moved; please use its current URL.")
        raise GitHubApiError(f"Unexpected GitHub API response (HTTP {status}).")

    def _stream_to_file(self, url: str, part_path: Path, deadline: float) -> None:
        current = url
        for _ in range(self._settings.max_redirects + 1):
            self._check_url_allowed(current)
            try:
                with self._client.stream(
                    "GET",
                    current,
                    headers=self._headers_for(current),
                    follow_redirects=False,
                ) as response:
                    if response.status_code in _REDIRECT_STATUSES:
                        location = response.headers.get("location")
                        if not location:
                            raise DownloadFailedError(
                                "Redirect response had no location."
                            )
                        current = urljoin(current, location)
                        continue
                    self._raise_for_status(response)
                    self._write_body(response, part_path, deadline)
                    return
            except httpx.TimeoutException as exc:
                raise GitHubTimeoutError("Repository download timed out.") from exc
            except httpx.HTTPError as exc:
                raise DownloadFailedError(
                    "Network error while downloading the repository."
                ) from exc
        raise DownloadFailedError(
            "Too many redirects while downloading the repository."
        )

    def _write_body(
        self, response: httpx.Response, part_path: Path, deadline: float
    ) -> None:
        limit = self._settings.max_archive_bytes
        declared = response.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > limit:
            raise ArchiveLimitExceededError(
                "Repository archive exceeds the allowed download size."
            )

        total = 0
        with part_path.open("wb") as handle:
            for chunk in response.iter_bytes(
                chunk_size=self._settings.download_chunk_size
            ):
                total += len(chunk)
                if total > limit:
                    raise ArchiveLimitExceededError(
                        "Repository archive exceeds the allowed download size."
                    )
                if (
                    time.monotonic() > deadline
                ):  # overall deadline, checked on every chunk
                    raise GitHubTimeoutError("Repository download timed out.")
                handle.write(chunk)
