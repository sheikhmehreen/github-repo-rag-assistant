"""Tests for app.ingestion.github_client. All HTTP is mocked; no network access."""

import io
import zipfile
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
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
from app.ingestion import github_client
from app.ingestion.github_client import GitHubClient
from pydantic import SecretStr

SHA = "a" * 40
CODELOAD_URL = f"https://codeload.github.com/openai/openai-python/legacy.zip/{SHA}"


def zip_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("repo-abc/main.py", "print(1)")
    return buffer.getvalue()


def repo_json(**overrides) -> dict:
    data = {
        "name": "openai-python",
        "size": 12_000,
        "default_branch": "main",
        "private": False,
        "owner": {"login": "openai"},
    }
    data.update(overrides)
    return data


def make_client(handler, settings) -> GitHubClient:
    return GitHubClient(
        settings, client=httpx.Client(transport=httpx.MockTransport(handler))
    )


def metadata_handler(meta: dict | None = None, commit: dict | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/repos/openai/openai-python":
            return httpx.Response(200, json=meta if meta is not None else repo_json())
        if request.url.path == "/repos/openai/openai-python/commits/main":
            return httpx.Response(
                200, json=commit if commit is not None else {"sha": SHA}
            )
        return httpx.Response(404)

    return handler


# ---------------------------------------------------------------- T2.2 metadata


def test_get_repository_info_success(make_settings):
    client = make_client(metadata_handler(), make_settings())

    info = client.get_repository_info("openai", "openai-python")

    assert info.owner == "openai"
    assert info.repo == "openai-python"
    assert info.size_kb == 12_000
    assert info.default_branch == "main"
    assert info.commit_sha == SHA


def test_branch_names_with_slashes_keep_their_slashes(make_settings):
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.raw_path.decode())
        return httpx.Response(200, json={"sha": SHA})

    make_client(handler, make_settings()).get_latest_commit_sha("o", "r", "feature/x y")

    assert seen == ["/repos/o/r/commits/feature/x%20y"]


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (404, RepositoryNotFoundError),
        (401, GitHubAuthError),
        (403, GitHubApiError),  # forbidden but not a rate limit
        (409, GitHubApiError),  # empty repository
        (301, GitHubApiError),  # moved repository
        (500, GitHubApiError),
    ],
)
def test_error_status_codes_are_mapped(make_settings, status, expected):
    client = make_client(lambda request: httpx.Response(status), make_settings())

    with pytest.raises(expected):
        client.get_repository_info("openai", "openai-python")


def test_not_found_message_is_ambiguous(make_settings):
    client = make_client(lambda request: httpx.Response(404), make_settings())

    with pytest.raises(RepositoryNotFoundError, match="not found or not accessible"):
        client.get_repository_info("openai", "openai-python")


def test_rate_limit_with_reset_time(make_settings):
    headers = {"x-ratelimit-remaining": "0", "x-ratelimit-reset": "1700000000"}
    client = make_client(
        lambda request: httpx.Response(403, headers=headers), make_settings()
    )

    with pytest.raises(GitHubRateLimitError) as excinfo:
        client.get_repository_info("openai", "openai-python")

    assert excinfo.value.reset_at == 1_700_000_000


def test_secondary_rate_limit_uses_retry_after(make_settings):
    client = make_client(
        lambda request: httpx.Response(429, headers={"retry-after": "30"}),
        make_settings(),
    )

    with pytest.raises(GitHubRateLimitError):
        client.get_repository_info("openai", "openai-python")


def test_timeout_is_mapped(make_settings):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    with pytest.raises(GitHubTimeoutError):
        make_client(handler, make_settings()).get_repository_info(
            "openai", "openai-python"
        )


def test_network_error_is_mapped(make_settings):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    with pytest.raises(GitHubApiError):
        make_client(handler, make_settings()).get_repository_info(
            "openai", "openai-python"
        )


def test_oversized_repository_is_rejected(make_settings):
    client = make_client(metadata_handler(repo_json(size=60_000)), make_settings())

    with pytest.raises(RepositoryTooLargeError):
        client.get_repository_info("openai", "openai-python")


def test_private_repository_looks_like_not_found(make_settings):
    client = make_client(metadata_handler(repo_json(private=True)), make_settings())

    with pytest.raises(RepositoryNotFoundError):
        client.get_repository_info("openai", "openai-python")


@pytest.mark.parametrize(
    "meta",
    [
        repo_json(default_branch=None),
        repo_json(default_branch=""),
        repo_json(size="big"),
        repo_json(size=True),
        repo_json(owner={}),
        repo_json(name=None),
    ],
)
def test_invalid_metadata_fields_raise_controlled_error(make_settings, meta):
    client = make_client(metadata_handler(meta), make_settings())

    with pytest.raises(GitHubApiError):
        client.get_repository_info("openai", "openai-python")


@pytest.mark.parametrize(
    "commit", [{"sha": "xyz"}, {}, {"sha": 123}, {"sha": "A" * 40}]
)
def test_invalid_commit_sha_raises_controlled_error(make_settings, commit):
    client = make_client(metadata_handler(commit=commit), make_settings())

    with pytest.raises(GitHubApiError):
        client.get_repository_info("openai", "openai-python")


def test_invalid_json_raises_controlled_error(make_settings):
    client = make_client(
        lambda request: httpx.Response(200, content=b"not json"), make_settings()
    )

    with pytest.raises(GitHubApiError):
        client.get_repository_info("openai", "openai-python")


def test_token_is_sent_when_configured(make_settings):
    captured: dict[str, str | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("authorization")
        captured["ua"] = request.headers.get("user-agent")
        return (
            httpx.Response(200, json=repo_json())
            if "commits" not in request.url.path
            else httpx.Response(200, json={"sha": SHA})
        )

    settings = make_settings(github_token=SecretStr("secret-token"))
    make_client(handler, settings).get_repository_info("openai", "openai-python")

    assert captured["auth"] == "Bearer secret-token"
    assert captured["ua"]


def test_token_is_absent_when_not_configured(make_settings):
    captured: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request.headers.get("authorization"))
        return (
            httpx.Response(200, json=repo_json())
            if "commits" not in request.url.path
            else httpx.Response(200, json={"sha": SHA})
        )

    make_client(handler, make_settings()).get_repository_info("openai", "openai-python")

    assert captured and all(value is None for value in captured)


def test_token_never_appears_in_error_messages(make_settings):
    settings = make_settings(github_token=SecretStr("secret-token"))
    client = make_client(lambda request: httpx.Response(401), settings)

    with pytest.raises(GitHubAuthError) as excinfo:
        client.get_repository_info("openai", "openai-python")

    assert "secret-token" not in str(excinfo.value)


# ------------------------------------------------------------ T2.3 download


def download_handler(body, seen_auth: dict | None = None):
    """API host redirects to codeload; codeload serves ``body`` (bytes or iterator)."""

    def handler(request: httpx.Request) -> httpx.Response:
        if seen_auth is not None:
            seen_auth[request.url.host] = request.headers.get("authorization")
        if request.url.host == "api.github.com" and request.url.path.endswith(
            f"/zipball/{SHA}"
        ):
            return httpx.Response(302, headers={"location": CODELOAD_URL})
        if request.url.host == "codeload.github.com":
            return httpx.Response(200, content=body)
        return httpx.Response(404)

    return handler


def leftovers(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir())


def test_download_saves_zip(tmp_path, make_settings):
    client = make_client(download_handler(zip_bytes()), make_settings())
    destination = tmp_path / "repo.zip"

    result = client.download_repository_archive(
        "openai", "openai-python", SHA, destination
    )

    assert result == destination
    assert zipfile.is_zipfile(destination)
    assert leftovers(tmp_path) == ["repo.zip"]  # no .part file


def test_token_is_sent_to_api_host_only(tmp_path, make_settings):
    seen: dict[str, str | None] = {}
    settings = make_settings(github_token=SecretStr("secret-token"))
    client = make_client(download_handler(zip_bytes(), seen), settings)

    client.download_repository_archive(
        "openai", "openai-python", SHA, tmp_path / "repo.zip"
    )

    assert seen["api.github.com"] == "Bearer secret-token"
    assert seen["codeload.github.com"] is None


def test_download_aborts_when_stream_exceeds_limit(tmp_path, make_settings):
    def chunks():  # generator body: no Content-Length, so the loop must enforce the limit
        for _ in range(10):
            yield b"a" * 100

    client = make_client(
        download_handler(chunks()), make_settings(max_archive_bytes=250)
    )

    with pytest.raises(ArchiveLimitExceededError):
        client.download_repository_archive(
            "openai", "openai-python", SHA, tmp_path / "repo.zip"
        )

    assert leftovers(tmp_path) == []  # partial file removed


def test_download_rejects_declared_content_length_over_limit(tmp_path, make_settings):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.github.com":
            return httpx.Response(302, headers={"location": CODELOAD_URL})
        return httpx.Response(
            200, content=b"x" * 10, headers={"content-length": "999999"}
        )

    client = make_client(handler, make_settings(max_archive_bytes=1000))

    with pytest.raises(ArchiveLimitExceededError):
        client.download_repository_archive(
            "openai", "openai-python", SHA, tmp_path / "repo.zip"
        )

    assert leftovers(tmp_path) == []


def test_download_aborts_when_overall_deadline_passes(
    tmp_path, make_settings, monkeypatch
):
    calls = {"n": 0}

    def fake_monotonic() -> float:
        calls["n"] += 1
        return 0.0 if calls["n"] == 1 else 10_000.0  # first call sets the deadline

    monkeypatch.setattr(
        github_client, "time", SimpleNamespace(monotonic=fake_monotonic)
    )
    client = make_client(
        download_handler(zip_bytes()), make_settings(download_timeout_seconds=5)
    )

    with pytest.raises(GitHubTimeoutError):
        client.download_repository_archive(
            "openai", "openai-python", SHA, tmp_path / "repo.zip"
        )

    assert leftovers(tmp_path) == []


def test_download_read_timeout_is_mapped(tmp_path, make_settings):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(GitHubTimeoutError):
        make_client(handler, make_settings()).download_repository_archive(
            "openai", "openai-python", SHA, tmp_path / "repo.zip"
        )

    assert leftovers(tmp_path) == []


def test_download_404_is_clear_error(tmp_path, make_settings):
    client = make_client(lambda request: httpx.Response(404), make_settings())

    with pytest.raises(RepositoryNotFoundError):
        client.download_repository_archive(
            "openai", "openai-python", SHA, tmp_path / "repo.zip"
        )


def test_download_network_failure_is_mapped(tmp_path, make_settings):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    with pytest.raises(DownloadFailedError):
        make_client(handler, make_settings()).download_repository_archive(
            "openai", "openai-python", SHA, tmp_path / "repo.zip"
        )


def test_partial_download_is_cleaned_up(tmp_path, make_settings):
    def broken_body():
        yield b"PK\x03\x04partial"
        raise httpx.ReadError("connection dropped")

    client = make_client(download_handler(broken_body()), make_settings())

    with pytest.raises(DownloadFailedError):
        client.download_repository_archive(
            "openai", "openai-python", SHA, tmp_path / "repo.zip"
        )

    assert leftovers(tmp_path) == []


@pytest.mark.parametrize(
    "location",
    [
        "https://evil.example.com/repo.zip",  # host not allowlisted
        "http://codeload.github.com/repo.zip",  # downgrade to http
        "https://codeload.github.com.evil.com/repo.zip",  # lookalike host
    ],
)
def test_redirect_to_disallowed_location_is_rejected(tmp_path, make_settings, location):
    client = make_client(
        lambda request: httpx.Response(302, headers={"location": location}),
        make_settings(),
    )

    with pytest.raises(DownloadFailedError):
        client.download_repository_archive(
            "openai", "openai-python", SHA, tmp_path / "repo.zip"
        )

    assert leftovers(tmp_path) == []


def test_too_many_redirects_are_rejected(tmp_path, make_settings):
    client = make_client(
        lambda request: httpx.Response(302, headers={"location": CODELOAD_URL}),
        make_settings(max_redirects=2),
    )

    with pytest.raises(DownloadFailedError):
        client.download_repository_archive(
            "openai", "openai-python", SHA, tmp_path / "repo.zip"
        )


def test_non_zip_body_is_rejected_and_removed(tmp_path, make_settings):
    client = make_client(download_handler(b"definitely not a zip"), make_settings())

    with pytest.raises(DownloadFailedError):
        client.download_repository_archive(
            "openai", "openai-python", SHA, tmp_path / "repo.zip"
        )

    assert leftovers(tmp_path) == []


def test_invalid_commit_sha_is_rejected_before_any_request(tmp_path, make_settings):
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request should be made")

    with pytest.raises(GitHubApiError):
        make_client(handler, make_settings()).download_repository_archive(
            "openai", "openai-python", "main", tmp_path / "repo.zip"
        )
