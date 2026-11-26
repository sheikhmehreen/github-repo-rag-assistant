"""End-to-end test of the Epic 2 pipeline using mocked GitHub responses."""

import io
import zipfile

import httpx
import pytest

from app.core.exceptions import InvalidUrlError, NoUsableFilesError
from app.ingestion.github_client import GitHubClient
from app.ingestion.pipeline import ingest_repository

SHA = "b" * 40


def build_zip(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buffer.getvalue()


def make_client(settings, archive: bytes) -> GitHubClient:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/repos/octo/demo":
            return httpx.Response(
                200,
                json={
                    "name": "demo",
                    "size": 10,
                    "default_branch": "main",
                    "private": False,
                    "owner": {"login": "octo"},
                },
            )
        if path == "/repos/octo/demo/commits/main":
            return httpx.Response(200, json={"sha": SHA})
        if path == f"/repos/octo/demo/zipball/{SHA}":
            return httpx.Response(
                302,
                headers={
                    "location": f"https://codeload.github.com/octo/demo/legacy.zip/{SHA}"
                },
            )
        if request.url.host == "codeload.github.com":
            return httpx.Response(200, content=archive)
        return httpx.Response(404)

    return GitHubClient(
        settings, client=httpx.Client(transport=httpx.MockTransport(handler))
    )


def test_full_pipeline_returns_clean_files(make_settings):
    archive = build_zip(
        {
            "demo-bbbb/main.py": b"print('hello')\n",
            "demo-bbbb/README.md": b"# Demo\n",
            "demo-bbbb/.env": b"API_KEY=super-secret\n",
            "demo-bbbb/node_modules/lib/index.js": b"module.exports = 1\n",
            "demo-bbbb/logo.png": b"\x89PNG\x00\x00",
        }
    )
    settings = make_settings()

    result = ingest_repository(
        "https://github.com/octo/demo", settings, make_client(settings, archive)
    )

    assert result.repository.commit_sha == SHA
    assert {f.path for f in result.files} == {"main.py", "README.md"}
    assert all("super-secret" not in f.content for f in result.files)
    assert result.skipped["sensitive"] == 1
    assert result.skipped["excluded_dir"] == 1


def test_invalid_url_fails_before_any_network_call(make_settings):
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("network must not be used")

    settings = make_settings()
    client = GitHubClient(
        settings, client=httpx.Client(transport=httpx.MockTransport(handler))
    )

    with pytest.raises(InvalidUrlError):
        ingest_repository("http://github.com/octo/demo", settings, client)


def test_repository_without_usable_files_raises(make_settings):
    settings = make_settings()
    archive = build_zip({"demo-bbbb/logo.png": b"\x89PNG\x00\x00"})

    with pytest.raises(NoUsableFilesError):
        ingest_repository(
            "https://github.com/octo/demo", settings, make_client(settings, archive)
        )
