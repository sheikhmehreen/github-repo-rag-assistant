import logging

import pytest
from fastapi.testclient import TestClient

from app.core.exceptions import (
    DownloadFailedError,
    IngestionFailedError,
    InvalidUrlError,
    RepoNotFoundError,
)
from app.main import create_app


def build_client(error_class) -> TestClient:
    application = create_app()

    @application.get("/boom")
    async def boom() -> None:
        raise error_class(context={"url": "https://github.com"})

    return TestClient(application)


@pytest.mark.parametrize(
    ("error_class", "status", "code"),
    [
        (InvalidUrlError, 422, "INVALID_URL"),
        (RepoNotFoundError, 404, "REPO_NOT_FOUND"),
        (DownloadFailedError, 502, "DOWNLOAD_FAILED"),
        (IngestionFailedError, 500, "INGESTION_FAILED"),
    ],
)
def test_app_error_becomes_structured_response(error_class, status, code):
    response = build_client(error_class).get("/boom")
    assert response.status_code == status
    body = response.json()["error"]
    assert body["code"] == code
    assert body["message"]
    assert "url" not in response.text  # context must not leak


def test_client_error_logs_single_warning(caplog):
    with caplog.at_level(logging.INFO):
        build_client(RepoNotFoundError).get("/boom")
    records = [
        r for r in caplog.records if getattr(r, "event", None) == "application_error"
    ]
    assert len(records) == 1
    assert records[0].levelname == "WARNING"
    assert records[0].error_code == "REPO_NOT_FOUND"


def test_server_error_logs_single_error_with_traceback(caplog):
    with caplog.at_level(logging.INFO):
        build_client(IngestionFailedError).get("/boom")
    records = [
        r for r in caplog.records if getattr(r, "event", None) == "application_error"
    ]
    assert len(records) == 1
    assert records[0].levelname == "ERROR"
    assert records[0].exc_info is not None
