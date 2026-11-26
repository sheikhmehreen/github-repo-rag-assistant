import pytest
from app.core.exceptions import (
    AppError,
    DownloadFailedError,
    ExtractionFailedError,
    IngestionFailedError,
    InvalidUrlError,
    RepoNotFoundError,
    RepoTooLargeError,
)

CASES = [
    (InvalidUrlError, "INVALID_URL", "Invalid GitHub repository URL.", 422),
    (
        RepoNotFoundError,
        "REPO_NOT_FOUND",
        "Repository not found or not accessible.",
        404,
    ),
    (
        RepoTooLargeError,
        "REPO_TOO_LARGE",
        "Repository exceeds the maximum allowed size.",
        422,
    ),
    (DownloadFailedError, "DOWNLOAD_FAILED", "Unable to download the repository.", 502),
    (
        ExtractionFailedError,
        "EXTRACTION_FAILED",
        "Unable to extract the repository archive.",
        500,
    ),
    (
        IngestionFailedError,
        "INGESTION_FAILED",
        "Repository ingestion failed. Please try again.",
        500,
    ),
]


@pytest.mark.parametrize(("error_class", "code", "message", "status"), CASES)
def test_error_defaults(error_class, code, message, status):
    error = error_class()
    assert error.code == code
    assert error.message == message
    assert str(error) == message
    assert error.status_code == status
    assert isinstance(error, AppError)


def test_error_codes_are_unique():
    codes = [case[1] for case in CASES]
    assert len(codes) == len(set(codes))


def test_custom_message_and_context_are_kept():
    error = RepoNotFoundError(
        "No such repo.", context={"url": "https://github.com/u/p"}
    )
    assert str(error) == "No such repo."
    assert error.context == {"url": "https://github.com/u/p"}


def test_raising_and_catching_via_base_class():
    with pytest.raises(AppError):
        raise InvalidUrlError()
