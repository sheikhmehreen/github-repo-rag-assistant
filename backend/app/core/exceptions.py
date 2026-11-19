"""Application-specific exceptions.

Exceptions here are plain data. They do not log. The API layer logs once when it handles them.
"""

from typing import Any


class AppError(Exception):
    """Base class for all expected, application-level errors."""

    code: str = "APPLICATION_ERROR"
    default_message: str = "An unexpected application error occurred."
    status_code: int = 500

    def __init__(
        self,
        message: str | None = None,
        *,
        context: dict[str, Any] | None = None,
    ) -> None:
        self.message = message or self.default_message
        self.context = context or {}
        super().__init__(self.message)


class InvalidUrlError(AppError):
    """The submitted value is not a valid GitHub repository URL."""

    code = "INVALID_URL"
    default_message = "Invalid GitHub repository URL."
    status_code = 422


class RepoNotFoundError(AppError):
    """The repository does not exist or is not publicly accessible."""

    code = "REPO_NOT_FOUND"
    default_message = "Repository not found or not accessible."
    status_code = 404


class RepoTooLargeError(AppError):
    """The repository exceeds the configured size limit."""

    code = "REPO_TOO_LARGE"
    default_message = "Repository exceeds the maximum allowed size."
    status_code = 422


class DownloadFailedError(AppError):
    """Downloading the repository archive from GitHub failed."""

    code = "DOWNLOAD_FAILED"
    default_message = "Unable to download the repository."
    status_code = 502


class ExtractionFailedError(AppError):
    """The downloaded archive could not be extracted."""

    code = "EXTRACTION_FAILED"
    default_message = "Unable to extract the repository archive."
    status_code = 500


class IngestionFailedError(AppError):
    """The ingestion pipeline failed after the repository was fetched."""

    code = "INGESTION_FAILED"
    default_message = "Repository ingestion failed. Please try again."
    status_code = 500
