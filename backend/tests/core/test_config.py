"""Tests for application settings used by Epic 2 ingestion."""

from app.core.config import Settings


def test_epic_two_limits_have_bounded_defaults() -> None:
    settings = Settings(_env_file=None, debug=False)

    assert settings.github_api_base_url == "https://api.github.com"
    assert settings.max_repo_size_kb == 51_200
    assert settings.max_archive_bytes > 0
    assert settings.max_extracted_bytes > 0
    assert settings.max_files == 2_000
    assert settings.max_file_bytes > 0
    assert settings.max_compression_ratio > 0
    assert settings.allowed_redirect_hosts == {"api.github.com", "codeload.github.com"}


def test_epic_two_limits_can_be_overridden() -> None:
    settings = Settings(
        _env_file=None,
        debug=False,
        max_repo_size_kb=1_000,
        max_archive_bytes=2_000,
        max_extracted_bytes=3_000,
        max_file_bytes=400,
        max_files=5,
        max_compression_ratio=20,
        allowed_redirect_hosts={"api.example.test"},
    )

    assert settings.max_repo_size_kb == 1_000
    assert settings.max_archive_bytes == 2_000
    assert settings.max_extracted_bytes == 3_000
    assert settings.max_file_bytes == 400
    assert settings.max_files == 5
    assert settings.max_compression_ratio == 20
    assert settings.allowed_redirect_hosts == {"api.example.test"}
