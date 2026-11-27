"""Tests for application settings used by Epic 2 ingestion."""

import pytest
from app.core.config import PROJECT_ROOT, Settings
from pydantic import ValidationError


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


def test_epic_four_defaults_include_batched_embeddings_and_storage_paths() -> None:
    settings = Settings(_env_file=None, debug=False)

    assert settings.embedding_batch_size == 64
    assert settings.ingestion_timeout_seconds == 300
    assert settings.ingestion_max_repo_size_kb == 51_200
    assert settings.ingestion_max_chunks == 20_000
    assert settings.chroma_persist_dir == PROJECT_ROOT / "data/chroma"
    assert settings.sqlite_path == PROJECT_ROOT / "data/repos.db"
    assert settings.ingestion_temp_dir == PROJECT_ROOT / "data/tmp"


def test_epic_five_retrieval_and_generation_limits_are_bounded() -> None:
    settings = Settings(_env_file=None, debug=False)

    assert settings.retrieval_top_k == 5
    assert settings.retrieval_min_similarity == 0.29
    assert settings.max_question_chars == 1000
    assert settings.max_prompt_chars == 12_000
    assert settings.llm_provider == "ollama"
    assert settings.ollama_model == "llama3.2:latest"

    with pytest.raises(ValidationError):
        Settings(_env_file=None, retrieval_top_k=21)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, retrieval_min_similarity=1.1)
