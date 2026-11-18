import pytest
from pydantic import ValidationError

from core.config import Settings, get_settings


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    """Remove every setting's env var so the host machine cannot affect tests."""
    for field_name in Settings.model_fields:
        monkeypatch.delenv(field_name.upper(), raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_defaults():
    config = Settings(_env_file=None)
    assert config.max_repo_size_mb == 50
    assert config.max_files == 2000
    assert config.max_file_size_kb == 500
    assert config.chunk_size == 200
    assert config.chunk_overlap == 30
    assert config.top_k == 5
    assert config.llm_timeout_seconds == 60
    assert config.ingestion_timeout_seconds == 120
    assert config.llm_provider == "ollama"
    assert config.llm_model == "llama3.2"
    assert config.github_token is None


def test_environment_variables_override_defaults(monkeypatch):
    monkeypatch.setenv("MAX_FILES", "3000")
    monkeypatch.setenv("DEBUG", "true")
    monkeypatch.setenv("LLM_MODEL", "custom-model")
    config = Settings(_env_file=None)
    assert config.max_files == 3000  # str converted to int
    assert config.debug is True  # str converted to bool
    assert config.llm_model == "custom-model"


def test_env_file_is_loaded(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("MAX_FILES=3500\nLLM_MODEL=from-file\n", encoding="utf-8")
    config = Settings(_env_file=env_file)
    assert config.max_files == 3500
    assert config.llm_model == "from-file"


def test_environment_beats_env_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("MAX_FILES=3500\n", encoding="utf-8")
    monkeypatch.setenv("MAX_FILES", "4000")
    assert Settings(_env_file=env_file).max_files == 4000


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("max_files", -10),
        ("top_k", 0),
        ("chunk_size", 0),
        ("retrieval_score_threshold", 1.5),
        ("app_env", "prodution"),
    ],
)
def test_invalid_values_are_rejected(name, value):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{name: value})


def test_non_numeric_env_value_is_rejected(monkeypatch):
    monkeypatch.setenv("MAX_FILES", "many")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_chunk_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, chunk_size=100, chunk_overlap=100)


def test_download_timeout_cannot_exceed_ingestion_timeout():
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            repo_download_timeout_seconds=300,
            ingestion_timeout_seconds=120,
        )


def test_blank_github_token_becomes_none(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "")
    assert Settings(_env_file=None).github_token is None


def test_github_token_is_not_exposed_in_repr():
    config = Settings(_env_file=None, github_token="ghp_example_secret")
    assert "ghp_example_secret" not in repr(config)
    assert config.github_token.get_secret_value() == "ghp_example_secret"


def test_relative_paths_are_anchored_to_backend():
    config = Settings(_env_file=None, data_dir="data")
    assert config.data_dir.is_absolute()
    assert config.data_dir.name == "data"


def test_unit_helpers():
    config = Settings(_env_file=None)
    assert config.max_repo_size_bytes == 50 * 1024 * 1024
    assert config.max_file_size_bytes == 500 * 1024


def test_get_settings_is_cached():
    assert get_settings() is get_settings()
