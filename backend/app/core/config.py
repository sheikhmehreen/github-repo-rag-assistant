"""Central application configuration.

Precedence (highest first): init arguments, environment variables, the project-root .env file, then the defaults declared below.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "GitHub Repo RAG Assistant"
    app_env: Literal["development", "testing", "staging", "production"] = "development"
    debug: bool = False

    # Repository limits
    max_repo_size_mb: int = Field(default=50, gt=0)
    # Epic 2 ingestion limits. Keep the KB value explicit because GitHub's API
    # reports repository size in KB; these settings can all be overridden by env.
    max_repo_size_kb: int = Field(default=51_200, gt=0)
    max_archive_bytes: int = Field(default=52_428_800, gt=0)
    max_extracted_bytes: int = Field(default=209_715_200, gt=0)
    max_files: int = Field(default=2000, gt=0)
    max_file_size_kb: int = Field(default=500, gt=0)
    max_file_bytes: int = Field(default=1_048_576, gt=0)
    max_compression_ratio: float = Field(default=100.0, gt=0)
    download_chunk_size: int = Field(default=65_536, gt=0)
    max_redirects: int = Field(default=3, ge=0)
    allowed_redirect_hosts: set[str] = Field(
        default_factory=lambda: {"api.github.com", "codeload.github.com"}
    )

    # Chunking (units: characters)
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=150, ge=0)
    chroma_persist_dir: Path = Field(default=Path("data/chroma"))
    max_chunks_per_repo: int = Field(default=5000, gt=0)

    # Retrieval
    retrieval_top_k: int = Field(default=5, gt=0)
    max_question_length: int = Field(default=1000, gt=0)

    # Cosine SIMILARITY
    retrieval_score_threshold: float = Field(default=0.35, ge=-1.0, le=1.0)

    # Timeouts (seconds)
    github_api_timeout_seconds: int = Field(default=15, gt=0)
    github_http_timeout_seconds: float = Field(default=15.0, gt=0)
    repo_download_timeout_seconds: int = Field(default=60, gt=0)
    download_timeout_seconds: float = Field(default=60.0, gt=0)
    ingestion_timeout_seconds: int = Field(default=300, gt=0)
    llm_timeout_seconds: int = Field(default=60, gt=0)

    # GitHub Secrets
    github_token: SecretStr | None = Field(default=None)
    github_api_base_url: str = "https://api.github.com"

    # LLM Settings
    llm_provider: str = "ollama"
    llm_model: str = "llama3.2"
    llm_base_url: str = "http://localhost:11434"
    llm_max_output_tokens: int = 512
    llm_max_retries: int = 2

    # Embeddings
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = Field(default=64, gt=0)
    sqlite_path: Path = Field(default=Path("data/repos.db"))
    ingestion_max_repo_size_kb: int = Field(default=51_200, gt=0)
    ingestion_max_chunks: int = Field(default=20_000, gt=0)
    ingestion_temp_dir: Path = Field(default=Path("data/tmp"))

    # Storage Paths
    data_dir: Path = Field(default=Path("data"))

    # Field Validators
    @field_validator("github_token", mode="before")
    @classmethod
    def blank_github_token_becomes_none(cls, v: any) -> any:
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

    @field_validator("data_dir", mode="before")
    @classmethod
    def relative_paths_are_anchored_to_backend(cls, v: any) -> Path:
        path_val = Path(v)
        if not path_val.is_absolute():
            return BACKEND_DIR / path_val
        return path_val

    @field_validator(
        "chroma_persist_dir", "sqlite_path", "ingestion_temp_dir", mode="before"
    )
    @classmethod
    def relative_storage_path_is_anchored_to_backend(cls, v: any) -> Path:
        path_val = Path(v)
        if not path_val.is_absolute():
            return PROJECT_ROOT / path_val
        return path_val

    # Model Validators
    @model_validator(mode="after")
    def validate_dependencies_and_thresholds(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.repo_download_timeout_seconds > self.ingestion_timeout_seconds:
            raise ValueError("download_timeout cannot exceed ingestion_timeout")
        return self

    # Helper Unit Properties
    @property
    def max_repo_size_bytes(self) -> int:
        return self.max_repo_size_mb * 1024 * 1024

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_kb * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
