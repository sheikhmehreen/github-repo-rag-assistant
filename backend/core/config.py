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
    max_files: int = Field(default=2000, gt=0)
    max_file_size_kb: int = Field(default=500, gt=0)

    # Chunking (units: tokens)
    chunk_size: int = Field(default=200, gt=0)
    chunk_overlap: int = Field(default=30, ge=0)

    # Retrieval
    top_k: int = Field(default=5, gt=0)
    max_question_length: int = Field(default=1000, gt=0)

    # Cosine SIMILARITY
    retrieval_score_threshold: float = Field(default=0.35, ge=-1.0, le=1.0)

    # Timeouts (seconds)
    github_api_timeout_seconds: int = Field(default=15, gt=0)
    repo_download_timeout_seconds: int = Field(default=60, gt=0)
    ingestion_timeout_seconds: int = Field(default=120, gt=0)
    llm_timeout_seconds: int = Field(default=60, gt=0)

    # GitHub Secrets
    github_token: SecretStr | None = Field(default=None)

    # LLM Settings
    llm_provider: str = "ollama"
    llm_model: str = "llama3.2"
    llm_base_url: str = "http://localhost:11434"
    llm_max_output_tokens: int = 512
    llm_max_retries: int = 2

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Storage Paths
    data_dir: Path = Field(default=Path("data"))

    # Field Validators
    @field_validator("github_token", mode="before")
    @classmethod
    def blank_github_token_becomes_none(cls, v: any) -> any:
        """Convert empty environment string inputs into a clean None value."""
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

    @field_validator("data_dir", mode="before")
    @classmethod
    def relative_paths_are_anchored_to_backend(cls, v: any) -> Path:
        """Resolve paths relative to the backend directory, making them absolute."""
        path_val = Path(v)
        if not path_val.is_absolute():
            return BACKEND_DIR / path_val
        return path_val

    # Model Validators
    @model_validator(mode="after")
    def validate_dependencies_and_thresholds(self) -> "Settings":
        """Verify cross-field dependencies and configuration logic."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.repo_download_timeout_seconds > self.ingestion_timeout_seconds:
            raise ValueError("download_timeout cannot exceed ingestion_timeout")
        return self

    # Helper Unit Properties
    @property
    def max_repo_size_bytes(self) -> int:
        """Convert megabytes limit to bytes."""
        return self.max_repo_size_mb * 1024 * 1024

    @property
    def max_file_size_bytes(self) -> int:
        """Convert kilobytes limit to bytes."""
        return self.max_file_size_kb * 1024


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton instance of the application settings."""
    return Settings()
