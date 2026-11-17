"""Smoke tests for the backend Python environment."""

import sys
from importlib.util import find_spec

REQUIRED_MODULES: tuple[str, ...] = (
    "fastapi",
    "uvicorn",
    "pydantic",
    "pydantic_settings",
    "httpx",
    "sentence_transformers",
    "chromadb",
    "langchain",
    "langchain_text_splitters",
    "pytest",
    "ruff",
)


def test_python_version_is_3_12() -> None:
    """The project targets Python 3.12."""
    assert sys.version_info[:2] == (3, 12)


def test_running_inside_virtual_environment() -> None:
    """Tests must run inside the project's virtual environment."""
    assert sys.prefix != sys.base_prefix


def test_required_dependencies_are_installed() -> None:
    """Verify that all required backend dependencies are available."""
    missing = [module for module in REQUIRED_MODULES if find_spec(module) is None]
    assert not missing, f"Missing required dependencies: {', '.join(missing)}"
