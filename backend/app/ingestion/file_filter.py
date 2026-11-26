"""Select useful repository files for indexing (Epic 2, T2.5, part A).

Policy: an explicit extension allowlist, pruned excluded directories, and a
sensitive-file denylist that always wins (secrets must never be indexed or sent
to an LLM).
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.core.config import Settings

EXCLUDED_DIRS = frozenset(
    {
        ".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build",
        ".next", "target", "vendor", ".idea", ".vscode", ".pytest_cache",
        ".mypy_cache", ".ruff_cache", ".tox", ".gradle", "coverage",
    }
)  # fmt: skip

LOCK_FILES = frozenset(
    {
        "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock",
        "pipfile.lock", "cargo.lock", "composer.lock", "gemfile.lock", "uv.lock",
    }
)  # fmt: skip

ALLOWED_EXTENSIONS = frozenset(
    {
        ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".rs", ".c", ".cpp",
        ".h", ".hpp", ".cs", ".rb", ".php", ".kt", ".swift", ".sql", ".sh",
        ".md", ".rst", ".txt", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
        ".html", ".css", ".scss",
    }
)  # fmt: skip

# Extensionless or dot-files that are useful and safe to index.
ALLOWED_FILENAMES = frozenset(
    {
        "dockerfile", "makefile", "license", "readme", "codeowners",
        ".env.example", ".env.sample", ".env.template",
    }
)  # fmt: skip

SENSITIVE_EXTENSIONS = frozenset({".pem", ".key", ".p12", ".pfx", ".jks", ".keystore"})
SENSITIVE_PREFIXES = (
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "secrets.",
    "secret.",
)
SAFE_ENV_TEMPLATES = frozenset({".env.example", ".env.sample", ".env.template"})


@dataclass(frozen=True)
class FilterResult:
    """Selected files (relative to the root, sorted) and why others were skipped."""

    files: list[Path]
    skipped: dict[str, int]


def filter_repository_files(root: Path, settings: Settings) -> FilterResult:
    """Walk ``root`` and return the files worth indexing."""
    root = root.resolve()
    kept: list[Path] = []
    skipped: Counter[str] = Counter()

    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        remaining: list[str] = []
        for name in dirnames:
            lowered = name.lower()
            if lowered in EXCLUDED_DIRS or lowered.endswith(".egg-info"):
                skipped["excluded_dir"] += 1
            else:
                remaining.append(name)
        dirnames[:] = sorted(remaining)  # in-place edit prunes the walk

        for filename in sorted(filenames):
            path = Path(dirpath) / filename
            reason = _skip_reason(path, filename.lower(), settings)
            if reason:
                skipped[reason] += 1
            else:
                kept.append(path.relative_to(root))

    kept.sort()
    return FilterResult(files=kept, skipped=dict(skipped))


def is_sensitive(name: str) -> bool:
    """Return True for filenames that commonly hold secrets (name must be lowercase)."""
    if name == ".env" or (name.startswith(".env.") and name not in SAFE_ENV_TEMPLATES):
        return True
    if Path(name).suffix in SENSITIVE_EXTENSIONS:
        return True
    if name.startswith(SENSITIVE_PREFIXES):
        return True
    return name.startswith("credentials") and name.endswith(".json")


def _skip_reason(path: Path, name: str, settings: Settings) -> str | None:
    if path.is_symlink() or not path.is_file():
        return "not_regular_file"
    if is_sensitive(name):
        return "sensitive"
    if name in LOCK_FILES:
        return "lock_file"
    if name.endswith((".min.js", ".min.css", ".map")):
        return "minified_or_generated"
    if name not in ALLOWED_FILENAMES and Path(name).suffix not in ALLOWED_EXTENSIONS:
        return "extension_not_allowed"
    try:
        if path.stat().st_size > settings.max_file_bytes:
            return "too_large"
    except OSError:
        return "unreadable"
    return None
