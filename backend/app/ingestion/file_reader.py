"""Read filtered repository files into memory (Epic 2, T2.5, part B)."""

from __future__ import annotations

import hashlib
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.core.config import Settings

logger = logging.getLogger(__name__)

_BINARY_SNIFF_BYTES = 8192


@dataclass(frozen=True)
class RepoFile:
    """A text file from the repository."""

    path: str  # POSIX-style path relative to the repository root
    content: str
    size_bytes: int
    extension: str  # lowercase suffix, or lowercase filename when there is none
    sha256: str  # hash of the raw bytes (used for change detection in Epic 4)


@dataclass(frozen=True)
class ReadResult:
    """Files that were read and counts of files skipped by reason."""

    files: list[RepoFile]
    skipped: dict[str, int]


def read_repository_files(
    root: Path, files: list[Path], settings: Settings
) -> ReadResult:
    """Read each relative path in ``files`` under ``root``; one bad file never fails the batch."""
    root = root.resolve()
    results: list[RepoFile] = []
    skipped: Counter[str] = Counter()

    for relative in files:
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or path.is_symlink():
            skipped["unsafe_path"] += 1
            continue
        try:
            if path.stat().st_size > settings.max_file_bytes:
                skipped["too_large"] += 1
                continue
            raw = path.read_bytes()
        except OSError:
            logger.warning("Could not read %s", relative.as_posix())
            skipped["unreadable"] += 1
            continue

        if b"\x00" in raw[:_BINARY_SNIFF_BYTES]:
            skipped["binary"] += 1
            continue
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            skipped["not_utf8"] += 1
            continue

        text = text.replace("\r\n", "\n").replace("\r", "\n")
        if not text.strip():
            skipped["empty"] += 1
            continue

        results.append(
            RepoFile(
                path=relative.as_posix(),
                content=text,
                size_bytes=len(raw),
                extension=relative.suffix.lower() or relative.name.lower(),
                sha256=hashlib.sha256(raw).hexdigest(),
            )
        )

    return ReadResult(files=results, skipped=dict(skipped))
