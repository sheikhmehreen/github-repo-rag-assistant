"""Safe ZIP extraction (Epic 2, T2.4).

Treats every archive as hostile: rejects path traversal, absolute and drive paths,
encrypted entries and duplicates; skips symlinks and other special entries; and
enforces file-count, size and compression-ratio limits on both declared and
actually written bytes. Nothing from the archive is ever executed.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from app.core.exceptions import ArchiveLimitExceededError, UnsafeArchiveError

if TYPE_CHECKING:
    from app.core.config import Settings

logger = logging.getLogger(__name__)

_S_IFMT = 0o170000
_S_IFLNK = 0o120000
_S_IFREG = 0o100000
_S_IFDIR = 0o040000
_COPY_CHUNK_BYTES = 64 * 1024
_MAX_NAME_LENGTH = 400
_RATIO_CHECK_MIN_BYTES = 64 * 1024  # tiny files cannot hurt; avoid false positives
_DRIVE_PATH_RE = re.compile(r"^[A-Za-z]:")
_WINDOWS_RESERVED = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
)


@dataclass(frozen=True)
class ExtractionResult:
    """Outcome of a successful extraction."""

    root: Path  # directory holding the repository files (top-level folder removed)
    files_extracted: int
    bytes_extracted: int
    special_entries_skipped: int  # symlinks, devices and other non-regular entries


@dataclass(frozen=True)
class _PlannedFile:
    info: zipfile.ZipInfo
    target: Path


def extract_archive(
    archive: Path, destination: Path, settings: Settings
) -> ExtractionResult:
    """Safely extract ``archive`` into the empty directory ``destination``.

    On any failure the destination directory is removed.

    Raises:
        UnsafeArchiveError: malicious, corrupted or unsupported archive.
        ArchiveLimitExceededError: a configured limit was exceeded.
    """
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise UnsafeArchiveError("Extraction destination must be an empty directory.")
    root = destination.resolve()

    try:
        with zipfile.ZipFile(archive) as zf:
            planned, top_levels, skipped = _plan(zf.infolist(), root, settings)
            total_bytes = _write(zf, planned, settings)
    except (
        zipfile.BadZipFile,
        RuntimeError,
        NotImplementedError,
        EOFError,
        zlib.error,
    ) as exc:
        shutil.rmtree(root, ignore_errors=True)
        raise UnsafeArchiveError(
            "Archive is corrupted or uses unsupported features."
        ) from exc
    except OSError as exc:
        shutil.rmtree(root, ignore_errors=True)
        raise UnsafeArchiveError(
            "Archive contains conflicting or invalid paths."
        ) from exc
    except BaseException:
        shutil.rmtree(root, ignore_errors=True)
        raise

    content_root = root
    if len(top_levels) == 1:
        only = root / next(iter(top_levels))
        if only.is_dir():
            content_root = only  # GitHub wraps everything in "<repo>-<sha>/"

    if skipped:
        logger.info("Skipped %d special archive entries (symlinks etc.)", skipped)
    return ExtractionResult(
        root=content_root,
        files_extracted=len(planned),
        bytes_extracted=total_bytes,
        special_entries_skipped=skipped,
    )


def _validate_name(name: str) -> PurePosixPath:
    """Reject unsafe entry names and return the parsed path."""
    if not name or "\x00" in name or "\\" in name or len(name) > _MAX_NAME_LENGTH:
        raise UnsafeArchiveError("Archive contains an invalid entry name.")
    if name.startswith("/") or _DRIVE_PATH_RE.match(name):
        raise UnsafeArchiveError("Archive contains an absolute path.")
    path = PurePosixPath(name)
    if ".." in path.parts:
        raise UnsafeArchiveError("Archive contains a path traversal entry.")
    return path


def _is_windows_reserved(path: PurePosixPath) -> bool:
    return os.name == "nt" and any(
        part.split(".")[0].upper() in _WINDOWS_RESERVED for part in path.parts
    )


def _plan(
    infos: list[zipfile.ZipInfo], root: Path, settings: Settings
) -> tuple[list[_PlannedFile], set[str], int]:
    """Validate every entry up front; nothing is written yet."""
    planned: list[_PlannedFile] = []
    top_levels: set[str] = set()
    seen: set[str] = set()
    skipped = 0
    declared_total = 0

    for info in infos:
        path = _validate_name(info.filename)
        mode = (info.external_attr >> 16) & _S_IFMT
        is_dir = info.filename.endswith("/") or mode == _S_IFDIR

        target = (root / info.filename).resolve()
        if not target.is_relative_to(root):
            raise UnsafeArchiveError("Archive entry escapes the extraction directory.")

        if is_dir:
            continue
        if mode not in (0, _S_IFREG) or _is_windows_reserved(path):
            skipped += 1  # symlink, device, FIFO, reserved device name...
            continue
        if info.flag_bits & 0x1:
            raise UnsafeArchiveError("Archive contains encrypted entries.")

        key = info.filename.lower()
        if key in seen:
            raise UnsafeArchiveError("Archive contains duplicate entries.")
        seen.add(key)

        if info.file_size > settings.max_file_bytes:
            raise ArchiveLimitExceededError(
                "Archive contains a file larger than the allowed size."
            )
        if (
            info.file_size >= _RATIO_CHECK_MIN_BYTES
            and info.file_size / max(info.compress_size, 1)
            > settings.max_compression_ratio
        ):
            raise ArchiveLimitExceededError(
                "Archive entry has a suspicious compression ratio."
            )
        declared_total += info.file_size
        if declared_total > settings.max_extracted_bytes:
            raise ArchiveLimitExceededError(
                "Archive would extract to more than the allowed size."
            )
        if len(planned) + 1 > settings.max_files:
            raise ArchiveLimitExceededError("Archive contains more files than allowed.")

        top_levels.add(path.parts[0])
        planned.append(_PlannedFile(info=info, target=target))

    # Top-level folders also come from directory entries and nested file paths.
    for info in infos:
        if info.filename and not info.filename.startswith("/"):
            top_levels.add(PurePosixPath(info.filename).parts[0])
    return planned, top_levels, skipped


def _write(zf: zipfile.ZipFile, planned: list[_PlannedFile], settings: Settings) -> int:
    """Write validated files in capped chunks, enforcing limits on real bytes."""
    total = 0
    for item in planned:
        item.target.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with (
            zf.open(item.info) as src,
            item.target.open("xb") as dst,
        ):  # "xb": never overwrite
            while chunk := src.read(_COPY_CHUNK_BYTES):
                written += len(chunk)
                total += len(chunk)
                if written > settings.max_file_bytes:
                    raise ArchiveLimitExceededError(
                        "A file exceeded the allowed size while extracting."
                    )
                if total > settings.max_extracted_bytes:
                    raise ArchiveLimitExceededError(
                        "Extracted data exceeded the allowed total size."
                    )
                dst.write(chunk)
    return total
