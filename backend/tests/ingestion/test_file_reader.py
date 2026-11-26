"""Tests for app.ingestion.file_reader."""

import hashlib
from pathlib import Path

from app.ingestion.file_reader import read_repository_files


def write(root: Path, relative: str, data: bytes) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return Path(relative)


def test_reads_utf8_file_with_metadata(tmp_path, make_settings):
    raw = b"print('hi')\n"
    rel = write(tmp_path, "pkg/main.py", raw)

    result = read_repository_files(tmp_path, [rel], make_settings())

    (file,) = result.files
    assert file.path == "pkg/main.py"  # POSIX style, relative
    assert file.content == "print('hi')\n"
    assert file.size_bytes == len(raw)
    assert file.extension == ".py"
    assert file.sha256 == hashlib.sha256(raw).hexdigest()


def test_utf8_bom_is_removed(tmp_path, make_settings):
    rel = write(tmp_path, "a.py", b"\xef\xbb\xbfx = 1")

    (file,) = read_repository_files(tmp_path, [rel], make_settings()).files

    assert file.content == "x = 1"


def test_crlf_is_normalized(tmp_path, make_settings):
    rel = write(tmp_path, "a.py", b"a\r\nb\r\n")

    (file,) = read_repository_files(tmp_path, [rel], make_settings()).files

    assert file.content == "a\nb\n"


def test_skips_binary_non_utf8_and_empty_files(tmp_path, make_settings):
    paths = [
        write(tmp_path, "bin.py", b"\x00\x01\x02"),
        write(tmp_path, "latin.py", "caf\xe9".encode("latin-1")),
        write(tmp_path, "empty.md", b"  \n"),
        write(tmp_path, "ok.py", b"x = 1"),
    ]

    result = read_repository_files(tmp_path, paths, make_settings())

    assert [f.path for f in result.files] == ["ok.py"]
    assert result.skipped == {"binary": 1, "not_utf8": 1, "empty": 1}


def test_oversized_file_is_skipped_at_read_time(tmp_path, make_settings):
    rel = write(tmp_path, "big.txt", b"a" * 5000)

    result = read_repository_files(tmp_path, [rel], make_settings(max_file_bytes=1000))

    assert result.files == []
    assert result.skipped == {"too_large": 1}


def test_one_unreadable_file_does_not_fail_the_batch(tmp_path, make_settings):
    good = write(tmp_path, "ok.py", b"x = 1")
    missing = Path("missing.py")

    result = read_repository_files(tmp_path, [missing, good], make_settings())

    assert [f.path for f in result.files] == ["ok.py"]
    assert result.skipped == {"unreadable": 1}


def test_path_escaping_the_root_is_rejected(tmp_path, make_settings):
    outside = tmp_path / "outside.py"
    outside.write_bytes(b"x = 1")
    root = tmp_path / "root"
    root.mkdir()

    result = read_repository_files(root, [Path("../outside.py")], make_settings())

    assert result.files == []
    assert result.skipped == {"unsafe_path": 1}
