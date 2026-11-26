"""Tests for app.ingestion.extractor."""

import io
import zipfile
from pathlib import Path

import pytest
from app.core.exceptions import ArchiveLimitExceededError, UnsafeArchiveError
from app.ingestion.extractor import extract_archive


def make_zip(path: Path, entries: list[tuple[str | zipfile.ZipInfo, bytes]]) -> Path:
    """Build a ZIP in code; ZipInfo entries let us fake symlinks and special modes."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries:
            zf.writestr(name, data)
    return path


def assert_nothing_outside(tmp_path: Path) -> None:
    """Only the archive and the extraction directory may exist next to each other."""
    assert {p.name for p in tmp_path.iterdir()} <= {"a.zip", "out"}


def test_extracts_normal_archive_and_strips_top_level_folder(tmp_path, make_settings):
    archive = make_zip(
        tmp_path / "a.zip",
        [
            ("repo-abc/", b""),
            ("repo-abc/main.py", b"print(1)"),
            ("repo-abc/docs/a.md", b"# a"),
        ],
    )

    result = extract_archive(archive, tmp_path / "out", make_settings())

    assert result.root.name == "repo-abc"
    assert result.files_extracted == 2
    assert result.bytes_extracted == len(b"print(1)") + len(b"# a")
    assert (result.root / "main.py").read_text() == "print(1)"
    assert (result.root / "docs" / "a.md").exists()


@pytest.mark.parametrize(
    "name",
    [
        "repo/../../evil.txt",  # path traversal
        "../evil.txt",
        "/etc/evil.txt",  # absolute path
        "C:\\evil.txt",  # Windows drive path
        "C:/evil.txt",
        "repo\\..\\evil.txt",  # backslash traversal
    ],
)
def test_rejects_malicious_paths(tmp_path, make_settings, name):
    archive = make_zip(tmp_path / "a.zip", [(name, b"x")])

    with pytest.raises(UnsafeArchiveError):
        extract_archive(archive, tmp_path / "out", make_settings())

    assert_nothing_outside(tmp_path)
    assert not (tmp_path / "out").exists()  # cleaned up on failure


def test_rejects_encrypted_entries(tmp_path, make_settings, monkeypatch):
    archive = make_zip(tmp_path / "a.zip", [("repo/secret.txt", b"secret")])
    original_infolist = zipfile.ZipFile.infolist

    def encrypted_infolist(self):
        entries = original_infolist(self)
        entries[0].flag_bits |= 0x1
        return entries

    monkeypatch.setattr(zipfile.ZipFile, "infolist", encrypted_infolist)

    with pytest.raises(UnsafeArchiveError, match="encrypted"):
        extract_archive(archive, tmp_path / "out", make_settings())

    assert not (tmp_path / "out").exists()


def test_rejects_too_many_files(tmp_path, make_settings):
    archive = make_zip(tmp_path / "a.zip", [(f"r/f{i}.txt", b"x") for i in range(5)])

    with pytest.raises(ArchiveLimitExceededError):
        extract_archive(archive, tmp_path / "out", make_settings(max_files=3))


def test_rejects_single_file_too_large(tmp_path, make_settings):
    archive = make_zip(tmp_path / "a.zip", [("r/big.txt", b"a" * 5000)])

    with pytest.raises(ArchiveLimitExceededError):
        extract_archive(archive, tmp_path / "out", make_settings(max_file_bytes=1000))


def test_enforces_actual_bytes_when_zip_metadata_understates_size(
    tmp_path, make_settings, monkeypatch
):
    archive = make_zip(tmp_path / "a.zip", [("r/file.txt", b"small")])
    original_open = zipfile.ZipFile.open

    def open_with_larger_payload(self, name, mode="r", pwd=None, *, force_zip64=False):
        if isinstance(name, zipfile.ZipInfo) and name.filename == "r/file.txt":
            return io.BytesIO(b"x" * 200)
        return original_open(self, name, mode, pwd, force_zip64=force_zip64)

    monkeypatch.setattr(zipfile.ZipFile, "open", open_with_larger_payload)

    with pytest.raises(ArchiveLimitExceededError):
        extract_archive(archive, tmp_path / "out", make_settings(max_file_bytes=100))

    assert not (tmp_path / "out").exists()


def test_rejects_total_size_exceeded(tmp_path, make_settings):
    archive = make_zip(
        tmp_path / "a.zip", [("r/a.txt", b"a" * 600), ("r/b.txt", b"b" * 600)]
    )

    with pytest.raises(ArchiveLimitExceededError):
        extract_archive(
            archive, tmp_path / "out", make_settings(max_extracted_bytes=1000)
        )


def test_rejects_zip_bomb_style_expansion(tmp_path, make_settings):
    archive = make_zip(tmp_path / "a.zip", [("r/zeros.bin", b"\x00" * 500_000)])

    with pytest.raises(ArchiveLimitExceededError):
        extract_archive(archive, tmp_path / "out", make_settings())


def test_corrupted_zip_is_rejected(tmp_path, make_settings):
    (tmp_path / "a.zip").write_bytes(b"this is not a zip file")

    with pytest.raises(UnsafeArchiveError):
        extract_archive(tmp_path / "a.zip", tmp_path / "out", make_settings())


def test_symlink_entries_are_skipped_not_extracted(tmp_path, make_settings):
    link = zipfile.ZipInfo("repo/link")
    link.external_attr = 0o120777 << 16  # Unix symlink mode bits (works on any OS)
    archive = make_zip(
        tmp_path / "a.zip", [("repo/ok.py", b"x = 1"), (link, b"/etc/passwd")]
    )

    result = extract_archive(archive, tmp_path / "out", make_settings())

    assert result.files_extracted == 1
    assert result.special_entries_skipped == 1
    assert (result.root / "ok.py").exists()
    assert not (result.root / "link").exists()


def test_rejects_duplicate_entries_including_case_collisions(tmp_path, make_settings):
    archive = make_zip(tmp_path / "a.zip", [("r/a.txt", b"1"), ("r/A.txt", b"2")])

    with pytest.raises(UnsafeArchiveError):
        extract_archive(archive, tmp_path / "out", make_settings())


def test_rejects_file_and_directory_conflict(tmp_path, make_settings):
    archive = make_zip(tmp_path / "a.zip", [("r/a", b"1"), ("r/a/b", b"2")])

    with pytest.raises(UnsafeArchiveError):
        extract_archive(archive, tmp_path / "out", make_settings())


def test_rejects_non_empty_destination(tmp_path, make_settings):
    archive = make_zip(tmp_path / "a.zip", [("r/a.txt", b"1")])
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "existing.txt").write_text("hi")

    with pytest.raises(UnsafeArchiveError):
        extract_archive(archive, tmp_path / "out", make_settings())

    assert (tmp_path / "out" / "existing.txt").read_text() == "hi"
