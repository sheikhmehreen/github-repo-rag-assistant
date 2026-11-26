"""Tests for app.ingestion.file_filter."""

from pathlib import Path

from app.ingestion import file_filter
from app.ingestion.file_filter import filter_repository_files, is_sensitive


def write(root: Path, relative: str, data: bytes = b"x = 1\n") -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def kept(root: Path, settings) -> set[str]:
    return {p.as_posix() for p in filter_repository_files(root, settings).files}


def test_keeps_source_docs_and_config(tmp_path, make_settings):
    for name in [
        "main.py",
        "src/app.ts",
        "README.md",
        "config.yaml",
        "Dockerfile",
        "LICENSE",
    ]:
        write(tmp_path, name)

    assert kept(tmp_path, make_settings()) == {
        "main.py", "src/app.ts", "README.md", "config.yaml", "Dockerfile", "LICENSE",
    }  # fmt: skip


def test_skips_translated_readme_duplicates(tmp_path, make_settings):
    for name in ["README.md", "README-fr.md", "README-zh-Hant.md", "README.de.md"]:
        write(tmp_path, name)

    result = filter_repository_files(tmp_path, make_settings())

    assert {path.as_posix() for path in result.files} == {"README.md"}
    assert result.skipped["translated_document"] == 3


def test_excluded_directories_are_pruned_and_never_entered(
    tmp_path, make_settings, monkeypatch
):
    write(tmp_path, "src/a.py")
    write(tmp_path, "node_modules/pkg/index.js")
    write(tmp_path, ".git/config")
    write(tmp_path, "venv/lib/x.py")

    walked: list[str] = []
    real_walk = file_filter.os.walk

    def spy(top, *args, **kwargs):
        for dirpath, dirnames, filenames in real_walk(top, *args, **kwargs):
            walked.append(Path(dirpath).name)
            yield dirpath, dirnames, filenames

    monkeypatch.setattr(file_filter.os, "walk", spy)

    result = filter_repository_files(tmp_path, make_settings())

    assert [p.as_posix() for p in result.files] == ["src/a.py"]
    assert (
        "node_modules" not in walked and ".git" not in walked and "venv" not in walked
    )
    assert result.skipped["excluded_dir"] == 3


def test_skips_lock_minified_and_binary_extensions(tmp_path, make_settings):
    for name in [
        "package-lock.json",
        "poetry.lock",
        "app.min.js",
        "site.min.css",
        "logo.png",
        "a.exe",
        "b.dll",
    ]:
        write(tmp_path, name)
    write(tmp_path, "keep.py")

    assert kept(tmp_path, make_settings()) == {"keep.py"}


def test_sensitive_files_are_excluded_but_env_templates_kept(tmp_path, make_settings):
    for name in [
        ".env",
        ".env.local",
        "server.pem",
        "private.key",
        "id_rsa",
        "secrets.yaml",
        "credentials.json",
    ]:
        write(tmp_path, name)
    write(tmp_path, ".env.example")
    write(tmp_path, "main.py")

    result = filter_repository_files(tmp_path, make_settings())

    assert {p.as_posix() for p in result.files} == {".env.example", "main.py"}
    assert result.skipped["sensitive"] >= 5


def test_is_sensitive_unit():
    assert is_sensitive(".env")
    assert is_sensitive(".env.production")
    assert not is_sensitive(".env.example")
    assert is_sensitive("cert.pem")
    assert is_sensitive("id_ed25519")
    assert not is_sensitive("main.py")


def test_oversized_files_are_skipped(tmp_path, make_settings):
    write(tmp_path, "small.json", b"{}")
    write(tmp_path, "huge.json", b"a" * 5000)

    result = filter_repository_files(tmp_path, make_settings(max_file_bytes=1000))

    assert [p.as_posix() for p in result.files] == ["small.json"]
    assert result.skipped["too_large"] == 1


def test_output_is_sorted_and_relative(tmp_path, make_settings):
    for name in ["z.py", "a.py", "m/b.py"]:
        write(tmp_path, name)

    files = filter_repository_files(tmp_path, make_settings()).files

    assert files == sorted(files)
    assert all(not p.is_absolute() for p in files)


def test_symlinks_are_skipped(tmp_path, make_settings):
    write(tmp_path, "real.py")
    link = tmp_path / "link.py"
    try:
        link.symlink_to(tmp_path / "real.py")
    except (OSError, NotImplementedError):  # symlinks need privileges on Windows
        return

    assert kept(tmp_path, make_settings()) == {"real.py"}
