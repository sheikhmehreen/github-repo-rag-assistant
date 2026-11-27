import pytest
from app.rag.chunker import RepoChunker
from app.rag.models import SourceFile


def test_rejects_invalid_chunk_settings() -> None:
    with pytest.raises(ValueError):
        RepoChunker(chunk_size=0, chunk_overlap=0)
    with pytest.raises(ValueError):
        RepoChunker(chunk_size=100, chunk_overlap=100)


def test_blank_file_yields_no_chunks() -> None:
    assert RepoChunker().chunk_file("o/r", SourceFile("a.py", " \n\n")) == []


def test_small_file_has_source_metadata() -> None:
    chunks = RepoChunker().chunk_file(
        "o/r", SourceFile("src/app.py", "def f():\n return 1\n")
    )
    assert len(chunks) == 1
    assert chunks[0].page_content.startswith("# File: src/app.py\n")
    assert chunks[0].metadata == {
        "chunk_id": "o/r:src/app.py:0",
        "repo_id": "o/r",
        "path": "src/app.py",
        "language": "python",
        "chunk_index": 0,
        "start_line": 1,
    }


def test_unknown_extension_uses_plain_text() -> None:
    chunks = RepoChunker().chunk_file("o/r", SourceFile("notes.xyz", "hello\nworld"))
    assert chunks[0].metadata["language"] == "text"


def test_large_file_splits_with_increasing_lines_and_unique_ids() -> None:
    content = "\n".join(f"def func_{i}():\n return {i}\n" for i in range(100))
    chunks = RepoChunker(chunk_size=200, chunk_overlap=20).chunk_file(
        "o/r", SourceFile("big.py", content)
    )
    lines = [chunk.metadata["start_line"] for chunk in chunks]
    assert len(chunks) > 1
    assert lines == sorted(lines)
    assert lines[0] == 1 and lines[-1] > 1
    assert len({chunk.metadata["chunk_id"] for chunk in chunks}) == len(chunks)


def test_windows_line_endings_are_normalized() -> None:
    chunks = RepoChunker().chunk_file("o/r", SourceFile("a.py", "a = 1\r\nb = 2\r\n"))
    assert "\r" not in chunks[0].page_content
