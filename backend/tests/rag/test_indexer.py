import pytest
from app.rag.chunker import RepoChunker
from app.rag.errors import IndexingError
from app.rag.indexer import IndexingService
from app.rag.models import SourceFile
from app.rag.vector_store import RepoVectorStore
from langchain_core.embeddings import DeterministicFakeEmbedding


def build(tmp_path, max_chunks: int = 5000):
    store = RepoVectorStore(
        str(tmp_path / "chroma"), DeterministicFakeEmbedding(size=64)
    )
    return IndexingService(RepoChunker(), store, max_chunks), store


FILES = [SourceFile("a.py", "def alpha():\n return 1\n"), SourceFile("empty.py", " \n")]


def test_index_repo_reports_stats(tmp_path) -> None:
    service, store = build(tmp_path)
    result = service.index_repo("o/r", FILES)
    assert result.files_indexed == 1
    assert result.files_skipped == 1
    assert result.chunks_indexed == store.count("o/r") == 1
    assert result.duration_seconds >= 0


def test_reindexing_replaces_instead_of_duplicating(tmp_path) -> None:
    service, store = build(tmp_path)
    service.index_repo("o/r", FILES)
    service.index_repo("o/r", FILES)
    assert store.count("o/r") == 1


def test_empty_repo_fails(tmp_path) -> None:
    service, _ = build(tmp_path)
    with pytest.raises(IndexingError):
        service.index_repo("o/r", [SourceFile("empty.py", "")])


def test_chunk_limit_is_enforced_before_replacing_index(tmp_path) -> None:
    service, store = build(tmp_path)
    service.index_repo("o/r", FILES)
    large = SourceFile("big.py", "\n".join(f"x_{i} = {i}" for i in range(2000)))
    limited = IndexingService(RepoChunker(), store, max_chunks_per_repo=2)
    with pytest.raises(IndexingError):
        limited.index_repo("o/r", [large])
    assert store.count("o/r") == 1
