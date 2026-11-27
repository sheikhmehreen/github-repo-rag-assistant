import re

import pytest
from app.rag.chunker import RepoChunker
from app.rag.models import SourceFile
from app.rag.vector_store import RepoVectorStore
from langchain_core.embeddings import DeterministicFakeEmbedding


@pytest.fixture
def store(tmp_path) -> RepoVectorStore:
    return RepoVectorStore(
        str(tmp_path / "chroma"), DeterministicFakeEmbedding(size=64)
    )


def make_docs(repo_id: str = "o/r"):
    sources = [
        SourceFile("a.py", "def alpha():\n return 1\n"),
        SourceFile("b.py", "def beta():\n return 2\n"),
    ]
    chunker = RepoChunker()
    return [doc for source in sources for doc in chunker.chunk_file(repo_id, source)]


def test_collection_name_is_stable_safe_and_distinct() -> None:
    name = RepoVectorStore.collection_name("Owner/My_Repo.git")
    assert name == RepoVectorStore.collection_name("Owner/My_Repo.git")
    assert re.fullmatch(r"[a-z0-9][a-z0-9-]{1,60}[a-z0-9]", name)
    assert name != RepoVectorStore.collection_name("owner/my-repo-git")


def test_add_count_and_search(store: RepoVectorStore) -> None:
    docs = make_docs()
    assert store.add_documents("o/r", docs) == 2
    assert store.count("o/r") == 2
    results = store.similarity_search("o/r", docs[1].page_content, k=2)
    assert results[0].path == "b.py"
    assert results[0].start_line == 1
    assert results[0].score > 0.99


def test_add_nothing_and_repo_isolation(store: RepoVectorStore) -> None:
    assert store.add_documents("o/r", []) == 0
    store.add_documents("o/r", make_docs())
    assert store.similarity_search("x/y", "alpha", k=3) == []


def test_delete_repo_removes_collection(store: RepoVectorStore) -> None:
    store.add_documents("o/r", make_docs())
    store.delete_repo("o/r")
    assert store.count("o/r") == 0


def test_empty_query_is_rejected(store: RepoVectorStore) -> None:
    with pytest.raises(ValueError):
        store.similarity_search("o/r", " ")
