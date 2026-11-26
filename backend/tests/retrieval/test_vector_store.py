import threading
import time

import pytest
from app.retrieval.vector_store import (
    ChromaVectorStore,
    CollectionNotFoundError,
    _UsageTracker,
    collection_name_for,
)


@pytest.fixture
def store(tmp_path) -> ChromaVectorStore:
    return ChromaVectorStore(tmp_path / "chroma")


def _add(store: ChromaVectorStore, collection: str, rows: list[tuple]) -> None:
    store.add_chunks(
        collection,
        ids=[row[0] for row in rows],
        texts=[row[1] for row in rows],
        embeddings=[row[2] for row in rows],
        metadatas=[{"file_path": row[0]} for row in rows],
    )


def test_add_then_search_returns_best_chunk(store: ChromaVectorStore) -> None:
    name = collection_name_for("repoa")
    _add(store, name, [("a", "auth code", [1.0, 0.0, 0.0]), ("b", "db", [0, 1, 0])])
    hits = store.search(name, [0.9, 0.1, 0.0], top_k=1)
    assert [hit.chunk_id for hit in hits] == ["a"]
    assert hits[0].text == "auth code"
    assert hits[0].metadata == {"file_path": "a"}
    assert hits[0].score > 0.9


def test_collections_are_isolated(store: ChromaVectorStore) -> None:
    repo_a, repo_b = collection_name_for("repoa"), collection_name_for("repob")
    _add(store, repo_a, [("only-a", "a text", [1.0, 0.0, 0.0])])
    _add(store, repo_b, [("only-b", "b text", [1.0, 0.0, 0.0])])
    ids_a = {hit.chunk_id for hit in store.search(repo_a, [1, 0, 0], top_k=10)}
    ids_b = {hit.chunk_id for hit in store.search(repo_b, [1, 0, 0], top_k=10)}
    assert ids_a == {"only-a"}
    assert ids_b == {"only-b"}


def test_top_k_larger_than_collection_is_safe(store: ChromaVectorStore) -> None:
    name = collection_name_for("repoa")
    _add(store, name, [("a", "x", [1.0, 0.0])])
    assert len(store.search(name, [1.0, 0.0], top_k=50)) == 1


def test_missing_collection_raises(store: ChromaVectorStore) -> None:
    with pytest.raises(CollectionNotFoundError):
        store.search(collection_name_for("nothere"), [1.0, 0.0])


def test_delete_is_idempotent_and_removes_collection(store: ChromaVectorStore) -> None:
    name = collection_name_for("repoa")
    _add(store, name, [("a", "x", [1.0, 0.0])])
    store.delete_collection(name)
    store.delete_collection(name)
    with pytest.raises(CollectionNotFoundError):
        store.search(name, [1.0, 0.0])


def test_upsert_replaces_existing_chunk(store: ChromaVectorStore) -> None:
    name = collection_name_for("repoa")
    _add(store, name, [("a", "first", [1.0, 0.0])])
    _add(store, name, [("a", "second", [1.0, 0.0])])
    assert [hit.text for hit in store.search(name, [1, 0], top_k=5)] == ["second"]


def test_delete_waits_for_active_readers() -> None:
    tracker = _UsageTracker()
    reader_inside = threading.Event()
    release_reader = threading.Event()
    deleted = threading.Event()

    def reader() -> None:
        with tracker.reading("collection"):
            reader_inside.set()
            release_reader.wait(timeout=5)

    def deleter() -> None:
        with tracker.deleting("collection"):
            deleted.set()

    reader_thread = threading.Thread(target=reader)
    reader_thread.start()
    assert reader_inside.wait(timeout=5)
    delete_thread = threading.Thread(target=deleter)
    delete_thread.start()
    time.sleep(0.2)
    assert not deleted.is_set()
    release_reader.set()
    reader_thread.join(5)
    delete_thread.join(5)
    assert deleted.is_set()


def test_new_readers_wait_while_delete_is_running() -> None:
    tracker = _UsageTracker()
    delete_started = threading.Event()
    finish_delete = threading.Event()
    reader_entered = threading.Event()

    def deleter() -> None:
        with tracker.deleting("collection"):
            delete_started.set()
            finish_delete.wait(timeout=5)

    def reader() -> None:
        with tracker.reading("collection"):
            reader_entered.set()

    delete_thread = threading.Thread(target=deleter)
    delete_thread.start()
    assert delete_started.wait(timeout=5)
    reader_thread = threading.Thread(target=reader)
    reader_thread.start()
    time.sleep(0.2)
    assert not reader_entered.is_set()
    finish_delete.set()
    delete_thread.join(5)
    reader_thread.join(5)
    assert reader_entered.is_set()
