import math

import pytest
from app.retrieval.retriever import RawHit, Retriever

QUESTION_VECTORS = {
    "how does login work?": (1.0, 0.0, 0.0),
    "what is the weather today?": (0.0, 0.0, 1.0),
    "login and weather": (0.6, 0.0, 0.8),
}
CHUNKS = [
    ((0.95, 0.10, 0.0), "def login(user): ...", "app/auth.py", 10, 30),
    ((0.10, 0.95, 0.0), "def connect_db(): ...", "app/db.py", 1, 20),
]


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


class FakeVectorSearch:
    def __init__(self, items):
        self.items = items
        self.calls = []

    def query(self, repo_id, embedding, top_k):
        self.calls.append((repo_id, top_k))
        hits = [
            RawHit(text, path, start, end, distance=1.0 - _cosine(vector, embedding))
            for vector, text, path, start, end in self.items
        ]
        hits.sort(key=lambda hit: hit.distance)
        return hits[:top_k]


def _retriever(**kwargs):
    embed_calls = []

    def embed(question):
        embed_calls.append(question)
        return QUESTION_VECTORS[question]

    search = FakeVectorSearch(CHUNKS)
    return Retriever(embed, search, **kwargs), search, embed_calls


def test_relevant_chunk_is_returned():
    retriever, _, _ = _retriever()
    result = retriever.retrieve("repo-1", "how does login work?")
    assert [chunk.path for chunk in result] == ["app/auth.py"]
    assert result[0].start_line == 10
    assert result[0].end_line == 30
    assert result[0].similarity == pytest.approx(0.994, abs=0.01)


def test_unrelated_question_returns_empty():
    retriever, _, _ = _retriever()
    assert retriever.retrieve("repo-1", "what is the weather today?") == []


def test_threshold_changes_results():
    retriever, _, _ = _retriever()
    question = "login and weather"
    assert len(retriever.retrieve("repo-1", question, min_similarity=0.0)) == 2
    assert len(retriever.retrieve("repo-1", question, min_similarity=0.3)) == 1
    assert retriever.retrieve("repo-1", question, min_similarity=0.7) == []


def test_results_are_best_first_and_limited_by_top_k():
    retriever, search, _ = _retriever(min_similarity=-1.0)
    result = retriever.retrieve("repo-1", "how does login work?", top_k=1)
    assert [chunk.path for chunk in result] == ["app/auth.py"]
    assert search.calls == [("repo-1", 1)]
    both = retriever.retrieve("repo-1", "how does login work?", top_k=2)
    assert both[0].similarity >= both[1].similarity


def test_blank_question_returns_empty_without_embedding():
    retriever, search, embed_calls = _retriever()
    assert retriever.retrieve("repo-1", " ") == []
    assert embed_calls == []
    assert search.calls == []


@pytest.mark.parametrize(
    ("top_k", "min_similarity"),
    [(0, 0.3), (21, 0.3), (5, 1.5), (5, -1.5)],
)
def test_invalid_settings_are_rejected(top_k, min_similarity):
    with pytest.raises(ValueError):
        _retriever(top_k=top_k, min_similarity=min_similarity)
