import numpy as np
import pytest
from app.retrieval.embedding_service import EMBEDDING_DIMENSION, EmbeddingService


@pytest.fixture(scope="module")
def service() -> EmbeddingService:
    return EmbeddingService(batch_size=4)


def test_embeddings_have_expected_dimension(service: EmbeddingService) -> None:
    vectors = service.embed_documents(["hello world", "def add(a, b): return a + b"])
    assert len(vectors) == 2
    assert all(len(vector) == EMBEDDING_DIMENSION for vector in vectors)
    assert len(service.embed_query("hello")) == EMBEDDING_DIMENSION


def test_batch_and_single_results_match(service: EmbeddingService) -> None:
    texts = [f"sample text number {index}" for index in range(10)]
    batched = service.embed_documents(texts)
    singles = [service.embed_query(text) for text in texts]
    assert np.allclose(batched, singles, atol=1e-5)


def test_similar_sentences_score_higher_than_unrelated(
    service: EmbeddingService,
) -> None:
    query = service.embed_query("How do I install the package with pip?")
    related = service.embed_query("Run pip install to set up the library")
    unrelated = service.embed_query("The cat slept on the warm sofa all afternoon")
    assert np.dot(query, related) > np.dot(query, unrelated)


def test_vectors_are_normalized(service: EmbeddingService) -> None:
    vector = service.embed_query("unit length please")
    assert np.isclose(np.linalg.norm(vector), 1.0, atol=1e-5)


def test_empty_input_returns_empty_list(service: EmbeddingService) -> None:
    assert service.embed_documents([]) == []


def test_model_is_loaded_only_once(service: EmbeddingService) -> None:
    service.embed_query("warm up")
    first = service._get_model()
    service.embed_query("again")
    assert service._get_model() is first
