import langchain_huggingface
import pytest
from app.rag.embeddings import build_embeddings
from app.rag.errors import EmbeddingError


def test_model_load_failure_is_wrapped_without_network(monkeypatch) -> None:
    def fail_to_load(**kwargs):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(langchain_huggingface, "HuggingFaceEmbeddings", fail_to_load)
    build_embeddings.cache_clear()
    with pytest.raises(EmbeddingError):
        build_embeddings("test/model")
    build_embeddings.cache_clear()
