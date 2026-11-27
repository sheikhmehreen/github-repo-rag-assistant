"""Lazy, cached loader for the local Hugging Face embedding model."""

import logging
from functools import lru_cache

from langchain_core.embeddings import Embeddings

from app.rag.errors import EmbeddingError

logger = logging.getLogger(__name__)


@lru_cache(maxsize=2)
def build_embeddings(model_name: str) -> Embeddings:
    """Load a CPU embedding model once per process."""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        logger.info("Loading embedding model %s", model_name)
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as exc:
        raise EmbeddingError(f"Could not load embedding model {model_name!r}") from exc
