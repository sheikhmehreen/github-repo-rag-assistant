"""Errors raised by the RAG indexing components."""


class RagError(Exception):
    """Base class for indexing and retrieval errors."""


class EmbeddingError(RagError):
    """The embedding model could not be loaded or used."""


class VectorStoreError(RagError):
    """ChromaDB read or write failed."""


class IndexingError(RagError):
    """A repository could not be indexed."""
