"""Application factories that connect RAG components to configured settings."""

from app.core.config import get_settings
from app.rag.chunker import RepoChunker
from app.rag.embeddings import build_embeddings
from app.rag.indexer import IndexingService
from app.rag.vector_store import RepoVectorStore


def build_vector_store() -> RepoVectorStore:
    settings = get_settings()
    return RepoVectorStore(
        persist_dir=str(settings.chroma_persist_dir),
        embeddings=build_embeddings(settings.embedding_model_name),
    )


def build_indexing_service() -> IndexingService:
    settings = get_settings()
    return IndexingService(
        chunker=RepoChunker(settings.chunk_size, settings.chunk_overlap),
        store=build_vector_store(),
        max_chunks_per_repo=settings.max_chunks_per_repo,
    )
