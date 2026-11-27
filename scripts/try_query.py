"""Inspect raw retrieval scores and execute a real Epic 5 query.

Run from the repository root:
  python scripts/try_query.py REPO_ID "How does login work?"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

from app.core.config import get_settings
from app.database.repository import RepoRepository
from app.generation.llm_client import build_llm_client
from app.generation.query_service import build_query_service
from app.retrieval.chroma_search import ChromaVectorSearch
from app.retrieval.embedding_service import EmbeddingService
from app.retrieval.retriever import distance_to_similarity
from app.retrieval.vector_store import ChromaVectorStore, collection_name_for


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_id")
    parser.add_argument("question")
    args = parser.parse_args()

    settings = get_settings()
    embedding = EmbeddingService(settings.embedding_model_name, settings.embedding_batch_size)
    store = ChromaVectorStore(settings.chroma_persist_dir)
    search = ChromaVectorSearch(store._client, collection_name_for)
    query_vector = embedding.embed_query(args.question)
    raw = search.query(args.repo_id, query_vector, settings.retrieval_top_k)
    print("Raw retrieval scores (cosine similarity; threshold not applied):")
    for hit in raw:
        print(f"  {distance_to_similarity(hit.distance):.4f}  {hit.path}:{hit.start_line}-{hit.end_line}")

    repository = RepoRepository(settings.sqlite_path)
    service = build_query_service(
        settings,
        embedding.embed_query,
        store._client,
        collection_name_for,
        lambda repo_id: repository.get(repo_id).status,
        build_llm_client(settings),
    )
    result = service.query(args.repo_id, args.question)
    print("\nAnswer:")
    print(result.answer)
    print("Found:", result.found)
    print("Citations:")
    for citation in result.citations:
        print(" ", citation.label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
