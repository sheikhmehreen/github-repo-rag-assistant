"""Adapter from the Epic 4 Chroma client to the retriever interface."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from app.retrieval.retriever import RawHit


class ChromaVectorSearch:
    """Translate Chroma query results into plain ``RawHit`` values."""

    def __init__(
        self, client: Any, collection_name_for: Callable[[str], str]
    ) -> None:
        self._client = client
        self._name_for = collection_name_for

    def query(
        self, repo_id: str, embedding: Sequence[float], top_k: int
    ) -> list[RawHit]:
        collection = self._client.get_collection(self._name_for(repo_id))
        result = collection.query(
            query_embeddings=[list(embedding)],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        documents = result["documents"][0] or []
        metadatas = result["metadatas"][0] or []
        distances = result["distances"][0] or []
        hits: list[RawHit] = []
        for document, metadata, distance in zip(
            documents, metadatas, distances, strict=True
        ):
            meta = metadata or {}
            hits.append(
                RawHit(
                    text=document,
                    path=str(meta.get("path") or meta.get("file_path") or ""),
                    start_line=int(meta["start_line"]),
                    end_line=int(meta["end_line"]),
                    distance=float(distance),
                )
            )
        return hits
