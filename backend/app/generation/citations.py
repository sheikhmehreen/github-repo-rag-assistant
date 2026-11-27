"""Build citations strictly from retrieved source metadata."""

from __future__ import annotations

from dataclasses import dataclass

from app.retrieval.retriever import RetrievedChunk


@dataclass(frozen=True)
class Citation:
    path: str
    start_line: int
    end_line: int

    @property
    def label(self) -> str:
        return f"{self.path}:{self.start_line}-{self.end_line}"


class CitationBuilder:
    def build(self, chunks: tuple[RetrievedChunk, ...] | list[RetrievedChunk]) -> list[Citation]:
        citations: list[Citation] = []
        seen: set[tuple[str, int, int]] = set()
        for chunk in chunks:
            key = (chunk.path, chunk.start_line, chunk.end_line)
            if key not in seen:
                seen.add(key)
                citations.append(Citation(*key))
        return citations
