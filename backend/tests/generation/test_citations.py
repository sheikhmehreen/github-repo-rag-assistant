import inspect

from app.generation.citations import CitationBuilder
from app.retrieval.retriever import RetrievedChunk


def test_citations_are_deduplicated_in_source_order():
    chunks = [RetrievedChunk("a", "a.py", 1, 4, .9), RetrievedChunk("b", "a.py", 1, 4, .8),
              RetrievedChunk("c", "b.py", 2, 3, .7)]
    citations = CitationBuilder().build(chunks)
    assert [citation.label for citation in citations] == ["a.py:1-4", "b.py:2-3"]


def test_builder_does_not_accept_model_answer():
    assert list(inspect.signature(CitationBuilder.build).parameters) == ["self", "chunks"]
