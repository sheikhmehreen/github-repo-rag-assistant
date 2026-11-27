import pytest
from app.core.config import Settings
from app.database.models import RepoStatus
from app.database.repository import RepoNotFoundError as DatabaseNotFound
from app.generation.errors import (
    QuestionInvalidError,
    QuestionTooLongError,
    RepoNotFoundError,
    RepoNotReadyError,
)
from app.generation.llm_client import FakeLLMClient
from app.generation.prompt_builder import NOT_FOUND_ANSWER
from app.generation.query_service import QueryService, build_query_service
from app.retrieval.retriever import RawHit, Retriever


class Search:
    def __init__(self, hits):
        self.hits = hits

    def query(self, repo_id, embedding, top_k):
        return self.hits[:top_k]


class Collection:
    def query(self, **kwargs):
        return {
            "documents": [["login uses JWT"]],
            "metadatas": [[{"file_path": "auth.py", "start_line": 4, "end_line": 8}]],
            "distances": [[0.05]],
        }


class Chroma:
    def get_collection(self, name):
        return Collection()


def make_service(
    status=RepoStatus.READY, answer="JWT is used.", hits=None, max_chars=100
):
    search = Search(
        hits if hits is not None else [RawHit("login uses JWT", "auth.py", 4, 8, 0.05)]
    )
    retriever = Retriever(lambda _q: [1.0], search, min_similarity=0.3)
    llm = FakeLLMClient(answer)
    service = QueryService(
        retriever,
        lambda _id: status,
        llm,
        __import__(
            "app.generation.prompt_builder", fromlist=["PromptBuilder"]
        ).PromptBuilder(),
        __import__(
            "app.generation.citations", fromlist=["CitationBuilder"]
        ).CitationBuilder(),
        max_question_chars=max_chars,
    )
    return service, llm


def test_query_returns_answer_and_citation_from_used_chunk():
    service, llm = make_service()
    result = service.query("repo", "How does login work?")
    assert result.answer == "JWT is used."
    assert result.found is True
    assert [citation.label for citation in result.citations] == ["auth.py:4-8"]
    assert "login uses JWT" in llm.calls[0][1]


def test_empty_retrieval_returns_canonical_answer_without_llm():
    service, llm = make_service(hits=[])
    result = service.query("repo", "unrelated")
    assert result == type(result)(NOT_FOUND_ANSWER, [], False)
    assert llm.calls == []


def test_not_found_llm_answer_is_canonical_and_has_no_citations():
    service, _ = make_service(answer=f"{NOT_FOUND_ANSWER} extra")
    result = service.query("repo", "question")
    assert result.answer == NOT_FOUND_ANSWER
    assert result.found is False
    assert result.citations == []


@pytest.mark.parametrize(
    "question,error", [(" ", QuestionInvalidError), ("x" * 101, QuestionTooLongError)]
)
def test_question_validation_precedes_repo_lookup(question, error):
    service, _ = make_service(max_chars=100)
    with pytest.raises(error):
        service.query("repo", question)


def test_missing_and_not_ready_repos_are_rejected():
    missing = QueryService(
        Retriever(lambda _q: [1], Search([])),
        lambda _id: (_ for _ in ()).throw(DatabaseNotFound("x")),
        FakeLLMClient(),
        __import__(
            "app.generation.prompt_builder", fromlist=["PromptBuilder"]
        ).PromptBuilder(),
        __import__(
            "app.generation.citations", fromlist=["CitationBuilder"]
        ).CitationBuilder(),
    )
    with pytest.raises(RepoNotFoundError):
        missing.query("x", "q")
    service, _ = make_service(status=RepoStatus.PROCESSING)
    with pytest.raises(RepoNotReadyError):
        service.query("repo", "q")


def test_factory_wires_chroma_fake_and_settings():
    service = build_query_service(
        Settings(retrieval_min_similarity=0.3),
        lambda _q: [1.0],
        Chroma(),
        lambda repo: f"repo_{repo}",
        lambda _id: "ready",
        FakeLLMClient(),
    )
    assert (
        service.query("abc", "how does login work?").citations[0].label == "auth.py:4-8"
    )
