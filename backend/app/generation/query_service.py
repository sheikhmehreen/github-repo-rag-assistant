"""Repository-aware retrieval and answer generation orchestration."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from app.core.config import Settings
from app.database.models import RepoStatus
from app.database.repository import RepoNotFoundError as DatabaseRepoNotFoundError
from app.database.repository import RepoRepository
from app.generation.citations import Citation, CitationBuilder
from app.generation.errors import (
    QuestionInvalidError,
    QuestionTooLongError,
    RepoNotFoundError,
    RepoNotReadyError,
)
from app.generation.llm_client import LLMClient, build_llm_client
from app.generation.prompt_builder import NOT_FOUND_ANSWER, PromptBuilder
from app.retrieval.chroma_search import ChromaVectorSearch
from app.retrieval.embedding_service import EmbeddingService
from app.retrieval.retriever import Retriever

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QueryResult:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    found: bool = True


class QueryService:
    def __init__(
        self,
        retriever: Retriever,
        repo_status: Callable[[str], object],
        llm: LLMClient,
        prompt_builder: PromptBuilder,
        citation_builder: CitationBuilder,
        *,
        max_question_chars: int = 1000,
    ) -> None:
        self._retriever = retriever
        self._repo_status = repo_status
        self._llm = llm
        self._prompts = prompt_builder
        self._citations = citation_builder
        self._max_question_chars = max_question_chars

    def query(self, repo_id: str, question: str) -> QueryResult:
        if not question or not question.strip():
            raise QuestionInvalidError()
        if len(question) > self._max_question_chars:
            raise QuestionTooLongError()
        try:
            status = self._repo_status(repo_id)
        except (DatabaseRepoNotFoundError, KeyError) as exc:
            raise RepoNotFoundError() from exc
        if isinstance(status, RepoStatus):
            status = status.value
        if status is None:
            raise RepoNotFoundError()
        if status != "ready":
            raise RepoNotReadyError()
        chunks = self._retriever.retrieve(repo_id, question)
        logger.info(
            "query repo_id=%s question_chars=%d retrieved=%d",
            repo_id,
            len(question),
            len(chunks),
        )
        if not chunks:
            return QueryResult(NOT_FOUND_ANSWER, [], False)
        prompt = self._prompts.build(question, chunks)
        answer = self._llm.generate(prompt.system, prompt.user)
        if answer.strip().casefold().startswith(NOT_FOUND_ANSWER.casefold()):
            return QueryResult(NOT_FOUND_ANSWER, [], False)
        citations = self._citations.build(prompt.used_chunks)
        return QueryResult(answer, citations, True)


def build_query_service(
    settings: Settings,
    embed_query: Callable[[str], list[float]],
    chroma_client: object,
    collection_name_for: Callable[[str], str],
    repo_status: Callable[[str], object],
    llm: LLMClient | None = None,
) -> QueryService:
    from app.retrieval.retriever import Retriever

    retriever = Retriever(
        embed_query,
        ChromaVectorSearch(chroma_client, collection_name_for),
        top_k=settings.retrieval_top_k,
        min_similarity=settings.retrieval_min_similarity,
    )
    return QueryService(
        retriever,
        repo_status,
        llm or build_llm_client(settings),
        PromptBuilder(settings.max_prompt_chars),
        CitationBuilder(),
        max_question_chars=settings.max_question_chars,
    )


def build_default_query_service(
    settings: Settings, llm: LLMClient | None = None
) -> QueryService:
    embedding = EmbeddingService(
        settings.embedding_model_name, settings.embedding_batch_size
    )
    from app.retrieval.vector_store import ChromaVectorStore, collection_name_for

    store = ChromaVectorStore(settings.chroma_persist_dir)
    repository = RepoRepository(settings.sqlite_path)
    return build_query_service(
        settings,
        embedding.embed_query,
        store._client,
        collection_name_for,
        lambda repo_id: repository.get(repo_id).status,
        llm,
    )
