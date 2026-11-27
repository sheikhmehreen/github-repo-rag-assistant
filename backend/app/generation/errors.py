"""Typed errors for retrieval and answer generation."""

from app.core.exceptions import AppError


class GenerationError(AppError):
    """Base class for Epic 5 query errors."""

    code = "GENERATION_ERROR"
    default_message = "Question answering failed."
    status_code = 500


class LLMError(GenerationError):
    """The provider request failed or retries were exhausted."""

    code = "LLM_ERROR"
    default_message = "The language model request failed."
    status_code = 502


class LLMTimeoutError(LLMError):
    """The provider did not answer before the configured timeout."""

    code = "LLM_TIMEOUT"
    default_message = "The language model request timed out."
    status_code = 504


class LLMResponseError(LLMError):
    """The provider response could not be read as an answer."""

    code = "LLM_RESPONSE_ERROR"
    default_message = "The language model returned an invalid response."


class LLMConfigError(LLMError):
    """The configured provider or its credentials are invalid."""

    code = "LLM_CONFIG_ERROR"
    default_message = "The language model provider is misconfigured."
    status_code = 503


class QuestionInvalidError(GenerationError):
    """The question is empty or whitespace only."""

    code = "QUESTION_INVALID"
    default_message = "Question must not be empty."
    status_code = 422


class QuestionTooLongError(QuestionInvalidError):
    """The question exceeds the configured character limit."""

    code = "QUESTION_TOO_LONG"
    default_message = "Question exceeds the configured length limit."


class RepoNotFoundError(GenerationError):
    """The repository id does not exist in the repository database."""

    code = "REPO_NOT_FOUND"
    default_message = "Repository not found."
    status_code = 404


class RepoNotReadyError(GenerationError):
    """The repository has not completed ingestion."""

    code = "REPO_NOT_READY"
    default_message = "Repository is not ready for questions."
    status_code = 409


class PromptTooLargeError(GenerationError):
    """The system prompt and question leave no room for retrieved context."""

    code = "PROMPT_TOO_LARGE"
    default_message = "The question and prompt exceed the configured size limit."
    status_code = 422
