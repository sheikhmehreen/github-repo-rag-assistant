"""Language-aware splitting of repository files into searchable chunks."""

from pathlib import PurePosixPath

from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from app.rag.models import SourceFile

_EXTENSION_LANGUAGE: dict[str, Language] = {
    ".py": Language.PYTHON,
    ".js": Language.JS,
    ".jsx": Language.JS,
    ".ts": Language.TS,
    ".tsx": Language.TS,
    ".java": Language.JAVA,
    ".go": Language.GO,
    ".rs": Language.RUST,
    ".md": Language.MARKDOWN,
    ".html": Language.HTML,
    ".cpp": Language.CPP,
    ".cs": Language.CSHARP,
    ".rb": Language.RUBY,
    ".php": Language.PHP,
    ".kt": Language.KOTLIN,
    ".swift": Language.SWIFT,
}


class RepoChunker:
    """Split repository files into overlapping chunks with source metadata."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and < chunk_size")
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._splitters: dict[Language | None, RecursiveCharacterTextSplitter] = {}

    def _splitter_for(
        self, language: Language | None
    ) -> RecursiveCharacterTextSplitter:
        if language not in self._splitters:
            options = {
                "chunk_size": self._chunk_size,
                "chunk_overlap": self._chunk_overlap,
                "add_start_index": True,
            }
            self._splitters[language] = (
                RecursiveCharacterTextSplitter(**options)
                if language is None
                else RecursiveCharacterTextSplitter.from_language(language, **options)
            )
        return self._splitters[language]

    def chunk_file(self, repo_id: str, source: SourceFile) -> list[Document]:
        text = source.content.replace("\r\n", "\n")
        if not text.strip():
            return []
        language = _EXTENSION_LANGUAGE.get(PurePosixPath(source.path).suffix.lower())
        pieces = self._splitter_for(language).create_documents([text])
        header = f"# File: {source.path}\n"
        return [
            Document(
                page_content=header + piece.page_content,
                metadata={
                    "chunk_id": f"{repo_id}:{source.path}:{index}",
                    "repo_id": repo_id,
                    "path": source.path,
                    "language": language.value if language else "text",
                    "chunk_index": index,
                    "start_line": text.count("\n", 0, piece.metadata["start_index"])
                    + 1,
                },
            )
            for index, piece in enumerate(pieces)
        ]
