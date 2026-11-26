from app.ingestion.file_reader import RepoFile
from app.rag.adapter import to_source_files
from app.rag.models import SourceFile


def test_ingestion_files_convert_to_rag_sources() -> None:
    files = [RepoFile("src/main.py", "print('hi')", 11, ".py", "abc")]
    assert to_source_files(files) == [SourceFile("src/main.py", "print('hi')")]
