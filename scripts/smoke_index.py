"""Index a local source folder and print its most relevant chunks."""

import argparse
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.rag.factory import build_indexing_service, build_vector_store
from app.rag.models import SourceFile

ALLOWED = {
    ".py",
    ".md",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".go",
    ".rs",
    ".html",
    ".txt",
}
SKIP_DIRS = {".git", "node_modules", ".venv", "__pycache__", "data"}
MAX_BYTES = 200_000


def read_folder(root: Path) -> list[SourceFile]:
    files: list[SourceFile] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in ALLOWED:
            continue
        if SKIP_DIRS.intersection(path.parts) or path.stat().st_size > MAX_BYTES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        files.append(SourceFile(path.relative_to(root).as_posix(), text))
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    parser.add_argument("query")
    args = parser.parse_args()
    repo_id = "local/smoke"
    result = build_indexing_service().index_repo(
        repo_id, read_folder(Path(args.folder))
    )
    print(result)
    for hit in build_vector_store().similarity_search(repo_id, args.query, k=3):
        print(f"\n[{hit.score:.3f}] {hit.path}:{hit.start_line}")
        print(hit.content[:300])


if __name__ == "__main__":
    main()
