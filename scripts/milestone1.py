"""Ingest three public repositories and inspect their top retrieval results."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.database.models import RepoStatus
from app.ingestion.wiring import build_pipeline, get_embedding_service, get_vector_store
from app.retrieval.vector_store import collection_name_for

TOP_K = 5
SAMPLES = [
    (
        "small",
        "https://github.com/kennethreitz/records",
        [
            "How do I run a SQL query and read the rows?",
            "How do I export results to CSV?",
            "Which databases are supported?",
        ],
    ),
    (
        "medium",
        "https://github.com/pallets/flask",
        [
            "How do I define a route with a URL variable?",
            "How do I organise an app with blueprints?",
            "How is configuration loaded?",
        ],
    ),
    (
        "docs-heavy",
        "https://github.com/jlevy/the-art-of-command-line",
        [
            "How do I find files by name?",
            "How do I see which process is using a port?",
            "How do I work with JSON on the command line?",
        ],
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=[sample[0] for sample in SAMPLES])
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    pipeline = build_pipeline()
    embedder = get_embedding_service()
    store = get_vector_store()
    samples = [
        sample for sample in SAMPLES if args.only is None or sample[0] == args.only
    ]
    for label, url, questions in samples:
        print("=" * 78)
        print(f"[{label}] {url}")
        started = time.perf_counter()
        record = pipeline.ingest(url)
        elapsed = time.perf_counter() - started
        print(
            f"status={record.status.value} files={record.files_count} "
            f"chunks={record.chunks_count} time={elapsed:.1f}s"
        )
        if record.status is not RepoStatus.READY:
            print(f"ERROR: {record.error}")
            continue
        collection = collection_name_for(record.repo_id)
        for question in questions:
            print(f"\nQ: {question}")
            hits = store.search(collection, embedder.embed_query(question), top_k=TOP_K)
            for rank, hit in enumerate(hits, start=1):
                path = hit.metadata.get("file_path")
                start = hit.metadata.get("start_line")
                end = hit.metadata.get("end_line")
                preview = " ".join(hit.text.split())[:240]
                print(f" {rank}. score={hit.score:.3f} {path}:{start}-{end}")
                print(f"    {preview}")


if __name__ == "__main__":
    main()
