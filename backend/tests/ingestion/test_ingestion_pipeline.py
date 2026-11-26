import hashlib
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from app.database.models import RepoStatus
from app.database.repository import RepoRepository
from app.ingestion.pipeline import (
    IngestionPipeline,
    PipelineConfig,
    PipelineDeps,
    RepoTooLargeError,
)
from app.ingestion.types import Chunk, RepoInfo, SourceFile
from app.retrieval.vector_store import (
    ChromaVectorStore,
    CollectionNotFoundError,
    collection_name_for,
)

URL = "https://github.com/octo/mini"
SHA = "a" * 40
FAIL_POINTS = {
    "download": "download_archive",
    "extract": "extract_archive",
    "filter": "list_files",
    "read": "read_file",
    "chunk": "chunk_file",
}


class FakeEmbedder:
    """Stable 16-dimensional unit vectors for deterministic pipeline checks."""

    def _vec(self, text: str) -> list[float]:
        raw = [value / 255 for value in hashlib.sha256(text.encode()).digest()[:16]]
        norm = sum(value * value for value in raw) ** 0.5
        return [value / norm for value in raw]

    def embed_documents(self, texts):
        return [self._vec(text) for text in texts]

    def embed_query(self, text):
        return self._vec(text)


class ExplodingEmbedder:
    def embed_documents(self, texts):
        raise RuntimeError("boom")


class ExplodingStore:
    """Delegate reads and deletes to Chroma while making writes fail."""

    def __init__(self, inner):
        self._inner = inner

    def add_chunks(self, *args, **kwargs):
        raise RuntimeError("boom")

    def search(self, *args, **kwargs):
        return self._inner.search(*args, **kwargs)

    def delete_collection(self, *args, **kwargs):
        return self._inner.delete_collection(*args, **kwargs)


def _boom(*_args, **_kwargs):
    raise RuntimeError("boom")


def _fake_download(fixture_root: Path):
    def download(owner, name, sha, dest_dir, timeout):
        archive = dest_dir / "repo.zip"
        with zipfile.ZipFile(archive, "w") as output:
            for path in fixture_root.rglob("*"):
                if path.is_file():
                    inner = path.relative_to(fixture_root).as_posix()
                    output.write(path, f"{owner}-{name}-{sha[:7]}/{inner}")
        return archive

    return download


def _extract(archive: Path, destination: Path) -> Path:
    with zipfile.ZipFile(archive) as source:
        source.extractall(destination)
    (root,) = [path for path in destination.iterdir() if path.is_dir()]
    return root


def _list_files(root: Path):
    return sorted(path for path in root.rglob("*") if path.is_file())


def _read(root: Path, path: Path) -> SourceFile:
    return SourceFile(
        path.relative_to(root).as_posix(), path.read_text(encoding="utf-8")
    )


def _chunk(source: SourceFile, lines_per_chunk: int = 20) -> list[Chunk]:
    lines = source.content.splitlines()
    return [
        Chunk(
            text="\n".join(lines[start : start + lines_per_chunk]),
            file_path=source.path,
            start_line=start + 1,
            end_line=min(start + lines_per_chunk, len(lines)),
        )
        for start in range(0, len(lines), lines_per_chunk)
    ]


@dataclass
class Env:
    repository: RepoRepository
    store: ChromaVectorStore
    temp_root: Path
    config: PipelineConfig
    deps: PipelineDeps

    def broken(self, step: str) -> PipelineDeps:
        if step == "embed":
            return replace(self.deps, embedder=ExplodingEmbedder())
        if step == "store":
            return replace(self.deps, vector_store=ExplodingStore(self.store))
        return replace(self.deps, **{FAIL_POINTS[step]: _boom})


@pytest.fixture
def fixture_repo(tmp_path: Path) -> Path:
    root = tmp_path / "fixture_repo"
    (root / "src").mkdir(parents=True)
    (root / "docs").mkdir()
    (root / "README.md").write_text("# Mini\n\nA tiny demo project.\n" * 3)
    (root / "src" / "app.py").write_text("def add(a, b):\n return a + b\n\n" * 10)
    (root / "docs" / "guide.md").write_text("Install with pip.\n" * 30)
    return root


@pytest.fixture
def env(tmp_path: Path, fixture_repo: Path) -> Env:
    repository = RepoRepository(tmp_path / "repos.db")
    store = ChromaVectorStore(tmp_path / "chroma")
    deps = PipelineDeps(
        parse_url=lambda url: ("octo", "mini"),
        fetch_metadata=lambda owner, name: RepoInfo(
            owner=owner, name=name, commit_sha=SHA, size_kb=10
        ),
        download_archive=_fake_download(fixture_repo),
        extract_archive=_extract,
        list_files=_list_files,
        read_file=_read,
        chunk_file=_chunk,
        embedder=FakeEmbedder(),
        vector_store=store,
        repository=repository,
    )
    config = PipelineConfig(
        timeout_seconds=60, temp_root=tmp_path / "tmp", embed_batch_size=2
    )
    return Env(repository, store, tmp_path / "tmp", config, deps)


def test_end_to_end_pipeline_indexes_and_retrieves_fixture_repo(env: Env) -> None:
    record = IngestionPipeline(env.deps, env.config).ingest(URL)
    assert record.status is RepoStatus.READY
    assert record.error is None
    assert record.files_count == 3
    assert record.chunks_count == 5
    assert list(env.temp_root.iterdir()) == []
    question = "\n".join(["Install with pip."] * 20)
    hits = env.store.search(
        collection_name_for(record.repo_id),
        env.deps.embedder.embed_query(question),
        top_k=3,
    )
    assert hits[0].metadata["file_path"] == "docs/guide.md"
    assert hits[0].metadata["start_line"] == 1
    assert hits[0].score > 0.99


def test_same_repo_and_commit_is_deduplicated(env: Env) -> None:
    pipeline = IngestionPipeline(env.deps, env.config)
    first = pipeline.ingest(URL)
    again = pipeline.submit(URL)
    assert not again.claimed
    assert again.record.repo_id == first.repo_id
    assert again.record.status is RepoStatus.READY


@pytest.mark.parametrize(
    "step", ["download", "extract", "filter", "read", "chunk", "embed", "store"]
)
def test_pipeline_failures_mark_failed_and_clean_up(env: Env, step: str) -> None:
    record = IngestionPipeline(env.broken(step), env.config).ingest(URL)
    assert record.status is RepoStatus.FAILED
    assert f"[{step}]" in record.error
    assert "boom" in record.error
    assert list(env.temp_root.iterdir()) == []
    with pytest.raises(CollectionNotFoundError):
        env.store.search(collection_name_for(record.repo_id), [0.1] * 16)


def test_failed_job_can_be_retried(env: Env) -> None:
    failed = IngestionPipeline(env.broken("download"), env.config).ingest(URL)
    assert failed.status is RepoStatus.FAILED
    retried = IngestionPipeline(env.deps, env.config).ingest(URL)
    assert retried.repo_id == failed.repo_id
    assert retried.status is RepoStatus.READY
    assert retried.error is None


def test_oversized_repository_is_rejected_before_claim(env: Env) -> None:
    deps = replace(
        env.deps,
        fetch_metadata=lambda owner, name: RepoInfo(
            owner=owner, name=name, commit_sha=SHA, size_kb=10**9
        ),
    )
    with pytest.raises(RepoTooLargeError):
        IngestionPipeline(deps, env.config).submit(URL)


def test_overall_deadline_fails_job_and_cleans_up(env: Env) -> None:
    config = replace(env.config, timeout_seconds=0)
    record = IngestionPipeline(env.deps, config).ingest(URL)
    assert record.status is RepoStatus.FAILED
    assert "deadline" in record.error
    assert list(env.temp_root.iterdir()) == []
