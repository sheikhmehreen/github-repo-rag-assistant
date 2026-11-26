import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from app.database.models import RepoStatus
from app.database.repository import RepoNotFoundError, RepoRepository

URL = "https://github.com/octo/mini"
JOB = {"url": URL, "owner": "octo", "name": "mini", "commit_sha": "a" * 40}


@pytest.fixture
def repo(tmp_path) -> RepoRepository:
    return RepoRepository(tmp_path / "repos.db")


def test_first_submit_creates_pending_record_and_claims(repo: RepoRepository) -> None:
    result = repo.submit(**JOB)
    assert result.claimed
    assert result.record.status is RepoStatus.PENDING
    assert len(result.record.repo_id) == 32


def test_duplicate_request_returns_existing_record(repo: RepoRepository) -> None:
    first = repo.submit(**JOB)
    repo.mark_processing(first.record.repo_id)
    second = repo.submit(**JOB)
    assert not second.claimed
    assert second.record.repo_id == first.record.repo_id
    assert second.record.status is RepoStatus.PROCESSING


def test_owner_and_name_are_case_insensitive(repo: RepoRepository) -> None:
    first = repo.submit(url=URL, owner="Octo", name="Mini", commit_sha="b" * 40)
    second = repo.submit(url=URL, owner="octo", name="mini", commit_sha="b" * 40)
    assert first.record.repo_id == second.record.repo_id


def test_new_commit_gets_its_own_record(repo: RepoRepository) -> None:
    first = repo.submit(**JOB)
    second = repo.submit(**{**JOB, "commit_sha": "c" * 40})
    assert second.claimed
    assert second.record.repo_id != first.record.repo_id


def test_simultaneous_requests_create_one_job(repo: RepoRepository) -> None:
    workers = 8
    barrier = threading.Barrier(workers, timeout=10)

    def attempt(_index: int):
        barrier.wait()
        return repo.submit(**JOB)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(attempt, range(workers)))
    assert sum(result.claimed for result in results) == 1
    assert len({result.record.repo_id for result in results}) == 1


def test_failed_job_can_be_retried(repo: RepoRepository) -> None:
    repo_id = repo.submit(**JOB).record.repo_id
    repo.mark_processing(repo_id)
    failed = repo.mark_failed(repo_id, "[download] boom")
    assert failed.status is RepoStatus.FAILED
    assert failed.error == "[download] boom"
    retry = repo.submit(**JOB)
    assert retry.claimed
    assert retry.record.repo_id == repo_id
    assert retry.record.status is RepoStatus.PENDING
    assert retry.record.error is None
    assert not repo.submit(**JOB).claimed


def test_ready_repo_is_not_reclaimed(repo: RepoRepository) -> None:
    repo_id = repo.submit(**JOB).record.repo_id
    repo.mark_ready(repo_id, files_count=1, chunks_count=1)
    again = repo.submit(**JOB)
    assert not again.claimed
    assert again.record.status is RepoStatus.READY


def test_stale_processing_job_can_be_reclaimed(tmp_path) -> None:
    repo = RepoRepository(tmp_path / "repos.db", stale_after_seconds=0)
    repo_id = repo.submit(**JOB).record.repo_id
    repo.mark_processing(repo_id)
    time.sleep(0.01)
    assert repo.submit(**JOB).claimed


def test_status_changes_persist_across_instances(tmp_path) -> None:
    db = tmp_path / "repos.db"
    first = RepoRepository(db)
    repo_id = first.submit(**JOB).record.repo_id
    first.mark_processing(repo_id)
    first.mark_ready(repo_id, files_count=12, chunks_count=340)
    reopened = RepoRepository(db).get(repo_id)
    assert reopened.status is RepoStatus.READY
    assert (reopened.files_count, reopened.chunks_count) == (12, 340)
    assert reopened.error is None


def test_unknown_repo_raises(repo: RepoRepository) -> None:
    with pytest.raises(RepoNotFoundError):
        repo.get("missing")
    with pytest.raises(RepoNotFoundError):
        repo.mark_processing("missing")


def test_long_errors_are_truncated(repo: RepoRepository) -> None:
    repo_id = repo.submit(**JOB).record.repo_id
    failed = repo.mark_failed(repo_id, "x" * 5000)
    assert len(failed.error) == 1000
