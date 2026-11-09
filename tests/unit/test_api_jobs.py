import threading

import pytest

pytest.importorskip("fastapi")

from api.jobs import InMemoryJobStore, JobRunner  # noqa: E402
from baseline.schema import DocumentRecord  # noqa: E402

RECORD = DocumentRecord.model_construct(doc_id="0" * 16)


def _wait_for(store, job_id, state, timeout=5.0):
    import time

    end = time.monotonic() + timeout
    while store.get(job_id).state != state and time.monotonic() < end:
        time.sleep(0.01)
    return store.get(job_id)


def test_new_job_is_queued():
    store = InMemoryJobStore()
    job = store.get(store.create())
    assert job.state == "queued" and job.result is None and job.error is None


def test_unknown_job_is_none():
    assert InMemoryJobStore().get("nope") is None


def test_job_runs_then_is_done():
    gate = threading.Event()
    runner = JobRunner(InMemoryJobStore())

    def work():
        gate.wait(5)
        return RECORD

    job_id = runner.submit(work)
    assert _wait_for(runner.store, job_id, "running").state == "running"
    gate.set()
    runner.shutdown()
    job = runner.store.get(job_id)
    assert job.state == "done" and job.result is RECORD


def test_second_job_waits_for_the_single_worker():
    gate = threading.Event()
    runner = JobRunner(InMemoryJobStore(), workers=1)
    runner.submit(lambda: gate.wait(5) and RECORD)
    second = runner.submit(lambda: RECORD)
    try:
        assert runner.store.get(second).state == "queued"
    finally:
        gate.set()
        runner.shutdown()


def test_failed_job_keeps_only_the_error_type():
    runner = JobRunner(InMemoryJobStore())

    def work():
        raise ValueError("secret document text")

    job_id = runner.submit(work)
    runner.shutdown()
    job = runner.store.get(job_id)
    assert job.state == "failed" and job.error == "processing failed: ValueError"
