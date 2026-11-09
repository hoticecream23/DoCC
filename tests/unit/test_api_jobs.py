import threading

import pytest

pytest.importorskip("fastapi")

from api.jobs import Busy, InMemoryJobStore, JobRunner  # noqa: E402
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


def test_full_queue_raises_busy_and_frees_a_slot_when_a_job_ends():
    gate = threading.Event()
    runner = JobRunner(InMemoryJobStore(), max_pending=1)
    first = runner.submit(lambda: gate.wait(5) and RECORD)
    with pytest.raises(Busy):
        runner.submit(lambda: RECORD)
    gate.set()
    _wait_for(runner.store, first, "done")
    assert _submit_when_free(runner)
    runner.shutdown()


def _submit_when_free(runner, timeout=5.0):
    import time

    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            return runner.submit(lambda: RECORD)
        except Busy:
            time.sleep(0.01)
    return None


def test_oldest_finished_job_is_evicted():
    store = InMemoryJobStore(max_finished=1)
    old, new, waiting = store.create(), store.create(), store.create()
    store.set_done(old, RECORD)
    store.set_failed(new, "x")
    assert store.get(old) is None and store.get(new).state == "failed" and store.get(waiting).state == "queued"


def _sqlite_store(tmp_path, **kwargs):
    from api.jobs import SqliteJobStore

    return SqliteJobStore(tmp_path / "docc.sqlite", **kwargs)


def _real_record():
    from test_db import _record

    return _record("a" * 16)


def test_sqlite_job_is_visible_from_a_second_store_on_the_same_file(tmp_path):
    first = _sqlite_store(tmp_path)
    job_id = first.create("trace-1", "x")
    first.set_running(job_id)
    first.set_done(job_id, _real_record())
    second = _sqlite_store(tmp_path)
    job = second.get(job_id)
    assert job.state == "done" and job.result.doc_id == "a" * 16 and job.error is None
    assert second.get("nope") is None
    first.close()
    second.close()


def test_sqlite_job_left_running_is_failed_on_reopen(tmp_path):
    from api.jobs import STALE_JOB_ERROR

    first = _sqlite_store(tmp_path)
    running, queued = first.create(), first.create()
    first.set_running(running)
    first.close()
    second = _sqlite_store(tmp_path)
    for job_id in (running, queued):
        assert second.get(job_id).state == "failed" and second.get(job_id).error == STALE_JOB_ERROR
    second.close()


def test_sqlite_oldest_finished_job_is_evicted(tmp_path):
    store = _sqlite_store(tmp_path, max_finished=1)
    old, new, waiting = store.create(), store.create(), store.create()
    store.set_failed(old, "x")
    store.set_failed(new, "y")
    assert store.get(old) is None and store.get(new).error == "y" and store.get(waiting).state == "queued"
    store.close()
