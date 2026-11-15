"""Regression tests for the job sweep and job state writes a review reproduced."""

import sqlite3
from contextlib import closing

import pytest

pytest.importorskip("fastapi")

from api.jobs import JOB_STATE_ATTEMPTS, JOB_STATE_ERROR, InMemoryJobStore, JobRunner, SqliteJobStore  # noqa: E402
from baseline import db  # noqa: E402
from baseline.schema import DocumentRecord  # noqa: E402

RECORD = DocumentRecord.model_construct(doc_id="0" * 16)


def _real_record():
    from test_db import _record

    return _record("a" * 16)


def test_opening_the_store_from_another_process_leaves_a_live_job_queued(tmp_path):
    live = SqliteJobStore(tmp_path / "docc.sqlite", boot_id="live")
    job_id = live.create()
    db.connect(tmp_path / "docc.sqlite").close()
    other = SqliteJobStore(tmp_path / "docc.sqlite", boot_id="other")
    assert other.get(job_id).state == "queued"
    live.close()
    other.close()


def test_a_sweep_leaves_its_own_boots_jobs_alone(tmp_path):
    store = SqliteJobStore(tmp_path / "docc.sqlite", boot_id="live")
    job_id = store.create()
    assert store.sweep_stale() == 0 and store.get(job_id).state == "queued"
    store.close()


def test_a_job_failed_by_a_sweep_is_not_flipped_back_to_done(tmp_path):
    old = SqliteJobStore(tmp_path / "docc.sqlite", boot_id="old")
    job_id = old.create()
    old.set_running(job_id)
    new = SqliteJobStore(tmp_path / "docc.sqlite", boot_id="new")
    new.sweep_stale()
    old.set_done(job_id, _real_record())
    assert new.get(job_id).state == "failed"
    old.close()
    new.close()


class _Flaky(InMemoryJobStore):
    """Raises sqlite3.Error on the first `fails` calls of one method."""

    def __init__(self, method, fails):
        super().__init__()
        self.method, self.fails = method, fails

    def _maybe_fail(self, name):
        if name == self.method and self.fails > 0:
            self.fails -= 1
            raise sqlite3.OperationalError("database is locked")

    def set_running(self, job_id):
        self._maybe_fail("set_running")
        super().set_running(job_id)

    def set_done(self, job_id, result):
        self._maybe_fail("set_done")
        super().set_done(job_id, result)


def test_a_state_write_that_fails_once_is_retried(tmp_path):
    runner = JobRunner(_Flaky("set_running", 1))
    job_id = runner.submit(lambda: RECORD)
    runner.shutdown()
    assert runner.get(job_id).state == "done"


def test_a_state_write_that_keeps_failing_marks_the_job_failed_in_memory():
    runner = JobRunner(_Flaky("set_done", JOB_STATE_ATTEMPTS))
    job_id = runner.submit(lambda: RECORD)
    runner.shutdown()
    job = runner.get(job_id)
    assert (job.state, job.error) == ("failed", JOB_STATE_ERROR)


def test_a_failed_running_write_releases_the_queue_slot():
    runner = JobRunner(_Flaky("set_running", JOB_STATE_ATTEMPTS), max_pending=1)
    runner.submit(lambda: RECORD)
    runner.shutdown()
    assert runner._slots.acquire(blocking=False)


def test_a_new_job_carries_the_stores_boot_id(tmp_path):
    store = SqliteJobStore(tmp_path / "docc.sqlite", boot_id="b1")
    job_id = store.create()
    store.close()
    with closing(db.connect(tmp_path / "docc.sqlite")) as conn:
        assert conn.execute("SELECT boot_id FROM jobs WHERE job_id = ?", (job_id,)).fetchone()[0] == "b1"
