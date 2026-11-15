"""Background jobs for the cascade route.

Over 400 lines: the job stores, the one-process-per-store claim and the runner share one state model.

Job state lives in a JobStore: InMemoryJobStore (lost on restart) or
SqliteJobStore (the `jobs` table of the SQLite store). The runner is in
process, so a job cannot resume after a restart. Each job row carries the
boot id of the process that made it; the API calls `SqliteJobStore.sweep_stale`
once at startup, which fails only the queued or running jobs of another boot id.
A finished job (done or failed) is never written again.

One API process per store: at startup the API claims the store with
`SqliteJobStore.claim` (a row of `api_instances` with the store's own
instance id, a uuid4 per SqliteJobStore so per app, and a heartbeat that a
timer refreshes). A second app, in this process or another, fails with
ConfigError while the first one's heartbeat is live: wall-clock seconds no
older than API_INSTANCE_TIMEOUT_S and no further than that in the future (a
clock step back cannot keep a dead claim alive). Close removes only its own row.

A state write that raises sqlite3.Error is retried JOB_STATE_ATTEMPTS times
(each attempt already waits out the busy timeout). If every attempt fails, the
runner marks the job failed in memory, logs it, and `JobRunner.get` reports it.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import sqlite3
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel

from baseline import db
from baseline.errors import ConfigError, Unavailable, summarize_exception
from baseline.schema import DEFAULT_CABINET, DocumentRecord

log = logging.getLogger(__name__)

DEFAULT_WORKERS = 1  # one GPU: one cascade document at a time
MAX_PENDING_JOBS = 8  # queued plus running; more answers 503
MAX_FINISHED_JOBS = 1000  # finished jobs kept for polling; the oldest go first
STALE_JOB_ERROR = "server restarted before the job finished"
JOB_STATE_ATTEMPTS = 3
JOB_STATE_ERROR = "job state could not be saved"
# Identifies this process's jobs, so a sweep leaves a live process's jobs alone.
BOOT_ID = uuid.uuid4().hex
# Another API instance on the store counts as live while its heartbeat is younger than this.
API_INSTANCE_TIMEOUT_S = 30.0
HEARTBEAT_S = API_INSTANCE_TIMEOUT_S / 3
SECOND_INSTANCE_ERROR = (
    "another API process is serving this store; run one API process per store "
    "(no uvicorn --workers above 1), or wait {timeout:.0f} s after it stopped"
)


class Busy(Unavailable):
    """The job queue is full. The API answers 503 with Retry-After."""


JobState = Literal["queued", "running", "done", "failed"]


class Job(BaseModel):
    """One background job, as stored and as returned by GET /jobs/{job_id}."""

    job_id: str
    state: JobState = "queued"
    result: DocumentRecord | None = None
    error: str | None = None


class JobStore(Protocol):
    """Where job state lives. Every method is called from worker threads too."""

    def create(self, trace_id: str = "", cabinet_id: str = DEFAULT_CABINET) -> str:
        """Create a queued job and return its id."""

    def set_running(self, job_id: str) -> None:
        """Mark a job as running."""

    def set_done(self, job_id: str, result: DocumentRecord) -> None:
        """Mark a job as done with its record."""

    def set_failed(self, job_id: str, error: str) -> None:
        """Mark a job as failed with a message safe to show."""

    def get(self, job_id: str) -> Job | None:
        """Return a copy of the job, or None for an unknown id."""


class InMemoryJobStore:
    """A JobStore in a dict. Lost on restart."""

    def __init__(self, max_finished: int = MAX_FINISHED_JOBS) -> None:
        self._jobs: dict[str, Job] = {}
        self._finished: deque[str] = deque()
        self._max_finished = max_finished
        self._lock = threading.Lock()

    def create(self, trace_id: str = "", cabinet_id: str = DEFAULT_CABINET) -> str:
        """Create a queued job.

        Args:
            trace_id: Not kept in memory.
            cabinet_id: Not kept in memory.

        Returns:
            The new job id.
        """
        job_id = uuid.uuid4().hex
        with self._lock:
            self._jobs[job_id] = Job(job_id=job_id)
        return job_id

    def _update(self, job_id: str, **changes: object) -> None:
        with self._lock:
            self._jobs[job_id] = self._jobs[job_id].model_copy(update=changes)
            if changes["state"] in ("done", "failed"):
                self._finished.append(job_id)
                while len(self._finished) > self._max_finished:
                    del self._jobs[self._finished.popleft()]

    def set_running(self, job_id: str) -> None:
        """Mark a job as running.

        Args:
            job_id: The job.
        """
        self._update(job_id, state="running")

    def set_done(self, job_id: str, result: DocumentRecord) -> None:
        """Mark a job as done.

        Args:
            job_id: The job.
            result: The document record.
        """
        self._update(job_id, state="done", result=result)

    def set_failed(self, job_id: str, error: str) -> None:
        """Mark a job as failed.

        Args:
            job_id: The job.
            error: A message safe to show; never document text.
        """
        self._update(job_id, state="failed", error=error)

    def get(self, job_id: str) -> Job | None:
        """Look up a job.

        Args:
            job_id: The job.

        Returns:
            The job, or None for an unknown id.
        """
        with self._lock:
            return self._jobs.get(job_id)


def _now() -> str:
    # Microseconds, so eviction order holds for jobs that finish in the same second.
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


_EVICT = (
    "DELETE FROM jobs WHERE state IN ('done', 'failed') AND job_id NOT IN (SELECT job_id FROM jobs "
    "WHERE state IN ('done', 'failed') ORDER BY updated_at DESC, job_id DESC LIMIT ?)"
)


class SqliteJobStore:
    """A JobStore in the `jobs` table. Survives a restart; sweep_stale then fails another boot's unfinished jobs."""

    def __init__(
        self,
        path: str | Path,
        max_finished: int = MAX_FINISHED_JOBS,
        boot_id: str = BOOT_ID,
        instance_id: str | None = None,
    ) -> None:
        """Open the store and drop the oldest finished jobs.

        Args:
            path: SQLite file, shared with the document store.
            max_finished: Finished jobs kept; the oldest rows are deleted.
            boot_id: Stamped on every job this store creates; the process's BOOT_ID by default.
            instance_id: Key of this store's `api_instances` claim; None for a new uuid4.
        """
        self._conn = db.connect(path, threads=True)
        self._max_finished = max_finished
        self._boot_id = boot_id
        self.instance_id = instance_id or uuid.uuid4().hex
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._beater: threading.Thread | None = None
        with self._lock, self._conn:
            self._conn.execute(_EVICT, (max_finished,))

    def claim(self, timeout_s: float = API_INSTANCE_TIMEOUT_S) -> None:
        """Make this app the store's one API instance and start its heartbeat timer.

        Another instance's row is stale when its heartbeat is older than
        timeout_s or more than timeout_s in the future; stale rows are replaced.

        Args:
            timeout_s: Age after which another instance's heartbeat no longer counts.

        Raises:
            ConfigError: Another instance, in any process, has a live heartbeat.
        """
        now = time.time()
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                # The column is named boot_id; it holds the instance id.
                live = self._conn.execute(
                    "SELECT 1 FROM api_instances WHERE boot_id <> ? AND heartbeat_at > ? AND heartbeat_at < ?",
                    (self.instance_id, now - timeout_s, now + timeout_s),
                ).fetchone()
                if live is not None:
                    raise ConfigError(SECOND_INSTANCE_ERROR.format(timeout=timeout_s))
                self._conn.execute("DELETE FROM api_instances WHERE boot_id <> ?", (self.instance_id,))
                self._conn.execute(
                    "INSERT INTO api_instances (boot_id, started_at, heartbeat_at) VALUES (?, ?, ?) "
                    "ON CONFLICT (boot_id) DO UPDATE SET heartbeat_at = excluded.heartbeat_at",
                    (self.instance_id, _now(), now),
                )
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise
        self._beater = threading.Thread(target=self._heartbeat, name="docc-heartbeat", daemon=True)
        self._beater.start()

    def _heartbeat(self) -> None:
        """Refresh this instance's heartbeat every HEARTBEAT_S until close."""
        while not self._stop.wait(HEARTBEAT_S):
            try:
                with self._lock, self._conn:
                    self._conn.execute(
                        "UPDATE api_instances SET heartbeat_at = ? WHERE boot_id = ?", (time.time(), self.instance_id)
                    )
            except sqlite3.Error as exc:
                log.warning("api heartbeat not saved", extra={"error": summarize_exception(exc)})

    def sweep_stale(self) -> int:
        """Fail the queued or running jobs another boot left behind. The API runs it once, at startup.

        Returns:
            How many jobs were marked failed.
        """
        with self._lock, self._conn:
            stale = self._conn.execute(
                "UPDATE jobs SET state = 'failed', error = ?, updated_at = ? "
                "WHERE state IN ('queued', 'running') AND (boot_id IS NULL OR boot_id <> ?)",
                (STALE_JOB_ERROR, _now(), self._boot_id),
            ).rowcount
            self._conn.execute(_EVICT, (self._max_finished,))
        if stale:
            log.warning("stale jobs marked failed", extra={"count": stale})
        return int(stale)

    def create(self, trace_id: str = "", cabinet_id: str = DEFAULT_CABINET) -> str:
        """Create a queued job.

        Args:
            trace_id: Request id, stored with the job.
            cabinet_id: Cabinet the job writes into.

        Returns:
            The new job id.
        """
        job_id, now = uuid.uuid4().hex, _now()
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO jobs (job_id, cabinet_id, state, trace_id, created_at, updated_at, boot_id) "
                "VALUES (?, ?, 'queued', ?, ?, ?, ?)",
                (job_id, cabinet_id, trace_id, now, now, self._boot_id),
            )
        return job_id

    def _update(
        self, job_id: str, state: JobState, result: DocumentRecord | None = None, error: str | None = None
    ) -> None:
        doc_id = result.doc_id if result is not None else None
        body = result.model_dump_json() if result is not None else None
        # A finished job, such as one a sweep failed, is left as it is.
        with self._lock, self._conn:
            changed = self._conn.execute(
                "UPDATE jobs SET state = ?, doc_id = ?, error = ?, result_json = ?, updated_at = ? "
                "WHERE job_id = ? AND state IN ('queued', 'running')",
                (state, doc_id, error, body, _now(), job_id),
            ).rowcount
            if state in ("done", "failed"):
                self._conn.execute(_EVICT, (self._max_finished,))
        if not changed:
            log.warning("job already finished, state not changed", extra={"job_id": job_id, "state": state})

    def set_running(self, job_id: str) -> None:
        """Mark a job as running.

        Args:
            job_id: The job.
        """
        self._update(job_id, "running")

    def set_done(self, job_id: str, result: DocumentRecord) -> None:
        """Mark a job as done.

        Args:
            job_id: The job.
            result: The document record.
        """
        self._update(job_id, "done", result=result)

    def set_failed(self, job_id: str, error: str) -> None:
        """Mark a job as failed.

        Args:
            job_id: The job.
            error: A message safe to show; never document text.
        """
        self._update(job_id, "failed", error=error)

    def get(self, job_id: str) -> Job | None:
        """Look up a job.

        Args:
            job_id: The job.

        Returns:
            The job, or None for an unknown id.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT state, error, result_json FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        if row is None:
            return None
        result = DocumentRecord.model_validate_json(row["result_json"]) if row["result_json"] else None
        return Job(job_id=job_id, state=row["state"], result=result, error=row["error"])

    def close(self) -> None:
        """Stop the heartbeat, give up a claim, and close the connection."""
        self._stop.set()
        if self._beater is not None:
            self._beater.join()
            with self._lock, contextlib.suppress(sqlite3.Error), self._conn:
                self._conn.execute("DELETE FROM api_instances WHERE boot_id = ?", (self.instance_id,))
        with self._lock:
            self._conn.close()


class JobRunner:
    """Runs jobs on a thread pool and records their state in a JobStore."""

    def __init__(self, store: JobStore, workers: int = DEFAULT_WORKERS, max_pending: int = MAX_PENDING_JOBS) -> None:
        self.store = store
        self._lost: dict[str, Job] = {}
        self._slots = threading.BoundedSemaphore(max_pending)
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="docc-job")

    def submit(
        self,
        fn: Callable[[], DocumentRecord],
        trace_id: str = "",
        cabinet_id: str = DEFAULT_CABINET,
        cleanup: Callable[[], None] | None = None,
    ) -> str:
        """Queue a job.

        Args:
            fn: Produces the record; runs on a worker thread.
            trace_id: Request id, added to the job's log lines.
            cabinet_id: Cabinet the job writes into.
            cleanup: Runs once the job ends, whether fn ran or not; not run if submit raises.

        Returns:
            The job id.

        Raises:
            Busy: MAX_PENDING_JOBS jobs are already queued or running.
        """
        if not self._slots.acquire(blocking=False):
            raise Busy("job queue is full; retry later")
        try:
            job_id = self.store.create(trace_id, cabinet_id)
            # The job runs in the submitter's context, so the API's log scrubbing sees its upload.
            self._pool.submit(contextvars.copy_context().run, self._run, job_id, fn, trace_id, cleanup)
        except Exception:  # noqa: BLE001 - free the queue slot on any submit failure, then re-raise
            self._slots.release()
            raise
        return job_id

    def _run(
        self, job_id: str, fn: Callable[[], DocumentRecord], trace_id: str, cleanup: Callable[[], None] | None
    ) -> None:
        try:
            self._execute(job_id, fn, trace_id)
        finally:
            try:
                if cleanup is not None:
                    cleanup()
            finally:
                self._slots.release()

    def get(self, job_id: str) -> Job | None:
        """Look up a job; one whose state could not be saved reads as failed.

        Args:
            job_id: The job.

        Returns:
            The job, or None for an unknown id.
        """
        return self._lost.get(job_id) or self.store.get(job_id)

    def _save(self, job_id: str, write: Callable[[], None]) -> bool:
        """Run one state write, retried; after the last failure mark the job failed in memory."""
        for attempt in range(1, JOB_STATE_ATTEMPTS + 1):
            try:
                write()
                return True
            except sqlite3.Error as exc:
                extra = {"job_id": job_id, "attempt": attempt, "error": summarize_exception(exc)}
                log.warning("job state write failed", extra=extra)
        self._lost[job_id] = Job(job_id=job_id, state="failed", error=JOB_STATE_ERROR)
        log.error("job state not saved, marked failed in memory", extra={"job_id": job_id})
        return False

    def _execute(self, job_id: str, fn: Callable[[], DocumentRecord], trace_id: str) -> None:
        if not self._save(job_id, lambda: self.store.set_running(job_id)):
            return
        try:
            record = fn()
        except Exception as exc:  # noqa: BLE001 - job boundary: any failure marks the job failed
            log.error("job failed", extra={"job_id": job_id, "trace_id": trace_id, "error": summarize_exception(exc)})
            # Only the type: a message can embed document text.
            error = f"processing failed: {type(exc).__name__}"
            self._save(job_id, lambda: self.store.set_failed(job_id, error))
            return
        if self._save(job_id, lambda: self.store.set_done(job_id, record)):
            log.info("job done", extra={"job_id": job_id, "trace_id": trace_id})

    def shutdown(self) -> None:
        """Wait for running jobs and stop the pool."""
        self._pool.shutdown(wait=True)
