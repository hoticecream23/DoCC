"""Background jobs for the cascade route.

Jobs live in memory only and do not survive a restart. A DB-backed store
replaces InMemoryJobStore later; it only has to satisfy the JobStore protocol.
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Literal, Protocol

from pydantic import BaseModel

from baseline.errors import Unavailable, summarize_exception
from baseline.schema import DocumentRecord

log = logging.getLogger(__name__)

DEFAULT_WORKERS = 1  # one GPU: one cascade document at a time
MAX_PENDING_JOBS = 8  # queued plus running; more answers 503
MAX_FINISHED_JOBS = 1000  # finished jobs kept for polling; the oldest go first


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

    def create(self) -> str:
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

    def create(self) -> str:
        """Create a queued job.

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


class JobRunner:
    """Runs jobs on a thread pool and records their state in a JobStore."""

    def __init__(self, store: JobStore, workers: int = DEFAULT_WORKERS, max_pending: int = MAX_PENDING_JOBS) -> None:
        self.store = store
        self._slots = threading.BoundedSemaphore(max_pending)
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="docc-job")

    def submit(self, fn: Callable[[], DocumentRecord], trace_id: str = "") -> str:
        """Queue a job.

        Args:
            fn: Produces the record; runs on a worker thread.
            trace_id: Request id, added to the job's log lines.

        Returns:
            The job id.

        Raises:
            Busy: MAX_PENDING_JOBS jobs are already queued or running.
        """
        if not self._slots.acquire(blocking=False):
            raise Busy("job queue is full; retry later")
        try:
            job_id = self.store.create()
            self._pool.submit(self._run, job_id, fn, trace_id)
        except Exception:
            self._slots.release()
            raise
        return job_id

    def _run(self, job_id: str, fn: Callable[[], DocumentRecord], trace_id: str) -> None:
        try:
            self._execute(job_id, fn, trace_id)
        finally:
            self._slots.release()

    def _execute(self, job_id: str, fn: Callable[[], DocumentRecord], trace_id: str) -> None:
        self.store.set_running(job_id)
        try:
            record = fn()
        except Exception as exc:  # noqa: BLE001 - job boundary: any failure marks the job failed
            log.error("job failed", extra={"job_id": job_id, "trace_id": trace_id, "error": summarize_exception(exc)})
            # Only the type: a message can embed document text.
            self.store.set_failed(job_id, f"processing failed: {type(exc).__name__}")
            return
        self.store.set_done(job_id, record)
        log.info("job done", extra={"job_id": job_id, "trace_id": trace_id})

    def shutdown(self) -> None:
        """Wait for running jobs and stop the pool."""
        self._pool.shutdown(wait=True)
