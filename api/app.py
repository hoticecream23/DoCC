"""FastAPI shell over the baseline and cascade pipelines.

Baseline runs synchronously in the request. Cascade runs as a background job
(12-40 s per document), polled at GET /jobs/{job_id}. Without a VLM model the
cascade route answers 503, like the demo server.

With `db.path` set (a relative path resolves in `root`), every record the API
produces is upserted into the SQLite store, jobs live in its `jobs` table, and
the document read routes serve from it. With `db.path: null` jobs stay in
memory and the read routes answer 503.
"""

# Over 400 lines: the routes, the error mapping and the body limit form one HTTP surface; splitting would scatter it.

from __future__ import annotations

import contextlib
import logging
import re
import shutil
import sqlite3
import tempfile
import threading
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, FastAPI, Form, HTTPException, Query, Request, Response, UploadFile, status
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from api.jobs import DEFAULT_WORKERS, Busy, InMemoryJobStore, Job, JobRunner, SqliteJobStore
from baseline import db
from baseline.config import DEFAULT_DB_NAME, Config, load_config
from baseline.errors import BadInput, ConfigError, DoccError, Unavailable, summarize_exception
from baseline.extract import SUPPORTED_EXT
from baseline.pipeline import Models, process_document
from baseline.schema import DocumentRecord
from cascade.config import CascadeConfig, VLMSettings, load_cascade_config, require_model, vlm_settings
from cascade.runner import Provenance
from cascade.runner import process_document as cascade_process
from cascade.vlm import Backend, VLMClient

log = logging.getLogger(__name__)

PREFIX = "/api/v1"
MEGABYTE = 1024 * 1024
MAX_UPLOAD_MB = 50
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * MEGABYTE
CHUNK_BYTES = MEGABYTE
# Room for multipart boundaries and part headers on top of the file itself.
MULTIPART_OVERHEAD_BYTES = 64 * 1024
RETRY_AFTER_S = 30
HTTP_TOO_LARGE = 413
TRACE_HEADER = "X-Request-ID"
# A client id is echoed in a header and logged, so only a plain token is kept.
TRACE_ID_RE = re.compile(r"[A-Za-z0-9._-]{1,64}")
# A cabinet id is stored and filtered on, so it is held to the same plain token.
CABINET_ID_RE = TRACE_ID_RE
DEFAULT_PAGE_LIMIT = 50
MAX_PAGE_LIMIT = 500
LIST_COLUMNS = ("doc_id", "cabinet_id", "doc_type", "confidence", "filename", "updated_at")

trace_id: ContextVar[str] = ContextVar("trace_id", default="")


def _discard(path: Path) -> None:
    shutil.rmtree(path.parent, ignore_errors=True)


def _copy_limited(upload: UploadFile, target: Path) -> None:
    size = 0
    with target.open("wb") as out:
        while chunk := upload.file.read(CHUNK_BYTES):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                raise BadInput(f"file larger than {MAX_UPLOAD_MB} MB")
            out.write(chunk)
    if size == 0:
        raise BadInput("empty file")


@dataclass
class ApiState:
    """Config, models, the job runner, the store and the lazily built VLM backend."""

    cfg: Config
    ccfg: CascadeConfig
    models: Models
    vlm: VLMSettings
    upload_dir: Path | None
    runner: JobRunner
    conn: sqlite3.Connection | None = None
    _backend: list[Backend] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    db_lock: threading.Lock = field(default_factory=threading.Lock)

    def backend(self) -> Backend:
        """Return the VLM backend, building it on first use.

        Returns:
            The shared backend.

        Raises:
            ConfigError: No model or endpoint is set, or the host is not allowed.
        """
        require_model(self.vlm)
        with self._lock:
            if not self._backend:
                try:
                    self._backend.append(VLMClient(self.vlm))
                except ValueError as e:
                    raise ConfigError(f"VLM settings rejected: {e}") from e
        return self._backend[0]

    def cabinet(self, value: str | None) -> str:
        """The cabinet a request writes into.

        Args:
            value: The form field; None or empty for the config's cabinet_id.

        Returns:
            The cabinet id.

        Raises:
            BadInput: The id is not a plain token.
        """
        if not value:
            return self.cfg.cabinet_id
        if not CABINET_ID_RE.fullmatch(value):
            raise BadInput("cabinet_id must be 1-64 letters, digits, '.', '_' or '-'")
        return value

    def persist(self, record: DocumentRecord, provenance: Provenance | None = None) -> None:
        """Upsert a record, and its cascade provenance, into the store if there is one.

        A store failure is logged, never raised: the caller still gets the record.

        Args:
            record: The record the API produced.
            provenance: The cascade provenance line, if any.
        """
        if self.conn is None:
            return
        try:
            with self.db_lock:
                db.upsert_document(self.conn, record)
                if provenance is not None:
                    db.upsert_provenance(self.conn, dict(provenance))
        except sqlite3.Error as exc:
            extra = {"trace_id": trace_id.get(), "doc_id": record.doc_id, "error": summarize_exception(exc)}
            log.error("db write failed, record returned but not stored", extra=extra)

    def read(self, fn: Callable[[sqlite3.Connection], Any]) -> Any:
        """Run a read against the store under its lock.

        Args:
            fn: Takes the connection and returns the result.

        Returns:
            What fn returns.

        Raises:
            Unavailable: db.path is null, so there is no store.
        """
        if self.conn is None:
            raise Unavailable("document store is off; set db.path in pipeline.yaml")
        with self.db_lock:
            return fn(self.conn)

    def save(self, upload: UploadFile) -> Path:
        """Write an upload to a new temp directory under its own base name.

        Args:
            upload: The uploaded file.

        Returns:
            The saved file; remove its parent directory when done.

        Raises:
            BadInput: Bad name, unsupported extension, empty, or over MAX_UPLOAD_BYTES.
        """
        name = Path((upload.filename or "").replace("\\", "/")).name
        if not name or name.startswith("."):
            raise BadInput(f"invalid file name: {upload.filename!r}")
        if Path(name).suffix.lower() not in SUPPORTED_EXT:
            raise BadInput(f"unsupported file type; use one of {sorted(SUPPORTED_EXT)}")
        target = Path(tempfile.mkdtemp(prefix="docc_api_", dir=self.upload_dir)) / name
        try:
            _copy_limited(upload, target)
        except BadInput:
            _discard(target)
            raise
        return target


CabinetForm = Annotated[str | None, Form()]


def _routes(api: ApiState) -> APIRouter:
    router = APIRouter(prefix=PREFIX)

    @router.get("/health")
    def health() -> dict[str, bool | str]:
        return {"status": "ok", "baseline": True, "cascade": bool(api.vlm.model and api.vlm.base_url)}

    @router.post("/documents/process", response_model=DocumentRecord)
    def process(file: UploadFile, cabinet_id: CabinetForm = None) -> DocumentRecord:
        cabinet = api.cabinet(cabinet_id)
        path = api.save(file)
        try:
            record, _ = process_document(path, api.cfg, api.models, cabinet)
        finally:
            _discard(path)
        api.persist(record)
        return record

    @router.post("/documents/process_async", status_code=status.HTTP_202_ACCEPTED)
    def process_async(file: UploadFile, cabinet_id: CabinetForm = None) -> dict[str, str]:
        cabinet = api.cabinet(cabinet_id)
        backend = api.backend()  # 503 before anything is saved
        path = api.save(file)

        def work() -> DocumentRecord:
            try:
                result = cascade_process(path, api.cfg, api.ccfg, backend, api.models, cabinet)
            finally:
                _discard(path)
            api.persist(result.record, result.provenance)
            return result.record

        try:
            return {"job_id": api.runner.submit(work, trace_id.get(), cabinet)}
        except Busy:
            _discard(path)
            raise

    @router.get("/jobs/{job_id}", response_model=Job)
    def job(job_id: str) -> Job:
        found = api.runner.store.get(job_id)
        if found is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown job id")
        return found

    router.include_router(_read_routes(api))
    return router


def _read_routes(api: ApiState) -> APIRouter:
    router = APIRouter()

    @router.get("/documents")
    def documents(
        cabinet_id: str | None = None,
        limit: Annotated[int, Query(ge=1, le=MAX_PAGE_LIMIT)] = DEFAULT_PAGE_LIMIT,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> list[dict[str, Any]]:
        rows = api.read(lambda conn: db.list_documents(conn, cabinet_id, limit, offset))
        return [{k: row[k] for k in LIST_COLUMNS} for row in rows]

    @router.get("/documents/{doc_id}", response_model=DocumentRecord)
    def document(doc_id: str) -> DocumentRecord:
        found: DocumentRecord | None = api.read(lambda conn: db.get_document(conn, doc_id))
        if found is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown doc_id")
        return found

    @router.get("/documents/{doc_id}/review_events")
    def review_events(doc_id: str) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = api.read(lambda conn: db.list_review_events(conn, doc_id))
        return events

    return router


async def _on_docc_error(request: Request, exc: Exception) -> JSONResponse:
    """The one error mapping. Messages of 400 and 503 are ours; a 500 shows nothing."""
    if isinstance(exc, BadInput):
        code, msg = status.HTTP_400_BAD_REQUEST, str(exc)
    elif isinstance(exc, (Unavailable, ConfigError)):
        code, msg = status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)
    else:
        code, msg = status.HTTP_500_INTERNAL_SERVER_ERROR, "request failed; see server log"
    extra = {"trace_id": trace_id.get(), "path": request.url.path, "status": code, "error": summarize_exception(exc)}
    log.warning("request failed", extra=extra)
    headers = {"Retry-After": str(RETRY_AFTER_S)} if isinstance(exc, Busy) else None
    return JSONResponse({"error": msg}, status_code=code, headers=headers)


class _TooLarge(Exception):  # noqa: N818 - internal signal, never leaves BodyLimit
    """The request body passed the limit while it was read."""


class BodyLimit:
    """Answer 413 for a body over MAX_UPLOAD_BYTES plus multipart overhead.

    A Content-Length over the limit is refused before any body is read. A
    body without one (chunked) is counted as it streams and cut off at the limit.
    """

    def __init__(self, app: ASGIApp) -> None:
        """Wrap an ASGI app.

        Args:
            app: The inner app.
        """
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Run the inner app with a counting receive.

        Args:
            scope: The ASGI scope.
            receive: The ASGI receive channel.
            send: The ASGI send channel.
        """
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        limit = MAX_UPLOAD_BYTES + MULTIPART_OVERHEAD_BYTES
        length = dict(scope["headers"]).get(b"content-length", b"")
        if length.isdigit() and int(length) > limit:
            await _too_large(scope, receive, send)
            return
        seen = 0
        over = False

        async def counted() -> Message:
            nonlocal seen, over
            message = await receive()
            seen += len(message.get("body", b""))
            if seen > limit:
                over = True
                raise _TooLarge
            return message

        async def guarded(message: Message) -> None:
            if not over:  # a cut-off body may still yield a parse error; 413 replaces it
                await send(message)

        with contextlib.suppress(_TooLarge):
            await self.app(scope, counted, guarded)
        if over:
            await _too_large(scope, receive, send)


async def _too_large(scope: Scope, receive: Receive, send: Send) -> None:
    msg = f"request larger than {MAX_UPLOAD_MB} MB"
    await JSONResponse({"error": msg}, status_code=HTTP_TOO_LARGE)(scope, receive, send)


async def _trace(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    """Take X-Request-ID or make one, expose it in trace_id, and echo it back."""
    sent = request.headers.get(TRACE_HEADER, "")
    tid = sent if TRACE_ID_RE.fullmatch(sent) else uuid.uuid4().hex
    token = trace_id.set(tid)
    try:
        response = await call_next(request)
    finally:
        trace_id.reset(token)
    response.headers[TRACE_HEADER] = tid
    return response


def create_app(
    config_dir: str | Path | None = None,
    *,
    root: Path | None = None,
    upload_dir: Path | None = None,
    workers: int = DEFAULT_WORKERS,
) -> FastAPI:
    """Load config and models once, open the store, and build the app.

    Args:
        config_dir: Config directory; None for the default.
        root: Directory the model paths and a relative db.path are relative to; None for the working directory.
        upload_dir: Where uploads are written; None for the system temp directory.
        workers: Cascade jobs run at once.

    Returns:
        The app. A missing VLM model is not an error here.
    """
    cfg = load_config(config_dir)
    ccfg = load_cascade_config(config_dir)
    if upload_dir is not None:
        upload_dir.mkdir(parents=True, exist_ok=True)
    base = root or Path.cwd()
    # db_path resolves a relative db.path beside a results file; here that file would sit in root.
    db_file = cfg.db_path(base / DEFAULT_DB_NAME)
    jobs = SqliteJobStore(db_file) if db_file else None
    conn = db.connect(db_file, threads=True) if db_file else None
    runner = JobRunner(jobs or InMemoryJobStore(), workers)
    api = ApiState(cfg, ccfg, Models.load(cfg, base), vlm_settings(ccfg), upload_dir, runner, conn)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        runner.shutdown()
        if jobs is not None:
            jobs.close()
        if conn is not None:
            with api.db_lock:
                conn.close()

    app = FastAPI(title="DoCC", lifespan=lifespan)
    app.include_router(_routes(api))
    app.add_exception_handler(DoccError, _on_docc_error)
    app.middleware("http")(_trace)
    app.add_middleware(BodyLimit)
    return app
