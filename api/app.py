"""FastAPI shell over the baseline and cascade pipelines.

Over 400 lines: the routes, the error mapping and the body limit form one HTTP surface.

Baseline runs synchronously in the request. Cascade runs as a background job
(12-40 s per document), polled at GET /jobs/{job_id}. Without a VLM model the
cascade route answers 503, like the demo server.

With `db.path` set (a relative path resolves in `root`), every record the API
produces is upserted into the SQLite store, jobs live in its `jobs` table, and
the document read routes serve from it. With `db.path: null` jobs stay in
memory and the read routes answer 503. One API process per store: a second
process on the same store fails at startup with ConfigError (api/jobs.py).

Server upload paths never leave the API: they are replaced by the client's
file name in diagnostics.errors, in cascade provenance errors and in the
API's own log lines (the `uploads` context variable feeds that log filter).

Security limits: no authentication and no tenant check. Bind to loopback (the
default); any caller can read every cabinet. The optional `cabinet_id` query
on the document routes is a guard against mix-ups, not access control.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import shutil
import sqlite3
import tempfile
import threading
import unicodedata
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, FastAPI, Form, HTTPException, Query, Request, Response, UploadFile, status
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from api.jobs import DEFAULT_WORKERS, Busy, InMemoryJobStore, Job, JobRunner, SqliteJobStore
from baseline import db, db_registry
from baseline.config import DEFAULT_DB_NAME, Config, load_config
from baseline.errors import (
    BadInput,
    ConfigError,
    DoccError,
    DocumentInOtherCabinet,
    Unavailable,
    summarize_exception,
)
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
HTTP_UNREADABLE = 422
TRACE_HEADER = "X-Request-ID"
# A client id is echoed in a header and logged, so only a plain token is kept.
TRACE_ID_RE = re.compile(r"[A-Za-z0-9._-]{1,64}")
# A cabinet id is a slug, as client packs make them: lower-case letters, digits, '_' or '-'.
# The store compares ids case-sensitively, so "Acme" would be a second cabinet; it is refused.
CABINET_ID_RE = re.compile(r"[\w-]{1,64}")
CABINET_ID_ERROR = "cabinet_id must be 1-64 lower-case letters, digits, '_' or '-'"
DEFAULT_PAGE_LIMIT = 50
ROUTING_PATTERN = "^(auto|review|failed)$"
MAX_PAGE_LIMIT = 500
LIST_COLUMNS = ("doc_id", "cabinet_id", "doc_type", "confidence", "filename", "updated_at", "routing_status", "case_id")
API_ACTOR = "api"
MAX_FILENAME_CHARS = 255
# A client file name is kept as metadata only; it may not hold control or path characters.
SAFE_FILENAME_RE = re.compile(r'[^<>:"/\\|?*\x00-\x1f\x7f]+')
# Unicode control, format and line/paragraph separators (U+202E, U+2028, "\x85"...) are removed from a name.
STRIPPED_CATEGORIES = frozenset({"Cc", "Cf", "Zl", "Zp"})
# Every upload is written under this stem plus its extension, never under the client's name.
UPLOAD_STEM = "upload"
UPLOAD_PREFIX = "docc_api_"
OTHER_CABINET_ERROR = "document already stored under another cabinet"
UNREADABLE_RECORD_ERROR = "the stored record is unreadable"
# Windows paths compare case-insensitively, so their scrubbing does too.
_PATH_FLAGS = re.IGNORECASE if os.name == "nt" else 0

trace_id: ContextVar[str] = ContextVar("trace_id", default="")
# This request's uploads: saved file -> client file name. A dict set per request, filled by save().
uploads: ContextVar[dict[Path, str] | None] = ContextVar("uploads", default=None)


def _discard(path: Path) -> None:
    shutil.rmtree(path.parent, ignore_errors=True)


def _path_forms(path: Path) -> set[str]:
    """How a path can appear in a message: both slash styles and the repr() doubled backslashes."""
    forms = set()
    for p in {path, path.resolve()}:
        s = str(p)
        forms |= {s, p.as_posix(), s.replace("\\", "\\\\")}
    return forms


def scrub(text: str, names: Mapping[Path, str]) -> str:
    """Replace each saved upload's file path, temp directory and temp directory name with the client's name.

    Args:
        text: A message, such as an extractor error.
        names: Saved file -> client file name.

    Returns:
        The text with no server upload path left in it.
    """
    for path, name in names.items():
        targets = _path_forms(path) | _path_forms(path.parent) | {path.parent.name}
        # Longest first, so the file path goes before the directory it sits in.
        pattern = "|".join(re.escape(t) for t in sorted(targets, key=len, reverse=True))
        text = re.sub(pattern, lambda _, n=name: n, text, flags=_PATH_FLAGS)  # a function: no backslash escapes
    return text


class _ScrubUploads(logging.Filter):
    """Removes the current request's upload paths from the API's own log lines."""

    def filter(self, record: logging.LogRecord) -> bool:
        names = uploads.get()
        if names:
            for key, value in list(vars(record).items()):
                if isinstance(value, str):
                    setattr(record, key, scrub(value, names))
            if isinstance(record.args, tuple):
                record.args = tuple(scrub(a, names) if isinstance(a, str) else a for a in record.args)
        return True


for _logger in (log, logging.getLogger("api.jobs")):
    _logger.addFilter(_ScrubUploads())


def _clean_name(raw: str) -> str:
    """The client's base name without control or format characters, NFC; "upload" plus the extension if empty."""
    base = Path(raw.replace("\\", "/")).name
    kept = "".join(c for c in base if unicodedata.category(c) not in STRIPPED_CATEGORIES)
    name = unicodedata.normalize("NFC", kept).strip()
    stem, dot, ext = name.rpartition(".")
    return name if stem.strip() or not dot else UPLOAD_STEM + dot + ext


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
            BadInput: The id is not a slug (CABINET_ID_RE, no upper case).
        """
        if not value:
            return self.cfg.cabinet_id
        if not CABINET_ID_RE.fullmatch(value) or value != value.lower():
            raise BadInput(CABINET_ID_ERROR)
        return value

    def persist(self, record: DocumentRecord, provenance: Provenance | None = None, *, trace: str = "") -> None:
        """Upsert a record, and its cascade provenance, into the store if there is one.

        Each record gets its own run carrying the request trace id, and an
        audit row. The run, cabinet, document and provenance rows are one
        transaction, so a refused document leaves none of them. A store
        failure is logged, never raised: the caller still gets the record. The
        one exception is a document stored under another cabinet, which the
        API answers with 409.

        Args:
            record: The record the API produced.
            provenance: The cascade provenance line, if any.
            trace: The request trace id; empty for the current request's.

        Raises:
            DocumentInOtherCabinet: The doc_id is stored under another cabinet.
        """
        if self.conn is None:
            return
        tid = trace or trace_id.get() or None
        try:
            with self.db_lock:
                prov = None if provenance is None else dict(provenance)
                run_id = db.store_record(self.conn, record, self.cfg.dump(), trace_id=tid, provenance=prov)
                detail = {"run_id": run_id, "trace_id": tid}
                db_registry.record_audit(
                    self.conn,
                    actor=API_ACTOR,
                    action="document.create",
                    cabinet_id=record.cabinet_id,
                    doc_id=record.doc_id,
                    detail=detail,
                )
        except DocumentInOtherCabinet:
            raise
        except sqlite3.Error as exc:
            extra = {"trace_id": tid, "doc_id": record.doc_id, "error": summarize_exception(exc)}
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

    def save(self, upload: UploadFile) -> tuple[Path, str]:
        """Write an upload to a new temp directory under a generated name.

        The client's name never reaches the file system: the file is
        UPLOAD_STEM plus the validated extension. The client's base name,
        cleaned by _clean_name, is returned only as metadata, for the record,
        and is noted in `uploads` so the API's logs show it, not the server path.

        Args:
            upload: The uploaded file.

        Returns:
            The saved file (remove its parent directory when done) and the client's base name.

        Raises:
            BadInput: Bad or over-long name, unsupported extension, empty, or over MAX_UPLOAD_BYTES.
        """
        name = _clean_name(upload.filename or "")
        if len(name) > MAX_FILENAME_CHARS:
            raise BadInput(f"file name longer than {MAX_FILENAME_CHARS} characters")
        if name.startswith(".") or not SAFE_FILENAME_RE.fullmatch(name):
            raise BadInput('invalid file name; control characters and <>:"/\\|?* are not allowed')
        suffix = Path(name).suffix.lower()
        if suffix not in SUPPORTED_EXT:
            raise BadInput(f"unsupported file type; use one of {sorted(SUPPORTED_EXT)}")
        target = Path(tempfile.mkdtemp(prefix=UPLOAD_PREFIX, dir=self.upload_dir)) / (UPLOAD_STEM + suffix)
        names = uploads.get()
        if names is not None:
            names[target] = name
        try:
            _copy_limited(upload, target)
        except BaseException:
            _discard(target)
            raise
        return target, name


CabinetForm = Annotated[str | None, Form()]


def _routes(api: ApiState) -> APIRouter:
    router = APIRouter(prefix=PREFIX)

    @router.get("/health")
    def health() -> dict[str, bool | str]:
        return {"status": "ok", "baseline": True, "cascade": bool(api.vlm.model and api.vlm.base_url)}

    @router.post("/documents/process", response_model=DocumentRecord)
    def process(file: UploadFile, cabinet_id: CabinetForm = None) -> DocumentRecord:
        cabinet = api.cabinet(cabinet_id)
        path, name = api.save(file)
        try:
            record, _ = process_document(path, api.cfg, api.models, cabinet)
        finally:
            _discard(path)
        record = _cleaned(record, path, name)
        api.persist(record)
        return record

    @router.post("/documents/process_async", status_code=status.HTTP_202_ACCEPTED)
    def process_async(file: UploadFile, cabinet_id: CabinetForm = None) -> dict[str, str]:
        cabinet = api.cabinet(cabinet_id)
        backend = api.backend()  # 503 before anything is saved
        path, name = api.save(file)
        tid = trace_id.get()  # the job thread does not see this request's context

        def work() -> DocumentRecord:
            result = cascade_process(path, api.cfg, api.ccfg, backend, api.models, cabinet)
            record = _cleaned(result.record, path, name)
            result.provenance["filename"] = name
            errors = result.provenance.get("errors") or []
            result.provenance["errors"] = [scrub(e, {path: name}) for e in errors]
            api.persist(record, result.provenance, trace=tid)
            return record

        try:
            # The runner removes the upload when the job ends, even one that never ran.
            return {"job_id": api.runner.submit(work, tid, cabinet, cleanup=lambda: _discard(path))}
        except BaseException:  # the job never started, so the upload is this route's to remove
            _discard(path)
            raise

    router.include_router(_read_routes(api))
    return router


def _read_routes(api: ApiState) -> APIRouter:
    router = APIRouter()

    @router.get("/jobs/{job_id}", response_model=Job)
    def job(job_id: str) -> Job:
        found = api.runner.get(job_id)
        if found is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown job id")
        return found

    @router.get("/documents")
    def documents(
        cabinet_id: str | None = None,
        limit: Annotated[int, Query(ge=1, le=MAX_PAGE_LIMIT)] = DEFAULT_PAGE_LIMIT,
        offset: Annotated[int, Query(ge=0)] = 0,
        routing_status: Annotated[str | None, Query(pattern=ROUTING_PATTERN)] = None,
        case_id: str | None = None,
    ) -> list[dict[str, Any]]:
        rows = api.read(
            lambda conn: db.list_documents(
                conn, cabinet_id, limit, offset, routing_status=routing_status, case_id=case_id
            )
        )
        return [{k: row[k] for k in LIST_COLUMNS} for row in rows]

    @router.get("/documents/{doc_id}", response_model=DocumentRecord)
    def document(doc_id: str, cabinet_id: str | None = None) -> DocumentRecord:
        found: DocumentRecord | None = _of_document(api, doc_id, cabinet_id, db.get_document)
        if found is None:  # stored, but its record_json no longer parses
            raise HTTPException(HTTP_UNREADABLE, UNREADABLE_RECORD_ERROR)
        return _named(found, found.filename)

    @router.get("/documents/{doc_id}/review_events")
    def review_events(doc_id: str, cabinet_id: str | None = None) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = _of_document(api, doc_id, cabinet_id, db.list_review_events)
        return events

    @router.get("/documents/{doc_id}/fields")
    def fields(doc_id: str, cabinet_id: str | None = None) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = _of_document(api, doc_id, cabinet_id, db_registry.effective_fields)
        return found

    router.include_router(_cabinet_routes(api))
    return router


def _cabinet_routes(api: ApiState) -> APIRouter:
    router = APIRouter()

    @router.get("/cabinets")
    def cabinets() -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = api.read(db_registry.list_cabinets)
        return found

    @router.get("/cabinets/{cabinet_id}/doc_types")
    def doc_types(cabinet_id: str) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] | None = api.read(lambda conn: _doc_types(conn, cabinet_id))
        if found is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown cabinet_id")
        return found

    return router


def _named(record: DocumentRecord, name: str) -> DocumentRecord:
    """The record as the API shows and stores it: the client's file name, never a server path."""
    return record.model_copy(update={"filename": name, "source_path": name})


def _cleaned(record: DocumentRecord, path: Path, name: str) -> DocumentRecord:
    """_named, with the upload's server paths in diagnostics.errors replaced by the client's name."""
    errors = [scrub(e, {path: name}) for e in record.diagnostics.errors]
    diagnostics = record.diagnostics.model_copy(update={"errors": errors})
    return _named(record, name).model_copy(update={"diagnostics": diagnostics})


def _of_document(
    api: ApiState, doc_id: str, cabinet_id: str | None, fn: Callable[[sqlite3.Connection, str], Any]
) -> Any:
    """fn(conn, doc_id) for a stored document; 404 when it is unknown or held by a cabinet other than cabinet_id."""

    def read(conn: sqlite3.Connection) -> tuple[bool, Any]:
        stored = db.document_cabinet(conn, doc_id)
        if stored is None or (cabinet_id is not None and cabinet_id != stored):
            return False, None
        return True, fn(conn, doc_id)

    found, value = api.read(read)
    if not found:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown doc_id")
    return value


def _doc_types(conn: sqlite3.Connection, cabinet_id: str) -> list[dict[str, Any]] | None:
    """A cabinet's document types, each with its field definitions; None for an unknown cabinet."""
    if not db_registry.cabinet_exists(conn, cabinet_id):
        return None
    by_type: dict[str, list[dict[str, Any]]] = {}
    for f in db_registry.list_field_defs(conn, cabinet_id):
        by_type.setdefault(f.pop("doc_type"), []).append(f)
    return [{**t, "fields": by_type.get(t["name"], [])} for t in db_registry.list_doc_types(conn, cabinet_id)]


async def _on_docc_error(request: Request, exc: Exception) -> JSONResponse:
    """The one error mapping. Messages of 400 and 503 are ours; a 500 shows nothing."""
    body: dict[str, str] = {}
    if isinstance(exc, DocumentInOtherCabinet):
        # The owning cabinet is not the caller's, so it is not named.
        code, msg, body = status.HTTP_409_CONFLICT, OTHER_CABINET_ERROR, {"doc_id": exc.doc_id}
    elif isinstance(exc, BadInput):
        code, msg = status.HTTP_400_BAD_REQUEST, str(exc)
    elif isinstance(exc, (Unavailable, ConfigError)):
        code, msg = status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)
    else:
        code, msg = status.HTTP_500_INTERNAL_SERVER_ERROR, "request failed; see server log"
    extra = {"trace_id": trace_id.get(), "path": request.url.path, "status": code, "error": summarize_exception(exc)}
    log.warning("request failed", extra=extra)
    headers = {"Retry-After": str(RETRY_AFTER_S)} if isinstance(exc, Busy) else None
    return JSONResponse({"error": msg, **body}, status_code=code, headers=headers)


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
    token, names = trace_id.set(tid), uploads.set({})
    try:
        response = await call_next(request)
    finally:
        trace_id.reset(token)
        uploads.reset(names)
    response.headers[TRACE_HEADER] = tid
    return response


def _open_store(db_file: Path | None) -> tuple[SqliteJobStore | None, sqlite3.Connection | None]:
    """The job store, claimed for this process with another boot's unfinished jobs failed, and the document store.

    On any failure the claim is given up, so a retry is not refused.
    """
    if db_file is None:
        return None, None
    jobs = SqliteJobStore(db_file)
    try:
        jobs.claim()
        jobs.sweep_stale()  # only the API, once at startup: a pipeline run opening the store fails nothing
        return jobs, db.connect(db_file, threads=True)
    except BaseException:
        jobs.close()
        raise


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

    Raises:
        ConfigError: Another API process is serving the same store (one API process per store).
    """
    cfg = load_config(config_dir)
    ccfg = load_cascade_config(config_dir)
    if upload_dir is not None:
        upload_dir.mkdir(parents=True, exist_ok=True)
    base = root or Path.cwd()
    models = Models.load(cfg, base)  # before the store is claimed, so a failure here leaves no claim
    # db_path resolves a relative db.path beside a results file; here that file would sit in root.
    jobs, conn = _open_store(cfg.db_path(base / DEFAULT_DB_NAME))
    runner = JobRunner(jobs or InMemoryJobStore(), workers)
    api = ApiState(cfg, ccfg, models, vlm_settings(ccfg), upload_dir, runner, conn)

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
