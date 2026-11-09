"""FastAPI shell over the baseline and cascade pipelines.

Baseline runs synchronously in the request. Cascade runs as a background job
(12-40 s per document), polled at GET /jobs/{job_id}. Without a VLM model the
cascade route answers 503, like the demo server.
"""

from __future__ import annotations

import logging
import re
import shutil
import tempfile
import threading
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException, Request, Response, UploadFile, status
from fastapi.responses import JSONResponse

from api.jobs import DEFAULT_WORKERS, InMemoryJobStore, Job, JobRunner
from baseline.config import Config, load_config
from baseline.errors import BadInput, ConfigError, DoccError, Unavailable, summarize_exception
from baseline.extract import SUPPORTED_EXT
from baseline.pipeline import Models, process_document
from baseline.schema import DocumentRecord
from cascade.config import CascadeConfig, VLMSettings, load_cascade_config
from cascade.runner import process_document as cascade_process
from cascade.vlm import Backend, VLMClient
from demo.server import require_model, vlm_settings

log = logging.getLogger(__name__)

PREFIX = "/api/v1"
MEGABYTE = 1024 * 1024
MAX_UPLOAD_MB = 50
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * MEGABYTE
CHUNK_BYTES = MEGABYTE
TRACE_HEADER = "X-Request-ID"
# A client id is echoed in a header and logged, so only a plain token is kept.
TRACE_ID_RE = re.compile(r"[A-Za-z0-9._-]{1,64}")

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
    """Config, models, the job runner and the lazily built VLM backend."""

    cfg: Config
    ccfg: CascadeConfig
    models: Models
    vlm: VLMSettings
    upload_dir: Path | None
    runner: JobRunner
    _backend: list[Backend] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

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


def _routes(api: ApiState) -> APIRouter:
    router = APIRouter(prefix=PREFIX)

    @router.get("/health")
    def health() -> dict[str, bool | str]:
        return {"status": "ok", "baseline": True, "cascade": bool(api.vlm.model and api.vlm.base_url)}

    @router.post("/documents/process", response_model=DocumentRecord)
    def process(file: UploadFile) -> DocumentRecord:
        path = api.save(file)
        try:
            record, _ = process_document(path, api.cfg, api.models)
        finally:
            _discard(path)
        return record

    @router.post("/documents/process_async", status_code=status.HTTP_202_ACCEPTED)
    def process_async(file: UploadFile) -> dict[str, str]:
        backend = api.backend()  # 503 before anything is saved
        path = api.save(file)

        def work() -> DocumentRecord:
            try:
                return cascade_process(path, api.cfg, api.ccfg, backend, api.models).record
            finally:
                _discard(path)

        return {"job_id": api.runner.submit(work, trace_id.get())}

    @router.get("/jobs/{job_id}", response_model=Job)
    def job(job_id: str) -> Job:
        found = api.runner.store.get(job_id)
        if found is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown job id")
        return found

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
    return JSONResponse({"error": msg}, status_code=code)


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
    """Load config and models once and build the app.

    Args:
        config_dir: Config directory; None for the default.
        root: Directory the model paths are relative to; None for the working directory.
        upload_dir: Where uploads are written; None for the system temp directory.
        workers: Cascade jobs run at once.

    Returns:
        The app. A missing VLM model is not an error here.
    """
    cfg = load_config(config_dir)
    ccfg = load_cascade_config(config_dir)
    if upload_dir is not None:
        upload_dir.mkdir(parents=True, exist_ok=True)
    runner = JobRunner(InMemoryJobStore(), workers)
    api = ApiState(cfg, ccfg, Models.load(cfg, root or Path.cwd()), vlm_settings(ccfg), upload_dir, runner)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        runner.shutdown()

    app = FastAPI(title="DoCC", lifespan=lifespan)
    app.include_router(_routes(api))
    app.add_exception_handler(DoccError, _on_docc_error)
    app.middleware("http")(_trace)
    return app
