"""Local demo server for the baseline pipeline. Stdlib only, binds 127.0.0.1.

The server starts without a VLM model. Baseline mode always works; cascade
mode answers 503 until a model is configured. There is no default model.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import shutil
import tempfile
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, TypedDict
from urllib.parse import parse_qs, urlparse

from baseline.config import Config, load_config_cached
from baseline.errors import BadInput, ConfigError, Unavailable
from baseline.pipeline import Models, process_document
from cascade.config import CascadeConfig, VLMSettings, load_cascade_config
from cascade.runner import process_document as cascade_process
from cascade.vlm import Backend, VLMClient, VLMError

log = logging.getLogger("demo")
MAX_BYTES = 25 * 1024 * 1024
MAX_PAGES = 20
PAGE_DPI = 110
PROBE_TIMEOUT_S = 2
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

HTTP_OK = 200
HTTP_BAD_REQUEST = 400
HTTP_FORBIDDEN = 403
HTTP_NOT_FOUND = 404
HTTP_TOO_LARGE = 413
HTTP_SERVER_ERROR = 500
HTTP_BAD_GATEWAY = 502
HTTP_UNAVAILABLE = 503


make_backend = VLMClient  # seam: tests swap in a fake backend


class PageImage(TypedDict):
    """One rendered page for the browser."""

    index: int
    png_base64: str
    width: int
    height: int


def vlm_settings(ccfg: CascadeConfig) -> VLMSettings:
    """Return the VLM settings: cascade.yaml, then env overrides.

    The model may be unset here; require_model checks it when cascade runs.

    Args:
        ccfg: The cascade config.

    Returns:
        The settings with DOCC_VLM_BASE_URL and DOCC_VLM_MODEL applied.
    """
    over = {
        k: v
        for k, v in (("base_url", os.environ.get("DOCC_VLM_BASE_URL")), ("model", os.environ.get("DOCC_VLM_MODEL")))
        if v
    }
    return ccfg.vlm.model_copy(update=over)


def require_model(settings: VLMSettings) -> None:
    """Refuse cascade mode without a model or endpoint. There is no default.

    Args:
        settings: The VLM settings.

    Raises:
        ConfigError: No model is set in cascade.yaml vlm.model or DOCC_VLM_MODEL,
            or no endpoint is set.
    """
    if not settings.model:
        raise ConfigError("demo needs a VLM model: set DOCC_VLM_MODEL or vlm.model in config/cascade.yaml")
    if not settings.base_url:
        raise ConfigError("demo needs a VLM endpoint: set DOCC_VLM_BASE_URL or vlm.base_url in config/cascade.yaml")


def reachable(settings: VLMSettings) -> bool:
    """Probe the server's /models endpoint.

    Args:
        settings: The VLM settings.

    Returns:
        True if the server answered with a success status.
    """
    req = urllib.request.Request(settings.base_url.rstrip("/") + "/models")
    key = os.environ.get(settings.api_key_env) if settings.api_key_env else None
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=PROBE_TIMEOUT_S):
            return True
    except OSError as exc:  # URLError, HTTPError (a 401 still means unusable), timeouts
        log.warning("VLM probe failed", extra={"base_url": settings.base_url, "error": type(exc).__name__})
        return False


def render_pages(path: Path) -> list[PageImage]:
    """Render the first MAX_PAGES pages of a PDF or image as PNG for the browser.

    Args:
        path: The document.

    Returns:
        One entry per page; empty for a file type with no image.
    """
    ext = path.suffix.lower()
    if ext == ".pdf":
        import fitz

        out = []
        with fitz.open(path) as doc:
            for i, page in enumerate(doc):
                if i >= MAX_PAGES:
                    break
                pix = page.get_pixmap(dpi=PAGE_DPI, alpha=False)
                out.append(_page(i, pix.tobytes("png"), pix.width, pix.height))
        return out
    if ext in IMAGE_EXT:
        from PIL import Image

        with Image.open(path) as im:
            buf = io.BytesIO()
            im.convert("RGB").save(buf, "PNG")  # ponytail: first frame only for multi-page TIFF
            return [_page(0, buf.getvalue(), im.width, im.height)]
    return []


def _page(i: int, png: bytes, w: int, h: int) -> PageImage:
    return {"index": i, "png_base64": base64.b64encode(png).decode("ascii"), "width": w, "height": h}


def _safe_name(name: str) -> str:
    if not name or "/" in name or "\\" in name or ".." in name or name.startswith("."):
        raise BadInput(f"invalid file name: {name!r}")
    return name


@dataclass
class DemoApp:
    """Config, models and the lazily built VLM backend, shared by every request."""

    root: Path
    cfg: Config
    models: Models
    ccfg: CascadeConfig
    vlm: VLMSettings
    _backend: list[Backend] = field(default_factory=list)  # built on the first cascade request
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def corpus(self) -> Path:
        """The directory of sample documents."""
        return self.root / "corpus"

    def backend(self) -> Backend:
        """Return the VLM backend, building it on first use.

        Returns:
            The shared backend.

        Raises:
            ConfigError: No model is configured.
            Unavailable: The server does not answer.
        """
        require_model(self.vlm)
        if not reachable(self.vlm):
            raise Unavailable(f"VLM server not reachable at {self.vlm.base_url}")
        with self._lock:
            if not self._backend:
                self._backend.append(make_backend(self.vlm))
        return self._backend[0]

    def run(self, path: Path, mode: str) -> dict[str, Any]:
        """Process one document and return the response body.

        Args:
            path: The document.
            mode: "baseline" or "cascade".

        Returns:
            The mode, record, rendered pages, total time and, for cascade, provenance.
        """
        t0 = time.perf_counter()
        out: dict[str, Any] = {"mode": mode}
        if mode == "cascade":
            res = cascade_process(path, self.cfg, self.ccfg, self.backend(), self.models)
            record = res.record
            prov = res.provenance
            out["provenance"] = {
                "model": prov["model"],
                "finish_reason": prov["finish_reason"],
                "pages_sent": prov["pages_sent"],
                "images_sent": prov["images_sent"],
                "fallback": prov["fallback"],
                "errors": prov["errors"],
                "timings_ms": prov["timings_ms"],
            }
        else:
            record, _ = process_document(path, self.cfg, self.models)
        out.update(
            record=record.model_dump(mode="json"), pages=render_pages(path), total_ms=(time.perf_counter() - t0) * 1000
        )
        return out


class DemoServer(ThreadingHTTPServer):
    """HTTP server that carries the shared DemoApp."""

    def __init__(self, port: int, app: DemoApp) -> None:
        self.app = app
        super().__init__(("127.0.0.1", port), Handler)

    @property
    def vlm(self) -> VLMSettings:
        """The VLM settings the cascade mode uses."""
        return self.app.vlm


class Handler(BaseHTTPRequestHandler):
    """Routes for the demo page and its JSON API."""

    server: DemoServer

    def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj: object) -> None:
        self._send(code, json.dumps(obj).encode("utf-8"))

    def _allowed(self, post: bool = False) -> bool:
        """Block DNS rebinding (Host) and cross-site POSTs (Origin)."""
        hosts = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        if self.headers.get("Host") not in hosts:
            self._json(HTTP_FORBIDDEN, {"error": "bad host"})
            return False
        origin = self.headers.get("Origin")
        if post and origin is not None and origin not in {f"http://{h}" for h in hosts}:
            self._json(HTTP_FORBIDDEN, {"error": "bad origin"})
            return False
        return True

    def do_GET(self) -> None:
        """Serve the page, the VLM status and the corpus list."""
        if not self._allowed():
            return
        app = self.server.app
        path = urlparse(self.path).path
        if path == "/":
            page = app.root / "demo" / "index.html"
            if not page.is_file():
                return self._send(HTTP_NOT_FOUND, b"demo/index.html not found", "text/plain; charset=utf-8")
            return self._send(HTTP_OK, page.read_bytes(), "text/html; charset=utf-8")
        if path == "/api/status":
            vlm = app.vlm
            status = {"base_url": vlm.base_url, "model": vlm.model, "reachable": reachable(vlm)}
            return self._json(HTTP_OK, {"cascade": status})
        if path == "/api/corpus":
            return self._json(HTTP_OK, sorted(p.name for p in app.corpus.iterdir() if p.is_file()))
        self._json(HTTP_NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:
        """Run a corpus file or an uploaded file through the chosen pipeline."""
        if not self._allowed(post=True):
            return
        url = urlparse(self.path)
        if url.path != "/api/run":
            return self._json(HTTP_NOT_FOUND, {"error": "not found"})
        query = {k: v[0] for k, v in parse_qs(url.query).items()}
        tmp: list[str] = []  # upload directory to remove afterwards
        try:
            mode = query.get("mode", "baseline")
            if mode not in ("baseline", "cascade"):
                raise BadInput(f"unknown mode: {mode!r}")
            target = self._target(query, tmp)
            if target is not None:
                self._json(HTTP_OK, self.server.app.run(target, mode))
        except BadInput as e:
            self._json(HTTP_BAD_REQUEST, {"error": str(e)})
        except (Unavailable, ConfigError) as e:
            self._json(HTTP_UNAVAILABLE, {"error": str(e)})
        except VLMError as e:  # ponytail: the runner falls back on VLM errors, so this rarely fires
            self._json(HTTP_BAD_GATEWAY, {"error": str(e)})
        except Exception:  # noqa: BLE001 - request boundary: any failure becomes a 500, server stays up
            log.exception("run failed")
            self._json(HTTP_SERVER_ERROR, {"error": "pipeline failed; see server log"})
        finally:
            for d in tmp:
                shutil.rmtree(d, ignore_errors=True)

    def _target(self, query: dict[str, str], tmp: list[str]) -> Path | None:
        """The file to run: a corpus file, or the upload saved under a new temp dir.

        Returns None after answering 413 itself.
        """
        if "corpus" in query:
            corpus = self.server.app.corpus
            target = (corpus / _safe_name(query["corpus"])).resolve()
            if target.parent != corpus.resolve() or not target.is_file():
                raise BadInput(f"no such corpus file: {query['corpus']!r}")
            return target
        if "filename" not in query:
            raise BadInput("pass ?corpus=<name> or ?filename=<name>")
        name = _safe_name(Path(query["filename"].replace("\\", "/")).name)
        size = int(self.headers.get("Content-Length") or 0)
        if size > MAX_BYTES:
            self._json(HTTP_TOO_LARGE, {"error": "file larger than 25 MB"})
            return None
        if size <= 0:
            raise BadInput("empty request body")
        tmp.append(tempfile.mkdtemp(prefix="docc_demo_"))
        target = Path(tmp[0]) / name
        target.write_bytes(self.rfile.read(size))
        return target

    def log_message(self, fmt: str, *args: Any) -> None:
        """Send request logs to the demo logger instead of stderr."""
        log.info(fmt, *args)


def make_server(port: int, root: Path) -> DemoServer:
    """Load config and models and bind the server. A missing VLM model is not an error here.

    Args:
        port: Port on 127.0.0.1; 0 picks a free one.
        root: Project root, holding config/, models/, corpus/ and demo/.

    Returns:
        The bound, not yet serving, server.
    """
    root = Path(root).resolve()
    cfg = load_config_cached(None)
    ccfg = load_cascade_config(None)
    app = DemoApp(root=root, cfg=cfg, models=Models.load(cfg, root), ccfg=ccfg, vlm=vlm_settings(ccfg))
    return DemoServer(port, app)


def serve(port: int, root: Path) -> None:
    """Start the server and block until Ctrl+C.

    Args:
        port: Port on 127.0.0.1.
        root: Project root.
    """
    logging.basicConfig(level=logging.INFO)
    t0 = time.perf_counter()
    srv = make_server(port, root)
    log.info("DoCC demo ready in %.1fs at http://127.0.0.1:%d/", time.perf_counter() - t0, srv.server_port)
    log.info("cascade VLM: %s model=%s", srv.vlm.base_url, srv.vlm.model or "(none: cascade mode answers 503)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
