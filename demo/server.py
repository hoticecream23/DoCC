"""Local demo server for the baseline pipeline. Stdlib only, binds 127.0.0.1."""

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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from baseline.config import load_config_cached
from baseline.pipeline import Models, process_document
from cascade.config import load_cascade_config
from cascade.runner import process_document as cascade_process
from cascade.vlm import VLMClient, VLMError

log = logging.getLogger("demo")
MAX_BYTES = 25 * 1024 * 1024
MAX_PAGES = 20
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


class BadInput(ValueError):
    pass


class Unavailable(RuntimeError):
    pass


make_backend = VLMClient  # seam: tests swap in a fake backend


def vlm_settings(ccfg):
    """cascade.yaml, then env overrides; Ollama on this laptop when nothing is set."""
    over = {
        k: v
        for k, v in (("base_url", os.environ.get("DOCC_VLM_BASE_URL")), ("model", os.environ.get("DOCC_VLM_MODEL")))
        if v
    }
    if ccfg.vlm.model is None and not over:
        over = {"base_url": "http://127.0.0.1:11434/v1", "model": "qwen3.5:9b-16k"}
    return ccfg.vlm.model_copy(update=over)


def reachable(settings) -> bool:
    req = urllib.request.Request(settings.base_url.rstrip("/") + "/models")
    key = os.environ.get(settings.api_key_env) if settings.api_key_env else None
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=2):
            return True
    except Exception:  # HTTPError included: a 401 still means unusable for the demo
        return False


def render_pages(path: Path) -> list[dict]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        import fitz

        out = []
        with fitz.open(path) as doc:
            for i, page in enumerate(doc):
                if i >= MAX_PAGES:
                    break
                pix = page.get_pixmap(dpi=110, alpha=False)
                out.append(_page(i, pix.tobytes("png"), pix.width, pix.height))
        return out
    if ext in IMAGE_EXT:
        from PIL import Image

        with Image.open(path) as im:
            buf = io.BytesIO()
            im.convert("RGB").save(buf, "PNG")  # ponytail: first frame only for multi-page TIFF
            return [_page(0, buf.getvalue(), im.width, im.height)]
    return []


def _page(i: int, png: bytes, w: int, h: int) -> dict:
    return {"index": i, "png_base64": base64.b64encode(png).decode("ascii"), "width": w, "height": h}


def _safe_name(name: str) -> str:
    if not name or "/" in name or "\\" in name or ".." in name or name.startswith("."):
        raise BadInput(f"invalid file name: {name!r}")
    return name


def make_server(port: int, root: Path) -> ThreadingHTTPServer:
    root = Path(root).resolve()
    corpus = root / "corpus"
    cfg = load_config_cached(None)
    models = Models.load(cfg, root)
    ccfg = load_cascade_config(None)
    vlm = vlm_settings(ccfg)
    backend: list = []  # built on the first cascade request, then reused
    lock = threading.Lock()

    def get_backend():
        if not vlm.model:
            raise Unavailable("no VLM model configured")
        if not reachable(vlm):
            raise Unavailable(f"VLM server not reachable at {vlm.base_url}")
        with lock:
            if not backend:
                backend.append(make_backend(vlm))
        return backend[0]

    def run(path: Path, mode: str) -> dict:
        t0 = time.perf_counter()
        out = {"mode": mode}
        if mode == "cascade":
            res = cascade_process(path, cfg, ccfg, get_backend(), models)
            record = res.record
            out["provenance"] = {
                k: res.provenance[k]
                for k in ("model", "finish_reason", "pages_sent", "images_sent", "fallback", "errors", "timings_ms")
            }
        else:
            record, _ = process_document(path, cfg, models)
        out.update(
            record=record.model_dump(mode="json"), pages=render_pages(path), total_ms=(time.perf_counter() - t0) * 1000
        )
        return out

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj) -> None:
            self._send(code, json.dumps(obj).encode("utf-8"))

        def _allowed(self, post: bool = False) -> bool:
            """Block DNS rebinding (Host) and cross-site POSTs (Origin)."""
            hosts = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            if self.headers.get("Host") not in hosts:
                self._json(403, {"error": "bad host"})
                return False
            origin = self.headers.get("Origin")
            if post and origin is not None and origin not in {f"http://{h}" for h in hosts}:
                self._json(403, {"error": "bad origin"})
                return False
            return True

        def do_GET(self) -> None:
            if not self._allowed():
                return
            path = urlparse(self.path).path
            if path == "/":
                page = root / "demo" / "index.html"
                if not page.is_file():
                    return self._send(404, b"demo/index.html not found", "text/plain; charset=utf-8")
                return self._send(200, page.read_bytes(), "text/html; charset=utf-8")
            if path == "/api/status":
                return self._json(
                    200, {"cascade": {"base_url": vlm.base_url, "model": vlm.model, "reachable": reachable(vlm)}}
                )
            if path == "/api/corpus":
                return self._json(200, sorted(p.name for p in corpus.iterdir() if p.is_file()))
            self._json(404, {"error": "not found"})

        def do_POST(self) -> None:
            if not self._allowed(post=True):
                return
            url = urlparse(self.path)
            if url.path != "/api/run":
                return self._json(404, {"error": "not found"})
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            tmp = None
            try:
                mode = q.get("mode", "baseline")
                if mode not in ("baseline", "cascade"):
                    raise BadInput(f"unknown mode: {mode!r}")
                if "corpus" in q:
                    target = (corpus / _safe_name(q["corpus"])).resolve()
                    if target.parent != corpus.resolve() or not target.is_file():
                        raise BadInput(f"no such corpus file: {q['corpus']!r}")
                elif "filename" in q:
                    name = _safe_name(Path(q["filename"].replace("\\", "/")).name)
                    size = int(self.headers.get("Content-Length") or 0)
                    if size > MAX_BYTES:
                        return self._json(413, {"error": "file larger than 25 MB"})
                    if size <= 0:
                        raise BadInput("empty request body")
                    tmp = tempfile.mkdtemp(prefix="docc_demo_")
                    target = Path(tmp) / name
                    target.write_bytes(self.rfile.read(size))
                else:
                    raise BadInput("pass ?corpus=<name> or ?filename=<name>")
                self._json(200, run(target, mode))
            except BadInput as e:
                self._json(400, {"error": str(e)})
            except Unavailable as e:
                self._json(503, {"error": str(e)})
            except VLMError as e:  # ponytail: the runner falls back on VLM errors, so this rarely fires
                self._json(502, {"error": str(e)})
            except Exception:
                log.exception("run failed")
                self._json(500, {"error": "pipeline failed; see server log"})
            finally:
                if tmp:
                    shutil.rmtree(tmp, ignore_errors=True)

        def log_message(self, fmt, *args) -> None:
            log.info(fmt, *args)

    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    srv.vlm = vlm
    return srv


def serve(port: int, root: Path) -> None:
    logging.basicConfig(level=logging.INFO)
    t0 = time.perf_counter()
    srv = make_server(port, root)
    print(f"DoCC demo ready in {time.perf_counter() - t0:.1f}s at http://127.0.0.1:{srv.server_port}/", flush=True)
    print(f"cascade VLM: {srv.vlm.base_url} model={srv.vlm.model}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
