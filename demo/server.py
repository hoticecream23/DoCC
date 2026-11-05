"""Local demo server for the baseline pipeline. Stdlib only, binds 127.0.0.1."""
from __future__ import annotations

import base64
import io
import json
import logging
import shutil
import tempfile
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from baseline.config import load_config_cached
from baseline.pipeline import Models, process_document

log = logging.getLogger("demo")
MAX_BYTES = 25 * 1024 * 1024
MAX_PAGES = 20
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


class BadInput(ValueError):
    pass


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

    def run(path: Path) -> dict:
        t0 = time.perf_counter()
        record, _ = process_document(path, cfg, models)
        pages = render_pages(path)
        return {"record": record.model_dump(mode="json"), "pages": pages,
                "total_ms": (time.perf_counter() - t0) * 1000}

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
                self._json(200, run(target))
            except BadInput as e:
                self._json(400, {"error": str(e)})
            except Exception:
                log.exception("run failed")
                self._json(500, {"error": "pipeline failed; see server log"})
            finally:
                if tmp:
                    shutil.rmtree(tmp, ignore_errors=True)

        def log_message(self, fmt, *args) -> None:
            log.info(fmt, *args)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def serve(port: int, root: Path) -> None:
    logging.basicConfig(level=logging.INFO)
    t0 = time.perf_counter()
    srv = make_server(port, root)
    print(f"DoCC demo ready in {time.perf_counter() - t0:.1f}s at http://127.0.0.1:{srv.server_port}/", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
