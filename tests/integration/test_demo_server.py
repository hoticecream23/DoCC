import base64
import json
import socket
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

import demo.server as server
from demo.server import make_server

ROOT = Path(__file__).resolve().parents[2]


def _closed_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def base():
    with pytest.MonkeyPatch.context() as mp:  # cascade points at a closed port: no model is ever called
        mp.setenv("DOCC_VLM_BASE_URL", f"http://127.0.0.1:{_closed_port()}/v1")
        mp.setenv("DOCC_VLM_MODEL", "test-model")
        srv = make_server(0, ROOT)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()
    srv.server_close()


def call(url, data=None, headers=None):
    req = urllib.request.Request(url, data=data, headers=headers or {}, method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_corpus_lists_files(base):
    code, names = call(base + "/api/corpus")
    assert code == 200 and "invoice_native.pdf" in names


def test_run_corpus_pdf(base):
    code, body = call(base + "/api/run?corpus=invoice_native.pdf", b"")
    assert code == 200
    assert isinstance(body["record"]["classification"]["label"], str) and body["record"]["classification"]["label"]
    assert len(body["pages"]) >= 1
    assert base64.b64decode(body["pages"][0]["png_base64"]).startswith(b"\x89PNG")


def test_run_rejects_traversal(base):
    code, body = call(base + "/api/run?corpus=../pyproject.toml", b"")
    assert code == 400 and "error" in body


def test_run_upload_txt(base):
    code, body = call(base + "/api/run?filename=x.txt", b"Invoice No: 123\nTotal: 50.00 USD\n")
    assert code == 200 and body["pages"] == []


def test_foreign_host_rejected(base):
    code, body = call(base + "/api/corpus", headers={"Host": "evil.example"})
    assert code == 403 and body["error"] == "bad host"


def test_foreign_origin_rejected(base):
    code, body = call(base + "/api/run?corpus=invoice_native.pdf", b"", {"Origin": "http://evil.example"})
    assert code == 403 and body["error"] == "bad origin"


def test_status_reports_unreachable_vlm(base):
    code, body = call(base + "/api/status")
    c = body["cascade"]
    assert code == 200 and set(c) == {"base_url", "model", "reachable"} and c["reachable"] is False


def test_cascade_unreachable_is_503(base):
    code, body = call(base + "/api/run?corpus=invoice_native.pdf&mode=cascade", b"")
    assert code == 503 and "not reachable" in body["error"]


def test_bogus_mode_is_400(base):
    code, body = call(base + "/api/run?corpus=invoice_native.pdf&mode=bogus", b"")
    assert code == 400 and "error" in body


def test_baseline_mode_is_labelled(base):
    code, body = call(base + "/api/run?filename=x.txt", b"Invoice No: 123")
    assert code == 200 and body["mode"] == "baseline" and "provenance" not in body


def test_cascade_run_with_stub_backend(base, monkeypatch):
    from test_cascade_runner import FakeBackend, _invoice_or_nothing

    monkeypatch.setattr(server, "reachable", lambda settings: True)
    monkeypatch.setattr(server, "make_backend", lambda settings: FakeBackend(_invoice_or_nothing))
    code, body = call(base + "/api/run?corpus=invoice_native.pdf&mode=cascade", b"")
    assert code == 200 and body["mode"] == "cascade"
    assert isinstance(body["provenance"], dict) and "raw" not in body["provenance"]
    assert body["provenance"]["model"] == "fake" and body["record"]["classification"]["label"]
