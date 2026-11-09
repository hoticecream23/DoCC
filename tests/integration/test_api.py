"""The FastAPI shell over the synthetic corpus, with no models/ directory (as on CI)."""

import json
import time

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

import api.app as api_app  # noqa: E402
from api.app import create_app  # noqa: E402

V1 = "/api/v1"


def _client(cfg, tmp_path, monkeypatch, base_url=None, model=None):
    for name, value in (("DOCC_VLM_BASE_URL", base_url), ("DOCC_VLM_MODEL", model)):
        if value:
            monkeypatch.setenv(name, value)
        else:
            monkeypatch.delenv(name, raising=False)
    app = create_app(cfg.config_dir, root=tmp_path, upload_dir=tmp_path / "uploads")
    return TestClient(app)


@pytest.fixture
def client(cfg, tmp_path, monkeypatch):
    with _client(cfg, tmp_path, monkeypatch) as c:
        yield c


def _upload(client, route, name, data):
    return client.post(f"{V1}/documents/{route}", files={"file": (name, data)})


def test_health(client, cfg):
    body = client.get(f"{V1}/health").json()
    assert body["status"] == "ok" and body["baseline"] is True and isinstance(body["cascade"], bool)


def test_process_pdf_returns_a_record(client, corpus):
    r = _upload(client, "process", "invoice_native.pdf", (corpus / "invoice_native.pdf").read_bytes())
    assert r.status_code == 200
    assert r.json()["filename"] == "invoice_native.pdf" and r.json()["classification"]["label"]


def test_upload_dir_is_emptied_after_processing(client, tmp_path):
    _upload(client, "process", "x.txt", b"Invoice No: 123")
    assert list((tmp_path / "uploads").iterdir()) == []


def test_wrong_extension_is_400(client):
    r = _upload(client, "process", "x.exe", b"MZ")
    assert r.status_code == 400 and "unsupported file type" in r.json()["error"]


def test_empty_file_is_400(client):
    assert _upload(client, "process", "x.txt", b"").status_code == 400


def test_oversize_is_400(client, monkeypatch):
    monkeypatch.setattr(api_app, "MAX_UPLOAD_BYTES", 4)
    r = _upload(client, "process", "x.txt", b"12345")
    assert r.status_code == 400 and "larger than" in r.json()["error"]


def test_process_async_without_a_model_is_503(client):
    if client.get(f"{V1}/health").json()["cascade"]:
        pytest.skip("cascade.yaml sets a model")
    r = _upload(client, "process_async", "x.txt", b"Invoice No: 123")
    assert r.status_code == 503 and "DOCC_VLM_MODEL" in r.json()["error"]


def test_unknown_job_is_404(client):
    assert client.get(f"{V1}/jobs/nope").status_code == 404


def test_request_id_is_echoed(client):
    r = client.get(f"{V1}/health", headers={"X-Request-ID": "abc-123"})
    assert r.headers["X-Request-ID"] == "abc-123"


def test_unsafe_request_id_is_replaced(client):
    r = client.get(f"{V1}/health", headers={"X-Request-ID": "a b<c>"})
    assert r.headers["X-Request-ID"] not in ("", "a b<c>")


def test_other_docc_error_is_a_generic_500(client, monkeypatch):
    from baseline.errors import DoccError

    def boom(*args):
        raise DoccError("secret document text")

    monkeypatch.setattr(api_app, "process_document", boom)
    r = _upload(client, "process", "x.txt", b"Invoice No: 123")
    assert r.status_code == 500 and "secret" not in r.text


def test_disallowed_vlm_host_is_503(cfg, tmp_path, monkeypatch):
    with _client(cfg, tmp_path, monkeypatch, "http://evil.example/v1", "test-model") as c:
        r = _upload(c, "process_async", "x.txt", b"Invoice No: 123")
    assert r.status_code == 503 and "VLM settings rejected" in r.json()["error"]


def test_process_async_runs_a_cascade_job(cfg, corpus, tmp_path, monkeypatch):
    from vlm_stub import chat, serve

    from cascade.prompt import response_schema

    names = response_schema(cfg)["properties"]["fields"]["required"]
    answer = chat(json.dumps({"class": "unknown", "fields": {n: [] for n in names}}))
    with serve([(200, answer, 0, None)]) as (url, _), _client(cfg, tmp_path, monkeypatch, url, "test-model") as c:
        r = _upload(c, "process_async", "invoice_native.pdf", (corpus / "invoice_native.pdf").read_bytes())
        assert r.status_code == 202
        job = _poll(c, r.json()["job_id"])
    assert job["state"] == "done" and job["error"] is None and job["result"]["filename"] == "invoice_native.pdf"
    assert list((tmp_path / "uploads").iterdir()) == []


def test_chunked_oversize_body_is_cut_off_with_413(client, tmp_path, monkeypatch):
    monkeypatch.setattr(api_app, "MAX_UPLOAD_BYTES", 10)
    monkeypatch.setattr(api_app, "MULTIPART_OVERHEAD_BYTES", 0)
    head = b'--b\r\nContent-Disposition: form-data; name="file"; filename="x.txt"\r\n\r\n'
    body = iter([head, b"x" * 1000, b"\r\n--b--\r\n"])
    r = client.post(
        f"{V1}/documents/process", content=body, headers={"Content-Type": "multipart/form-data; boundary=b"}
    )
    assert r.status_code == 413 and "larger than" in r.json()["error"]
    assert list((tmp_path / "uploads").iterdir()) == []


def test_full_queue_is_503_with_retry_after_and_drops_the_upload(cfg, tmp_path, monkeypatch):
    from api.jobs import Busy

    def busy(*args):
        raise Busy("job queue is full; retry later")

    monkeypatch.setattr(api_app.JobRunner, "submit", busy)
    with _client(cfg, tmp_path, monkeypatch, "http://127.0.0.1:9/v1", "test-model") as c:
        r = _upload(c, "process_async", "x.txt", b"Invoice No: 123")
    assert r.status_code == 503 and r.headers["Retry-After"] == str(api_app.RETRY_AFTER_S)
    assert list((tmp_path / "uploads").iterdir()) == []


def _poll(client, job_id, timeout=60.0):
    end = time.monotonic() + timeout
    while True:
        job = client.get(f"{V1}/jobs/{job_id}").json()
        if job["state"] in ("done", "failed") or time.monotonic() > end:
            return job
        time.sleep(0.1)
