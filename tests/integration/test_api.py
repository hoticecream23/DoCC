"""The FastAPI shell over the synthetic corpus, with no models/ directory (as on CI)."""

import json
import time
from contextlib import closing

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

import api.app as api_app  # noqa: E402
from api.app import create_app  # noqa: E402
from baseline import db  # noqa: E402

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
    with closing(db.connect(tmp_path / "docc.sqlite")) as conn:
        assert [r["doc_id"] for r in db.list_documents(conn)] == [job["result"]["doc_id"]]
        prov = conn.execute("SELECT doc_id FROM provenance").fetchall()
    assert [r["doc_id"] for r in prov] == [job["result"]["doc_id"]]


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


def _process(client, corpus, name, cabinet=None):
    data = {"cabinet_id": cabinet} if cabinet else None
    r = client.post(f"{V1}/documents/process", files={"file": (name, (corpus / name).read_bytes())}, data=data)
    assert r.status_code == 200
    return r.json()


def test_processed_record_is_stored_under_its_cabinet(client, corpus, tmp_path):
    rec = _process(client, corpus, "invoice_native.pdf", "x")
    assert rec["cabinet_id"] == "x"
    with closing(db.connect(tmp_path / "docc.sqlite")) as conn:
        assert [r["doc_id"] for r in db.list_documents(conn, cabinet_id="x")] == [rec["doc_id"]]


def test_bad_cabinet_id_is_400(client, corpus):
    r = client.post(f"{V1}/documents/process", files={"file": ("x.txt", b"Invoice No: 1")}, data={"cabinet_id": "a b"})
    assert r.status_code == 400 and "cabinet_id" in r.json()["error"]


def test_list_filters_by_cabinet_and_pages(client, corpus):
    a = _process(client, corpus, "invoice_native.pdf", "x")
    b = _process(client, corpus, "receipt.txt")
    rows = client.get(f"{V1}/documents").json()
    assert sorted(r["doc_id"] for r in rows) == sorted([a["doc_id"], b["doc_id"]])
    assert set(rows[0]) == set(api_app.LIST_COLUMNS) and rows[0]["filename"] in ("invoice_native.pdf", "receipt.txt")
    only_x = client.get(f"{V1}/documents", params={"cabinet_id": "x"}).json()
    assert [r["doc_id"] for r in only_x] == [a["doc_id"]]
    assert len(client.get(f"{V1}/documents", params={"limit": 1}).json()) == 1
    assert client.get(f"{V1}/documents", params={"offset": 2}).json() == []


def test_list_rejects_out_of_bounds_paging(client):
    for params in ({"limit": 0}, {"limit": api_app.MAX_PAGE_LIMIT + 1}, {"offset": -1}):
        assert client.get(f"{V1}/documents", params=params).status_code == 422


def test_get_document_returns_the_stored_record_or_404(client, corpus):
    rec = _process(client, corpus, "invoice_native.pdf")
    got = client.get(f"{V1}/documents/{rec['doc_id']}")
    assert got.status_code == 200 and got.json() == rec
    assert client.get(f"{V1}/documents/nope").status_code == 404


def test_review_events_of_a_document(client, corpus, tmp_path):
    rec = _process(client, corpus, "invoice_native.pdf")
    event = db.ReviewEvent(rec["doc_id"], "class", "invoice", "receipt", "review.xlsx")
    with closing(db.connect(tmp_path / "docc.sqlite")) as conn:
        db.record_review_events(conn, [event])
    events = client.get(f"{V1}/documents/{rec['doc_id']}/review_events").json()
    assert [(e["field"], e["old_value"], e["new_value"]) for e in events] == [("class", "invoice", "receipt")]


def test_jobs_survive_a_restart_of_the_app(cfg, tmp_path, monkeypatch):
    from api.jobs import STALE_JOB_ERROR, SqliteJobStore

    store = SqliteJobStore(tmp_path / "docc.sqlite")
    job_id = store.create()
    store.close()
    with _client(cfg, tmp_path, monkeypatch) as c:
        job = c.get(f"{V1}/jobs/{job_id}").json()
    assert job["state"] == "failed" and job["error"] == STALE_JOB_ERROR


def test_without_a_db_path_jobs_stay_in_memory_and_reads_are_503(cfg, tmp_path, monkeypatch):
    from baseline.config import Config

    monkeypatch.setattr(Config, "db_path", lambda self, results: None)
    with _client(cfg, tmp_path, monkeypatch) as c:
        assert c.post(f"{V1}/documents/process", files={"file": ("x.txt", b"Invoice No: 1")}).status_code == 200
        r = c.get(f"{V1}/documents")
    assert r.status_code == 503 and "db.path" in r.json()["error"]
    assert not (tmp_path / "docc.sqlite").exists()
