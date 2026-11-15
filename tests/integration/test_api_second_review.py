"""Regression tests for API bugs from the second review: server paths, names, refusals, instances, cabinet ids."""

import json
import logging
import sqlite3
import time
from contextlib import closing

import pytest

pytest.importorskip("fastapi")

from test_api import V1, _client, _upload  # noqa: E402

import api.app as api_app  # noqa: E402
from api.jobs import API_INSTANCE_TIMEOUT_S, SqliteJobStore  # noqa: E402
from baseline import db, db_registry  # noqa: E402
from baseline.errors import ConfigError, DoccError  # noqa: E402

DATA = b"Invoice No: 123"
GARBAGE = {"bad.pdf": b"%PDF-1.4 garbage", "bad.docx": b"garbage", "bad.png": b"garbage"}


@pytest.fixture
def client(cfg, tmp_path, monkeypatch):
    with _client(cfg, tmp_path, monkeypatch) as c:
        yield c


def _leaks(text, tmp_path):
    """Whether text holds the upload directory, in any slash or repr form, or a generated temp name."""
    low = text.lower().replace("\\\\", "\\")
    uploads = tmp_path / "uploads"
    return api_app.UPLOAD_PREFIX in low or str(uploads).lower() in low or uploads.as_posix().lower() in low


def _api_log(caplog):
    return "\n".join(json.dumps(vars(r), default=str) for r in caplog.records if r.name.startswith("api"))


# 1. The server temp path never reaches a client, the store or the API's logs.


@pytest.mark.parametrize("name", sorted(GARBAGE))
def test_a_garbage_upload_shows_no_server_path_in_the_response(client, tmp_path, name):
    r = _upload(client, "process", name, GARBAGE[name])
    assert r.json()["diagnostics"]["errors"] and not _leaks(r.text, tmp_path)


@pytest.mark.parametrize("name", sorted(GARBAGE))
def test_a_garbage_upload_stores_no_server_path(client, tmp_path, name):
    doc_id = _upload(client, "process", name, GARBAGE[name]).json()["doc_id"]
    with closing(db.connect(tmp_path / "docc.sqlite")) as conn:
        body = conn.execute("SELECT record_json FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()[0]
    assert not _leaks(body, tmp_path) and not _leaks(client.get(f"{V1}/documents/{doc_id}").text, tmp_path)


@pytest.mark.parametrize("name", sorted(GARBAGE))
def test_a_garbage_upload_names_the_client_file_in_its_errors(client, name):
    assert name in _upload(client, "process", name, GARBAGE[name]).json()["diagnostics"]["errors"][0]


def test_an_api_log_line_about_an_upload_shows_the_client_name_not_the_path(client, tmp_path, monkeypatch, caplog):
    def fail(path, *args):
        raise DoccError(f"cannot read {path!s} or {path!r}")

    monkeypatch.setattr(api_app, "process_document", fail)
    with caplog.at_level(logging.DEBUG):
        _upload(client, "process", "bad.pdf", GARBAGE["bad.pdf"])
    text = _api_log(caplog)
    assert "request failed" in text and "bad.pdf" in text and not _leaks(text, tmp_path)


def test_scrub_replaces_every_form_of_the_path():
    path = api_app.Path("C:/srv/uploads/docc_api_ab12/upload.pdf")
    text = f"{path} {path.as_posix()} {str(path)!r} {path.parent}"
    if api_app.os.name == "nt":  # Windows paths match case-insensitively
        text += f" {str(path).upper()}"
    assert "docc_api" not in api_app.scrub(text, {path: "bad.pdf"}).lower()


# 6. Control and format characters are removed from an upload name.


@pytest.mark.parametrize(
    ("raw", "kept"),
    [
        ("a\u202eb.txt", "ab.txt"),
        ("a\u2028b\u2029.txt", "ab.txt"),
        ("a\x85b.txt", "ab.txt"),
        ("e\u0301.txt", "\u00e9.txt"),
    ],
)
def test_an_upload_name_loses_control_and_format_characters(raw, kept):
    assert api_app._clean_name(raw) == kept


def test_an_upload_name_left_empty_falls_back_to_upload_and_its_extension():
    assert api_app._clean_name("\u202e\u200b.pdf") == "upload.pdf"


def test_the_stored_name_carries_no_format_character(client):
    assert _upload(client, "process", "in\u202evoice.txt", DATA).json()["filename"] == "invoice.txt"


# 5. A refused upload leaves no cabinet or run of the refusing cabinet.


def test_a_refused_upload_leaves_no_cabinet_or_run_row(client, tmp_path):
    for cabinet in ("north", "south"):
        client.post(f"{V1}/documents/process", files={"file": ("x.txt", DATA)}, data={"cabinet_id": cabinet})
    with closing(db.connect(tmp_path / "docc.sqlite")) as conn:
        runs = conn.execute("SELECT COUNT(*) FROM runs WHERE cabinet_id = 'south'").fetchone()[0]
        assert ([c["cabinet_id"] for c in db_registry.list_cabinets(conn)], runs) == (["north"], 0)


# 7. A stored record that no longer parses is 422.


def test_an_unreadable_stored_record_is_422(client, tmp_path):
    doc_id = _upload(client, "process", "x.txt", DATA).json()["doc_id"]
    with closing(sqlite3.connect(tmp_path / "docc.sqlite")) as conn, conn:
        conn.execute("UPDATE documents SET record_json = '{broken'")
    r = client.get(f"{V1}/documents/{doc_id}")
    assert (r.status_code, r.json()["detail"]) == (422, api_app.UNREADABLE_RECORD_ERROR)


# 8. One API process per store.


def _other_instance(tmp_path, heartbeat_at):
    store = SqliteJobStore(tmp_path / "docc.sqlite", boot_id="other-process")
    with closing(store._conn) as conn, conn:
        conn.execute("INSERT INTO api_instances VALUES ('other-process', 't', ?)", (heartbeat_at,))


def test_a_second_api_process_on_the_same_store_fails_at_startup(cfg, tmp_path, monkeypatch):
    _other_instance(tmp_path, time.time())
    with pytest.raises(ConfigError, match="one API process per store"):
        _client(cfg, tmp_path, monkeypatch)


def test_a_stale_api_instance_is_replaced(cfg, tmp_path, monkeypatch):
    _other_instance(tmp_path, time.time() - API_INSTANCE_TIMEOUT_S - 1)
    with _client(cfg, tmp_path, monkeypatch) as c:
        assert c.get(f"{V1}/health").status_code == 200


def test_an_api_that_stopped_gives_up_its_claim(cfg, tmp_path, monkeypatch):
    with _client(cfg, tmp_path, monkeypatch):
        pass
    with closing(db.connect(tmp_path / "docc.sqlite")) as conn:
        assert conn.execute("SELECT COUNT(*) FROM api_instances").fetchone()[0] == 0


# 9. A cabinet id must be a slug.


@pytest.mark.parametrize("cabinet", ["Acme", "acme.ltd", "a b"])
def test_a_cabinet_id_that_is_not_a_slug_is_400(client, cabinet):
    r = client.post(f"{V1}/documents/process", files={"file": ("x.txt", DATA)}, data={"cabinet_id": cabinet})
    assert r.status_code == 400 and r.json()["error"] == api_app.CABINET_ID_ERROR


def test_a_pack_style_slug_is_a_valid_cabinet_id(client):
    r = client.post(f"{V1}/documents/process", files={"file": ("x.txt", DATA)}, data={"cabinet_id": "acme_ltd_2"})
    assert r.json()["cabinet_id"] == "acme_ltd_2"
