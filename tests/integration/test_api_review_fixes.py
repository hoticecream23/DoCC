"""Regression tests for the API bugs a review reproduced: cabinets, uploads, paths and unknown ids."""

import json
import sqlite3

import pytest

pytest.importorskip("fastapi")

from test_api import V1, _client, _process, _upload  # noqa: E402

import api.app as api_app  # noqa: E402

DATA = b"Invoice No: 123"


@pytest.fixture
def client(cfg, tmp_path, monkeypatch):
    with _client(cfg, tmp_path, monkeypatch) as c:
        yield c


def _post(client, name, cabinet):
    return client.post(f"{V1}/documents/process", files={"file": (name, DATA)}, data={"cabinet_id": cabinet})


def test_the_same_file_under_a_second_cabinet_is_409_without_naming_the_first(client):
    first = _post(client, "x.txt", "north").json()
    r = _post(client, "x.txt", "south")
    assert r.status_code == 409
    assert r.json() == {"error": api_app.OTHER_CABINET_ERROR, "doc_id": first["doc_id"]}


def test_the_first_cabinet_keeps_a_document_another_cabinet_uploaded_again(client):
    _post(client, "x.txt", "north")
    _post(client, "x.txt", "south")
    assert [d["cabinet_id"] for d in client.get(f"{V1}/documents").json()] == ["north"]


def test_an_over_long_file_name_is_400_and_leaves_no_upload(client, tmp_path):
    r = _upload(client, "process", "x" * 300 + ".txt", DATA)
    assert r.status_code == 400 and "longer than" in r.json()["error"]
    assert list((tmp_path / "uploads").iterdir()) == []


def test_a_file_name_with_reserved_characters_is_400(client, tmp_path):
    r = _upload(client, "process", "a<b>.txt", DATA)
    assert r.status_code == 400 and "invalid file name" in r.json()["error"]
    assert list((tmp_path / "uploads").iterdir()) == []


def test_the_saved_file_is_named_by_the_server_not_the_client(client, monkeypatch):
    seen = []
    real = api_app.process_document

    def spy(path, *args):
        seen.append(path.name)
        return real(path, *args)

    monkeypatch.setattr(api_app, "process_document", spy)
    assert _upload(client, "process", "Quarterly Bill.TXT", DATA).json()["filename"] == "Quarterly Bill.TXT"
    assert seen == [api_app.UPLOAD_STEM + ".txt"]


def test_a_job_store_error_on_submit_still_removes_the_upload(cfg, tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(api_app.JobRunner, "submit", broken)
    with _client(cfg, tmp_path, monkeypatch, "http://127.0.0.1:9/v1", "test-model") as c:
        with pytest.raises(sqlite3.OperationalError):
            _upload(c, "process_async", "x.txt", DATA)
    assert list((tmp_path / "uploads").iterdir()) == []


def test_no_response_shows_a_server_path(client, tmp_path):
    rec = _upload(client, "process", "x.txt", DATA).json()
    bodies = [json.dumps(rec)] + [client.get(f"{V1}{p}").text for p in ("/documents", f"/documents/{rec['doc_id']}")]
    uploads = tmp_path / "uploads"
    for body in bodies:
        assert str(uploads) not in body.replace("\\\\", "\\") and uploads.as_posix() not in body
    assert rec["source_path"] == "x.txt"


@pytest.mark.parametrize("route", ["fields", "review_events"])
def test_a_document_route_for_an_unknown_doc_id_is_404(client, route):
    assert client.get(f"{V1}/documents/nope/{route}").status_code == 404


def test_review_events_of_a_known_document_without_any_are_an_empty_list(client):
    rec = _upload(client, "process", "x.txt", DATA).json()
    r = client.get(f"{V1}/documents/{rec['doc_id']}/review_events")
    assert (r.status_code, r.json()) == (200, [])


def test_doc_types_of_an_unknown_cabinet_are_404(client):
    assert client.get(f"{V1}/cabinets/nope/doc_types").status_code == 404


def test_doc_types_of_a_known_cabinet_without_any_are_an_empty_list(client):
    _post(client, "x.txt", "north")
    r = client.get(f"{V1}/cabinets/north/doc_types")
    assert (r.status_code, r.json()) == (200, [])


@pytest.mark.parametrize("route", ["", "/fields", "/review_events"])
def test_a_document_asked_for_under_another_cabinet_is_404(client, route):
    rec = _post(client, "x.txt", "north").json()
    assert client.get(f"{V1}/documents/{rec['doc_id']}{route}", params={"cabinet_id": "south"}).status_code == 404


def test_a_document_asked_for_under_its_own_cabinet_is_found(client, corpus):
    rec = _process(client, corpus, "receipt.txt", "north")
    assert client.get(f"{V1}/documents/{rec['doc_id']}", params={"cabinet_id": "north"}).status_code == 200
