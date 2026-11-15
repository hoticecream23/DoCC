"""Regression tests for API bugs from the confirmation review: instance claims, path scrubbing, cabinet ids."""

import sqlite3
import time
from contextlib import closing
from urllib.parse import quote

import pytest

pytest.importorskip("fastapi")

from test_api import V1, _client  # noqa: E402

import api.app as api_app  # noqa: E402
from api.jobs import API_INSTANCE_TIMEOUT_S, SqliteJobStore  # noqa: E402
from baseline.errors import ConfigError  # noqa: E402

CLIENT_NAME = "bad.pdf"


# 3. One API instance per store, in any process.


def _claims(path):
    with closing(sqlite3.connect(path)) as conn:
        return [r[0] for r in conn.execute("SELECT boot_id FROM api_instances")]


def _foreign_claim(path, heartbeat_at):
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("INSERT INTO api_instances VALUES ('other', 't', ?)", (heartbeat_at,))


def test_a_second_app_in_the_same_process_on_the_same_store_is_refused(cfg, tmp_path, monkeypatch):
    with _client(cfg, tmp_path, monkeypatch), pytest.raises(ConfigError, match="one API process per store"):
        _client(cfg, tmp_path, monkeypatch)


def test_a_refused_second_app_leaves_the_first_apps_claim(cfg, tmp_path, monkeypatch):
    with _client(cfg, tmp_path, monkeypatch) as c:
        with pytest.raises(ConfigError):
            _client(cfg, tmp_path, monkeypatch)
        assert len(_claims(tmp_path / "docc.sqlite")) == 1 and c.get(f"{V1}/health").status_code == 200


def test_close_removes_only_the_callers_own_claim(tmp_path):
    path = tmp_path / "docc.sqlite"
    store = SqliteJobStore(path)
    store.claim()
    _foreign_claim(path, time.time())
    store.close()
    assert _claims(path) == ["other"]


def test_a_claim_far_in_the_future_is_stale(tmp_path):
    path = tmp_path / "docc.sqlite"
    SqliteJobStore(path).close()
    _foreign_claim(path, time.time() + API_INSTANCE_TIMEOUT_S + 60)
    store = SqliteJobStore(path)
    store.claim()
    store.close()


def test_a_claim_slightly_in_the_future_is_live(tmp_path):
    path = tmp_path / "docc.sqlite"
    SqliteJobStore(path).close()
    _foreign_claim(path, time.time() + API_INSTANCE_TIMEOUT_S / 2)
    store = SqliteJobStore(path)
    with closing(store), pytest.raises(ConfigError):
        store.claim()


# 4. Every form of the upload paths is scrubbed; ordinary words are not.


@pytest.fixture
def upload(tmp_path):
    path = tmp_path / "up loads" / "docc_api_ab12" / "upload.pdf"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"x")
    return path


def _scrubbed(text, upload):
    return api_app.scrub(text, {upload: CLIENT_NAME})


def test_scrub_removes_the_upload_root(upload):
    root = upload.parent.parent
    assert str(root).lower() not in _scrubbed(f"cannot list {root}", upload).lower()


def test_scrub_removes_the_file_uri_of_the_upload(upload):
    assert "docc_api" not in _scrubbed(f"open {upload.as_uri()} failed", upload)


def test_scrub_removes_the_file_uri_of_the_upload_root(upload):
    root_uri = upload.parent.parent.as_uri()
    assert root_uri.lower() not in _scrubbed(f"open {root_uri} failed", upload).lower()


def test_scrub_removes_a_percent_encoded_upload_root(upload):
    encoded = quote(upload.parent.parent.as_posix())
    assert encoded.lower() not in _scrubbed(f"see {encoded}", upload).lower()


def test_scrub_removes_any_generated_temp_directory_name(upload):
    assert "docc_api" not in _scrubbed("left docc_api_zz99xy behind", upload)


def test_scrub_leaves_the_last_path_component_as_an_ordinary_word(upload):
    assert _scrubbed("the up loads folder is full", upload) == "the up loads folder is full"


def test_scrub_removes_the_windows_short_path(upload):
    short = api_app._short_path(upload)
    if short is None or str(short).lower() == str(upload).lower():
        pytest.skip("no 8.3 short path here")
    assert "docc_api" not in _scrubbed(f"cannot read {short}", upload).lower()


def test_short_path_is_none_for_a_missing_path(tmp_path):
    assert api_app._short_path(tmp_path / "absent") is None


# 5. The cabinet id rule is named and stated.


def test_a_cabinet_id_of_the_maximum_length_matches():
    assert api_app.CABINET_ID_RE.fullmatch("a" * api_app.CABINET_ID_MAX_CHARS)


def test_a_cabinet_id_over_the_maximum_length_does_not_match():
    assert not api_app.CABINET_ID_RE.fullmatch("a" * (api_app.CABINET_ID_MAX_CHARS + 1))


def test_the_cabinet_id_error_states_the_limit():
    assert f"1-{api_app.CABINET_ID_MAX_CHARS}" in api_app.CABINET_ID_ERROR
