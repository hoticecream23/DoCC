"""Regression tests for store bugs from the second review: sidecars, corrections, locking, refusals, migration."""

import json
import logging
import sqlite3
from contextlib import closing

import pytest
from test_db_registry import A, _old_store, _record

from baseline import db, db_migrate, db_registry
from baseline.errors import DocumentInOtherCabinet
from baseline.pipeline import dumps

FIELD = "bill no"
SOURCE = "wb.xlsx#Metadata"
SHORT_BUSY_S = 0.2


@pytest.fixture
def conn(tmp_path):
    with closing(db.connect(tmp_path / "s.sqlite")) as c:
        yield c


def _events(*pairs):
    return [db.ReviewEvent(A, FIELD, old, new, SOURCE) for old, new in pairs]


def _effective(conn):
    return db_registry.effective_fields(conn, A)[0]["effective_value"]


# 2. Sidecar lines of a refused document are not written.


def _table_line(doc_id, method):
    table = {"table_id": "t0", "page": 0, "n_rows": 1, "n_cols": 1, "method": method, "confidence": 0.5}
    return {"doc_id": doc_id, "source_path": "x", "filename": "x", "tables": [table], "diagnostics": {}}


def _store_owner(conn, tmp_path):
    """Store A under north with its provenance and tables rows."""
    db.upsert_document(conn, _record(cabinet="north"))
    db.upsert_provenance(conn, {"doc_id": A, "model": "owner-model", "fallback": False})
    db._upsert_tables(conn, db.TableRecord.model_validate(_table_line(A, "text_grid")), tmp_path / "own.jsonl")


def _import_intruder(conn, tmp_path):
    """Import A for south with intruding provenance and tables lines."""
    src, prov, tables = tmp_path / "r.jsonl", tmp_path / "p.jsonl", tmp_path / "t.jsonl"
    src.write_text(dumps(_record(cabinet="south")) + "\n", encoding="utf-8")
    prov.write_text(json.dumps({"doc_id": A, "model": "intruder-model", "fallback": True, "errors": ["x"]}) + "\n")
    tables.write_text(json.dumps(_table_line(A, "geometry")) + "\n", encoding="utf-8")
    return db.import_jsonl(conn, src, provenance_path=prov, tables_path=tables)


def _sidecar_rows(conn):
    return [tuple(r) for t in ("provenance", "tables") for r in conn.execute(f"SELECT * FROM {t}").fetchall()]


def test_sidecar_lines_of_a_refused_document_leave_the_owners_rows_identical(conn, tmp_path):
    _store_owner(conn, tmp_path)
    before = _sidecar_rows(conn)
    _import_intruder(conn, tmp_path)
    assert _sidecar_rows(conn) == before


def test_sidecar_lines_of_a_refused_document_are_skipped_and_counted(conn, tmp_path):
    _store_owner(conn, tmp_path)
    stats = _import_intruder(conn, tmp_path)
    assert (stats["refused"], stats["skipped"], stats["tables"], stats["provenance"]) == (1, 2, 0, 0)


# 3. One import of corrections is idempotent as a whole.


def test_a_workbook_with_two_rows_for_one_field_adds_nothing_when_repeated(conn):
    db.upsert_document(conn, _record())
    w1 = _events(("X1", "X2"), ("X2", "X3"))
    assert [db.record_review_events(conn, w1) for _ in range(3)] == [2, 0, 0]


def test_a_workbook_imported_again_after_another_is_re_applied_and_wins(conn):
    db.upsert_document(conn, _record())
    w1, w2 = _events(("X1", "X2"), ("X2", "X3")), _events(("X1", "Y9"))
    assert [db.record_review_events(conn, w) for w in (w1, w2, w1)] == [2, 1, 2]
    assert _effective(conn) == "X3"


def test_a_correction_and_its_undo_across_three_imports_ends_at_the_last(conn):
    db.upsert_document(conn, _record())
    for pair in (("A", "B"), ("B", "A"), ("A", "B")):
        db.record_review_events(conn, _events(pair))
    assert _effective(conn) == "B"


# 4. Opening a current store takes no write lock.


def test_opening_a_current_store_does_not_wait_behind_a_writer(tmp_path, monkeypatch):
    path = tmp_path / "s.sqlite"
    db.connect(path).close()
    with closing(sqlite3.connect(path, isolation_level=None)) as writer:
        writer.execute("BEGIN IMMEDIATE")
        monkeypatch.setattr(db, "BUSY_TIMEOUT_S", SHORT_BUSY_S)
        db.connect(path).close()  # raised "database is locked" before
        writer.execute("ROLLBACK")


# 5. A refused API write leaves no cabinet, run or provenance row.


def _refused_store_record(conn):
    db.upsert_document(conn, _record(cabinet="north"))
    with pytest.raises(DocumentInOtherCabinet):
        db.store_record(conn, _record(cabinet="south"), {}, trace_id="t1", provenance={"doc_id": A, "fallback": True})


def test_a_refused_store_record_leaves_no_cabinet_or_run_row(conn):
    _refused_store_record(conn)
    cabinets = [c["cabinet_id"] for c in db_registry.list_cabinets(conn)]
    runs = conn.execute("SELECT COUNT(*) FROM runs WHERE cabinet_id = 'south'").fetchone()[0]
    assert (cabinets, runs) == (["north"], 0)


def test_a_refused_store_record_leaves_no_provenance_row(conn):
    _refused_store_record(conn)
    assert conn.execute("SELECT COUNT(*) FROM provenance").fetchone()[0] == 0


def test_a_refused_store_record_still_writes_its_audit_row(conn):
    _refused_store_record(conn)
    assert db_registry.list_audit(conn, doc_id=A)[0]["action"] == "upsert_refused"


def test_store_record_writes_the_run_and_the_document_together(conn):
    run_id = db.store_record(conn, _record(cabinet="north"), {}, trace_id="t1")
    assert db.list_documents(conn)[0]["run_id"] == run_id


# 7. Unreadable stored records go to review, counted once.


def _old_store_with(tmp_path, body):
    path = _old_store(tmp_path / "old.sqlite", _record())
    with closing(sqlite3.connect(path)) as c, c:
        c.execute("UPDATE documents SET record_json = ?", (body,))
    return path


def _older_version_body():
    row = json.loads(dumps(_record()))
    row["schema_version"] = "0.9.0"
    return json.dumps(row)


@pytest.mark.parametrize("body", ["{not json", _older_version_body()], ids=["unparseable", "older-version"])
def test_an_unreadable_record_is_routed_to_review_on_upgrade(tmp_path, body):
    with closing(db.connect(_old_store_with(tmp_path, body))) as c:
        assert [r["doc_id"] for r in db.list_documents(c, routing_status="review")] == [A]


def test_unreadable_records_get_one_audit_row_with_their_count_only(tmp_path):
    with closing(db.connect(_old_store_with(tmp_path, "{not json"))) as c:
        rows = [r for r in db_registry.list_audit(c) if r["action"] == db_migrate.UNREADABLE_ACTION]
    assert [(r["detail"], r["doc_id"], r["cabinet_id"]) for r in rows] == [({"count": 1}, None, None)]


def test_a_failed_route_given_to_an_unreadable_record_before_v3_becomes_review(conn):
    db.upsert_document(conn, _record())
    with conn:
        conn.execute("UPDATE documents SET record_json = '{not json', routing_status = 'failed'")
        conn.execute("DELETE FROM schema_migrations WHERE version = 3")
    assert db_migrate.migrate(conn) == [3]
    assert db.list_documents(conn)[0]["routing_status"] == "review"


def test_get_document_of_an_unparseable_row_is_none_and_logs_the_doc_id_only(conn, caplog):
    db.upsert_document(conn, _record())
    with conn:
        conn.execute("UPDATE documents SET record_json = '{SECRET-VALUE'")
    with caplog.at_level(logging.WARNING):
        assert db.get_document(conn, A) is None
    got = [r for r in caplog.records if r.getMessage() == "stored record is unreadable"]
    assert [r.doc_id for r in got] == [A] and "SECRET" not in caplog.text
