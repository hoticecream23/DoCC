"""Regression tests for the store bugs a review reproduced: cabinets, migrations, corrections and packs."""

import logging
import sqlite3
import threading
from contextlib import closing

import pytest
from test_db_registry import A, B, _old_store, _pack, _record

from baseline import db, db_registry, packmode
from baseline import pipeline as pipeline_mod
from baseline.errors import DocumentInOtherCabinet, PackRegistrationError
from baseline.pipeline import dumps
from baseline.schema import MetadataField, MetadataMethod

RACERS = 4
RACE_ROUNDS = 3


@pytest.fixture
def conn(tmp_path):
    with closing(db.connect(tmp_path / "s.sqlite")) as c:
        yield c


def _correct(conn, old, new, source="wb.xlsx#Metadata"):
    return db.record_review_events(conn, [db.ReviewEvent(A, "bill no", old, new, source, "corrected")])


# 1. A document is never moved between cabinets.


def test_upserting_a_stored_document_under_another_cabinet_is_refused(conn):
    db.upsert_document(conn, _record(cabinet="north"))
    with pytest.raises(DocumentInOtherCabinet) as caught:
        db.upsert_document(conn, _record(cabinet="south"))
    assert (caught.value.doc_id, caught.value.stored_cabinet_id, caught.value.cabinet_id) == (A, "north", "south")


def test_a_refused_upsert_leaves_the_first_cabinet_its_document_and_events(conn):
    db.upsert_document(conn, _record(cabinet="north"))
    _correct(conn, "X1", "X2")
    with pytest.raises(DocumentInOtherCabinet):
        db.upsert_document(conn, _record(cabinet="south", value="Z9"))
    assert [r["doc_id"] for r in db.list_documents(conn, "north")] == [A]
    assert db.list_documents(conn, "south") == []
    assert db_registry.effective_fields(conn, A)[0]["value"] == "X1"
    assert len(db.list_review_events(conn, A)) == 1


def test_a_refused_upsert_writes_an_audit_row_with_ids_only(conn):
    db.upsert_document(conn, _record(cabinet="north"))
    with pytest.raises(DocumentInOtherCabinet):
        db.upsert_document(conn, _record(cabinet="south"))
    row = db_registry.list_audit(conn, doc_id=A)[0]
    assert (row["action"], row["cabinet_id"], row["detail"]) == (
        "upsert_refused",
        "south",
        {"stored_cabinet_id": "north"},
    )


def test_import_counts_a_document_of_another_cabinet_as_refused(conn, tmp_path):
    db.upsert_document(conn, _record(cabinet="north"))
    src = tmp_path / "r.jsonl"
    src.write_text(dumps(_record(cabinet="south")) + "\n" + dumps(_record(B, cabinet="south")) + "\n", encoding="utf-8")
    stats = db.import_jsonl(conn, src)
    assert (stats["documents"], stats["refused"]) == (1, 1)


def test_the_pipeline_mirror_logs_a_refused_document_and_carries_on(conn, tmp_path, caplog):
    db.upsert_document(conn, _record(cabinet="north"))
    store = pipeline_mod._Store(conn, db.start_run(conn, {}, "south"), tmp_path)
    with caplog.at_level(logging.WARNING):
        store.mirror(dumps(_record(cabinet="south")))
    assert "another cabinet" in caplog.text
    assert db.document_cabinet(conn, A) == "north"


# 13. Import counts distinct documents.


def test_import_counts_a_repeated_doc_id_once_and_reports_the_duplicate(conn, tmp_path):
    src = tmp_path / "r.jsonl"
    src.write_text((dumps(_record()) + "\n") * 2, encoding="utf-8")
    stats = db.import_jsonl(conn, src)
    assert (stats["documents"], stats["duplicates"]) == (1, 1)


# 2. The upgrade backfills routing_status.


def test_an_upgraded_store_backfills_routing_status_from_the_record(tmp_path):
    path = _old_store(tmp_path / "old.sqlite", _record())
    with closing(db.connect(path)) as c:
        assert [r["doc_id"] for r in db.list_documents(c, routing_status="auto")] == [A]


def test_an_upgraded_store_gives_old_events_a_seq_and_a_kind(tmp_path):
    path = _old_store(tmp_path / "old.sqlite", _record())
    with closing(db.connect(path)) as c:
        got = db.list_review_events(c)[0]
    assert (got["seq"], got["kind"]) == (1, "field")


# 3 and 4. The latest correction is the last one recorded.


def test_a_correction_made_again_after_another_is_the_effective_value(conn):
    db.upsert_document(conn, _record())
    for old, new in (("X1", "X2"), ("X2", "X1"), ("X1", "X2")):
        assert _correct(conn, old, new) == 1
    assert db_registry.effective_fields(conn, A)[0]["effective_value"] == "X2"


def test_two_corrections_in_the_same_second_keep_their_import_order(conn, monkeypatch):
    from baseline import db_review

    monkeypatch.setattr(db_review, "_now", lambda: "2026-01-01T00:00:00+00:00")
    db.upsert_document(conn, _record())
    _correct(conn, "X1", "X3")
    _correct(conn, "X3", "X2")
    assert db_registry.effective_fields(conn, A)[0]["effective_value"] == "X2"


def test_recording_the_same_corrections_twice_in_a_row_adds_nothing(conn):
    db.upsert_document(conn, _record())
    events = [db.ReviewEvent(A, "bill no", "X1", "X2", "wb.xlsx#Metadata")]
    db.record_review_events(conn, events)
    assert db.record_review_events(conn, events) == 0


# 12. Only field corrections overlay a field, matched on the normalised name.


def test_a_class_correction_does_not_overlay_a_field_named_class(conn):
    field = MetadataField(
        field="Class",
        value="A",
        normalized_value="A",
        page=0,
        char_start=0,
        char_end=1,
        confidence=0.9,
        method=MetadataMethod.ANCHOR,
        validated=True,
    )
    db.upsert_document(conn, _record().model_copy(update={"metadata": [field]}))
    db.record_review_events(conn, [db.ReviewEvent(A, "class", "Supplier Bill", "other", "wb.xlsx#Documents")])
    got = db_registry.effective_fields(conn, A)[0]
    assert (got["effective_value"], got["corrected"]) == ("A", False)


def test_an_event_without_a_sheet_takes_its_kind_from_the_field_name():
    kinds = [db.event_kind(db.ReviewEvent(A, f, None, "x", "review.xlsx")) for f in ("class", "tag:urgent", "total")]
    assert kinds == ["class", "tag", "field"]


# 5. Registering a pack keeps stored cabinet values and is all or nothing.


def test_register_pack_keeps_the_cabinets_tenant_and_root_path(conn):
    db_registry.upsert_tenant(conn, "t1", "Northwind Holdings")
    db_registry.upsert_cabinet(conn, "borealis", tenant_id="t1", root_path="/data/borealis")
    db_registry.register_pack(conn, _pack())
    got = db_registry.list_cabinets(conn)[0]
    assert (got["tenant_id"], got["root_path"], got["name"]) == ("t1", "/data/borealis", "Borealis Bills")


def test_a_pack_with_a_field_named_twice_changes_nothing_and_names_it(conn):
    db_registry.register_pack(conn, _pack())
    bad = _pack()
    bad.fields["Supplier Bill"].append(bad.fields["Supplier Bill"][0])
    bad.cabinet.name = "Renamed"
    with pytest.raises(PackRegistrationError, match="Supplier Bill/Vendor Code"):
        db_registry.register_pack(conn, bad)
    assert len(db_registry.list_field_defs(conn, "borealis")) == 2
    assert db_registry.list_cabinets(conn)[0]["name"] == "Borealis Bills"


# 6. Two connections migrating an old store at once.


def _race(path):
    barrier = threading.Barrier(RACERS)
    errors = []

    def open_once():
        barrier.wait()
        try:
            db.connect(path).close()
        except sqlite3.Error as exc:
            errors.append(exc)

    threads = [threading.Thread(target=open_once) for _ in range(RACERS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return errors


def test_connections_racing_to_migrate_an_old_store_all_succeed(tmp_path):
    errors = [e for i in range(RACE_ROUNDS) for e in _race(_old_store(tmp_path / f"old{i}.sqlite", _record()))]
    assert errors == []


def test_a_failed_migration_closes_its_connection(tmp_path, monkeypatch):
    closed = []

    class Spy(sqlite3.Connection):
        def close(self):
            closed.append(True)
            super().close()

    def boom(conn):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(db, "migrate", boom)
    real_connect = sqlite3.connect
    monkeypatch.setattr(db.sqlite3, "connect", lambda *a, **k: real_connect(*a, factory=Spy, **k))
    with pytest.raises(sqlite3.OperationalError):
        db.connect(tmp_path / "s.sqlite")
    assert closed == [True]


# 11. A store that cannot be opened turns itself off.


def test_a_pack_registration_error_turns_the_store_off(cfg, tmp_path, monkeypatch):
    def refuse(conn, cfg):
        raise PackRegistrationError("pack cabinet x names these twice")

    monkeypatch.setattr(packmode, "register", refuse)
    with pipeline_mod._open_store(cfg, tmp_path / "r.jsonl", tmp_path, "x", tmp_path) as store:
        assert store is None


def test_an_os_error_opening_the_store_turns_it_off(cfg, tmp_path, monkeypatch):
    def fail(path):
        raise PermissionError("no access")

    monkeypatch.setattr(db, "connect", fail)
    with pipeline_mod._open_store(cfg, tmp_path / "r.jsonl", tmp_path, "x", tmp_path) as store:
        assert store is None
