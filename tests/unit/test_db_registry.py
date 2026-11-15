"""Store additions: migration, cabinets, field definitions, routing, verdicts and the audit log."""

import csv
import json
import sqlite3
from contextlib import closing

import make_pack
import pytest

from baseline import db, db_migrate, db_registry, packmode
from baseline.clientpack import CabinetEntry, ClientPack, FieldSpec, ManifestRow, TaxonomyClass
from baseline.pipeline import _start_store, dumps
from baseline.schema import (
    Classification,
    ClassifyMethod,
    Diagnostics,
    DocumentRecord,
    ExtractionMethod,
    MetadataField,
    MetadataMethod,
    TextRecord,
)

A, B = "aaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbb"
SECRET = "SECRET-VALUE-42"


def _record(
    doc_id=A, cabinet="borealis", label="Supplier Bill", method=ClassifyMethod.RULE, conf=0.95, errors=(), value="X1"
):
    text = f"Bill No: {value}"
    return DocumentRecord(
        doc_id=doc_id,
        cabinet_id=cabinet,
        source_path=f"in/{doc_id}.txt",
        filename=f"{doc_id}.txt",
        text=TextRecord(
            full=text,
            pages=[text],
            method=ExtractionMethod.NATIVE,
            char_count=len(text),
            extraction_confidence=0.98,
            page_methods=[ExtractionMethod.NATIVE],
        ),
        classification=Classification(label=label, confidence=conf, all_scores={label: conf}, method=method),
        metadata=[
            MetadataField(
                field="Bill No",
                value=value,
                normalized_value=value,
                page=0,
                char_start=9,
                char_end=9 + len(value),
                confidence=0.95,
                method=MetadataMethod.ANCHOR,
                validated=True,
            )
        ],
        diagnostics=Diagnostics(page_count=1, is_scanned=False, has_native_text=True, errors=list(errors)),
    )


def _pack(classes=("Supplier Bill", "other")):
    fields = {
        "Supplier Bill": [
            FieldSpec(name="Vendor Code", dtype="text", role="required", origin="erp"),
            FieldSpec(name="Bill No", dtype="text", role="key", origin="document"),
        ]
    }
    return ClientPack(
        cabinet=CabinetEntry(slug="borealis", name="Borealis Bills", documents=1, types_with_samples=1),
        classes=[
            TaxonomyClass(name=c, aliases=["Tax Bill"] if c != "other" else [], has_samples=True) for c in classes
        ],
        fields=fields,
        manifest=[ManifestRow(doc_id=A, rel_path="x/a.pdf", case_id="CASE-7", size_kb=1)],
    )


@pytest.fixture
def conn(tmp_path):
    with closing(db.connect(tmp_path / "s.sqlite")) as c:
        yield c


def _old_store(path, record):
    """A store made with the pre-migration DDL, holding one document."""
    with closing(sqlite3.connect(path)) as c, c:
        for stmt in db.SCHEMA_DDL:
            c.execute(stmt)
        c.execute(
            "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, ?, ?, ?)",
            (record.doc_id, record.cabinet_id, record.source_path, record.filename, "Supplier Bill", 0.95, "rule")
            + (record.schema_version, 1, dumps(record), "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00"),
        )
        c.execute("INSERT INTO review_events VALUES ('e1', ?, 'Bill No', 'X1', 'X2', 'wb', 't')", (record.doc_id,))
    return path


def _columns(c, table):
    return {r[0] for r in c.execute("SELECT name FROM pragma_table_info(?)", (table,))}


def test_an_old_store_gains_the_new_columns_and_keeps_its_rows(tmp_path):
    path = _old_store(tmp_path / "old.sqlite", _record())
    with closing(db.connect(path)) as c:
        assert {"routing_status", "case_id"} <= _columns(c, "documents")
        assert {"verdict", "is_gold"} <= _columns(c, "review_events")
        assert "trace_id" in _columns(c, "runs")
        assert c.execute("SELECT doc_id FROM documents").fetchall()[0][0] == A
        assert c.execute("SELECT is_gold FROM review_events").fetchone()[0] == 0


def test_a_migrated_store_lists_its_existing_cabinet(tmp_path):
    path = _old_store(tmp_path / "old.sqlite", _record())
    with closing(db.connect(path)) as c:
        assert [(r["cabinet_id"], r["documents"]) for r in db_registry.list_cabinets(c)] == [("borealis", 1)]


def test_migrate_is_a_no_op_the_second_time(conn):
    assert db_migrate.migrate(conn) == []


def test_a_half_applied_step_is_safe_to_run_again(conn):
    with conn:
        conn.execute("DELETE FROM schema_migrations")
    assert db_migrate.migrate(conn) == [1, 2]


def test_export_of_a_migrated_store_is_unchanged(tmp_path):
    record = _record()
    path = _old_store(tmp_path / "old.sqlite", record)
    out = tmp_path / "out.jsonl"
    with closing(db.connect(path)) as c:
        db.export_jsonl(c, out)
    assert out.read_text(encoding="utf-8") == dumps(record) + "\n"


def test_upsert_document_adds_its_cabinet(conn):
    db.upsert_document(conn, _record())
    db.upsert_document(conn, _record(B))
    assert [(r["cabinet_id"], r["name"], r["documents"]) for r in db_registry.list_cabinets(conn)] == [
        ("borealis", "borealis", 2)
    ]


def test_start_run_stores_the_trace_id_and_adds_the_cabinet(conn):
    run_id = db.start_run(conn, {}, "acme", trace_id="req-1")
    assert conn.execute("SELECT trace_id FROM runs WHERE run_id = ?", (run_id,)).fetchone()[0] == "req-1"
    assert [r["cabinet_id"] for r in db_registry.list_cabinets(conn)] == ["acme"]


def test_a_cabinet_can_belong_to_a_tenant(conn):
    db_registry.upsert_tenant(conn, "t1", "Acme Holdings", domain_type="finance")
    db_registry.upsert_cabinet(conn, "acme", name="Acme Bills", tenant_id="t1")
    assert db_registry.list_cabinets(conn)[0]["tenant_id"] == "t1"


def test_register_pack_twice_leaves_the_same_rows(conn):
    db_registry.register_pack(conn, _pack())
    first = (db_registry.list_doc_types(conn, "borealis"), db_registry.list_field_defs(conn, "borealis"))
    db_registry.register_pack(conn, _pack())
    assert (db_registry.list_doc_types(conn, "borealis"), db_registry.list_field_defs(conn, "borealis")) == first


def test_register_pack_replaces_a_dropped_class(conn):
    db_registry.register_pack(conn, _pack(("Supplier Bill", "other")))
    db_registry.register_pack(conn, _pack(("Supplier Bill",)))
    assert [t["name"] for t in db_registry.list_doc_types(conn, "borealis")] == ["Supplier Bill"]


def test_key_field_is_the_types_key_or_none(conn):
    db_registry.register_pack(conn, _pack())
    assert db_registry.key_field(conn, "borealis", "Supplier Bill") == "Bill No"
    assert db_registry.key_field(conn, "borealis", "other") is None


def test_list_field_defs_filters_by_type(conn):
    db_registry.register_pack(conn, _pack())
    assert db_registry.list_field_defs(conn, "borealis", "other") == []


def test_effective_fields_take_the_latest_correction(conn, monkeypatch):
    db.upsert_document(conn, _record())
    for at, new in (("2026-01-01T00:00:00+00:00", "X2"), ("2026-01-02T00:00:00+00:00", "X3")):
        monkeypatch.setattr(db, "_now", lambda at=at: at)
        db.record_review_events(conn, [db.ReviewEvent(A, "bill no", "X1", new, "wb.xlsx#Metadata", "corrected")])
    got = db_registry.effective_fields(conn, A)[0]
    assert (got["value"], got["effective_value"], got["corrected"], got["verdict"]) == ("X1", "X3", True, "corrected")


def test_effective_fields_without_a_correction_keep_the_stored_value(conn):
    db.upsert_document(conn, _record())
    got = db_registry.effective_fields(conn, A)[0]
    assert (got["effective_value"], got["corrected"], got["role"]) == ("X1", False, None)


def test_effective_fields_join_the_cabinets_field_definition(conn):
    db_registry.register_pack(conn, _pack())
    db.upsert_document(conn, _record())
    got = db_registry.effective_fields(conn, A)[0]
    assert (got["role"], got["origin"]) == ("key", "document")


def test_a_record_with_errors_routes_to_failed(conn):
    db.upsert_document(conn, _record(errors=["ocr: boom"]))
    assert db.list_documents(conn)[0]["routing_status"] == "failed"


def test_an_abstention_routes_to_review(conn):
    db.upsert_document(conn, _record(label="other", method=ClassifyMethod.FALLBACK, conf=0.9))
    assert db.list_documents(conn)[0]["routing_status"] == "review"


def test_a_confident_validated_record_routes_to_auto(conn):
    db.upsert_document(conn, _record())
    assert db.list_documents(conn)[0]["routing_status"] == "auto"


def test_list_documents_filters_by_routing_status(conn):
    db.upsert_document(conn, _record(A))
    db.upsert_document(conn, _record(B, errors=["x"]))
    assert [r["doc_id"] for r in db.list_documents(conn, routing_status="failed")] == [B]


def test_list_documents_filters_by_case_id(conn):
    db.upsert_document(conn, _record(A), case_id="CASE-7")
    db.upsert_document(conn, _record(B))
    assert [r["doc_id"] for r in db.list_documents(conn, case_id="CASE-7")] == [A]


def test_an_upsert_without_a_case_id_keeps_the_stored_one(conn):
    db.upsert_document(conn, _record(A), case_id="CASE-7")
    db.upsert_document(conn, _record(A))
    assert db.list_documents(conn)[0]["case_id"] == "CASE-7"


def test_verdict_and_gold_are_stored_and_the_event_id_ignores_them(conn):
    db.upsert_document(conn, _record())
    plain = db.ReviewEvent(A, "bill no", "X1", "X2", "wb.xlsx#Metadata")
    db.record_review_events(conn, [plain._replace(verdict="corrected", is_gold=True)])
    assert db.record_review_events(conn, [plain]) == 0
    got = db.list_review_events(conn)[0]
    assert (got["verdict"], got["is_gold"]) == ("corrected", 1)


def test_audit_rows_name_ids_and_counts_never_values(conn, tmp_path):
    src = tmp_path / "r.jsonl"
    src.write_text(dumps(_record(value=SECRET)) + "\n", encoding="utf-8")
    db.import_jsonl(conn, src)
    db_registry.register_pack(conn, _pack())
    db.record_review_events(conn, [db.ReviewEvent(A, "bill no", SECRET, SECRET + "9", "wb.xlsx#Metadata")])
    rows = db_registry.list_audit(conn)
    assert {r["action"] for r in rows} == {"db.import", "pack.register", "review_events.record"}
    assert SECRET not in json.dumps(rows) and "Bill No:" not in json.dumps(rows)


def test_list_audit_filters_by_document(conn):
    db_registry.record_audit(conn, actor="api", action="document.create", cabinet_id="acme", doc_id=A)
    db_registry.record_audit(conn, actor="api", action="document.create", cabinet_id="acme", doc_id=B)
    assert [r["audit_id"] for r in db_registry.list_audit(conn, doc_id=B)] == [2]


def test_a_pack_run_registers_its_cabinet_and_files_the_case(conn, cfg, tmp_path):
    pack_dir = make_pack.build(tmp_path / "pack")
    manifest = pack_dir / make_pack.SLUG / "manifest.csv"
    with open(manifest, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    rows[0]["case_id"] = "CASE-1"
    with open(manifest, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=make_pack.HEADER)
        writer.writeheader()
        writer.writerows(rows)
    pcfg = packmode.pack_config(cfg, pack_dir, make_pack.CABINET)
    store = _start_store(conn, pcfg, tmp_path, make_pack.SLUG, tmp_path)
    rec = _record(cabinet=make_pack.SLUG).model_copy(update={"filename": f"{rows[0]['doc_id']}.pdf"})
    store.mirror(dumps(rec))
    assert db.list_documents(conn)[0]["case_id"] == "CASE-1"
    assert db_registry.key_field(conn, make_pack.SLUG, "Supplier Bill") == "Invoice No"
