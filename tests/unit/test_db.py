"""SQLite store: schema, upsert, import/export round trip, cabinet filter."""

import json
from contextlib import closing

import pytest

from baseline import db
from baseline.pipeline import dumps
from baseline.schema import (
    Classification,
    ClassifyMethod,
    Diagnostics,
    DocumentRecord,
    ExtractionMethod,
    MetadataField,
    MetadataMethod,
    Tag,
    TagMethod,
    TextRecord,
)

TABLES = ("documents", "fields", "tags", "tables", "provenance", "review_events", "runs", "models")


def _record(doc_id: str, cabinet: str = "default", label: str = "invoice") -> DocumentRecord:
    text = "Invoice No: INV-1"
    return DocumentRecord(
        doc_id=doc_id,
        cabinet_id=cabinet,
        source_path=f"corpus/{doc_id}.txt",
        filename=f"{doc_id}.txt",
        text=TextRecord(
            full=text,
            pages=[text],
            method=ExtractionMethod.NATIVE,
            char_count=len(text),
            extraction_confidence=0.98,
            page_methods=[ExtractionMethod.NATIVE],
        ),
        classification=Classification(label=label, confidence=0.9, all_scores={label: 0.9}, method=ClassifyMethod.RULE),
        metadata=[
            MetadataField(
                field="invoice_number",
                value="INV-1",
                normalized_value="INV-1",
                page=0,
                char_start=12,
                char_end=17,
                confidence=0.95,
                method=MetadataMethod.ANCHOR,
                validated=False,
            )
        ],
        tags=[Tag(tag="financial", confidence=0.8, method=TagMethod.KEYWORD)],
        diagnostics=Diagnostics(page_count=1, is_scanned=False, has_native_text=True),
    )


def _write(path, records):
    path.write_text("".join(dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def _counts(conn):
    return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in TABLES}


@pytest.fixture
def conn(tmp_path):
    with closing(db.connect(tmp_path / "s.sqlite")) as c:
        yield c


def test_connect_creates_every_table_and_is_idempotent(tmp_path):
    path = tmp_path / "s.sqlite"
    db.connect(path).close()
    with closing(db.connect(path)) as c:
        names = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert set(TABLES) <= names


def test_upsert_twice_leaves_one_document_and_its_children(conn):
    rec = _record("aaaaaaaaaaaaaaaa")
    db.upsert_document(conn, rec)
    db.upsert_document(conn, rec)
    counts = _counts(conn)
    assert (counts["documents"], counts["fields"], counts["tags"]) == (1, 1, 1)


def test_import_twice_gives_equal_counts(conn, tmp_path):
    src = _write(tmp_path / "r.jsonl", [_record("aaaaaaaaaaaaaaaa"), _record("bbbbbbbbbbbbbbbb")])
    db.import_jsonl(conn, src)
    first = _counts(conn)
    db.import_jsonl(conn, src)
    assert _counts(conn) == first


def test_export_round_trips_to_equal_records(conn, tmp_path):
    records = [_record("bbbbbbbbbbbbbbbb"), _record("aaaaaaaaaaaaaaaa", cabinet="x")]
    db.import_jsonl(conn, _write(tmp_path / "r.jsonl", records))
    out = tmp_path / "out.jsonl"
    db.export_jsonl(conn, out)
    got = [DocumentRecord.model_validate_json(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert got == sorted(records, key=lambda r: r.doc_id)


def test_export_filters_by_cabinet(conn, tmp_path):
    records = [_record("aaaaaaaaaaaaaaaa", cabinet="x"), _record("bbbbbbbbbbbbbbbb", cabinet="y")]
    db.import_jsonl(conn, _write(tmp_path / "r.jsonl", records))
    out = tmp_path / "out.jsonl"
    assert db.export_jsonl(conn, out, cabinet_id="y") == 1
    assert json.loads(out.read_text(encoding="utf-8"))["doc_id"] == "bbbbbbbbbbbbbbbb"


def test_list_documents_filters_by_cabinet(conn):
    db.upsert_document(conn, _record("aaaaaaaaaaaaaaaa", cabinet="x"))
    db.upsert_document(conn, _record("bbbbbbbbbbbbbbbb", cabinet="y", label="receipt"))
    rows = db.list_documents(conn, cabinet_id="y")
    assert [(r["doc_id"], r["doc_type"]) for r in rows] == [("bbbbbbbbbbbbbbbb", "receipt")]


def test_old_jsonl_without_cabinet_id_imports_as_default(conn, tmp_path):
    row = _record("aaaaaaaaaaaaaaaa").model_dump(mode="json")
    del row["cabinet_id"]
    src = tmp_path / "old.jsonl"
    src.write_text(json.dumps(row) + "\n", encoding="utf-8")
    db.import_jsonl(conn, src)
    assert db.list_documents(conn)[0]["cabinet_id"] == "default"


def test_sidecars_import_and_skip_unknown_doc_ids(conn, tmp_path):
    src = _write(tmp_path / "r.jsonl", [_record("aaaaaaaaaaaaaaaa")])
    table = {"table_id": "t0", "page": 0, "n_rows": 1, "n_cols": 1, "method": "text_grid", "confidence": 0.5}
    tables = tmp_path / "tables.jsonl"
    tables.write_text(
        "".join(
            json.dumps({"doc_id": d, "source_path": "x", "filename": "x", "tables": [table], "diagnostics": {}}) + "\n"
            for d in ("aaaaaaaaaaaaaaaa", "cccccccccccccccc")
        ),
        encoding="utf-8",
    )
    prov = tmp_path / "r.jsonl.provenance"
    prov.write_text(json.dumps({"doc_id": "aaaaaaaaaaaaaaaa", "fallback": True, "errors": ["down"]}) + "\n")
    stats = db.import_jsonl(conn, src, provenance_path=prov, tables_path=tables)
    assert (stats["tables"], stats["provenance"], stats["skipped"]) == (1, 1, 1)
    assert conn.execute("SELECT abstention_reason FROM provenance").fetchone()[0] == "down"


def test_start_run_and_record_models_write_rows(conn):
    run_id = db.start_run(conn, {"pipeline": {}}, "default")
    db.record_models(conn, {"classifier/model.joblib": "ab" * 32})
    db.record_models(conn, {"classifier/model.joblib": "ab" * 32})
    assert conn.execute("SELECT run_id FROM runs").fetchone()[0] == run_id
    assert _counts(conn)["models"] == 1


def test_review_events_are_idempotent_and_skip_unknown_documents(conn):
    db.upsert_document(conn, _record("aaaaaaaaaaaaaaaa"))
    db.upsert_document(conn, _record("bbbbbbbbbbbbbbbb"))
    events = [
        db.ReviewEvent("aaaaaaaaaaaaaaaa", "invoice_number", "INV-1", "INV-2", "wb.xlsx#Metadata"),
        db.ReviewEvent("bbbbbbbbbbbbbbbb", "class", "invoice", "receipt", "wb.xlsx#Documents"),
        db.ReviewEvent("cccccccccccccccc", "class", "invoice", "receipt", "wb.xlsx#Documents"),
    ]
    assert db.record_review_events(conn, events) == 2
    assert db.record_review_events(conn, events) == 0
    got = db.list_review_events(conn, doc_id="aaaaaaaaaaaaaaaa")
    assert [(e["field"], e["old_value"], e["new_value"]) for e in got] == [("invoice_number", "INV-1", "INV-2")]
    assert len(db.list_review_events(conn)) == 2


def test_a_cascade_store_failure_is_logged_not_raised(tmp_path, caplog):
    from baseline.pipeline import _Store
    from cascade.runner import Result, _mirror

    closed = db.connect(tmp_path / "s.sqlite")
    closed.close()
    rec = _record("aaaaaaaaaaaaaaaa")
    _mirror(_Store(closed, "run", tmp_path), Result(rec, None, {"doc_id": rec.doc_id}))
    assert "db write failed" in caplog.text
