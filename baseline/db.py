"""SQLite store that mirrors the results JSONL.

Over 400 lines: one schema plus its import and export keep the DDL and the row mappers together.

The JSONL file is the log and the source of truth for a run. This store is a
queryable copy of it, for review and cabinet state. It can always be rebuilt
from the JSONL with `import_jsonl`, and every import is idempotent.

The DDL is portable on purpose: TEXT, INTEGER and REAL columns only, JSON kept
as TEXT, composite primary keys instead of autoincrement, and upserts written
as `INSERT ... ON CONFLICT ... DO UPDATE`, which PostgreSQL also accepts. WAL
mode and foreign keys are connection pragmas, not part of the schema.

The `jobs` table holds API job state (api/jobs.py SqliteJobStore); it is not
part of the JSONL mirror. `SCHEMA_DDL` is the base shape; the tables and
columns added since (cabinets, field definitions, routing, verdicts, audit log)
live in `db_registry.py` and reach an existing store through the migrations
in `db_migrate.py`.

A document stays in the cabinet that first stored it: doc_id is the content
hash and the primary key, so an upsert naming another cabinet is refused with
DocumentInOtherCabinet. A composite (cabinet_id, doc_id) key is a later change.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, TypedDict

from .db_migrate import migrate
from .db_registry import DEFAULT_ACTOR, ensure_cabinet, record_audit, routing_status
from .db_registry import utc_now as _now
from .db_review import KIND_CLASS as KIND_CLASS
from .db_review import KIND_FIELD as KIND_FIELD
from .db_review import KIND_TAG as KIND_TAG
from .db_review import ReviewEvent as ReviewEvent
from .db_review import document_cabinet
from .db_review import event_kind as event_kind
from .db_review import list_review_events as list_review_events
from .db_review import record_review_events as record_review_events
from .errors import DocumentInOtherCabinet
from .io import atomic_open
from .logging_setup import get_logger
from .schema import SCHEMA_VERSION, DocumentRecord, TableRecord, read_jsonl

log = get_logger(__name__)

DEFAULT_LIST_LIMIT = 100
_HASH_CHUNK = 1 << 20
# How long a connection waits for another writer's lock before it fails.
BUSY_TIMEOUT_S = 30.0
WAL_RETRY_S = 0.05


SCHEMA_DDL: tuple[str, ...] = (
    """CREATE TABLE IF NOT EXISTS runs (
        run_id TEXT PRIMARY KEY,
        started_at TEXT NOT NULL,
        config_hash TEXT NOT NULL,
        schema_version TEXT NOT NULL,
        cabinet_id TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS models (
        name TEXT PRIMARY KEY,
        version TEXT,
        sha256 TEXT NOT NULL,
        recorded_at TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS documents (
        doc_id TEXT PRIMARY KEY,
        cabinet_id TEXT NOT NULL,
        source_path TEXT NOT NULL,
        filename TEXT NOT NULL,
        doc_type TEXT NOT NULL,
        confidence REAL NOT NULL,
        classify_method TEXT NOT NULL,
        schema_version TEXT NOT NULL,
        page_count INTEGER NOT NULL,
        run_id TEXT REFERENCES runs(run_id),
        bbox_path TEXT,
        bbox_sha256 TEXT,
        record_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS documents_cabinet ON documents (cabinet_id, doc_id)",
    """CREATE TABLE IF NOT EXISTS fields (
        doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
        ord INTEGER NOT NULL,
        name TEXT NOT NULL,
        value TEXT NOT NULL,
        normalized_value TEXT,
        page INTEGER NOT NULL,
        char_start INTEGER NOT NULL,
        char_end INTEGER NOT NULL,
        method TEXT NOT NULL,
        validated INTEGER NOT NULL,
        normalizer_version TEXT NOT NULL,
        confidence REAL NOT NULL,
        currency TEXT,
        PRIMARY KEY (doc_id, ord)
    )""",
    """CREATE TABLE IF NOT EXISTS tags (
        doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
        ord INTEGER NOT NULL,
        tag TEXT NOT NULL,
        confidence REAL NOT NULL,
        method TEXT NOT NULL,
        PRIMARY KEY (doc_id, ord)
    )""",
    """CREATE TABLE IF NOT EXISTS tables (
        doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
        table_id TEXT NOT NULL,
        page INTEGER NOT NULL,
        n_rows INTEGER NOT NULL,
        n_cols INTEGER NOT NULL,
        method TEXT NOT NULL,
        confidence REAL NOT NULL,
        table_json TEXT NOT NULL,
        sidecar_path TEXT NOT NULL,
        PRIMARY KEY (doc_id, table_id)
    )""",
    """CREATE TABLE IF NOT EXISTS provenance (
        doc_id TEXT PRIMARY KEY REFERENCES documents(doc_id) ON DELETE CASCADE,
        model TEXT,
        finish_reason TEXT,
        fallback INTEGER NOT NULL,
        abstention_reason TEXT,
        prov_json TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS review_events (
        event_id TEXT PRIMARY KEY,
        doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
        field TEXT NOT NULL,
        old_value TEXT,
        new_value TEXT,
        source TEXT NOT NULL,
        created_at TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS jobs (
        job_id TEXT PRIMARY KEY,
        cabinet_id TEXT NOT NULL,
        state TEXT NOT NULL,
        doc_id TEXT,
        error TEXT,
        trace_id TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        result_json TEXT
    )""",
    "CREATE INDEX IF NOT EXISTS jobs_state ON jobs (state, updated_at)",
)

_UPSERT_DOCUMENT = """
INSERT INTO documents (doc_id, cabinet_id, source_path, filename, doc_type, confidence, classify_method,
    schema_version, page_count, run_id, bbox_path, bbox_sha256, record_json, created_at, updated_at,
    routing_status, case_id)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT (doc_id) DO UPDATE SET
    source_path = excluded.source_path, filename = excluded.filename,
    doc_type = excluded.doc_type, confidence = excluded.confidence, classify_method = excluded.classify_method,
    schema_version = excluded.schema_version, page_count = excluded.page_count,
    run_id = COALESCE(excluded.run_id, documents.run_id),
    bbox_path = COALESCE(excluded.bbox_path, documents.bbox_path),
    bbox_sha256 = COALESCE(excluded.bbox_sha256, documents.bbox_sha256),
    record_json = excluded.record_json, updated_at = excluded.updated_at,
    routing_status = excluded.routing_status, case_id = COALESCE(excluded.case_id, documents.case_id)
WHERE documents.cabinet_id = excluded.cabinet_id
"""

_DOC_COLUMNS = (
    "doc_id, cabinet_id, source_path, filename, doc_type, confidence, classify_method, "
    "schema_version, page_count, run_id, bbox_path, bbox_sha256, created_at, updated_at, routing_status, case_id"
)


class ImportStats(TypedDict):
    """Counts one import returns."""

    documents: int
    tables: int
    provenance: int
    skipped: int
    refused: int
    duplicates: int


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_HASH_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _enable_wal(conn: sqlite3.Connection) -> None:
    """Turn on WAL, retrying until BUSY_TIMEOUT_S passes.

    Leaving the rollback journal needs every other connection gone, and SQLite
    answers "locked" at once instead of calling the busy handler, so two
    processes opening an old store together would otherwise fail.
    """
    deadline = time.monotonic() + BUSY_TIMEOUT_S
    while True:
        try:
            conn.execute("PRAGMA journal_mode = WAL")
            return
        except sqlite3.OperationalError:
            if time.monotonic() > deadline:
                raise
            time.sleep(WAL_RETRY_S)


def connect(path: str | Path, threads: bool = False) -> sqlite3.Connection:
    """Open the store, turn on WAL and foreign keys, and create the schema if missing.

    Args:
        path: Database file; its parent directory is created.
        threads: Allow use from other threads; the caller then serializes access with a lock.

    Returns:
        An open connection with rows as sqlite3.Row. The caller closes it.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p, check_same_thread=not threads, timeout=BUSY_TIMEOUT_S)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        _enable_wal(conn)
        with conn:
            for stmt in SCHEMA_DDL:
                conn.execute(stmt)
        migrate(conn)
    except BaseException:
        conn.close()  # a failed open must not keep the file open; Windows would hold it locked
        raise
    return conn


def start_run(conn: sqlite3.Connection, config: dict[str, Any], cabinet_id: str, *, trace_id: str | None = None) -> str:
    """Record a pipeline run and return its id. The cabinet gets a row if it has none.

    Args:
        conn: Open store.
        config: Every loaded config file; hashed, not stored.
        cabinet_id: Cabinet the run writes into.
        trace_id: Request trace id when an API request started the run.

    Returns:
        A new run_id.
    """
    run_id = uuid.uuid4().hex
    digest = hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    with conn:
        ensure_cabinet(conn, cabinet_id)
        conn.execute(
            "INSERT INTO runs (run_id, started_at, config_hash, schema_version, cabinet_id, trace_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, _now(), digest, SCHEMA_VERSION, cabinet_id, trace_id),
        )
    return run_id


def record_models(conn: sqlite3.Connection, manifest: dict[str, str]) -> None:
    """Upsert the model files a run loaded, from models/MANIFEST.sha256.

    The manifest carries no version, so version stays null until one exists.

    Args:
        conn: Open store.
        manifest: Relative model path to sha256, as the manifest reader returns.
    """
    now = _now()
    with conn:
        conn.executemany(
            "INSERT INTO models (name, version, sha256, recorded_at) VALUES (?, NULL, ?, ?) "
            "ON CONFLICT (name) DO UPDATE SET sha256 = excluded.sha256, recorded_at = excluded.recorded_at",
            [(name, digest, now) for name, digest in sorted(manifest.items())],
        )


def upsert_document(
    conn: sqlite3.Connection,
    record: DocumentRecord,
    run_id: str | None = None,
    bbox_path: Path | None = None,
    *,
    case_id: str | None = None,
) -> None:
    """Write one record and replace its field and tag rows, in one transaction.

    Re-importing the same record leaves the store unchanged except updated_at.
    Review events, tables and provenance are not touched. routing_status is
    derived from the record; the cabinet gets a row if it has none. A document
    is never moved between cabinets: when the doc_id is stored under another
    cabinet nothing is written except an "upsert_refused" audit row (ids only).

    Args:
        conn: Open store.
        record: The record, exactly as written to the JSONL.
        run_id: Run that produced it; None keeps whatever run is stored.
        bbox_path: Word-box sidecar; stored with its hash only if the file exists.
        case_id: Case the document belongs to (a pack manifest's case_id); None keeps the stored one.

    Raises:
        DocumentInOtherCabinet: The doc_id is stored under another cabinet.
    """
    bbox = bbox_path if bbox_path is not None and bbox_path.is_file() else None
    doc = _document_row(record, run_id, bbox, case_id)
    fields = [
        (record.doc_id, i, m.field, m.value, m.normalized_value, m.page, m.char_start, m.char_end, m.method.value)
        + (int(m.validated), m.normalizer_version, m.confidence, m.currency)
        for i, m in enumerate(record.metadata)
    ]
    tags = [(record.doc_id, i, t.tag, t.confidence, t.method.value) for i, t in enumerate(record.tags)]
    with conn:
        # The upsert's WHERE leaves a row of another cabinet alone and reports no change.
        stored = conn.execute(_UPSERT_DOCUMENT, doc).rowcount > 0
        if stored:
            ensure_cabinet(conn, record.cabinet_id)
            conn.execute("DELETE FROM fields WHERE doc_id = ?", (record.doc_id,))
            conn.executemany("INSERT INTO fields VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", fields)
            conn.execute("DELETE FROM tags WHERE doc_id = ?", (record.doc_id,))
            conn.executemany("INSERT INTO tags VALUES (?, ?, ?, ?, ?)", tags)
    if not stored:
        _refuse(conn, record.doc_id, record.cabinet_id)


def _refuse(conn: sqlite3.Connection, doc_id: str, cabinet_id: str) -> None:
    """Audit and raise a refused move of doc_id into cabinet_id."""
    owner = document_cabinet(conn, doc_id) or ""
    detail = {"stored_cabinet_id": owner}
    record_audit(
        conn, actor=DEFAULT_ACTOR, action="upsert_refused", cabinet_id=cabinet_id, doc_id=doc_id, detail=detail
    )
    log.warning("upsert refused, document is in another cabinet", extra={"doc_id": doc_id, "cabinet_id": cabinet_id})
    raise DocumentInOtherCabinet(doc_id, owner, cabinet_id)


def _document_row(
    record: DocumentRecord, run_id: str | None, bbox: Path | None, case_id: str | None
) -> tuple[Any, ...]:
    """Build the documents-table row for one record, with created_at and updated_at set to now."""
    now = _now()
    cls = record.classification
    return (
        record.doc_id,
        record.cabinet_id,
        record.source_path,
        record.filename,
        cls.label,
        cls.confidence,
        cls.method.value,
        record.schema_version,
        record.diagnostics.page_count,
        run_id,
        bbox.as_posix() if bbox else None,
        _file_sha256(bbox) if bbox else None,
        json.dumps(record.model_dump(mode="json"), ensure_ascii=False),
        now,
        now,
        routing_status(record),
        case_id or None,
    )


def _has_document(conn: sqlite3.Connection, doc_id: str) -> bool:
    return conn.execute("SELECT 1 FROM documents WHERE doc_id = ?", (doc_id,)).fetchone() is not None


def _upsert_tables(conn: sqlite3.Connection, rec: TableRecord, sidecar: Path) -> int:
    rows = [
        (rec.doc_id, t.table_id, t.page, t.n_rows, t.n_cols, t.method.value, t.confidence)
        + (json.dumps(t.model_dump(mode="json"), ensure_ascii=False), sidecar.as_posix())
        for t in rec.tables
    ]
    with conn:
        conn.execute("DELETE FROM tables WHERE doc_id = ?", (rec.doc_id,))
        conn.executemany("INSERT INTO tables VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    return len(rows)


def upsert_provenance(conn: sqlite3.Connection, prov: dict[str, Any]) -> None:
    """Write one cascade provenance line; its document must already be stored.

    Args:
        conn: Open store.
        prov: The provenance line, as written to the sidecar.
    """
    fallback = bool(prov.get("fallback"))
    reason = "; ".join(prov.get("errors") or []) if fallback else None
    row = (prov["doc_id"], prov.get("model"), prov.get("finish_reason"), int(fallback), reason)
    with conn:
        conn.execute(
            "INSERT INTO provenance VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT (doc_id) DO UPDATE SET "
            "model = excluded.model, finish_reason = excluded.finish_reason, fallback = excluded.fallback, "
            "abstention_reason = excluded.abstention_reason, prov_json = excluded.prov_json",
            (*row, json.dumps(prov, ensure_ascii=False, sort_keys=True)),
        )


def import_jsonl(
    conn: sqlite3.Connection,
    results_path: str | Path,
    provenance_path: str | Path | None = None,
    tables_path: str | Path | None = None,
    *,
    actor: str = DEFAULT_ACTOR,
) -> ImportStats:
    """Load a results file, and optionally its sidecars, into the store. Idempotent.

    Word-box sidecars are looked up in `<results>.bboxes/`. Sidecar lines for a
    doc_id the store does not hold are skipped and counted. A record whose
    doc_id is stored under another cabinet is counted as refused. documents
    counts distinct doc_ids; a doc_id seen again in the file counts as a
    duplicate, and its later line wins.

    Args:
        conn: Open store.
        results_path: Results JSONL.
        provenance_path: Cascade provenance JSONL, if any.
        tables_path: Tables sidecar JSONL, if any.
        actor: Who ran the import, for the audit log.

    Returns:
        Distinct documents, tables and provenance rows written; sidecar lines skipped; refused; duplicates.

    Raises:
        ValueError: A line is not valid JSON or not a valid record.
    """
    results = Path(results_path)
    bbox_dir = results.with_suffix(results.suffix + ".bboxes")
    stats: ImportStats = {"documents": 0, "tables": 0, "provenance": 0, "skipped": 0, "refused": 0, "duplicates": 0}
    seen: dict[str, str] = {}
    for row in read_jsonl(results):
        record = DocumentRecord.model_validate(row)
        try:
            upsert_document(conn, record, bbox_path=bbox_dir / f"{record.doc_id}.json.gz")
        except DocumentInOtherCabinet:
            stats["refused"] += 1
            continue
        stats["duplicates"] += record.doc_id in seen
        seen[record.doc_id] = record.cabinet_id
    stats["documents"] = len(seen)
    cabinets = Counter(seen.values())
    _import_sidecars(conn, stats, tables_path, provenance_path)
    for cabinet_id, n in sorted(cabinets.items()):
        detail = {"documents": n, "results": results.name}
        record_audit(conn, actor=actor, action="db.import", cabinet_id=cabinet_id, detail=detail)
    log.info("db import done", extra={"results": str(results), **stats})
    return stats


def _import_sidecars(
    conn: sqlite3.Connection,
    stats: ImportStats,
    tables_path: str | Path | None,
    provenance_path: str | Path | None,
) -> None:
    """Load the tables and provenance sidecars, counting writes and skips into stats."""
    if tables_path:
        for row in read_jsonl(tables_path):
            rec = TableRecord.model_validate(row)
            if not _has_document(conn, rec.doc_id):
                stats["skipped"] += 1
                continue
            stats["tables"] += _upsert_tables(conn, rec, Path(tables_path))
    if provenance_path:
        for prov in read_jsonl(provenance_path):
            if not _has_document(conn, str(prov.get("doc_id"))):
                stats["skipped"] += 1
                continue
            upsert_provenance(conn, prov)
            stats["provenance"] += 1


def export_jsonl(conn: sqlite3.Connection, out_path: str | Path, cabinet_id: str | None = None) -> int:
    """Write the stored records back out as results JSONL, ordered by doc_id.

    Args:
        conn: Open store.
        out_path: JSONL to write, all or nothing.
        cabinet_id: Only this cabinet; None for every cabinet.

    Returns:
        Number of records written.
    """
    sql = "SELECT record_json FROM documents"
    params: tuple[str, ...] = ()
    if cabinet_id is not None:
        sql += " WHERE cabinet_id = ?"
        params = (cabinet_id,)
    rows = conn.execute(sql + " ORDER BY doc_id", params).fetchall()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with atomic_open(out, newline="\n") as fh:
        for row in rows:
            fh.write(row["record_json"] + "\n")
    log.info("db export done", extra={"output": str(out), "documents": len(rows), "cabinet_id": cabinet_id})
    return len(rows)


def list_documents(
    conn: sqlite3.Connection,
    cabinet_id: str | None = None,
    limit: int = DEFAULT_LIST_LIMIT,
    offset: int = 0,
    *,
    routing_status: str | None = None,
    case_id: str | None = None,
) -> list[dict[str, Any]]:
    """Document summaries without the record body, ordered by doc_id.

    Args:
        conn: Open store.
        cabinet_id: Only this cabinet; None for every cabinet.
        limit: Maximum rows.
        offset: Rows to skip, for paging.
        routing_status: Only this routing status (auto, review, failed); None for all.
        case_id: Only this case; None for all.

    Returns:
        One dict per document, keyed by column name.
    """
    # Columns are a module constant and every value is bound, so this is not injectable.
    sql = "SELECT " + _DOC_COLUMNS + " FROM documents WHERE (? IS NULL OR cabinet_id = ?)"  # nosec B608
    sql += " AND (? IS NULL OR routing_status = ?) AND (? IS NULL OR case_id = ?) ORDER BY doc_id LIMIT ? OFFSET ?"
    params = [cabinet_id, cabinet_id, routing_status, routing_status, case_id, case_id, limit, offset]
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def get_document(conn: sqlite3.Connection, doc_id: str) -> DocumentRecord | None:
    """One stored record, exactly as written.

    Args:
        conn: Open store.
        doc_id: The document.

    Returns:
        The record, or None when the store does not hold it.
    """
    row = conn.execute("SELECT record_json FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
    return None if row is None else DocumentRecord.model_validate_json(row["record_json"])
