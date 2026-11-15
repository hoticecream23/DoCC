"""Store additions: cabinets, document types, field definitions, routing and an audit log.

Over 400 lines: the migration steps and the tables they create are read together.

``baseline/db.py`` holds the JSONL mirror. This module holds what sits beside
it and the in-place migration that adds it to an existing ``docc.sqlite``.
``db.connect`` runs ``migrate`` on every open.

Migrations are ordered steps in ``MIGRATIONS``. Each applied version is a row
of ``schema_migrations``. A step only creates tables with ``IF NOT EXISTS``
and adds a column only when it is missing, so a step that stopped half way is
safe to run again. The DDL stays portable (TEXT, INTEGER, REAL, JSON as TEXT).

Corrections stay append-only in ``review_events``. ``effective_fields`` reads
the latest correction over the stored value; nothing is overwritten.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any, NamedTuple

from .clientpack import ClientPack
from .logging_setup import get_logger
from .schema import ClassifyMethod, DocumentRecord

log = get_logger(__name__)

DEFAULT_ACTOR = "system"
DEFAULT_AUDIT_LIMIT = 100
ROUTE_AUTO = "auto"
ROUTE_REVIEW = "review"
ROUTE_FAILED = "failed"
_HIGH_PRIORITY = "HIGH"


class AddColumn(NamedTuple):
    """One guarded ALTER: run `ddl` only when `table` lacks `column`."""

    table: str
    column: str
    ddl: str


Step = str | AddColumn

_V1: tuple[Step, ...] = (
    """CREATE TABLE IF NOT EXISTS tenants (
        tenant_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        domain_type TEXT,
        is_active INTEGER NOT NULL DEFAULT 1
    )""",
    """CREATE TABLE IF NOT EXISTS cabinets (
        cabinet_id TEXT PRIMARY KEY,
        tenant_id TEXT REFERENCES tenants(tenant_id),
        name TEXT NOT NULL,
        root_path TEXT,
        created_at TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS doc_types (
        cabinet_id TEXT NOT NULL REFERENCES cabinets(cabinet_id) ON DELETE CASCADE,
        name TEXT NOT NULL,
        aliases TEXT NOT NULL,
        has_samples INTEGER NOT NULL,
        PRIMARY KEY (cabinet_id, name)
    )""",
    """CREATE TABLE IF NOT EXISTS field_defs (
        cabinet_id TEXT NOT NULL,
        doc_type TEXT NOT NULL,
        name TEXT NOT NULL,
        ord INTEGER NOT NULL,
        dtype TEXT NOT NULL,
        date_format TEXT,
        role TEXT NOT NULL,
        origin TEXT NOT NULL,
        PRIMARY KEY (cabinet_id, doc_type, name),
        FOREIGN KEY (cabinet_id, doc_type) REFERENCES doc_types(cabinet_id, name) ON DELETE CASCADE
    )""",
    """CREATE TABLE IF NOT EXISTS audit_log (
        audit_id INTEGER PRIMARY KEY,
        at TEXT NOT NULL,
        actor TEXT NOT NULL,
        action TEXT NOT NULL,
        cabinet_id TEXT,
        doc_id TEXT,
        detail TEXT
    )""",
    "CREATE INDEX IF NOT EXISTS audit_log_cabinet ON audit_log (cabinet_id, audit_id)",
    AddColumn("review_events", "verdict", "ALTER TABLE review_events ADD COLUMN verdict TEXT"),
    AddColumn("review_events", "is_gold", "ALTER TABLE review_events ADD COLUMN is_gold INTEGER NOT NULL DEFAULT 0"),
    AddColumn("runs", "trace_id", "ALTER TABLE runs ADD COLUMN trace_id TEXT"),
    AddColumn("documents", "routing_status", "ALTER TABLE documents ADD COLUMN routing_status TEXT"),
    AddColumn("documents", "case_id", "ALTER TABLE documents ADD COLUMN case_id TEXT"),
    "CREATE INDEX IF NOT EXISTS documents_routing ON documents (routing_status, doc_id)",
    # Cabinets that already hold documents or runs get their row, named by id.
    """INSERT INTO cabinets (cabinet_id, tenant_id, name, root_path, created_at)
       SELECT cabinet_id, NULL, cabinet_id, NULL, MIN(created_at) FROM documents GROUP BY cabinet_id
       ON CONFLICT (cabinet_id) DO NOTHING""",
    """INSERT INTO cabinets (cabinet_id, tenant_id, name, root_path, created_at)
       SELECT cabinet_id, NULL, cabinet_id, NULL, MIN(started_at) FROM runs GROUP BY cabinet_id
       ON CONFLICT (cabinet_id) DO NOTHING""",
)

# Ordered (version, steps). Append a new version; never edit an applied one.
MIGRATIONS: tuple[tuple[int, tuple[Step, ...]], ...] = ((1, _V1),)


def utc_now() -> str:
    """Current UTC time, ISO 8601 to the second.

    Returns:
        The timestamp the store writes.
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    rows = conn.execute("SELECT name FROM pragma_table_info(?)", (table,)).fetchall()
    return column in {r[0] for r in rows}


def _apply(conn: sqlite3.Connection, steps: Iterable[Step]) -> None:
    for step in steps:
        if isinstance(step, AddColumn):
            if not _has_column(conn, step.table, step.column):
                conn.execute(step.ddl)
        else:
            conn.execute(step)


def migrate(conn: sqlite3.Connection) -> list[int]:
    """Apply every migration the store has not recorded yet, in order.

    Args:
        conn: Open store whose base tables already exist.

    Returns:
        The versions applied by this call; empty when the store is current.
    """
    conn.execute("CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)")
    done = {r[0] for r in conn.execute("SELECT version FROM schema_migrations").fetchall()}
    applied = []
    for version, steps in MIGRATIONS:
        if version in done:
            continue
        with conn:
            _apply(conn, steps)
            conn.execute(
                "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?) ON CONFLICT (version) DO NOTHING",
                (version, utc_now()),
            )
        applied.append(version)
    if applied:
        log.info("store migrated", extra={"versions": applied})
    return applied


def routing_status(record: DocumentRecord) -> str:
    """Where a record goes next: failed, review or auto.

    failed: diagnostics carry errors. review: the class abstained (method
    fallback, or confidence below the review priority's HIGH edge) or any field
    has HIGH review priority, the rule the workbook export uses. Else auto.

    Args:
        record: The record as written to the JSONL.

    Returns:
        One of ROUTE_FAILED, ROUTE_REVIEW, ROUTE_AUTO.
    """
    # Imported here: the workbook package imports db, which imports this module.
    from .workbook.exporter import PRIORITY_HIGH_BELOW, review_priority

    if record.diagnostics.errors:
        return ROUTE_FAILED
    cls = record.classification
    if cls.method == ClassifyMethod.FALLBACK or cls.confidence < PRIORITY_HIGH_BELOW:
        return ROUTE_REVIEW
    if any(review_priority(m.model_dump(mode="json")) == _HIGH_PRIORITY for m in record.metadata):
        return ROUTE_REVIEW
    return ROUTE_AUTO


def ensure_cabinet(conn: sqlite3.Connection, cabinet_id: str) -> None:
    """Add a cabinets row named by its id when none exists; leave an existing one alone.

    Args:
        conn: Open store; the caller owns the transaction.
        cabinet_id: The cabinet.
    """
    conn.execute(
        "INSERT INTO cabinets (cabinet_id, tenant_id, name, root_path, created_at) VALUES (?, NULL, ?, NULL, ?) "
        "ON CONFLICT (cabinet_id) DO NOTHING",
        (cabinet_id, cabinet_id, utc_now()),
    )


def upsert_tenant(
    conn: sqlite3.Connection, tenant_id: str, name: str, domain_type: str | None = None, is_active: bool = True
) -> None:
    """Insert or update one tenant.

    Args:
        conn: Open store.
        tenant_id: The tenant.
        name: Display name.
        domain_type: Kind of business, if known.
        is_active: Whether the tenant is in use.
    """
    with conn:
        conn.execute(
            "INSERT INTO tenants (tenant_id, name, domain_type, is_active) VALUES (?, ?, ?, ?) "
            "ON CONFLICT (tenant_id) DO UPDATE SET name = excluded.name, domain_type = excluded.domain_type, "
            "is_active = excluded.is_active",
            (tenant_id, name, domain_type, int(is_active)),
        )


def upsert_cabinet(
    conn: sqlite3.Connection,
    cabinet_id: str,
    name: str | None = None,
    tenant_id: str | None = None,
    root_path: str | None = None,
) -> None:
    """Insert or update one cabinet; created_at is kept on update.

    Args:
        conn: Open store.
        cabinet_id: The cabinet.
        name: Display name; None for the id.
        tenant_id: Owning tenant, if any; it must exist.
        root_path: Folder the cabinet is read from, if any.
    """
    with conn:
        conn.execute(
            "INSERT INTO cabinets (cabinet_id, tenant_id, name, root_path, created_at) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT (cabinet_id) DO UPDATE SET tenant_id = excluded.tenant_id, name = excluded.name, "
            "root_path = excluded.root_path",
            (cabinet_id, tenant_id, name or cabinet_id, root_path, utc_now()),
        )


def list_cabinets(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Every cabinet with how many documents it holds, ordered by id.

    Args:
        conn: Open store.

    Returns:
        One dict per cabinet: its columns plus `documents`.
    """
    sql = (
        "SELECT c.cabinet_id, c.tenant_id, c.name, c.root_path, c.created_at, COUNT(d.doc_id) AS documents "
        "FROM cabinets c LEFT JOIN documents d ON d.cabinet_id = c.cabinet_id "
        "GROUP BY c.cabinet_id, c.tenant_id, c.name, c.root_path, c.created_at ORDER BY c.cabinet_id"
    )
    return [dict(r) for r in conn.execute(sql).fetchall()]


def record_audit(
    conn: sqlite3.Connection,
    *,
    actor: str,
    action: str,
    cabinet_id: str | None,
    doc_id: str | None = None,
    detail: dict[str, Any] | None = None,
) -> None:
    """Append one audit row. Detail holds ids, counts and names, never document text or values.

    The id is the next integer, taken in the same transaction, so no
    autoincrement is needed.

    Args:
        conn: Open store.
        actor: Who acted: "system", "api" or a user name.
        action: What was done, e.g. "review_events.record".
        cabinet_id: Cabinet acted on, if one.
        doc_id: Document acted on, if one.
        detail: Extra ids and counts, stored as JSON.
    """
    body = None if detail is None else json.dumps(detail, sort_keys=True, ensure_ascii=False)
    with conn:
        conn.execute(
            "INSERT INTO audit_log (audit_id, at, actor, action, cabinet_id, doc_id, detail) "
            "VALUES ((SELECT COALESCE(MAX(audit_id), 0) + 1 FROM audit_log), ?, ?, ?, ?, ?, ?)",
            (utc_now(), actor, action, cabinet_id, doc_id, body),
        )


def list_audit(
    conn: sqlite3.Connection, cabinet_id: str | None = None, doc_id: str | None = None, limit: int = DEFAULT_AUDIT_LIMIT
) -> list[dict[str, Any]]:
    """Audit rows, newest first, with detail parsed.

    Args:
        conn: Open store.
        cabinet_id: Only this cabinet; None for all.
        doc_id: Only this document; None for all.
        limit: Maximum rows.

    Returns:
        One dict per row.
    """
    sql = (
        "SELECT audit_id, at, actor, action, cabinet_id, doc_id, detail FROM audit_log "
        "WHERE (? IS NULL OR cabinet_id = ?) AND (? IS NULL OR doc_id = ?) ORDER BY audit_id DESC LIMIT ?"
    )
    rows = conn.execute(sql, (cabinet_id, cabinet_id, doc_id, doc_id, limit)).fetchall()
    return [{**dict(r), "detail": None if r["detail"] is None else json.loads(r["detail"])} for r in rows]


def register_pack(conn: sqlite3.Connection, pack: ClientPack, *, actor: str = DEFAULT_ACTOR) -> None:
    """Load a pack cabinet's classes and field definitions, replacing that cabinet's rows.

    Idempotent: one transaction deletes and re-inserts. The cabinet id is the
    pack slug, as in a pack-mode run.

    Args:
        conn: Open store.
        pack: One cabinet of a client pack.
        actor: Who registered it, for the audit log.
    """
    cab = pack.cabinet.slug
    types = [(cab, c.name, json.dumps(c.aliases, ensure_ascii=False), int(c.has_samples)) for c in pack.classes]
    known = {c.name for c in pack.classes}
    defs = [
        (cab, t, f.name, i, f.dtype, f.date_format, f.role, f.origin)
        for t, specs in pack.fields.items()
        if t in known
        for i, f in enumerate(specs)
    ]
    upsert_cabinet(conn, cab, name=pack.cabinet.name)
    with conn:
        conn.execute("DELETE FROM field_defs WHERE cabinet_id = ?", (cab,))
        conn.execute("DELETE FROM doc_types WHERE cabinet_id = ?", (cab,))
        conn.executemany("INSERT INTO doc_types (cabinet_id, name, aliases, has_samples) VALUES (?, ?, ?, ?)", types)
        conn.executemany(
            "INSERT INTO field_defs (cabinet_id, doc_type, name, ord, dtype, date_format, role, origin) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            defs,
        )
    skipped = sum(len(v) for t, v in pack.fields.items() if t not in known)
    detail = {"doc_types": len(types), "field_defs": len(defs), "skipped_field_defs": skipped}
    record_audit(conn, actor=actor, action="pack.register", cabinet_id=cab, detail=detail)


def list_doc_types(conn: sqlite3.Connection, cabinet_id: str) -> list[dict[str, Any]]:
    """A cabinet's document types, ordered by name.

    Args:
        conn: Open store.
        cabinet_id: The cabinet.

    Returns:
        One dict per type: name, aliases (a list) and has_samples (a bool).
    """
    rows = conn.execute(
        "SELECT name, aliases, has_samples FROM doc_types WHERE cabinet_id = ? ORDER BY name", (cabinet_id,)
    ).fetchall()
    return [
        {"name": r["name"], "aliases": json.loads(r["aliases"]), "has_samples": bool(r["has_samples"])} for r in rows
    ]


def list_field_defs(conn: sqlite3.Connection, cabinet_id: str, doc_type: str | None = None) -> list[dict[str, Any]]:
    """A cabinet's field definitions, by type then pack order.

    Args:
        conn: Open store.
        cabinet_id: The cabinet.
        doc_type: Only this type; None for all.

    Returns:
        One dict per field definition.
    """
    sql = (
        "SELECT doc_type, name, ord, dtype, date_format, role, origin FROM field_defs "
        "WHERE cabinet_id = ? AND (? IS NULL OR doc_type = ?) ORDER BY doc_type, ord"
    )
    return [dict(r) for r in conn.execute(sql, (cabinet_id, doc_type, doc_type)).fetchall()]


def key_field(conn: sqlite3.Connection, cabinet_id: str, doc_type: str) -> str | None:
    """Name of a type's key field.

    Args:
        conn: Open store.
        cabinet_id: The cabinet.
        doc_type: The document type.

    Returns:
        The first field with role key, or None when the type has none.
    """
    row = conn.execute(
        "SELECT name FROM field_defs WHERE cabinet_id = ? AND doc_type = ? AND role = 'key' ORDER BY ord LIMIT 1",
        (cabinet_id, doc_type),
    ).fetchone()
    return None if row is None else str(row["name"])


# The latest correction of a field: newest created_at, event_id breaking a tie in the same second.
_EFFECTIVE_FIELDS = """
SELECT f.ord, f.name, f.value, f.normalized_value, f.page, f.char_start, f.char_end, f.method,
    f.validated, f.confidence, fd.role, fd.origin,
    (SELECT COUNT(*) FROM review_events e WHERE e.doc_id = f.doc_id AND LOWER(e.field) = LOWER(f.name))
        AS corrections,
    (SELECT e.new_value FROM review_events e WHERE e.doc_id = f.doc_id AND LOWER(e.field) = LOWER(f.name)
        ORDER BY e.created_at DESC, e.event_id DESC LIMIT 1) AS corrected_value,
    (SELECT e.verdict FROM review_events e WHERE e.doc_id = f.doc_id AND LOWER(e.field) = LOWER(f.name)
        ORDER BY e.created_at DESC, e.event_id DESC LIMIT 1) AS verdict
FROM fields f
JOIN documents d ON d.doc_id = f.doc_id
LEFT JOIN field_defs fd ON fd.cabinet_id = d.cabinet_id AND fd.doc_type = d.doc_type AND fd.name = f.name
WHERE f.doc_id = ?
ORDER BY f.ord
"""


def effective_fields(conn: sqlite3.Connection, doc_id: str) -> list[dict[str, Any]]:
    """Each stored field of a document with the latest review correction laid over it.

    A pure read. `effective_value` is the latest correction's new value when a
    correction exists (None when the reviewer removed the field), else the
    normalized value, else the raw value. role and origin come from field_defs
    when the cabinet has them, else None. Fields a reviewer added that the
    record never held are not listed; see list_review_events.

    Args:
        conn: Open store.
        doc_id: The document.

    Returns:
        One dict per stored field, in record order.
    """
    out = []
    for r in conn.execute(_EFFECTIVE_FIELDS, (doc_id,)).fetchall():
        row = dict(r)
        corrected = row["corrections"] > 0
        row["validated"] = bool(row["validated"])
        row["corrected"] = corrected
        row["effective_value"] = row["corrected_value"] if corrected else row["normalized_value"] or row["value"]
        del row["corrections"]
        out.append(row)
    return out
