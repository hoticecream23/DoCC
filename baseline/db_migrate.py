"""In-place migration of an existing ``docc.sqlite`` to the current shape.

``db.connect`` checks ``is_current`` without a lock on every open and runs
``migrate`` only when a version is pending. Migrations are ordered steps in
``MIGRATIONS``. Each applied version is a row of ``schema_migrations``. A step
only creates tables with ``IF NOT EXISTS``, adds a column only when it is
missing, and backfills only rows that are still NULL, so a step that stopped
half way is safe to run again. The DDL stays portable (TEXT, INTEGER, REAL,
JSON as TEXT).

``migrate`` takes the write lock (``BEGIN IMMEDIATE``) before it reads the
applied versions and applies every pending version in that one transaction,
so two processes that open an old store at once do not both alter it.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterable
from typing import NamedTuple

from .db_registry import DEFAULT_ACTOR, ROUTE_REVIEW, insert_audit, routing_status, utc_now
from .logging_setup import get_logger
from .schema import SCHEMA_VERSION, DocumentRecord

log = get_logger(__name__)

BACKFILL_BATCH = 500
UNREADABLE_ACTION = "migration_unreadable_records"


class AddColumn(NamedTuple):
    """One guarded ALTER: run `ddl` only when `table` lacks `column`."""

    table: str
    column: str
    ddl: str


Step = str | AddColumn | Callable[[sqlite3.Connection], None]

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


def _route(body: str) -> str | None:
    """routing_status of a stored record; None when it does not parse or is from another schema version."""
    try:
        record = DocumentRecord.model_validate_json(body)
    except ValueError:
        return None
    return routing_status(record) if record.schema_version == SCHEMA_VERSION else None


def _reroute(conn: sqlite3.Connection, where: str) -> None:
    """Route the rows `where` selects from their record_json, in doc_id batches.

    Uses the same rule as upsert_document. A record that is unreadable (see
    _route) routes to review, so a person looks at it and the step always
    finishes; their count goes into one audit row, never their content.
    """
    after, unreadable = "", 0
    # `where` is one of this module's constants, never input, so this is not injectable.
    sql = "SELECT doc_id, record_json FROM documents WHERE " + where  # nosec B608
    sql += " AND doc_id > ? ORDER BY doc_id LIMIT ?"
    while rows := conn.execute(sql, (after, BACKFILL_BATCH)).fetchall():
        updates = []
        for doc_id, body in rows:
            status = _route(body)
            unreadable += status is None
            updates.append((status or ROUTE_REVIEW, doc_id))
        conn.executemany("UPDATE documents SET routing_status = ? WHERE doc_id = ?", updates)
        after = rows[-1][0]
    if unreadable:
        insert_audit(conn, actor=DEFAULT_ACTOR, action=UNREADABLE_ACTION, cabinet_id=None, detail={"count": unreadable})
        log.warning("unreadable stored records routed to review", extra={"count": unreadable})


def _backfill_routing(conn: sqlite3.Connection) -> None:
    """Set routing_status wherever it is NULL."""
    _reroute(conn, "routing_status IS NULL")


def _reroute_failed(conn: sqlite3.Connection) -> None:
    """Re-check rows routed to failed: before v3 an unreadable record was routed there silently."""
    _reroute(conn, "routing_status = 'failed'")


# The importer's convention: class corrections come from the Documents sheet as
# field "class", tag corrections from the Tags sheet as "tag:<name>", the rest
# from the Metadata sheet. The sheet decides when the source names one.
_KIND_FROM_CONVENTION = """UPDATE review_events SET kind = CASE
    WHEN source LIKE '%#Documents' THEN 'class'
    WHEN source LIKE '%#Tags' THEN 'tag'
    WHEN source LIKE '%#Metadata' THEN 'field'
    WHEN field = 'class' THEN 'class'
    WHEN field LIKE 'tag:%' THEN 'tag'
    ELSE 'field' END
WHERE kind IS NULL"""

_V2: tuple[Step, ...] = (
    AddColumn("review_events", "seq", "ALTER TABLE review_events ADD COLUMN seq INTEGER"),
    # Rows written so far keep the order they were inserted in.
    "UPDATE review_events SET seq = rowid WHERE seq IS NULL",
    "CREATE INDEX IF NOT EXISTS review_events_latest ON review_events (doc_id, field, seq)",
    AddColumn("review_events", "kind", "ALTER TABLE review_events ADD COLUMN kind TEXT"),
    _KIND_FROM_CONVENTION,
    AddColumn("jobs", "boot_id", "ALTER TABLE jobs ADD COLUMN boot_id TEXT"),
    _backfill_routing,
)

_V3: tuple[Step, ...] = (
    # One API process per store: the live one's boot id and heartbeat (api/jobs.py).
    """CREATE TABLE IF NOT EXISTS api_instances (
        boot_id TEXT PRIMARY KEY,
        started_at TEXT NOT NULL,
        heartbeat_at REAL NOT NULL
    )""",
    _reroute_failed,
)

# Ordered (version, steps). Append a new version; never edit an applied one.
MIGRATIONS: tuple[tuple[int, tuple[Step, ...]], ...] = ((1, _V1), (2, _V2), (3, _V3))
LATEST_VERSION = MIGRATIONS[-1][0]


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    rows = conn.execute("SELECT name FROM pragma_table_info(?)", (table,)).fetchall()
    return column in {r[0] for r in rows}


def _apply(conn: sqlite3.Connection, steps: Iterable[Step]) -> None:
    for step in steps:
        if isinstance(step, AddColumn):
            if not _has_column(conn, step.table, step.column):
                conn.execute(step.ddl)
        elif isinstance(step, str):
            conn.execute(step)
        else:
            step(conn)


def is_current(conn: sqlite3.Connection) -> bool:
    """Whether every migration is recorded, read without taking the write lock.

    Args:
        conn: Open store with no transaction open.

    Returns:
        True when LATEST_VERSION is applied; False for a new or older store.
    """
    known = conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'schema_migrations'").fetchone()
    if known is None:
        return False
    return conn.execute("SELECT 1 FROM schema_migrations WHERE version = ?", (LATEST_VERSION,)).fetchone() is not None


def migrate(conn: sqlite3.Connection) -> list[int]:
    """Apply every migration the store has not recorded yet, in order, under the write lock.

    The caller must have no transaction open. On any failure the whole
    migration rolls back and the error is raised.

    Args:
        conn: Open store whose base tables already exist.

    Returns:
        The versions applied by this call; empty when the store is current.
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
        )
        done = {r[0] for r in conn.execute("SELECT version FROM schema_migrations").fetchall()}
        applied = []
        for version, steps in MIGRATIONS:
            if version in done:
                continue
            _apply(conn, steps)
            conn.execute("INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)", (version, utc_now()))
            applied.append(version)
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    if applied:
        log.info("store migrated", extra={"versions": applied})
    return applied
