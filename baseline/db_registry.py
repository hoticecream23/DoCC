"""Store additions: cabinets, document types, field definitions, routing and an audit log.

``baseline/db.py`` holds the JSONL mirror. This module holds what sits beside
it. ``baseline/db_migrate.py`` adds these tables to an existing ``docc.sqlite``.

Corrections stay append-only in ``review_events``. ``effective_fields`` reads
the latest correction (highest ``seq``, which is import order) over the stored
value; nothing is overwritten.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from .clientpack import ClientPack
from .errors import PackRegistrationError
from .logging_setup import get_logger
from .schema import ClassifyMethod, DocumentRecord

log = get_logger(__name__)

DEFAULT_ACTOR = "system"
DEFAULT_AUDIT_LIMIT = 100
ROUTE_AUTO = "auto"
ROUTE_REVIEW = "review"
ROUTE_FAILED = "failed"
_HIGH_PRIORITY = "HIGH"


def utc_now() -> str:
    """Current UTC time, ISO 8601 to the second.

    Returns:
        The timestamp the store writes.
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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
    """Insert or update one cabinet. A value left as None never overwrites a stored one; created_at is kept.

    Args:
        conn: Open store.
        cabinet_id: The cabinet.
        name: Display name; None keeps the stored one, or the id for a new cabinet.
        tenant_id: Owning tenant, if any; it must exist. None keeps the stored one.
        root_path: Folder the cabinet is read from; None keeps the stored one.
    """
    with conn:
        _write_cabinet(conn, cabinet_id, name, tenant_id, root_path)


def _write_cabinet(
    conn: sqlite3.Connection, cabinet_id: str, name: str | None, tenant_id: str | None, root_path: str | None
) -> None:
    """The cabinet upsert, inside the caller's transaction."""
    conn.execute(
        "INSERT INTO cabinets (cabinet_id, tenant_id, name, root_path, created_at) VALUES (?, ?, ?, ?, ?) "
        "ON CONFLICT (cabinet_id) DO UPDATE SET tenant_id = COALESCE(excluded.tenant_id, cabinets.tenant_id), "
        "name = COALESCE(?, cabinets.name), root_path = COALESCE(excluded.root_path, cabinets.root_path)",
        (cabinet_id, tenant_id, name or cabinet_id, root_path, utc_now(), name),
    )


def cabinet_exists(conn: sqlite3.Connection, cabinet_id: str) -> bool:
    """Whether the store has a row for this cabinet.

    Args:
        conn: Open store.
        cabinet_id: The cabinet.

    Returns:
        True when the cabinet is known.
    """
    return conn.execute("SELECT 1 FROM cabinets WHERE cabinet_id = ?", (cabinet_id,)).fetchone() is not None


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

    Idempotent: one transaction upserts the cabinet (keeping its stored
    tenant and root path), deletes and re-inserts. A pack that fails changes
    nothing. The cabinet id is the pack slug, as in a pack-mode run.

    Args:
        conn: Open store.
        pack: One cabinet of a client pack.
        actor: Who registered it, for the audit log.

    Raises:
        PackRegistrationError: A class or a type's field is named twice, or the store refused a row.
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
    _check_unique(cab, [t[1] for t in types], [(d[1], d[2]) for d in defs])
    try:
        with conn:
            _write_cabinet(conn, cab, pack.cabinet.name, None, None)
            conn.execute("DELETE FROM field_defs WHERE cabinet_id = ?", (cab,))
            conn.execute("DELETE FROM doc_types WHERE cabinet_id = ?", (cab,))
            conn.executemany(
                "INSERT INTO doc_types (cabinet_id, name, aliases, has_samples) VALUES (?, ?, ?, ?)", types
            )
            conn.executemany(
                "INSERT INTO field_defs (cabinet_id, doc_type, name, ord, dtype, date_format, role, origin) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                defs,
            )
    except sqlite3.IntegrityError as e:
        raise PackRegistrationError(f"pack cabinet {cab} was not registered: the store refused a row") from e
    skipped = sum(len(v) for t, v in pack.fields.items() if t not in known)
    detail = {"doc_types": len(types), "field_defs": len(defs), "skipped_field_defs": skipped}
    record_audit(conn, actor=actor, action="pack.register", cabinet_id=cab, detail=detail)


def _check_unique(cabinet_id: str, type_names: list[str], field_keys: list[tuple[str, str]]) -> None:
    """Raise PackRegistrationError naming every class or (type, field) given twice."""
    types = sorted(n for n, c in Counter(type_names).items() if c > 1)
    fields = sorted(f"{t}/{f}" for (t, f), c in Counter(field_keys).items() if c > 1)
    if types or fields:
        raise PackRegistrationError(
            f"pack cabinet {cabinet_id} names these twice: types {types}, fields {fields}; nothing was registered"
        )


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


_FIELDS_WITH_DEFS = """
SELECT f.ord, f.name, f.value, f.normalized_value, f.page, f.char_start, f.char_end, f.method,
    f.validated, f.confidence, fd.role, fd.origin
FROM fields f
JOIN documents d ON d.doc_id = f.doc_id
LEFT JOIN field_defs fd ON fd.cabinet_id = d.cabinet_id AND fd.doc_type = d.doc_type AND fd.name = f.name
WHERE f.doc_id = ?
ORDER BY f.ord
"""

# Field corrections only, in import order; the last one per name wins.
_FIELD_EVENTS = "SELECT field, new_value, verdict FROM review_events WHERE doc_id = ? AND kind = 'field' ORDER BY seq"


def _norm(name: str) -> str:
    """A field name as the workbook importer stores it."""
    return name.casefold()


def effective_fields(conn: sqlite3.Connection, doc_id: str) -> list[dict[str, Any]]:
    """Each stored field of a document with the latest review correction laid over it.

    A pure read. Only field corrections (kind field) apply, matched on the
    casefolded name; class and tag corrections never overlay a field. The
    latest correction is the one with the highest seq, which is import order.
    `effective_value` is its new value when one exists (None when the reviewer
    removed the field), else the normalized value, else the raw value. role and
    origin come from field_defs when the cabinet has them, else None. Fields a
    reviewer added that the record never held are not listed; see
    list_review_events.

    Args:
        conn: Open store.
        doc_id: The document.

    Returns:
        One dict per stored field, in record order.
    """
    latest: dict[str, tuple[str | None, str | None]] = {}
    for e in conn.execute(_FIELD_EVENTS, (doc_id,)).fetchall():
        latest[_norm(e["field"])] = (e["new_value"], e["verdict"])
    out = []
    for r in conn.execute(_FIELDS_WITH_DEFS, (doc_id,)).fetchall():
        row = dict(r)
        found = latest.get(_norm(row["name"]))
        row["validated"] = bool(row["validated"])
        row["corrected"] = found is not None
        row["corrected_value"], row["verdict"] = found if found is not None else (None, None)
        row["effective_value"] = found[0] if found is not None else row["normalized_value"] or row["value"]
        out.append(row)
    return out
