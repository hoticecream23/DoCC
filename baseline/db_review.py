"""Review events: the human corrections the workbook importer records.

Split from ``baseline/db.py``, which re-exports every public name here.
Corrections are append-only. Each row has a ``seq``, the next integer at
insert, so "latest" is import order, never the clock. ``kind`` (class, field or
tag) keeps a class or tag correction from overlaying a field.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, NamedTuple

from .db_registry import DEFAULT_ACTOR, record_audit
from .db_registry import utc_now as _now
from .logging_setup import get_logger

log = get_logger(__name__)

KIND_CLASS = "class"
KIND_FIELD = "field"
KIND_TAG = "tag"
_SHEET_KIND = {"Documents": KIND_CLASS, "Tags": KIND_TAG, "Metadata": KIND_FIELD}


class ReviewEvent(NamedTuple):
    """One human correction: a field, class or tag that a review changed.

    verdict is correct, incorrect, corrected or added, when the source says;
    is_gold is whether the reviewer marked the row as gold. Neither is part of
    the event id. kind is class, field or tag; None derives it from the
    importer's convention (the source's sheet, else a field "class" or "tag:").
    """

    doc_id: str
    field: str
    old_value: str | None
    new_value: str | None
    source: str
    verdict: str | None = None
    is_gold: bool = False
    kind: str | None = None


def event_kind(event: ReviewEvent) -> str:
    """The event's kind: as given, else from the source sheet, else from the field name.

    Args:
        event: The correction.

    Returns:
        KIND_CLASS, KIND_FIELD or KIND_TAG.
    """
    if event.kind:
        return event.kind
    sheet = event.source.rpartition("#")[2] if "#" in event.source else ""
    if sheet in _SHEET_KIND:
        return _SHEET_KIND[sheet]
    if event.field == KIND_CLASS:
        return KIND_CLASS
    return KIND_TAG if event.field.startswith("tag:") else KIND_FIELD


def document_cabinet(conn: sqlite3.Connection, doc_id: str) -> str | None:
    """The cabinet that holds a document.

    Args:
        conn: Open store.
        doc_id: The document.

    Returns:
        The cabinet id, or None when the store does not hold the document.
    """
    row = conn.execute("SELECT cabinet_id FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
    return None if row is None else str(row["cabinet_id"])


def _event_id(event: ReviewEvent, predecessor: str | None) -> str:
    """Id of a correction; it names the event it follows, so a repeated later correction gets its own id."""
    key = json.dumps([event.doc_id, event.field, event.old_value, event.new_value, event.source, predecessor])
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


_LATEST_EVENT = (
    "SELECT event_id, old_value, new_value, source FROM review_events WHERE doc_id = ? AND field = ? "
    "ORDER BY seq DESC LIMIT 1"
)

# seq is the next integer, taken in the same transaction, so no autoincrement is needed.
_INSERT_EVENT = (
    "INSERT INTO review_events (event_id, doc_id, field, old_value, new_value, source, created_at, verdict, is_gold, "
    "kind, seq) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, (SELECT COALESCE(MAX(seq), 0) + 1 FROM review_events)) "
    "ON CONFLICT (event_id) DO NOTHING"
)


def _write_event(conn: sqlite3.Connection, e: ReviewEvent, now: str) -> int:
    """Insert one event unless it repeats the latest stored event for its field; return rows written."""
    latest = conn.execute(_LATEST_EVENT, (e.doc_id, e.field)).fetchone()
    if latest is not None and (latest["old_value"], latest["new_value"], latest["source"]) == (
        e.old_value,
        e.new_value,
        e.source,
    ):
        return 0
    predecessor = None if latest is None else str(latest["event_id"])
    row = (_event_id(e, predecessor), e.doc_id, e.field, e.old_value, e.new_value, e.source, now, e.verdict)
    return int(conn.execute(_INSERT_EVENT, (*row, int(e.is_gold), event_kind(e))).rowcount)


def record_review_events(conn: sqlite3.Connection, events: list[ReviewEvent], *, actor: str = DEFAULT_ACTOR) -> int:
    """Store review corrections, in order. An event identical to the latest stored one for its field is skipped.

    Identical means the same old value, new value and source; verdict and
    is_gold do not count, so a skipped row keeps its stored verdict. Importing
    the same workbook twice in a row adds nothing, while a correction made
    again after a different one (A to B, B to A, A to B) is stored and is
    then the latest. Events for a doc_id the store does not hold are skipped:
    the foreign key needs the document. One audit row per cabinet records the
    counts and the source file names.

    Args:
        conn: Open store.
        events: The corrections, in sheet order.
        actor: Who recorded them, for the audit log.

    Returns:
        Number of new rows written.
    """
    now = _now()
    written: Counter[str] = Counter()
    sources: dict[str, set[str]] = {}
    with conn:
        for e in events:
            cabinet_id = document_cabinet(conn, e.doc_id)
            if cabinet_id is None:
                continue
            written[cabinet_id] += _write_event(conn, e, now)
            sources.setdefault(cabinet_id, set()).add(Path(e.source.split("#")[0]).name)
    for cabinet_id, names in sorted(sources.items()):
        detail = {"written": written[cabinet_id], "sources": sorted(names)}
        record_audit(conn, actor=actor, action="review_events.record", cabinet_id=cabinet_id, detail=detail)
    log.info("review events recorded", extra={"events": len(events), "written": sum(written.values())})
    return sum(written.values())


def list_review_events(conn: sqlite3.Connection, doc_id: str | None = None) -> list[dict[str, Any]]:
    """Stored review events, ordered by doc_id, then seq (import order).

    Args:
        conn: Open store.
        doc_id: Only this document; None for every document.

    Returns:
        One dict per event, keyed by column name.
    """
    sql = "SELECT * FROM review_events WHERE (? IS NULL OR doc_id = ?) ORDER BY doc_id, seq"
    return [dict(r) for r in conn.execute(sql, [doc_id, doc_id]).fetchall()]
