"""Regression tests for store bugs from the confirmation review: replayed review history, truncation routing."""

from contextlib import closing

import pytest
from test_db_registry import A, _record

from baseline import db, db_registry

FIELD = "bill no"
SOURCE = "wb.xlsx#Metadata"
# One note per prefix, as baseline/extract.py and cascade/runner.py write them.
TRUNCATION_NOTES = [
    "read frame 1 of 4, 3 skipped (extraction.multi_frame_images)",
    "cascade: image mode sent 3 of 9 pages (render.max_pages)",
    "cascade: text mode sent 2 of 5 pages (vlm.text_total_chars)",
    "cascade: text mode cut the text of pages [1, 2] (vlm.text_page_chars, vlm.text_total_chars)",
]


@pytest.fixture
def conn(tmp_path):
    with closing(db.connect(tmp_path / "s.sqlite")) as c:
        db.upsert_document(c, _record())
        yield c


def _events(*pairs):
    return [db.ReviewEvent(A, FIELD, old, new, SOURCE) for old, new in pairs]


def _effective(conn):
    return db_registry.effective_fields(conn, A)[0]["effective_value"]


# 1. A replayed older workbook does not revert the value.


def test_the_same_workbook_imported_twice_adds_nothing_the_second_time(conn):
    w1 = _events(("X1", "X2"))
    assert [db.record_review_events(conn, w1) for _ in range(2)] == [1, 0]


def test_a_correction_and_its_undo_over_three_imports_ends_at_b(conn):
    for pair in (("A", "B"), ("B", "A"), ("A", "B")):
        db.record_review_events(conn, _events(pair))
    assert _effective(conn) == "B"


def test_a_two_correction_workbook_imported_three_times_writes_two_then_nothing(conn):
    w1 = _events(("X1", "X2"), ("X2", "X3"))
    assert [db.record_review_events(conn, w1) for _ in range(3)] == [2, 0, 0]


def test_an_extension_of_stored_history_writes_only_the_new_suffix(conn):
    db.record_review_events(conn, _events(("X1", "X2")))
    db.record_review_events(conn, _events(("X1", "X2"), ("X2", "X3")))
    assert [(e["old_value"], e["new_value"]) for e in db.list_review_events(conn, A)] == [("X1", "X2"), ("X2", "X3")]


def test_an_older_workbook_replayed_writes_nothing(conn):
    db.record_review_events(conn, _events(("X1", "X2"), ("X2", "X3")))
    assert db.record_review_events(conn, _events(("X1", "X2"))) == 0


def test_an_older_workbook_replayed_keeps_the_newest_value(conn):
    db.record_review_events(conn, _events(("X1", "X2"), ("X2", "X3")))
    db.record_review_events(conn, _events(("X1", "X2")))
    assert _effective(conn) == "X3"


def test_a_correction_from_a_value_never_stored_is_appended(conn):
    db.record_review_events(conn, _events(("X1", "X2")))
    assert db.record_review_events(conn, _events(("X9", "X4"))) == 1


# 2. A document with dropped pages or cut text goes to review.


def test_a_record_without_notes_routes_auto():
    assert db_registry.routing_status(_record()) == db_registry.ROUTE_AUTO


@pytest.mark.parametrize("note", TRUNCATION_NOTES)
def test_a_truncation_note_routes_the_record_to_review(note):
    record = _record()
    record.diagnostics.notes.append(note)
    assert db_registry.routing_status(record) == db_registry.ROUTE_REVIEW


def test_every_truncation_prefix_is_covered_by_a_test_note():
    assert {p for p in db_registry.TRUNCATION_NOTE_PREFIXES if any(n.startswith(p) for n in TRUNCATION_NOTES)} == set(
        db_registry.TRUNCATION_NOTE_PREFIXES
    )


def test_an_unrelated_note_keeps_the_record_auto():
    record = _record()
    record.diagnostics.notes.append("cascade: total value is only in text not sent but read anyway")
    assert db_registry.routing_status(record) == db_registry.ROUTE_AUTO
