"""Table extraction v0.

The premise matches the knowledge graph: a wrong grid is worse than no grid.
So most of these tests assert that a cell lands exactly where it should, or
that a band the extractor cannot read is refused rather than guessed at.
"""

import json

import pytest

from baseline.pipeline import dumps, process_document
from baseline.schema import TableMethod
from baseline.tables import (
    _blank_runs,
    _build_columns,
    _Cell,
    _is_header_row,
    _segment_cells,
    _Line,
    build_record,
    detect_text_grid,
)


@pytest.fixture(scope="module")
def tcfg(cfg):
    return cfg.tables


def _tables_for(path, cfg):
    """Run the pipeline on one file, then the table pass over its output."""
    record, sidecar = process_document(path, cfg)
    row = json.loads(dumps(record))
    words = [w.model_dump(mode="json") for w in sidecar.words]
    return row, build_record(row, words, cfg.tables)


def _grid(table):
    cells = {(c.row, c.col): c.text for c in table.cells}
    return [[cells.get((r, c), "") for c in range(table.n_cols)] for r in range(table.n_rows)]


# ------------------------------------------------------------- primitives

def _w(text, x0, x1, y0=0.0, y1=10.0, start=None):
    return {"text": text, "x0": x0, "x1": x1, "y0": y0, "y1": y1,
            "char_start": start, "char_end": None if start is None else start + len(text)}


def test_a_wide_gap_ends_a_cell_and_a_space_does_not():
    line = _Line(y0=0, y1=10, words=[_w("Widget", 0, 30), _w("assembly", 33, 70),
                                     _w("2500.00", 120, 160)])
    cells = _segment_cells(line, 0, gap_limit=8.0)
    assert [len(c.words) for c in cells] == [2, 1]


def test_a_column_needs_every_member_to_agree_not_just_one():
    """Complete linkage. Single linkage chains two columns through a straddler."""
    left = _Cell(row=0, x0=0, x1=40, y0=0, y1=10)
    straddler = _Cell(row=1, x0=30, x1=70, y0=10, y1=20)
    right = _Cell(row=2, x0=60, x1=100, y0=20, y1=30)
    cols, ambiguous = _build_columns([left, straddler, right], ratio=0.4)
    assert len(cols) > 1, "left and right must not be chained into one column"
    assert ambiguous == 0


def test_two_cells_from_one_row_never_share_a_column():
    a = _Cell(row=0, x0=0, x1=50, y0=0, y1=10)
    b = _Cell(row=0, x0=10, x1=60, y0=0, y1=10)
    cols, _ = _build_columns([a, b], ratio=0.4)
    assert len(cols) == 2


def test_blank_runs_are_the_columns_blank_on_every_line():
    lines = ["Date        Description   Debit",
             "01/03/2024  Opening        22.00"]
    runs = _blank_runs(lines, max(len(x) for x in lines), min_width=2)
    assert (10, 12) in runs


def test_a_row_of_numbers_is_not_a_header():
    from baseline.schema import TableCell

    row = [TableCell(row=0, col=0, text="45000.00", page=0),
           TableCell(row=0, col=1, text="7200.00", page=0)]
    assert not _is_header_row(row, 2, {"require_full_row": True, "max_digit_ratio": 0.1})


def test_a_full_row_of_labels_is_a_header():
    from baseline.schema import TableCell

    row = [TableCell(row=0, col=0, text="Qty", page=0),
           TableCell(row=0, col=1, text="Description", page=0)]
    assert _is_header_row(row, 2, {"require_full_row": True, "max_digit_ratio": 0.1})


# ------------------------------------------------------ geometry strategy

def test_invoice_line_items_come_out_of_the_boxes(corpus, cfg):
    row, rec = _tables_for(corpus / "invoice_native.pdf", cfg)
    assert len(rec.tables) == 1
    t = rec.tables[0]
    assert t.method == TableMethod.GEOMETRY
    assert (t.n_rows, t.n_cols) == (3, 4)
    assert t.header == ["Qty", "Description", "Rate", "Amount"]
    assert _grid(t)[1] == ["2", "Widget assembly", "2500.00", "5000.00"]
    assert _grid(t)[2] == ["1", "Freight and handling", "5000.00", "5000.00"]


def test_a_two_row_table_still_counts(corpus, cfg):
    """The purchase order has one line item. min_cells, not min_rows, is the gate."""
    _, rec = _tables_for(corpus / "purchase_order.pdf", cfg)
    assert len(rec.tables) == 1
    assert (rec.tables[0].n_rows, rec.tables[0].n_cols) == (2, 4)


def test_table_cells_slice_the_canonical_text_back(corpus, cfg):
    """The contract every offset in this repo lives under."""
    for name in ("invoice_native.pdf", "bank_statement.pdf",
                 "purchase_order.pdf", "salary_slip.docx"):
        row, rec = _tables_for(corpus / name, cfg)
        full = row["text"]["full"]
        assert rec.verify_offsets(full) == [], name
        placed = [c for t in rec.tables for c in t.cells if c.char_start is not None]
        assert placed, name
        for c in placed:
            assert full[c.char_start:c.char_end] == c.text


# ----------------------------------------------------- text grid strategy

def test_the_table_is_found_inside_a_band_with_no_blank_lines(corpus, cfg):
    """A PDF text layer drops blank lines, so the whole page arrives fused.

    The letterhead above the table destroys the separators, so the run has to
    be searched for rather than assumed to be the band.
    """
    _, rec = _tables_for(corpus / "bank_statement.pdf", cfg)
    assert len(rec.tables) == 1
    t = rec.tables[0]
    assert t.method == TableMethod.TEXT_GRID
    assert t.header == ["Date", "Description", "Debit", "Credit", "Balance"]
    assert (t.n_rows, t.n_cols) == (5, 5)
    # The letterhead lines above the table are not rows of it.
    assert all("IFSC" not in c.text for c in t.cells)


def test_sparse_cells_stay_in_their_own_column(corpus, cfg):
    """Debit and Credit are mostly empty. Nothing may slide between them."""
    _, rec = _tables_for(corpus / "bank_statement.pdf", cfg)
    grid = _grid(rec.tables[0])
    assert grid[3] == ["12/03/2024", "Vendor payment", "22000.00", "", "114800.00"]
    assert grid[2] == ["05/03/2024", "NEFT Globex", "", "11800.00", "136800.00"]


def test_a_document_with_no_geometry_at_all_still_gets_tables(corpus, cfg):
    """docx has no boxes, so text_grid is the only strategy that can run."""
    _, rec = _tables_for(corpus / "salary_slip.docx", cfg)
    assert len(rec.tables) == 1
    t = rec.tables[0]
    assert t.method == TableMethod.TEXT_GRID
    assert (t.n_rows, t.n_cols) == (5, 2)
    # No digits-only header was invented for a table that has no header.
    assert t.header == []
    assert _grid(t)[0] == ["Basic Pay", "45000.00"]


def test_text_grid_cells_pick_up_geometry_when_the_document_has_any(corpus, cfg):
    """Offsets already place the cell, so the boxes are looked up, not guessed."""
    _, rec = _tables_for(corpus / "bank_statement.pdf", cfg)
    cells = rec.tables[0].cells
    assert all(c.x0 is not None for c in cells)


def test_prose_does_not_become_a_table(corpus, cfg):
    _, rec = _tables_for(corpus / "contract.txt", cfg)
    assert rec.tables == []


def test_a_label_value_block_is_not_a_table(corpus, cfg):
    _, rec = _tables_for(corpus / "id_card.txt", cfg)
    assert rec.tables == []


def test_alignment_only_a_couple_of_rows_share_is_refused(tcfg):
    """Two aligned lines in a block of prose is a coincidence, not a column."""
    body = "\n".join([
        "alpha beta gamma delta epsilon zeta",
        "one  two",
        "three  four",
        "a much longer line of ordinary prose that runs on",
        "another long line of ordinary prose running on too",
    ])
    tables, _ = detect_text_grid(0, body, 0, tcfg)
    assert tables == []


# ------------------------------------------------------------- abstention

def test_a_band_that_does_not_line_up_is_refused_not_guessed(corpus, cfg):
    """Space aligned text rendered in a proportional font loses its columns.

    The header drifts off its own data, so geometry finds more columns than
    the widest row has cells. That is unreadable, and it must say so.
    """
    row, sidecar = process_document(corpus / "bank_statement.pdf", cfg)
    words = [w.model_dump(mode="json") for w in sidecar.words]
    from baseline.tables import detect_geometry

    tables, abstained = detect_geometry(0, words, row.text.full, cfg.tables)
    assert tables == []
    assert any("columns but the widest row holds" in a for a in abstained)


def test_an_abstention_is_recorded_with_its_reason(corpus, cfg):
    _, rec = _tables_for(corpus / "bank_statement.pdf", cfg)
    assert rec.diagnostics.abstained, "a refusal must leave a trace"


def test_geometry_declining_hands_over_to_text_grid(corpus, cfg):
    """First hit wins, so a strategy that abstains does not block the next."""
    _, rec = _tables_for(corpus / "bank_statement.pdf", cfg)
    assert rec.diagnostics.strategies_tried == ["geometry", "text_grid"]
    assert rec.tables and rec.tables[0].method == TableMethod.TEXT_GRID


def test_a_missing_sidecar_is_not_an_error(corpus, cfg):
    """Boxes are optional. Without them text_grid simply carries the load."""
    row, _ = _tables_for(corpus / "bank_statement.pdf", cfg)
    rec = build_record(row, [], cfg.tables)
    assert rec.diagnostics.errors == []
    assert len(rec.tables) == 1
    assert all(c.x0 is None for c in rec.tables[0].cells)


# ------------------------------------------------------------- guarantees

def test_the_pass_is_reproducible(corpus, cfg):
    row, _ = _tables_for(corpus / "invoice_native.pdf", cfg)
    record, sidecar = process_document(corpus / "invoice_native.pdf", cfg)
    words = [w.model_dump(mode="json") for w in sidecar.words]
    a = build_record(row, words, cfg.tables).model_dump(mode="json")
    b = build_record(row, words, cfg.tables).model_dump(mode="json")
    a["diagnostics"]["timings_ms"] = b["diagnostics"]["timings_ms"] = {}
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_the_document_record_is_not_touched(corpus, cfg):
    """Tables are a separate pass. The frozen record must come back unchanged."""
    record, sidecar = process_document(corpus / "invoice_native.pdf", cfg)
    before = dumps(record)
    words = [w.model_dump(mode="json") for w in sidecar.words]
    build_record(json.loads(before), words, cfg.tables)
    assert dumps(record) == before


def test_no_column_or_table_vocabulary_is_hardcoded():
    """Config over code, same rule the rest of the pipeline lives under."""
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent / "baseline" / "tables.py"
    body = src.read_text(encoding="utf-8").lower()
    for word in ("qty", "invoice", "debit", "credit", "balance", "hsn"):
        assert word not in body, f"{word!r} is domain vocabulary, it belongs in yaml"


def test_thresholds_come_from_config_not_defaults(corpus, cfg):
    """Raising min_cells past the table's size must suppress it."""
    strict = dict(cfg.tables)
    strict["min_cells"] = 999
    row, _ = _tables_for(corpus / "invoice_native.pdf", cfg)
    record, sidecar = process_document(corpus / "invoice_native.pdf", cfg)
    words = [w.model_dump(mode="json") for w in sidecar.words]
    assert build_record(row, words, strict).tables == []


def test_v0_never_claims_a_merged_cell(corpus, cfg):
    """Spans are in the schema and always 1. Saying so is the point of them."""
    for name in ("invoice_native.pdf", "bank_statement.pdf", "salary_slip.docx"):
        _, rec = _tables_for(corpus / name, cfg)
        for t in rec.tables:
            assert all(c.row_span == 1 and c.col_span == 1 for c in t.cells), name
