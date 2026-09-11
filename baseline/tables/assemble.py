"""A finished grid of cells, made into a Table. Shared by both strategies."""

from __future__ import annotations

from ..schema import Table, TableCell, TableMethod


def _is_header_row(cells: list[TableCell], n_cols: int, hcfg: dict) -> bool:
    """First row is a header when it is a full row of labels, not of values.

    Digits are the tell. When the test fails the table is emitted with no
    header at all rather than with a header that is really data.
    """
    row0 = [c for c in cells if c.row == 0]
    if not row0:
        return False
    if bool(hcfg.get("require_full_row", True)) and len(row0) < n_cols:
        return False
    body = "".join(c.text for c in row0)
    if not body:
        return False
    digits = sum(ch.isdigit() for ch in body)
    return (digits / len(body)) <= float(hcfg.get("max_digit_ratio", 0.1))


def _assemble(
    page: int, n_rows: int, n_cols: int, method: TableMethod,
    cells: list[TableCell], tcfg: dict,
) -> Table:
    header: list[str] = []
    if _is_header_row(cells, n_cols, dict(tcfg.get("header", {}))):
        by_col = {c.col: c.text for c in cells if c.row == 0}
        header = [by_col.get(i, "") for i in range(n_cols)]
        for c in cells:
            if c.row == 0:
                c.is_header = True

    placed = [c for c in cells if c.char_start is not None and c.char_end is not None]
    span = (
        (min(c.char_start for c in placed), max(c.char_end for c in placed))
        if len(placed) == len(cells) and placed
        else (None, None)
    )
    # Grid occupancy, nothing more. It orders candidates, it is not calibrated
    # against whether the table is right. A sparse but correct table scores
    # lower than a dense wrong one, so read it as density, not as trust.
    fill = round(len(cells) / (n_rows * n_cols), 4) if n_rows and n_cols else 0.0
    return Table(
        table_id="",  # filled in by the caller, which knows the doc_id
        page=page,
        n_rows=n_rows,
        n_cols=n_cols,
        method=method,
        confidence=min(1.0, fill),
        header=header,
        cells=cells,
        char_start=span[0],
        char_end=span[1],
    )
