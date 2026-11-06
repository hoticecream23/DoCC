"""The tab delimited strategy: rows the extractor wrote with tabs between cells.

Real .docx tables come through `_extract_docx` as one tab joined line per row.
A single tab is one character, so it never forms text_grid's two character
separator, and a .docx has no word boxes for geometry. Without this strategy
every Word table in the corpus was invisible.
"""

from __future__ import annotations

from typing import Any

from ..schema import Table, TableCell, TableMethod
from .assemble import _assemble


def detect_tab_delimited(
    page: int, page_text: str, page_start: int, tcfg: dict[str, Any]
) -> tuple[list[Table], list[str]]:
    """Tables on one page from runs of lines that carry the same tab count.

    The structural check is the tab count itself: every row of a Word table
    has one cell per column, so a run ends at the first line whose count
    differs. Merged cells repeat their text across the columns they span.

    Args:
        page: Page number.
        page_text: Canonical text of the page.
        page_start: Offset of the page in the canonical text.
        tcfg: Tables config.

    Returns:
        The tables found, and an always empty abstention list.
    """
    min_rows = int(tcfg.get("min_rows", 2))
    min_cols = int(tcfg.get("min_cols", 2))
    min_cells = int(tcfg.get("min_cells", 6))

    tables: list[Table] = []
    run: list[tuple[int, str]] = []  # (offset of the line, line)
    cursor = page_start
    for line in page_text.split("\n") + [""]:
        tabs = line.count("\t")
        if run and tabs != run[0][1].count("\t"):
            t = _run_to_table(page, run, min_rows, min_cols, min_cells, tcfg)
            if t is not None:
                tables.append(t)
            run = []
        if tabs:
            run.append((cursor, line))
        cursor += len(line) + 1
    return tables, []


def _run_to_table(
    page: int,
    run: list[tuple[int, str]],
    min_rows: int,
    min_cols: int,
    min_cells: int,
    tcfg: dict[str, Any],
) -> Table | None:
    """One run of equal tab count lines as a table, or None when too small."""
    if len(run) < min_rows:
        return None
    grid: list[list[TableCell | None]] = []
    for r, (line_start, line) in enumerate(run):
        row: list[TableCell | None] = []
        pos = 0
        for c, raw in enumerate(line.split("\t")):
            content = raw.strip()
            if content:
                start = line_start + pos + (len(raw) - len(raw.lstrip()))
                row.append(
                    TableCell(
                        row=r,
                        col=c,
                        text=content,
                        page=page,
                        char_start=start,
                        char_end=start + len(content),
                    )
                )
            else:
                row.append(None)
            pos += len(raw) + 1
        grid.append(row)

    # Drop columns nothing lands in, the same way text_grid drops margins.
    keep = [c for c in range(len(grid[0])) if any(row[c] is not None for row in grid)]
    cells = []
    for row in grid:
        for new_c, c in enumerate(keep):
            cell = row[c]
            if cell is not None:
                cell.col = new_c
                cells.append(cell)
    if len(keep) < min_cols or len(cells) < min_cells:
        return None
    return _assemble(page, len(grid), len(keep), TableMethod.TAB_DELIMITED, cells, tcfg)
