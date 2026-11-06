"""The text grid strategy: columns from whitespace in the canonical text.

The only signal available for .txt and .docx, which carry no geometry at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..schema import Table, TableCell, TableMethod
from .assemble import _assemble

# A row straddles a separator when it has at least this many cells.
MULTI_CELL_ROW = 2


@dataclass(frozen=True)
class _GridParams:
    """Thresholds for one page, read once from the tables config."""

    min_sep: int
    min_rows: int
    min_cols: int
    min_cells: int
    min_multi: float
    max_band_lines: int
    tcfg: dict[str, Any]

    @classmethod
    def from_config(cls, tcfg: dict[str, Any]) -> _GridParams:
        """Read the thresholds, with their defaults, from the tables config."""
        tcg = dict(tcfg.get("text_grid", {}))
        return cls(
            min_sep=int(tcg.get("min_separator_width", 2)),
            min_rows=int(tcfg.get("min_rows", 2)),
            min_cols=int(tcfg.get("min_cols", 2)),
            min_cells=int(tcfg.get("min_cells", 6)),
            min_multi=float(tcg.get("min_multi_cell_rows", 0.75)),
            max_band_lines=int(tcg.get("max_band_lines", 400)),
            tcfg=tcfg,
        )


@dataclass(frozen=True)
class _PageText:
    """One page split into lines, with the canonical offset of each line."""

    page: int
    lines: list[str]
    offsets: list[int]


def _blank_runs(lines: list[str], width: int, min_width: int) -> list[tuple[int, int]]:
    """Column ranges that are blank on every line of the band."""
    blank = []
    for j in range(width):
        if all(j >= len(ln) or ln[j] == " " for ln in lines):
            blank.append(j)
    runs = []
    start: int | None = None
    prev = 0
    for j in blank:
        if start is None:
            start = j
        elif j != prev + 1:
            runs.append((start, prev + 1))
            start = j
        prev = j
    if start is not None:
        runs.append((start, prev + 1))
    return [r for r in runs if (r[1] - r[0]) >= min_width]


def detect_text_grid(page: int, page_text: str, page_start: int, tcfg: dict[str, Any]) -> tuple[list[Table], list[str]]:
    """Tables on one page from whitespace columns in the canonical text.

    Works on anything with text, including documents that carry no geometry
    at all. It relies on the alignment surviving into the text, which is true
    of plain text and of most text layers, and false the moment a renderer
    reflows the content.

    Args:
        page: Page number.
        page_text: Canonical text of the page.
        page_start: Offset of the page in the canonical text.
        tcfg: Tables config.

    Returns:
        The tables found, and reasons for bands that looked like tables but
        were rejected.
    """
    params = _GridParams.from_config(tcfg)
    raw_lines = page_text.split("\n")
    offsets = []
    cursor = page_start
    for ln in raw_lines:
        offsets.append(cursor)
        cursor += len(ln) + 1
    text = _PageText(page, raw_lines, offsets)

    tables: list[Table] = []
    abstained: list[str] = []

    # Bands are runs of non blank lines. A blank line ends a block.
    band: list[int] = []
    for i, ln in enumerate(raw_lines + [""]):
        if ln.strip():
            band.append(i)
            continue
        if len(band) >= params.min_rows:
            t, a = _search_band(text, band, params)
            tables.extend(t)
            abstained.extend(a)
        band = []

    return tables, abstained


def _best_sub_run(text: _PageText, band: list[int], params: _GridParams) -> tuple[int, Table] | None:
    """Best table over every sub run of the band, with the run's start index."""
    best_of: list[tuple[int, int, int, Table]] = []
    for i in range(len(band)):
        for j in range(i + params.min_rows, len(band) + 1):
            got, _ = _text_grid_band(text, band[i:j], params)
            if got:
                best_of.append((-got[0].n_cols, -got[0].n_rows, i, got[0]))
    if not best_of:
        return None
    best_of.sort(key=lambda t: t[:3])
    _, _, start, table = best_of[0]
    return start, table


def _search_band(text: _PageText, band: list[int], params: _GridParams) -> tuple[list[Table], list[str]]:
    """Find the table inside a band, which is rarely the whole band.

    A blank line is the obvious boundary, but PDF text layers routinely drop
    blank lines, so a whole page arrives as one band with its letterhead and
    its table fused together. Any line that is not part of the table destroys
    the separators the table depends on, so the run has to be found rather
    than assumed.

    Objective: the most columns, then the most rows. Adding a line can only
    remove separators, never add one, so a candidate with more columns is
    always the finer reading of the same region.
    """
    if len(band) > params.max_band_lines:
        # Quadratic search is not worth it on a wall of text, and a wall of
        # text is not a table. Judge the band as it stands and move on.
        return _text_grid_band(text, band, params)

    best = _best_sub_run(text, band, params)
    # Nothing anywhere inside the band. Report why the band as a whole failed,
    # rather than one reason per rejected sub run, which would be noise.
    if best is None:
        _, why = _text_grid_band(text, band, params)
        return [], why

    # Take the best candidate, then look either side of it for another table.
    start, table = best
    used = {c.row + start for c in table.cells}
    lo, hi = min(used), max(used)
    tables = [table]
    abstained: list[str] = []
    for side in (band[:lo], band[hi + 1 :]):
        if len(side) >= params.min_rows:
            got, why = _search_band(text, side, params)
            tables.extend(got)
            abstained.extend(why)
    tables.sort(key=lambda t: (t.char_start if t.char_start is not None else 0))
    return tables, abstained


def _column_bounds(seps: list[tuple[int, int]], width: int) -> list[tuple[int, int]]:
    """Segments between separators. Margins fall out as empty columns later."""
    bounds: list[tuple[int, int]] = []
    cursor = 0
    for s0, s1 in seps:
        if s0 > cursor:
            bounds.append((cursor, s0))
        cursor = s1
    if cursor < width:
        bounds.append((cursor, width))
    return bounds


def _grid_rows(text: _PageText, band: list[int], bounds: list[tuple[int, int]]) -> list[list[TableCell | None]]:
    """Cut each band line at the column bounds. None where a slot is blank."""
    grid: list[list[TableCell | None]] = []
    for r, line_idx in enumerate(band):
        line = text.lines[line_idx]
        row_cells: list[TableCell | None] = []
        for c, (c0, c1) in enumerate(bounds):
            raw = line[c0:c1]
            content = raw.strip()
            if not content:
                row_cells.append(None)
                continue
            lead = len(raw) - len(raw.lstrip())
            start = text.offsets[line_idx] + c0 + lead
            row_cells.append(
                TableCell(
                    row=r,
                    col=c,
                    text=content,
                    page=text.page,
                    char_start=start,
                    char_end=start + len(content),
                )
            )
        grid.append(row_cells)
    return grid


def _compact(grid: list[list[TableCell | None]], keep: list[int]) -> tuple[list[TableCell], int]:
    """Renumber cells onto the kept columns. Also counts multi cell rows."""
    cells = []
    multi = 0
    for row in grid:
        present = [row[c] for c in keep if row[c] is not None]
        if len(present) >= MULTI_CELL_ROW:
            multi += 1
        for new_c, c in enumerate(keep):
            cell = row[c]
            if cell is None:
                continue
            cell.col = new_c
            cells.append(cell)
    return cells, multi


def _text_grid_band(text: _PageText, band: list[int], params: _GridParams) -> tuple[list[Table], list[str]]:
    """Judge one band as a single table. Returns it, or why it was rejected."""
    body = [text.lines[i] for i in band]
    width = max(len(ln) for ln in body)
    seps = _blank_runs(body, width, params.min_sep)
    if not seps:
        return [], []

    bounds = _column_bounds(seps, width)
    grid = _grid_rows(text, band, bounds)

    # Drop columns nothing lands in, which is what a margin looks like.
    keep = [c for c in range(len(bounds)) if any(row[c] is not None for row in grid)]
    if len(keep) < params.min_cols:
        return [], []

    cells, multi = _compact(grid, keep)
    where = f"page {text.page} lines {band[0]}-{band[-1]}"
    if len(cells) < params.min_cells:
        return [], []
    if multi / len(grid) < params.min_multi:
        # Alignment that only one or two lines take part in is a coincidence
        # in a block of prose, not a column structure.
        return [], [f"{where}: only {multi} of {len(grid)} rows straddle a separator"]

    cells.sort(key=lambda c: (c.row, c.col))
    return [_assemble(text.page, len(grid), len(keep), TableMethod.TEXT_GRID, cells, params.tcfg)], []
