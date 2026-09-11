"""The text grid strategy: columns from whitespace in the canonical text.

The only signal available for .txt and .docx, which carry no geometry at all.
"""

from __future__ import annotations

from ..schema import Table, TableCell, TableMethod
from .assemble import _assemble


def _blank_runs(lines: list[str], width: int, min_width: int) -> list[tuple[int, int]]:
    """Column ranges that are blank on every line of the band."""
    blank = []
    for j in range(width):
        if all(j >= len(ln) or ln[j] == " " for ln in lines):
            blank.append(j)
    runs = []
    start = None
    prev = None
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


def detect_text_grid(
    page: int, page_text: str, page_start: int, tcfg: dict
) -> tuple[list[Table], list[str]]:
    """Tables on one page from whitespace columns in the canonical text.

    Works on anything with text, including documents that carry no geometry
    at all. It relies on the alignment surviving into the text, which is true
    of plain text and of most text layers, and false the moment a renderer
    reflows the content.
    """
    tcg = dict(tcfg.get("text_grid", {}))
    min_rows = int(tcfg.get("min_rows", 2))
    min_cols = int(tcfg.get("min_cols", 2))
    min_cells = int(tcfg.get("min_cells", 6))
    min_sep = int(tcg.get("min_separator_width", 2))
    min_multi = float(tcg.get("min_multi_cell_rows", 0.75))

    raw_lines = page_text.split("\n")
    offsets = []
    cursor = page_start
    for ln in raw_lines:
        offsets.append(cursor)
        cursor += len(ln) + 1

    tables: list[Table] = []
    abstained: list[str] = []

    # Bands are runs of non blank lines. A blank line ends a block.
    band: list[int] = []
    for i, ln in enumerate(raw_lines + [""]):
        if ln.strip():
            band.append(i)
            continue
        if len(band) >= min_rows:
            t, a = _search_band(
                page, band, raw_lines, offsets, min_sep, min_rows, min_cols,
                min_cells, min_multi, int(tcg.get("max_band_lines", 400)), tcfg,
            )
            tables.extend(t)
            abstained.extend(a)
        band = []

    return tables, abstained


def _search_band(
    page: int,
    band: list[int],
    raw_lines: list[str],
    offsets: list[int],
    min_sep: int,
    min_rows: int,
    min_cols: int,
    min_cells: int,
    min_multi: float,
    max_band_lines: int,
    tcfg: dict,
) -> tuple[list[Table], list[str]]:
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
    if len(band) > max_band_lines:
        # Quadratic search is not worth it on a wall of text, and a wall of
        # text is not a table. Judge the band as it stands and move on.
        return _text_grid_band(
            page, band, raw_lines, offsets, min_sep, min_cols, min_cells, min_multi, tcfg
        )

    tables: list[Table] = []
    abstained: list[str] = []
    best_of: list[tuple] = []
    for i in range(len(band)):
        for j in range(i + min_rows, len(band) + 1):
            got, _ = _text_grid_band(
                page, band[i:j], raw_lines, offsets, min_sep, min_cols,
                min_cells, min_multi, tcfg,
            )
            if got:
                best_of.append((-got[0].n_cols, -got[0].n_rows, i, got[0]))

    # Nothing anywhere inside the band. Report why the band as a whole failed,
    # rather than one reason per rejected sub run, which would be noise.
    if not best_of:
        _, why = _text_grid_band(
            page, band, raw_lines, offsets, min_sep, min_cols, min_cells, min_multi, tcfg
        )
        return [], why

    # Take the best candidate, then look either side of it for another table.
    best_of.sort(key=lambda t: t[:3])
    _, _, start, table = best_of[0]
    used = {c.row + start for c in table.cells}
    lo, hi = min(used), max(used)
    tables.append(table)
    for side in (band[:lo], band[hi + 1:]):
        if len(side) >= min_rows:
            got, why = _search_band(
                page, side, raw_lines, offsets, min_sep, min_rows, min_cols,
                min_cells, min_multi, max_band_lines, tcfg,
            )
            tables.extend(got)
            abstained.extend(why)
    tables.sort(key=lambda t: (t.char_start if t.char_start is not None else 0))
    return tables, abstained


def _text_grid_band(
    page: int,
    band: list[int],
    raw_lines: list[str],
    offsets: list[int],
    min_sep: int,
    min_cols: int,
    min_cells: int,
    min_multi: float,
    tcfg: dict,
) -> tuple[list[Table], list[str]]:
    body = [raw_lines[i] for i in band]
    width = max(len(ln) for ln in body)
    seps = _blank_runs(body, width, min_sep)
    if not seps:
        return [], []

    # Segments between separators. Margins fall out as empty columns below.
    bounds: list[tuple[int, int]] = []
    cursor = 0
    for s0, s1 in seps:
        if s0 > cursor:
            bounds.append((cursor, s0))
        cursor = s1
    if cursor < width:
        bounds.append((cursor, width))

    grid: list[list[TableCell]] = []
    for r, line_idx in enumerate(band):
        line = raw_lines[line_idx]
        row_cells = []
        for c, (c0, c1) in enumerate(bounds):
            raw = line[c0:c1]
            content = raw.strip()
            if not content:
                row_cells.append(None)
                continue
            lead = len(raw) - len(raw.lstrip())
            start = offsets[line_idx] + c0 + lead
            row_cells.append(
                TableCell(
                    row=r, col=c, text=content, page=page,
                    char_start=start, char_end=start + len(content),
                )
            )
        grid.append(row_cells)

    # Drop columns nothing lands in, which is what a margin looks like.
    keep = [c for c in range(len(bounds)) if any(row[c] is not None for row in grid)]
    if len(keep) < min_cols:
        return [], []

    cells = []
    multi = 0
    for row in grid:
        present = [row[c] for c in keep if row[c] is not None]
        if len(present) >= 2:
            multi += 1
        for new_c, c in enumerate(keep):
            cell = row[c]
            if cell is None:
                continue
            cell.col = new_c
            cells.append(cell)

    where = f"page {page} lines {band[0]}-{band[-1]}"
    if len(cells) < min_cells:
        return [], []
    if multi / len(grid) < min_multi:
        # Alignment that only one or two lines take part in is a coincidence
        # in a block of prose, not a column structure.
        return [], [f"{where}: only {multi} of {len(grid)} rows straddle a separator"]

    cells.sort(key=lambda c: (c.row, c.col))
    return [_assemble(page, len(grid), len(keep), TableMethod.TEXT_GRID, cells, tcfg)], []
