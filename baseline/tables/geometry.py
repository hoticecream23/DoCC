"""The geometry strategy: columns from word bounding boxes.

The right signal for real PDFs and scans, where x positions are what the
layout actually is.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from ..schema import PAGE_SEP, Table, TableCell, TableMethod
from .assemble import _assemble

# --------------------------------------------------------------------------
# small numeric helpers
# --------------------------------------------------------------------------


def _percentile(values: list[float], pct: float) -> float:
    """Nearest rank percentile. No numpy, and deterministic on ties."""
    if not values:
        return 0.0
    s = sorted(values)
    if pct <= 0:
        return s[0]
    k = int(round((pct / 100.0) * (len(s) - 1)))
    return s[max(0, min(len(s) - 1, k))]


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    """Length of the shared interval. Zero when they do not touch."""
    return max(0.0, min(a1, b1) - max(a0, b0))


def _overlap_ratio(a0: float, a1: float, b0: float, b1: float) -> float:
    """Shared length as a fraction of the narrower interval."""
    narrow = min(a1 - a0, b1 - b0)
    if narrow <= 0:
        return 0.0
    return _overlap(a0, a1, b0, b1) / narrow


# --------------------------------------------------------------------------
# lines and cells
# --------------------------------------------------------------------------


@dataclass
class _Cell:
    """One run of words with no wide gap in it. A cell before it has a column."""

    row: int
    x0: float
    x1: float
    y0: float
    y1: float
    words: list[dict] = field(default_factory=list)


@dataclass
class _Line:
    y0: float
    y1: float
    words: list[dict] = field(default_factory=list)

    @property
    def height(self) -> float:
        return self.y1 - self.y0


def _group_lines(words: list[dict], overlap_ratio: float) -> list[_Line]:
    """Words into lines by vertical overlap, top to bottom.

    Overlap rather than a y threshold, so a line with mixed font sizes still
    holds together and two tight lines still come apart.
    """
    ordered = sorted(words, key=lambda w: (round(w["y0"], 3), round(w["x0"], 3), w["text"]))
    lines: list[_Line] = []
    for w in ordered:
        placed = False
        if lines:
            cur = lines[-1]
            share = _overlap(cur.y0, cur.y1, w["y0"], w["y1"])
            narrow = min(cur.y1 - cur.y0, w["y1"] - w["y0"])
            if narrow > 0 and share / narrow >= overlap_ratio:
                cur.words.append(w)
                cur.y0 = min(cur.y0, w["y0"])
                cur.y1 = max(cur.y1, w["y1"])
                placed = True
        if not placed:
            lines.append(_Line(y0=w["y0"], y1=w["y1"], words=[w]))
    for ln in lines:
        ln.words.sort(key=lambda w: (round(w["x0"], 3), w["text"]))
    return lines


def _estimate_space_width(lines: list[_Line], gcfg: dict) -> float:
    """One space, in points, for this page.

    Within line gaps are bimodal: spaces inside a cell, and the much wider
    runs that separate columns. A low percentile lands inside the space
    cluster without having to know the font.
    """
    gaps = []
    for ln in lines:
        for a, b in zip(ln.words, ln.words[1:]):
            g = b["x0"] - a["x1"]
            if g > 0:
                gaps.append(g)
    est = _percentile(gaps, float(gcfg.get("space_width_percentile", 25)))
    return max(est, float(gcfg.get("min_space_width", 1.0)))


def _segment_cells(line: _Line, row: int, gap_limit: float) -> list[_Cell]:
    """Split one line where the horizontal gap exceeds the limit."""
    cells: list[_Cell] = []
    cur: list[dict] = []
    for w in line.words:
        if cur and (w["x0"] - cur[-1]["x1"]) > gap_limit:
            cells.append(_make_cell(row, cur))
            cur = []
        cur.append(w)
    if cur:
        cells.append(_make_cell(row, cur))
    return cells


def _make_cell(row: int, words: list[dict]) -> _Cell:
    return _Cell(
        row=row,
        x0=min(w["x0"] for w in words),
        x1=max(w["x1"] for w in words),
        y0=min(w["y0"] for w in words),
        y1=max(w["y1"] for w in words),
        words=list(words),
    )


# --------------------------------------------------------------------------
# columns and bands
# --------------------------------------------------------------------------


def _build_columns(cells: list[_Cell], ratio: float) -> tuple[list[list[_Cell]], int]:
    """Cluster cells into columns. Returns the columns and an ambiguity count.

    Complete linkage: a cell joins a column only if it clears the overlap
    ratio against every cell already there, not merely against one of them.
    Single linkage would chain two adjacent numeric columns into one via a
    header that straddles both.

    A cell may not join a column that already holds a cell from its own row,
    since one row contributes at most one cell per column by definition.

    A cell that fits more than one column is counted as ambiguous. The caller
    decides what to do about it; this function does not guess quietly.
    """
    columns: list[list[_Cell]] = []
    ambiguous = 0
    # Left to right, then by row, so the result never depends on input order.
    for cell in sorted(cells, key=lambda c: (round(c.x0, 3), c.row, round(c.x1, 3))):
        fits = []
        for idx, col in enumerate(columns):
            if any(m.row == cell.row for m in col):
                continue
            if all(_overlap_ratio(cell.x0, cell.x1, m.x0, m.x1) >= ratio for m in col):
                fits.append(idx)
        if not fits:
            columns.append([cell])
            continue
        if len(fits) > 1:
            ambiguous += 1
        # Best overlap wins, lowest index breaks a tie.
        best = max(
            fits,
            key=lambda i: (
                round(
                    sum(_overlap_ratio(cell.x0, cell.x1, m.x0, m.x1) for m in columns[i])
                    / len(columns[i]),
                    6,
                ),
                -i,
            ),
        )
        columns[best].append(cell)

    columns.sort(key=lambda col: (round(min(m.x0 for m in col), 3),
                                  round(min(m.x1 for m in col), 3)))
    return columns, ambiguous


def _geometry_bands(
    lines: list[_Line], gcfg: dict, min_cols: int, min_rows: int
) -> list[list[tuple[int, list[_Cell]]]]:
    """Runs of consecutive lines that each segment into enough cells."""
    space = _estimate_space_width(lines, gcfg)
    gap_limit = space * float(gcfg.get("cell_gap_spaces", 1.6))
    heights = [ln.height for ln in lines if ln.height > 0]
    max_gap = _median(heights) * float(gcfg.get("max_row_gap_heights", 2.0))

    segmented = [(i, _segment_cells(ln, i, gap_limit)) for i, ln in enumerate(lines)]

    bands: list[list[tuple[int, list[_Cell]]]] = []
    cur: list[tuple[int, list[_Cell]]] = []
    for (i, cells) in segmented:
        wide_enough = len(cells) >= min_cols
        breaks = False
        if cur and wide_enough:
            prev_i = cur[-1][0]
            # A big vertical jump means a new block, not the next table row.
            breaks = (lines[i].y0 - lines[prev_i].y1) > max_gap
        if not wide_enough or breaks:
            if len(cur) >= min_rows:
                bands.append(cur)
            cur = [(i, cells)] if (wide_enough and breaks) else []
            continue
        cur.append((i, cells))
    if len(cur) >= min_rows:
        bands.append(cur)
    return bands


# --------------------------------------------------------------------------
# the strategy
# --------------------------------------------------------------------------


def detect_geometry(
    page: int, words: list[dict], full: str, tcfg: dict
) -> tuple[list[Table], list[str]]:
    """Tables on one page from its word boxes."""
    gcfg = dict(tcfg.get("geometry", {}))
    min_rows = int(tcfg.get("min_rows", 2))
    min_cols = int(tcfg.get("min_cols", 2))
    min_cells = int(tcfg.get("min_cells", 6))
    ratio = float(gcfg.get("column_overlap_ratio", 0.4))
    support = int(gcfg.get("min_column_support", 2))

    usable = [w for w in words if (w.get("text") or "").strip()]
    if not usable:
        return [], []

    lines = _group_lines(usable, float(gcfg.get("row_overlap_ratio", 0.5)))
    tables: list[Table] = []
    abstained: list[str] = []

    for band in _geometry_bands(lines, gcfg, min_cols, min_rows):
        if len(band) < min_rows:
            continue
        # Renumber rows so they are dense inside this band.
        flat: list[_Cell] = []
        for r, (_, cells) in enumerate(band):
            for c in cells:
                c.row = r
                flat.append(c)

        columns, ambiguous = _build_columns(flat, ratio)
        where = f"page {page} rows {band[0][0]}-{band[-1][0]}"

        widest = max(len(cells) for _, cells in band)
        if len(columns) < min_cols:
            continue
        if ambiguous:
            abstained.append(f"{where}: {ambiguous} cell(s) fit more than one column")
            continue
        if len(columns) > widest:
            # More columns than the widest row has cells means the rows do not
            # actually line up. Common when text is space aligned but rendered
            # in a proportional font, where the header drifts off its data.
            abstained.append(
                f"{where}: {len(columns)} columns but the widest row holds {widest} cells"
            )
            continue
        thin = [i for i, col in enumerate(columns) if len(col) < support]
        if thin:
            abstained.append(
                f"{where}: column(s) {thin} stand on fewer than {support} cells"
            )
            continue
        if len(flat) < min_cells:
            continue

        cells = []
        for ci, col in enumerate(columns):
            for c in col:
                cells.append(_cell_from_words(c.row, ci, page, c.words, full))
        cells.sort(key=lambda c: (c.row, c.col))
        tables.append(
            _assemble(page, len(band), len(columns), TableMethod.GEOMETRY, cells, tcfg)
        )

    return tables, abstained


def _cell_from_words(row: int, col: int, page: int, words: list[dict], full: str) -> TableCell:
    """One cell, offsets taken from the words when every one of them placed.

    The text is the canonical slice, not the words joined back together, so
    the offsets verify by construction. When any word is unplaced, or the span
    would cross a line, the offsets are null rather than approximate.
    """
    starts = [w.get("char_start") for w in words]
    ends = [w.get("char_end") for w in words]
    x0 = min(w["x0"] for w in words)
    y0 = min(w["y0"] for w in words)
    x1 = max(w["x1"] for w in words)
    y1 = max(w["y1"] for w in words)

    if all(s is not None for s in starts) and all(e is not None for e in ends):
        s, e = min(starts), max(ends)
        slice_ = full[s:e]
        if "\n" not in slice_ and PAGE_SEP not in slice_:
            return TableCell(
                row=row, col=col, text=slice_, page=page,
                char_start=s, char_end=e, x0=x0, y0=y0, x1=x1, y1=y1,
            )
    return TableCell(
        row=row, col=col, text=" ".join(w["text"] for w in words), page=page,
        char_start=None, char_end=None, x0=x0, y0=y0, x1=x1, y1=y1,
    )
