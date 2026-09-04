"""Table and line item extraction.

A separate pass over a results file plus its bbox sidecar, exactly like the
knowledge graph. It never touches DocumentRecord, so the record contract stays
frozen and tables can be rebuilt from any run at any time.

Two strategies, run in config order, first hit wins per document:

  geometry   Columns from word bounding boxes. The right signal for real PDFs
             and scans, where x positions are what the layout actually is.
  text_grid  Columns from runs of whitespace that are blank on every line of a
             band. The only signal available for .txt and .docx, which carry
             no geometry at all.

The v0 rule, borrowed from the graph: a wrong table is worse than a missing
one. Every structural check is a reason to abstain, and every abstention is
recorded with its reason rather than swallowed.
"""

from __future__ import annotations

import gzip
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .logging_setup import get_logger
from .schema import (
    PAGE_SEP,
    Table,
    TableCell,
    TableDiagnostics,
    TableMethod,
    TableRecord,
    page_spans,
)

log = get_logger(__name__)


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
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


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
# geometry strategy
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


# --------------------------------------------------------------------------
# text grid strategy
# --------------------------------------------------------------------------


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


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------


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


# --------------------------------------------------------------------------
# per document
# --------------------------------------------------------------------------


def _attach_boxes(tables: list[Table], words: list[dict]) -> None:
    """Give text_grid cells geometry, where the document has any.

    Derived from offsets the cell already owns, so nothing is guessed: a word
    is in the cell only if its span sits inside the cell's span.
    """
    placed = sorted(
        (w for w in words if w.get("char_start") is not None and w.get("char_end") is not None),
        key=lambda w: (w["char_start"], w["char_end"]),
    )
    if not placed:
        return
    for t in tables:
        for c in t.cells:
            if c.char_start is None or c.x0 is not None:
                continue
            inside = [
                w for w in placed
                if w["char_start"] >= c.char_start and w["char_end"] <= c.char_end
            ]
            if not inside:
                continue
            c.x0 = min(w["x0"] for w in inside)
            c.y0 = min(w["y0"] for w in inside)
            c.x1 = max(w["x1"] for w in inside)
            c.y1 = max(w["y1"] for w in inside)


def build_record(
    record: dict, words: list[dict], tcfg: dict, deterministic_timings: bool = False
) -> TableRecord:
    """All tables in one document. Strategies in config order, first hit wins."""
    doc_id = str(record.get("doc_id", ""))
    text = record.get("text", {}) or {}
    full = str(text.get("full", ""))
    pages = list(text.get("pages", []) or [])
    spans = page_spans(pages)

    diag = TableDiagnostics()
    tables: list[Table] = []

    for name in list(tcfg.get("strategies", []) or []):
        t0 = time.perf_counter()
        found: list[Table] = []
        try:
            if name == "geometry":
                for pno in range(len(pages)):
                    page_words = [w for w in words if w.get("page") == pno]
                    got, abst = detect_geometry(pno, page_words, full, tcfg)
                    found.extend(got)
                    diag.abstained.extend(abst)
            elif name == "text_grid":
                for pno, body in enumerate(pages):
                    start = spans[pno][0] if pno < len(spans) else 0
                    got, abst = detect_text_grid(pno, body, start, tcfg)
                    found.extend(got)
                    diag.abstained.extend(abst)
                _attach_boxes(found, words)
            else:
                diag.errors.append(f"unknown strategy {name!r}")
                continue
        except Exception as exc:  # a bad page never kills the batch
            log.exception("table strategy failed", extra={"doc_id": doc_id, "strategy": name})
            diag.errors.append(f"{name}: {type(exc).__name__}: {exc}")
        finally:
            elapsed = (time.perf_counter() - t0) * 1000.0
            # Same knob as the pipeline: timings are the one field that cannot
            # be reproducible, so there is a switch to zero them.
            diag.timings_ms[name] = 0.0 if deterministic_timings else round(elapsed, 3)
        diag.strategies_tried.append(name)
        if found:
            tables = found
            break

    for i, t in enumerate(tables):
        t.table_id = f"{doc_id}-p{t.page}-t{i}"

    rec = TableRecord(
        doc_id=doc_id,
        source_path=str(record.get("source_path", "")),
        filename=str(record.get("filename", "")),
        tables=tables,
        diagnostics=diag,
    )
    # Same discipline as the document record. An offset that does not slice
    # back is a bug, and it gets recorded rather than shipped silently.
    problems = rec.verify_offsets(full)
    if problems:
        log.error("table offset verification failed",
                  extra={"doc_id": doc_id, "n": len(problems)})
        rec.diagnostics.errors.extend(problems)
    return rec


# --------------------------------------------------------------------------
# io and reporting
# --------------------------------------------------------------------------


def load_results(path: str | Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_words(bbox_dir: Path, doc_id: str) -> list[dict]:
    """Word boxes for one document. Missing sidecar is normal, not an error."""
    p = bbox_dir / f"{doc_id}.json.gz"
    if not p.exists():
        return []
    try:
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            return list(json.load(fh).get("words", []))
    except Exception as exc:
        log.error("cannot read sidecar", extra={"path": str(p), "error": str(exc)})
        return []


def render_report(records: list[TableRecord], stats: dict, source: str) -> str:
    out: list[str] = []
    a = out.append
    a("# Tables")
    a("")
    a(f"Source: `{source}`")
    a("")
    a("| metric | value |")
    a("| --- | --- |")
    for k in ("documents", "documents_with_tables", "tables", "cells", "placed_cells",
              "abstentions"):
        a(f"| {k.replace('_', ' ')} | {stats[k]} |")
    a("")
    a("## By strategy")
    a("")
    a("| strategy | tables |")
    a("| --- | --- |")
    for k, v in sorted(stats["by_method"].items()):
        a(f"| {k} | {v} |")
    a("")

    a("## Tables found")
    a("")
    any_table = False
    for rec in records:
        for t in rec.tables:
            any_table = True
            a(f"### `{rec.filename}` page {t.page}")
            a("")
            a(f"{t.n_rows} rows x {t.n_cols} cols, {t.method.value}, "
              f"fill {t.confidence}, span {t.char_start}-{t.char_end}")
            a("")
            grid = {(c.row, c.col): c.text for c in t.cells}
            for r in range(t.n_rows):
                cells = [grid.get((r, c), "").replace("|", "\\|") for c in range(t.n_cols)]
                a("| " + " | ".join(cells) + " |")
                if r == 0 and t.header:
                    a("| " + " | ".join("---" for _ in range(t.n_cols)) + " |")
            a("")
    if not any_table:
        a("None.")
        a("")

    a("## Abstentions")
    a("")
    a("A band that looked like a table and failed a structural check. These are")
    a("findings, not noise: v0 would rather emit nothing than emit a wrong grid.")
    a("")
    rows = [(rec.filename, why) for rec in records for why in rec.diagnostics.abstained]
    if rows:
        a("| document | reason |")
        a("| --- | --- |")
        for fn, why in rows:
            a(f"| `{fn}` | {why} |")
    else:
        a("None.")
    a("")

    errs = [(rec.filename, e) for rec in records for e in rec.diagnostics.errors]
    if errs:
        a("## Errors")
        a("")
        for fn, e in errs:
            a(f"- `{fn}`: {e}")
        a("")
    return "\n".join(out)


def build(
    input_path: str | Path,
    output_path: str | Path,
    tcfg: dict,
    bbox_dir: str | Path | None = None,
    report_path: str | Path | None = None,
    deterministic_timings: bool = False,
) -> dict:
    """Whole pass: results in, one TableRecord per document out."""
    inp = Path(input_path)
    bdir = Path(bbox_dir) if bbox_dir else Path(str(inp) + ".bboxes")

    records = []
    for row in load_results(inp):
        words = load_words(bdir, str(row.get("doc_id", "")))
        records.append(build_record(row, words, tcfg, deterministic_timings))

    stats = {
        "documents": len(records),
        "documents_with_tables": sum(1 for r in records if r.tables),
        "tables": sum(len(r.tables) for r in records),
        "cells": sum(len(t.cells) for r in records for t in r.tables),
        "placed_cells": sum(
            1 for r in records for t in r.tables for c in t.cells if c.char_start is not None
        ),
        "abstentions": sum(len(r.diagnostics.abstained) for r in records),
        "errors": sum(len(r.diagnostics.errors) for r in records),
        "by_method": {},
    }
    for r in records:
        for t in r.tables:
            stats["by_method"][t.method.value] = stats["by_method"].get(t.method.value, 0) + 1

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        for r in records:
            fh.write(json.dumps(r.model_dump(mode="json"), ensure_ascii=False) + "\n")

    if report_path:
        rp = Path(report_path)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(render_report(records, stats, str(inp)), encoding="utf-8")

    return {"stats": stats, "records": records}
