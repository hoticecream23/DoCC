"""One document through the strategies, and the whole pass over a results file."""

from __future__ import annotations

import gzip
import json
import time
from pathlib import Path
from typing import Any

from ..errors import summarize_exception
from ..io import atomic_open, atomic_write_text
from ..logging_setup import get_logger
from ..schema import Table, TableDiagnostics, TableRecord, page_spans, read_jsonl
from .geometry import detect_geometry
from .tab_delimited import detect_tab_delimited
from .text_grid import detect_text_grid

log = get_logger(__name__)

Word = dict[str, Any]
# Summary counts shown, in this order, at the top of the report.
SUMMARY_KEYS = ("documents", "documents_with_tables", "tables", "cells", "placed_cells", "abstentions")

# --------------------------------------------------------------------------
# per document
# --------------------------------------------------------------------------


def _attach_boxes(tables: list[Table], words: list[Word]) -> None:
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
            inside = [w for w in placed if w["char_start"] >= c.char_start and w["char_end"] <= c.char_end]
            if not inside:
                continue
            c.x0 = min(w["x0"] for w in inside)
            c.y0 = min(w["y0"] for w in inside)
            c.x1 = max(w["x1"] for w in inside)
            c.y1 = max(w["y1"] for w in inside)


def _run_strategy(
    name: str, found: list[Table], words: list[Word], text: dict[str, Any], tcfg: dict[str, Any], diag: TableDiagnostics
) -> bool:
    """Run one strategy over every page, appending into `found`.

    Appends as it goes, so pages done before a failure are kept.

    Returns:
        False when the strategy name is unknown.
    """
    full = str(text.get("full", ""))
    pages = list(text.get("pages", []) or [])
    if name == "geometry":
        for pno in range(len(pages)):
            page_words = [w for w in words if w.get("page") == pno]
            got, abst = detect_geometry(pno, page_words, full, tcfg)
            found.extend(got)
            diag.abstained.extend(abst)
        return True
    if name in ("text_grid", "tab_delimited"):
        spans = page_spans(pages)
        detect = detect_text_grid if name == "text_grid" else detect_tab_delimited
        for pno, body in enumerate(pages):
            start = spans[pno][0] if pno < len(spans) else 0
            got, abst = detect(pno, body, start, tcfg)
            found.extend(got)
            diag.abstained.extend(abst)
        _attach_boxes(found, words)
        return True
    return False


def build_record(
    record: dict[str, Any], words: list[Word], tcfg: dict[str, Any], deterministic_timings: bool = False
) -> TableRecord:
    """All tables in one document. Strategies in config order, first hit wins.

    A strategy that raises is recorded in the diagnostics, not raised.

    Args:
        record: One document record from the results file.
        words: Word boxes of the document, possibly empty.
        tcfg: Tables config.
        deterministic_timings: Zero the timings, for reproducible output.

    Returns:
        The document's tables with diagnostics and offset checks.
    """
    doc_id = str(record.get("doc_id", ""))
    text = record.get("text", {}) or {}
    full = str(text.get("full", ""))

    diag = TableDiagnostics()
    tables: list[Table] = []

    for name in list(tcfg.get("strategies", []) or []):
        t0 = time.perf_counter()
        found: list[Table] = []
        try:
            if not _run_strategy(name, found, words, text, tcfg, diag):
                diag.errors.append(f"unknown strategy {name!r}")
                continue
        except Exception as exc:  # noqa: BLE001 - boundary: a bad page never kills the batch
            log.error(
                "table strategy failed", extra={"doc_id": doc_id, "strategy": name, "error": summarize_exception(exc)}
            )
            diag.errors.append(f"{name}: {summarize_exception(exc)}")
        finally:
            elapsed = (time.perf_counter() - t0) * 1000.0
            # Same knob as the pipeline: timings are the one field that cannot
            # be reproducible, so there is a switch to zero them.
            diag.timings_ms[name] = 0.0 if deterministic_timings else round(elapsed, 3)
        diag.strategies_tried.append(name)
        if found:
            tables = found
            break

    return _finish_record(record, tables, diag, full)


def _finish_record(record: dict[str, Any], tables: list[Table], diag: TableDiagnostics, full: str) -> TableRecord:
    """Number the tables, wrap them as a record, and verify their offsets."""
    doc_id = str(record.get("doc_id", ""))
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
        log.error("table offset verification failed", extra={"doc_id": doc_id, "n": len(problems)})
        rec.diagnostics.errors.extend(problems)
    return rec


# --------------------------------------------------------------------------
# io and reporting
# --------------------------------------------------------------------------


load_results = read_jsonl


def load_words(bbox_dir: Path, doc_id: str) -> list[Word]:
    """Word boxes for one document. Missing sidecar is normal, not an error.

    Args:
        bbox_dir: Directory of `<doc_id>.json.gz` sidecars.
        doc_id: Document to load.

    Returns:
        The words, or an empty list when the sidecar is missing or damaged.
    """
    p = bbox_dir / f"{doc_id}.json.gz"
    if not p.exists():
        return []
    try:
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            return list(json.load(fh).get("words", []))
    except (OSError, EOFError, ValueError, AttributeError) as exc:  # damaged gzip/JSON: no boxes, still logged
        log.error("cannot read sidecar", extra={"path": str(p), "error": summarize_exception(exc)})
        return []


def render_report(records: list[TableRecord], stats: dict[str, Any], source: str) -> str:
    """Render a tables pass as markdown.

    Args:
        records: One TableRecord per document.
        stats: Summary counts from `build`.
        source: Results path, shown in the header.

    Returns:
        The report text.
    """
    out = ["# Tables", "", f"Source: `{source}`", "", "| metric | value |", "| --- | --- |"]
    out += [f"| {k.replace('_', ' ')} | {stats[k]} |" for k in SUMMARY_KEYS]
    out += ["", "## By strategy", "", "| strategy | tables |", "| --- | --- |"]
    out += [f"| {k} | {v} |" for k, v in sorted(stats["by_method"].items())]
    out.append("")
    out += _report_tables_found(records)
    out += _report_abstentions(records)
    return "\n".join(out)


def _render_table(t: Table) -> list[str]:
    """One table as a markdown grid, header rule after row 0 when it has one."""
    out: list[str] = []
    grid = {(c.row, c.col): c.text for c in t.cells}
    for r in range(t.n_rows):
        cells = [grid.get((r, c), "").replace("|", "\\|") for c in range(t.n_cols)]
        out.append("| " + " | ".join(cells) + " |")
        if r == 0 and t.header:
            out.append("| " + " | ".join("---" for _ in range(t.n_cols)) + " |")
    return out


def _report_tables_found(records: list[TableRecord]) -> list[str]:
    """Every table found, document by document."""
    out: list[str] = []
    a = out.append
    a("## Tables found")
    a("")
    any_table = False
    for rec in records:
        for t in rec.tables:
            any_table = True
            a(f"### `{rec.filename}` page {t.page}")
            a("")
            a(
                f"{t.n_rows} rows x {t.n_cols} cols, {t.method.value}, "
                f"fill {t.confidence}, span {t.char_start}-{t.char_end}"
            )
            a("")
            out += _render_table(t)
            a("")
    if not any_table:
        a("None.")
        a("")
    return out


def _report_abstentions(records: list[TableRecord]) -> list[str]:
    """Abstentions table, then errors when there are any."""
    out: list[str] = []
    a = out.append
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
    return out


def build(
    input_path: str | Path,
    output_path: str | Path,
    tcfg: dict[str, Any],
    bbox_dir: str | Path | None = None,
    report_path: str | Path | None = None,
    deterministic_timings: bool = False,
) -> dict[str, Any]:
    """Whole pass: results in, one TableRecord per document out.

    Args:
        input_path: Results JSONL.
        output_path: Tables JSONL to write, atomically.
        tcfg: Tables config.
        bbox_dir: Word box sidecars. Defaults to `<input>.bboxes`.
        report_path: Optional markdown report to write.
        deterministic_timings: Zero the timings, for reproducible output.

    Returns:
        `stats` and the `records` written.
    """
    inp = Path(input_path)
    bdir = Path(bbox_dir) if bbox_dir else Path(str(inp) + ".bboxes")

    records = []
    for row in load_results(inp):
        words = load_words(bdir, str(row.get("doc_id", "")))
        records.append(build_record(row, words, tcfg, deterministic_timings))

    stats = _stats(records)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with atomic_open(out, newline="\n") as fh:
        for r in records:
            fh.write(json.dumps(r.model_dump(mode="json"), ensure_ascii=False) + "\n")

    if report_path:
        rp = Path(report_path)
        rp.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(rp, render_report(records, stats, str(inp)))

    return {"stats": stats, "records": records}


def _stats(records: list[TableRecord]) -> dict[str, Any]:
    """Summary counts over a whole pass."""
    by_method: dict[str, int] = {}
    for r in records:
        for t in r.tables:
            by_method[t.method.value] = by_method.get(t.method.value, 0) + 1
    return {
        "documents": len(records),
        "documents_with_tables": sum(1 for r in records if r.tables),
        "tables": sum(len(r.tables) for r in records),
        "cells": sum(len(t.cells) for r in records for t in r.tables),
        "placed_cells": sum(1 for r in records for t in r.tables for c in t.cells if c.char_start is not None),
        "abstentions": sum(len(r.diagnostics.abstained) for r in records),
        "errors": sum(len(r.diagnostics.errors) for r in records),
        "by_method": by_method,
    }
