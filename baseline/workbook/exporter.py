"""A results file, written out as the review workbook.

The other half of the loop. Correcting an exported row is far faster than
annotating from scratch, so until this existed the sheet had to be filled by
hand and the review effort had nowhere to come from.
"""

from __future__ import annotations

import datetime
import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..config import Config
from ..io import atomic_open

if TYPE_CHECKING:
    from openpyxl.workbook.workbook import Workbook
    from openpyxl.worksheet.worksheet import Worksheet
from .columns import (
    DOCUMENT_COLUMNS,
    LINE_ITEM_COLUMNS,
    METADATA_COLUMNS,
    PREVIEW_CHARS,
    TAG_COLUMNS,
    TAXONOMY_COLUMNS,
)
from .merge import (
    REVIEW_CELLS,
    PriorReview,
    _carry_document,
    _carry_metadata,
    _carry_tag,
    _rescue_metadata,
    _rescue_tags,
    load_prior_review,
)


def _preview(text: str) -> str:
    """A short, Excel safe look at the text. Never the text itself.

    Control characters are what make the full text unsafe to store, so they are
    the first thing to go: a form feed or a stray carriage return in a cell is
    exactly what would rewrite the string Excel hands back.
    """
    flat = " ".join(str(text or "").split())
    return flat[:PREVIEW_CHARS]


def _runner_up(scores: dict[str, Any], label: str | None) -> tuple[str, Any]:
    """The best scoring class that is not the one we picked.

    A reviewer deciding whether `purchase_order` should have been `invoice`
    wants to see what came second and by how much.
    """
    others = [(k, v) for k, v in (scores or {}).items() if k != label]
    if not others:
        return "", ""
    name, score = max(others, key=lambda kv: kv[1])
    return name, round(float(score), 4)


# Export-only, appended after every agreed column so no position moves. The
# importer and merge read by header name and never look at it.
PRIORITY_COLUMN = "REVIEW_PRIORITY"
_CHECKED_METHODS = {"regex_checksum", "regex_format"}
# Confidence edges of the review priority, matching the eval's calibration
# buckets: below the [0.7, 0.9) bucket is HIGH, the [0.9, 1.0] bucket can be LOW.
PRIORITY_HIGH_BELOW = 0.7
PRIORITY_LOW_FROM = 0.9


def review_priority(m: dict[str, Any]) -> str:
    """HIGH, MEDIUM or LOW for one field row, from the eval's confidence buckets.

    HIGH: below 0.7 (the [0.5,0.7) bucket and the unbucketed tail), or
    unvalidated with no checksum/format regex behind it. LOW: [0.9,1.0] and
    validated. Everything else MEDIUM.

    Args:
        m: One predicted metadata item.

    Returns:
        The priority label.
    """
    # ponytail: fixed bucket edges, read measured per-field precision per bucket
    # from a calibration file once one exists.
    try:
        conf = float(m["confidence"])
    except (KeyError, TypeError, ValueError):
        conf = 0.0
    validated = m.get("validated") is True
    if conf < PRIORITY_HIGH_BELOW or (not validated and m.get("method") not in _CHECKED_METHODS):
        return "HIGH"
    if conf >= PRIORITY_LOW_FROM and validated:
        return "LOW"
    return "MEDIUM"


def _write_row(ws: Worksheet, values: list[Any]) -> None:
    """Append a row, forcing every string to stay a string.

    openpyxl infers a cell's type from its value, so a string beginning with
    `=` is written as a formula. OCR text is untrusted input and does contain
    such strings, and a formula would change what the importer reads back as
    well as what Excel runs. Forcing the type keeps the value exact.
    """
    ws.append(values)
    for cell in ws[ws.max_row]:
        if isinstance(cell.value, str):
            cell.data_type = "s"


@dataclass(frozen=True)
class _Sheets:
    """The five tabs of the review workbook."""

    docs: Worksheet
    meta: Worksheet
    tags: Worksheet
    taxonomy: Worksheet
    items: Worksheet


def _new_workbook() -> tuple[Workbook, _Sheets]:
    """An empty workbook with every tab created and its header row written."""
    import openpyxl

    wb = openpyxl.Workbook()
    del wb["Sheet"]
    sheets = _Sheets(
        docs=wb.create_sheet("Documents"),
        meta=wb.create_sheet("Metadata"),
        tags=wb.create_sheet("Tags"),
        taxonomy=wb.create_sheet("Taxonomy"),
        items=wb.create_sheet("Line items"),
    )
    for ws, cols in (
        (sheets.docs, DOCUMENT_COLUMNS),
        (sheets.meta, [*METADATA_COLUMNS, PRIORITY_COLUMN]),
        (sheets.tags, TAG_COLUMNS),
        (sheets.taxonomy, TAXONOMY_COLUMNS),
        (sheets.items, LINE_ITEM_COLUMNS),
    ):
        _write_row(ws, cols)
    return wb, sheets


def _write_rescued(sheets: _Sheets, prior: PriorReview, seen_docs: set[str], counts: Counter[str]) -> None:
    """Reviewed rows the new run no longer produces.

    They are appended rather than dropped, because what the reviewer said is
    still true about the document even though the prediction that carried it
    is gone.
    """
    for _doc_id, row in _rescue_metadata(prior, seen_docs, counts):
        _write_row(sheets.meta, row)
        counts["metadata"] += 1
    for row in _rescue_tags(prior, seen_docs, counts):
        _write_row(sheets.tags, row)
        counts["tags"] += 1


def export_workbook(
    records: list[dict[str, Any]],
    cfg: Config,
    out_path: str | Path,
    tables: list[dict[str, Any]] | None = None,
    run_id: str = "",
    overwrite: bool = False,
    merge_from: str | Path | None = None,
) -> dict[str, int]:
    """Write a results file out as the review workbook.

    Args:
        records: Result records to export.
        cfg: Config, for the Taxonomy tab.
        out_path: Workbook to write.
        tables: Optional tables records, for the Line items tab.
        run_id: Run label. Defaults to one built from the current time.
        overwrite: Allow replacing an existing file.
        merge_from: An earlier reviewed workbook to carry onto this run (see merge.py).

    Returns:
        Row counts per tab, plus merge counts.

    Raises:
        FileExistsError: `out_path` exists and `overwrite` is not set.
    """
    out = Path(out_path)
    if out.exists() and not overwrite:
        # Re-exporting over a reviewed sheet destroys the review, the expensive
        # half. Pass merge_from to carry a review forward instead.
        raise FileExistsError(
            f"{out} exists. Exporting would overwrite any review already in it. "
            "Write to a new path, or pass overwrite if the file is disposable."
        )

    run_id = run_id or f"run_{datetime.datetime.now():%Y%m%dT%H%M%S}"
    extracted_at = f"{datetime.datetime.now():%Y-%m-%dT%H:%M:%S}"

    wb, sheets = _new_workbook()
    prior = load_prior_review(merge_from) if merge_from else PriorReview()
    counts: Counter[str] = Counter()
    seen_docs = _write_document_rows(records, sheets, prior, counts, run_id, extracted_at)
    _write_rescued(sheets, prior, seen_docs, counts)

    _write_taxonomy(sheets.taxonomy, cfg, counts)
    _write_line_items(sheets.items, tables, counts)

    out.parent.mkdir(parents=True, exist_ok=True)
    with atomic_open(out, mode="wb") as fh:
        wb.save(fh)
    return dict(counts)


def _document_row(
    rec: dict[str, Any], prior: PriorReview, counts: Counter[str], run_id: str, extracted_at: str
) -> list[Any]:
    """The Documents row of one record, review cells carried from the prior review."""
    doc_id = rec.get("doc_id", "")
    text = rec.get("text") or {}
    cls = rec.get("classification") or {}
    diag = rec.get("diagnostics") or {}
    full = text.get("full") or ""
    runner, runner_conf = _runner_up(cls.get("all_scores") or {}, cls.get("label"))
    errors = diag.get("errors") or []
    source = rec.get("source_path", "")
    return [
        doc_id,
        run_id,
        extracted_at,
        rec.get("schema_version", ""),
        source,
        rec.get("filename", ""),
        source.rsplit(".", 1)[-1].lower() if "." in source else "",
        diag.get("page_count", ""),
        text.get("method", ""),
        ", ".join(text.get("page_methods") or []),
        diag.get("has_native_text", ""),
        diag.get("is_scanned", ""),
        text.get("extraction_confidence", ""),
        text.get("char_count", ""),
        hashlib.sha256(full.encode("utf-8")).hexdigest(),
        _preview(full),
        cls.get("label", ""),
        cls.get("confidence", ""),
        cls.get("method", ""),
        runner,
        runner_conf,
        len(errors),
        "; ".join(errors),
        # LANGUAGE and QUALITY have no pipeline source. They are the
        # reviewer's, and inventing a value for them would be a guess
        # wearing the pipeline's clothes.
        "",
        "",
        *(
            _carry_document(prior.documents[doc_id], str(cls.get("label") or ""), counts)
            if doc_id in prior.documents
            else [""] * REVIEW_CELLS
        ),
    ]


def _metadata_row(doc_id: str, n: int, m: dict[str, Any], prior: PriorReview, counts: Counter[str]) -> list[Any]:
    """The Metadata row of one predicted field, with its review priority."""
    return [
        f"{doc_id}_{n}",
        doc_id,
        "pipeline",
        m.get("field", ""),
        m.get("value", ""),
        m.get("normalized_value", ""),
        m.get("currency", ""),
        m.get("page", ""),
        m.get("char_start", ""),
        m.get("char_end", ""),
        m.get("confidence", ""),
        # regex_checksum and regex_format stay distinct. Collapsing them
        # to "regex" reintroduces the first bug the eval harness ever
        # caught, at the human layer where nothing would catch it again.
        m.get("method", ""),
        m.get("validated", ""),
        m.get("normalizer_version", ""),
        *_carry_metadata(prior, doc_id, str(m.get("field") or "").casefold(), str(m.get("value") or ""), counts),
        review_priority(m),
    ]


def _tag_row(doc_id: str, t: dict[str, Any], prior: PriorReview, counts: Counter[str]) -> list[Any]:
    """The Tags row of one predicted tag."""
    return [
        doc_id,
        t.get("tag", ""),
        "pipeline",
        t.get("confidence", ""),
        t.get("method", ""),
        *_carry_tag(prior, doc_id, str(t.get("tag") or "").casefold(), counts),
    ]


def _write_document_rows(
    records: list[dict[str, Any]],
    sheets: _Sheets,
    prior: PriorReview,
    counts: Counter[str],
    run_id: str,
    extracted_at: str,
) -> set[str]:
    """One Documents row per record, plus its Metadata and Tags rows.

    Returns the doc_ids written, so the caller can tell which reviewed rows the
    new run no longer produces.
    """
    seen_docs: set[str] = set()
    for rec in records:
        doc_id = rec.get("doc_id", "")
        _write_row(sheets.docs, _document_row(rec, prior, counts, run_id, extracted_at))
        seen_docs.add(doc_id)
        counts["documents"] += 1

        for n, m in enumerate(rec.get("metadata") or [], start=1):
            _write_row(sheets.meta, _metadata_row(doc_id, n, m, prior, counts))
            counts["metadata"] += 1

        for t in rec.get("tags") or []:
            _write_row(sheets.tags, _tag_row(doc_id, t, prior, counts))
            counts["tags"] += 1
    return seen_docs


def _write_taxonomy(ws: Worksheet, cfg: Config, counts: Counter[str]) -> None:
    """The Taxonomy tab, generated from config and never typed.

    Typed, it drifts from the code within a month and the reviewer is checking
    against a vocabulary the pipeline no longer has.
    """
    for c in cfg.classes.get("classes", []):
        _write_row(ws, ["class", c.get("name", ""), c.get("description", ""), "", "", ""])
        counts["taxonomy"] += 1
    for f in cfg.field_defs:
        _write_row(
            ws,
            [
                "field",
                f.get("name", ""),
                f.get("description", ""),
                f.get("type", ""),
                (f.get("pattern") or {}).get("regex", ""),
                ", ".join(f.get("classes") or []) or "all",
            ],
        )
        counts["taxonomy"] += 1
    for t in cfg.tag_defs:
        _write_row(ws, ["tag", t.get("name", ""), t.get("description", ""), "bool", "", ""])
        counts["taxonomy"] += 1


def _write_line_items(ws: Worksheet, tables: list[dict[str, Any]] | None, counts: Counter[str]) -> None:
    """The Line items tab, one row per table cell."""
    for rec in tables or []:
        doc_id = rec.get("doc_id", "")
        for table in rec.get("tables") or []:
            for cell in table.get("cells") or []:
                _write_row(
                    ws,
                    [
                        doc_id,
                        table.get("table_id", ""),
                        table.get("page", ""),
                        table.get("method", ""),
                        table.get("n_rows", ""),
                        table.get("n_cols", ""),
                        cell.get("row", ""),
                        cell.get("col", ""),
                        cell.get("text", ""),
                        cell.get("char_start", ""),
                        cell.get("char_end", ""),
                        bool(table.get("header")) and cell.get("row") == 0,
                    ],
                )
                counts["line_items"] += 1
