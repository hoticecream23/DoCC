"""A results file, written out as the review workbook.

The other half of the loop. Correcting an exported row is far faster than
annotating from scratch, so until this existed the sheet had to be filled by
hand and the review effort had nowhere to come from.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from ..config import Config
from .columns import (
    DOCUMENT_COLUMNS,
    LINE_ITEM_COLUMNS,
    METADATA_COLUMNS,
    PREVIEW_CHARS,
    TAG_COLUMNS,
    TAXONOMY_COLUMNS,
)
from .merge import (
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


def _runner_up(scores: dict[str, Any], label: str) -> tuple[str, Any]:
    """The best scoring class that is not the one we picked.

    A reviewer deciding whether `purchase_order` should have been `invoice`
    wants to see what came second and by how much.
    """
    others = [(k, v) for k, v in (scores or {}).items() if k != label]
    if not others:
        return "", ""
    name, score = max(others, key=lambda kv: kv[1])
    return name, round(float(score), 4)


def _write_row(ws, values: list[Any]) -> None:
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


def export_workbook(
    records: list[dict[str, Any]],
    cfg: Config,
    out_path: str | Path,
    tables: list[dict[str, Any]] | None = None,
    run_id: str = "",
    overwrite: bool = False,
    merge_from: str | Path | None = None,
) -> dict[str, int]:
    """Write a results file out as the review workbook. Returns row counts.

    merge_from carries an earlier review onto this run. See merge.py for what
    survives a changed prediction and what does not.
    """
    import hashlib

    import openpyxl

    out = Path(out_path)
    if out.exists() and not overwrite:
        # Re-exporting over a reviewed sheet destroys the review, and the
        # review is the expensive half. Nothing here can merge corrections
        # forward yet, so refusing is the only safe answer.
        raise FileExistsError(
            f"{out} exists. Exporting would overwrite any review already in it. "
            "Write to a new path, or pass overwrite if the file is disposable."
        )

    run_id = run_id or f"run_{__import__('datetime').datetime.now():%Y%m%dT%H%M%S}"
    extracted_at = f"{__import__('datetime').datetime.now():%Y-%m-%dT%H:%M:%S}"

    wb = openpyxl.Workbook()
    del wb["Sheet"]
    docs = wb.create_sheet("Documents")
    meta = wb.create_sheet("Metadata")
    tags_ws = wb.create_sheet("Tags")
    taxo = wb.create_sheet("Taxonomy")
    items = wb.create_sheet("Line items")
    for ws, cols in (
        (docs, DOCUMENT_COLUMNS), (meta, METADATA_COLUMNS), (tags_ws, TAG_COLUMNS),
        (taxo, TAXONOMY_COLUMNS), (items, LINE_ITEM_COLUMNS),
    ):
        _write_row(ws, cols)

    prior = load_prior_review(merge_from) if merge_from else PriorReview()
    counts = Counter()
    seen_docs: set[str] = set()
    for rec in records:
        doc_id = rec.get("doc_id", "")
        text = rec.get("text") or {}
        cls = rec.get("classification") or {}
        diag = rec.get("diagnostics") or {}
        full = text.get("full") or ""
        runner, runner_conf = _runner_up(cls.get("all_scores") or {}, cls.get("label"))
        errors = diag.get("errors") or []
        source = rec.get("source_path", "")

        _write_row(docs, [
            doc_id, run_id, extracted_at, rec.get("schema_version", ""), source,
            rec.get("filename", ""),
            source.rsplit(".", 1)[-1].lower() if "." in source else "",
            diag.get("page_count", ""), text.get("method", ""),
            ", ".join(text.get("page_methods") or []),
            diag.get("has_native_text", ""), diag.get("is_scanned", ""),
            text.get("extraction_confidence", ""), text.get("char_count", ""),
            hashlib.sha256(full.encode("utf-8")).hexdigest(), _preview(full),
            cls.get("label", ""), cls.get("confidence", ""), cls.get("method", ""),
            runner, runner_conf, len(errors), "; ".join(errors),
            # LANGUAGE and QUALITY have no pipeline source. They are the
            # reviewer's, and inventing a value for them would be a guess
            # wearing the pipeline's clothes.
            "", "",
            *(_carry_document(prior.documents[doc_id], str(cls.get("label") or ""), counts)
              if doc_id in prior.documents else [""] * 6),
        ])
        seen_docs.add(doc_id)
        counts["documents"] += 1

        for n, m in enumerate(rec.get("metadata") or [], start=1):
            _write_row(meta, [
                f"{doc_id}_{n}", doc_id, "pipeline", m.get("field", ""),
                m.get("value", ""), m.get("normalized_value", ""), m.get("currency", ""),
                m.get("page", ""), m.get("char_start", ""), m.get("char_end", ""),
                m.get("confidence", ""),
                # regex_checksum and regex_format stay distinct. Collapsing them
                # to "regex" reintroduces the first bug the eval harness ever
                # caught, at the human layer where nothing would catch it again.
                m.get("method", ""),
                m.get("validated", ""), m.get("normalizer_version", ""),
                *_carry_metadata(prior, doc_id, str(m.get("field") or "").casefold(),
                                 str(m.get("value") or ""), counts),
            ])
            counts["metadata"] += 1

        for t in rec.get("tags") or []:
            _write_row(tags_ws, [
                doc_id, t.get("tag", ""), "pipeline",
                t.get("confidence", ""), t.get("method", ""),
                *_carry_tag(prior, doc_id, str(t.get("tag") or "").casefold(), counts),
            ])
            counts["tags"] += 1

    # Reviewed rows the new run no longer produces. They are appended rather
    # than dropped, because what the reviewer said is still true about the
    # document even though the prediction that carried it is gone.
    for _doc_id, row in _rescue_metadata(prior, seen_docs, counts):
        _write_row(meta, row)
        counts["metadata"] += 1
    for row in _rescue_tags(prior, seen_docs, counts):
        _write_row(tags_ws, row)
        counts["tags"] += 1

    # The Taxonomy tab is generated, never typed. Typed, it drifts from the
    # code within a month and the reviewer is checking against a vocabulary
    # the pipeline no longer has.
    for c in cfg.classes.get("classes", []):
        _write_row(taxo, ["class", c.get("name", ""), c.get("description", ""), "", "", ""])
        counts["taxonomy"] += 1
    for f in cfg.field_defs:
        _write_row(taxo, [
            "field", f.get("name", ""), f.get("description", ""), f.get("type", ""),
            (f.get("pattern") or {}).get("regex", ""),
            ", ".join(f.get("classes") or []) or "all",
        ])
        counts["taxonomy"] += 1
    for t in cfg.tag_defs:
        _write_row(taxo, ["tag", t.get("name", ""), t.get("description", ""), "bool", "", ""])
        counts["taxonomy"] += 1

    for rec in tables or []:
        doc_id = rec.get("doc_id", "")
        for table in rec.get("tables") or []:
            for cell in table.get("cells") or []:
                _write_row(items, [
                    doc_id, table.get("table_id", ""), table.get("page", ""),
                    table.get("method", ""), table.get("n_rows", ""), table.get("n_cols", ""),
                    cell.get("row", ""), cell.get("col", ""), cell.get("text", ""),
                    cell.get("char_start", ""), cell.get("char_end", ""),
                    bool(table.get("header")) and cell.get("row") == 0,
                ])
                counts["line_items"] += 1

    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
    return dict(counts)
