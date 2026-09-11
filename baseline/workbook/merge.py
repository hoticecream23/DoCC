"""Carrying an earlier review onto a new run.

Re-running the pipeline produces a new results file, and without this the
only way to carry an existing review onto it is by hand. That is what stands
between one reviewed sheet and a growing gold set.

The whole design turns on one distinction:

  A correction is a fact about the DOCUMENT.  "The total is 4044." That stays
  true no matter what the pipeline says next, so it always carries forward.

  A confirmation is a fact about a PREDICTION. "What you said is right." That
  is only meaningful while the prediction is the same, so it carries forward
  only when the new run says the same thing.

Getting that backwards in either direction is a silent data problem. Carrying
a confirmation onto a changed prediction marks a new, unreviewed answer as
human approved. Dropping a correction because the prediction moved throws
away the expensive half of the work.

A third case falls out of the same rule. When a confirmed value disappears
from the new run, the confirmation was still a statement that the value
belongs on the document, so it is rewritten as a human `missing` row rather
than discarded. The reviewer's finding survives the prediction that carried
it.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .columns import VERDICTS, _get, _read_tab


@dataclass
class PriorReview:
    """Everything a reviewer wrote, indexed for lookup against a new run."""

    documents: dict[str, dict[str, str]] = field(default_factory=dict)
    # Keyed on the exact value the reviewer was looking at, because that is
    # what a confirmation is about.
    metadata: dict[tuple[str, str, str], dict[str, str]] = field(default_factory=dict)
    # Same rows again, keyed only by field, so a correction can still be found
    # when the pipeline's value has moved underneath it.
    metadata_by_field: dict[tuple[str, str], list[dict[str, str]]] = field(default_factory=dict)
    tags: dict[tuple[str, str], dict[str, str]] = field(default_factory=dict)
    # Row ids already carried onto the new run, so nothing is used twice and
    # whatever is left over is exactly what the new run no longer produces.
    matched: set[int] = field(default_factory=set)
    matched_tags: set[tuple[str, str]] = field(default_factory=set)

    def rows(self) -> list[dict[str, str]]:
        out: list[dict[str, str]] = list(self.documents.values())
        for rows in self.metadata_by_field.values():
            out.extend(rows)
        out.extend(self.tags.values())
        return out


def _reviewed(row: dict[str, str]) -> bool:
    """Did a human actually write anything on this row."""
    return bool(row.get("VERDICT") or row.get("CLASS_VERDICT")
                or row.get("CORRECTED_VALUE") or row.get("CLASS_CORRECTED")
                or row.get("IS_GOLD"))


def load_prior_review(path: str | Path) -> PriorReview:
    """Index the reviewed columns of an existing workbook."""
    import openpyxl

    prior = PriorReview()
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    try:
        doc_i, doc_rows = _read_tab(wb, "Documents")
        meta_i, meta_rows = _read_tab(wb, "Metadata")
        tag_i, tag_rows = _read_tab(wb, "Tags")
    finally:
        wb.close()

    def as_dict(row, index):
        return {name: _get(row, index, name) for name in index}

    for row in doc_rows:
        data = as_dict(row, doc_i)
        if data.get("DOC_ID") and _reviewed(data):
            prior.documents[data["DOC_ID"]] = data

    for n, row in enumerate(meta_rows):
        data = as_dict(row, meta_i)
        doc_id, name = data.get("DOC_ID"), data.get("FIELD", "").casefold()
        if not doc_id or not name or not _reviewed(data):
            continue
        data["_id"] = n
        prior.metadata[(doc_id, name, data.get("VALUE", ""))] = data
        prior.metadata_by_field.setdefault((doc_id, name), []).append(data)

    for row in tag_rows:
        data = as_dict(row, tag_i)
        doc_id, name = data.get("DOC_ID"), data.get("TAG", "").casefold()
        if doc_id and name and _reviewed(data):
            prior.tags[(doc_id, name)] = data
    return prior


def _note(prior_note: str, added: str) -> str:
    """Append a machine written note without losing the reviewer's own."""
    prior_note = (prior_note or "").strip()
    return f"{prior_note} | {added}" if prior_note else added


def _carry_document(prior_row: dict[str, str], new_label: str,
                    counts: Counter) -> list[Any]:
    """The six review cells of a Documents row, carried or invalidated."""
    verdict = (prior_row.get("CLASS_VERDICT") or "").strip()
    corrected = (prior_row.get("CLASS_CORRECTED") or "").strip()
    was = (prior_row.get("DOC_CLASS") or "").strip()
    who, when = prior_row.get("REVIEWED_BY", ""), prior_row.get("REVIEW_DATE", "")
    notes = prior_row.get("NOTES", "")

    if corrected:
        # A correction is the reviewer's own statement of the class. It holds
        # whatever the pipeline now says.
        if new_label.casefold() == corrected.casefold():
            # The pipeline has caught up. There is nothing left to correct.
            counts["document_corrections_now_agreed"] += 1
            return ["", "OK", prior_row.get("IS_GOLD", ""), who, when,
                    _note(notes, f"pipeline now agrees, was {was or 'unset'}")]
        counts["document_corrections_carried"] += 1
        return [corrected, verdict or "wrong", prior_row.get("IS_GOLD", ""), who, when, notes]

    if was and new_label and was.casefold() != new_label.casefold():
        # A confirmation of a prediction that has since changed. The approval
        # cannot transfer to an answer nobody has looked at.
        counts["document_confirmations_invalidated"] += 1
        return ["", "", "", who, when,
                _note(notes, f"re-review: was {was}, confirmed; now {new_label}")]

    counts["document_confirmations_carried"] += 1
    return ["", verdict, prior_row.get("IS_GOLD", ""), who, when, notes]


def _carry_metadata(prior: PriorReview, doc_id: str, name: str, value: str,
                    counts: Counter) -> list[Any]:
    """The six review cells of a Metadata row.

    Exact match first, on the value the reviewer was actually looking at. Only
    a correction falls back to the field, because a correction is about the
    document and survives the pipeline changing its answer.
    """
    exact = prior.metadata.get((doc_id, name, value))
    if exact is not None:
        prior.matched.add(exact["_id"])
        verdict = (exact.get("VERDICT") or "").strip()
        corrected = (exact.get("CORRECTED_VALUE") or "").strip()
        counts["field_reviews_carried"] += 1
        return [corrected, verdict, exact.get("IS_GOLD", ""),
                exact.get("REVIEWED_BY", ""), exact.get("REVIEW_DATE", ""),
                exact.get("NOTES", "")]

    for row in prior.metadata_by_field.get((doc_id, name), []):
        if row["_id"] in prior.matched:
            continue
        corrected = (row.get("CORRECTED_VALUE") or "").strip()
        if not corrected:
            continue
        prior.matched.add(row["_id"])
        if corrected == value:
            # The pipeline now produces what the reviewer asked for.
            counts["field_corrections_now_agreed"] += 1
            return ["", "OK", row.get("IS_GOLD", ""), row.get("REVIEWED_BY", ""),
                    row.get("REVIEW_DATE", ""),
                    _note(row.get("NOTES", ""),
                          f"pipeline now agrees, was {row.get('VALUE') or 'unset'}")]
        counts["field_corrections_carried"] += 1
        return [corrected, row.get("VERDICT") or "wrong", row.get("IS_GOLD", ""),
                row.get("REVIEWED_BY", ""), row.get("REVIEW_DATE", ""),
                _note(row.get("NOTES", ""),
                      f"correction kept, pipeline was {row.get('VALUE') or 'unset'}")]

    return [""] * 6


def _carry_tag(prior: PriorReview, doc_id: str, name: str, counts: Counter) -> list[Any]:
    """The four review cells of a Tags row. A tag has no value, so the tag
    name is the whole of what was reviewed and a match is always exact."""
    row = prior.tags.get((doc_id, name))
    if row is None:
        return [""] * 4
    prior.matched_tags.add((doc_id, name))
    counts["tag_reviews_carried"] += 1
    return [row.get("VERDICT", ""), row.get("IS_GOLD", ""),
            row.get("REVIEWED_BY", ""), row.get("REVIEW_DATE", "")]


def _rescue_metadata(prior: PriorReview, doc_ids: set[str],
                     counts: Counter) -> list[tuple[str, list[Any]]]:
    """Reviewed field rows the new run no longer produces.

    Both a correction and a confirmation say something true about the document,
    so neither is discarded when the prediction that carried it disappears.
    They come back as human `missing` rows holding whichever value the reviewer
    stood behind, which is exactly what the importer turns into gold.

    A `spurious` row is the exception, and the happy one: the reviewer said the
    field did not belong and the pipeline has stopped emitting it. There is
    nothing left to review.
    """
    out: list[tuple[str, list[Any]]] = []
    for (doc_id, name), rows in sorted(prior.metadata_by_field.items()):
        if doc_id not in doc_ids:
            continue
        for row in rows:
            if row["_id"] in prior.matched:
                continue
            verdict = (row.get("VERDICT") or "").strip().casefold()
            if VERDICTS.get(verdict) == "spurious":
                counts["field_spurious_resolved"] += 1
                continue
            value = (row.get("CORRECTED_VALUE") or "").strip() \
                or (row.get("NORMALIZED_VALUE") or "").strip() \
                or (row.get("VALUE") or "").strip()
            if not value:
                continue
            counts["field_reviews_rescued"] += 1
            out.append((doc_id, [
                f"{doc_id}_human_{name}", doc_id, "human", name, value, value,
                "", "", "", "", "", "human", "", "",
                value, "missing", row.get("IS_GOLD", ""), row.get("REVIEWED_BY", ""),
                row.get("REVIEW_DATE", ""),
                _note(row.get("NOTES", ""),
                      "kept from an earlier review; the pipeline no longer finds this"),
            ]))
    return out


def _rescue_tags(prior: PriorReview, doc_ids: set[str],
                 counts: Counter) -> list[list[Any]]:
    """Confirmed tags the new run has stopped emitting.

    A confirmed tag is a statement that it belongs on the document, so it comes
    back as a human row rather than vanishing with the prediction.
    """
    out: list[list[Any]] = []
    for (doc_id, name), row in sorted(prior.tags.items()):
        if doc_id not in doc_ids or (doc_id, name) in prior.matched_tags:
            continue
        verdict = VERDICTS.get((row.get("VERDICT") or "").strip().casefold())
        if verdict in ("wrong", "spurious"):
            counts["tag_spurious_resolved"] += 1
            continue
        if verdict != "ok":
            continue
        counts["tag_reviews_rescued"] += 1
        out.append([
            doc_id, name, "human", "", "human", "missing", row.get("IS_GOLD", ""),
            row.get("REVIEWED_BY", ""), row.get("REVIEW_DATE", ""),
        ])
    return out
