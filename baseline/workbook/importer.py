# Over 400 lines because the module docstring is the written contract for
# how a reviewed sheet becomes gold (about 40 lines). The code itself is
# read, parse and merge steps that only make sense together.
"""The reviewed workbook, read back in as gold.

The workbook is the human facing mirror of `results.jsonl`: every document,
its class, its fields and its tags, exported so that non engineers can correct
them. Correcting an exported row is far faster than annotating from scratch,
which makes the sheet the annotation tool and this module the piece that turns
that review effort into a score.

Four things about the sheet shape the code more than the format does.

1. **A correction never overwrites a pipeline value.** Each tab carries the
   pipeline's answer, a `*_CORRECTED` column and a `VERDICT`, so the truth for
   a row is a function of the verdict rather than a single cell. `ok` means the
   pipeline was right and its own value is gold. `wrong` and `missing` mean the
   correction is gold. `spurious` means the pipeline invented something and the
   gold is its **absence**, so the row must produce nothing at all.

2. **A corrected value goes through the same normalizer as a predicted one.**
   The reviewer types what they see, so a corrected total reads `742.7` and a
   corrected date reads `12/22/2025`, while the pipeline emits `742.70` and
   `2025-12-22`. Comparing the two raw scores the pipeline wrong for being
   right, so corrections are normalised through the field's own declared
   normalizer before they become gold. Confirmed values are already normalised
   and are left alone.

3. **Only `ok` rows can carry offsets.** The pipeline drops any field whose
   offsets do not slice back to its value, so a confirmed value implies its
   span points at that value. A corrected value has no span: the reviewer typed
   an answer, not a location.

4. **What the sheet cannot say is as important as what it can.** Rows exist
   only where the pipeline produced something, unless a human added one. So a
   field or tag the pipeline never emitted leaves no trace, and gold built from
   this sheet can measure precision honestly but not recall. `render_summary`
   says so in the report rather than leaving the next reader to infer it.
   The summary itself lives in `summary.py`.

Names are read through the config the same way `labels.py` reads client labels
through `taxonomy.yaml`: a class, field or tag that is not declared is a loud
failure, never a quiet guess. `Aadhar_card` is not `aadhaar`, and deciding that
it is belongs to a human.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import Config
from ..logging_setup import get_logger
from ..metadata import NormCtx, normalize
from ..schema import compute_doc_id
from .columns import TRUTHY, VERDICTS, Row, _cell, _get, _read_tab

log = get_logger(__name__)

# One gold document: doc_id, label, metadata, tags, and optional path and filename.
GoldRow = dict[str, Any]


@dataclass
class WorkbookImport:
    """Gold rows read from a workbook, plus everything that did not import."""

    rows: list[GoldRow] = field(default_factory=list)
    # Everything the caller has to be told about rather than have guessed at.
    unknown_classes: list[str] = field(default_factory=list)
    unknown_fields: list[str] = field(default_factory=list)
    unknown_tags: list[str] = field(default_factory=list)
    unknown_verdicts: list[str] = field(default_factory=list)
    unresolved_doc_ids: list[str] = field(default_factory=list)
    confirmed_abstentions: list[str] = field(default_factory=list)
    orphan_rows: list[str] = field(default_factory=list)
    missing_correction: list[str] = field(default_factory=list)
    skipped_not_gold: int = 0
    spurious_rows: int = 0
    human_rows: int = 0
    counts: Counter[str] = field(default_factory=Counter)

    @property
    def problems(self) -> list[str]:
        """Anything that means the import should not be trusted as it stands."""
        out = []
        if self.unknown_classes:
            out.append(f"classes not in classes.yaml: {sorted(set(self.unknown_classes))}")
        if self.unknown_fields:
            out.append(f"fields not in fields.yaml: {sorted(set(self.unknown_fields))}")
        if self.unknown_tags:
            out.append(f"tags not in tags.yaml: {sorted(set(self.unknown_tags))}")
        if self.unknown_verdicts:
            out.append(f"unrecognised verdicts: {sorted(set(self.unknown_verdicts))}")
        return out


def _verdict(value: Any, res: WorkbookImport, where: str) -> str:
    """Normalise a verdict, or record that it was not one of the four."""
    raw = _cell(value)
    if not raw:
        return ""
    key = raw.casefold()
    if key not in VERDICTS:
        res.unknown_verdicts.append(f"{where}: {raw}")
        return ""
    return VERDICTS[key]


def _is_gold(value: Any) -> bool:
    """IS_GOLD cell reads as true."""
    return _cell(value).casefold() in TRUTHY


def _doc_id_to_path(root: Path) -> dict[str, str]:
    """Content address every file under root, so a stale row is caught loudly.

    The sheet's own SOURCE_PATH column points at wherever the reviewer ran the
    pipeline, so it cannot be trusted to locate anything here. doc_id can.
    """
    out: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            out.setdefault(compute_doc_id(path), str(path).replace("\\", "/"))
    return out


def _offsets(row: Row, index: dict[str, int]) -> dict[str, int]:
    """Page and span. Both ends of the span or neither, never one."""

    def as_int(column: str) -> int | None:
        raw = _get(row, index, column)
        try:
            return int(float(raw))
        except ValueError:
            return None

    out: dict[str, int] = {}
    page = as_int("PAGE")
    if page is not None:
        out["page"] = page
    start, end = as_int("CHAR_START"), as_int("CHAR_END")
    if start is not None and end is not None and end > start >= 0:
        out["char_start"] = start
        out["char_end"] = end
    return out


@dataclass
class _Tab:
    """One tab of the workbook: header index and non empty rows."""

    index: dict[str, int]
    rows: list[Row]


def _read_workbook(xlsx_path: str | Path) -> tuple[_Tab, _Tab, _Tab]:
    """Read step: the Documents, Metadata and Tags tabs, workbook closed after."""
    import openpyxl

    wb = openpyxl.load_workbook(xlsx_path, data_only=True, read_only=True)
    try:
        return (
            _Tab(*_read_tab(wb, "Documents")),
            _Tab(*_read_tab(wb, "Metadata")),
            _Tab(*_read_tab(wb, "Tags")),
        )
    finally:
        wb.close()


def _merge_gold(gold: dict[str, GoldRow], res: WorkbookImport) -> None:
    """Merge step: sort each gold row's contents and the rows themselves."""
    for row in gold.values():
        row["tags"].sort()
        row["metadata"].sort(key=_metadata_order)
    res.rows = [gold[k] for k in sorted(gold)]


def _metadata_order(entry: dict[str, Any]) -> tuple[str, str]:
    """Sort key for gold metadata: field, then value."""
    return entry["field"], entry["normalized_value"]


def import_workbook(xlsx_path: str | Path, root: str | Path, cfg: Config) -> WorkbookImport:
    """Read the reviewed workbook and emit one gold row per gold document.

    Read, then parse each tab against the config, then merge into gold rows.

    Args:
        xlsx_path: The reviewed workbook.
        root: Directory holding the documents, used to resolve doc_ids.
        cfg: Config declaring the classes, fields and tags.

    Returns:
        The gold rows and everything the caller has to be told about.

    Raises:
        NotADirectoryError: `root` is not a directory.
    """
    res = WorkbookImport()
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"--root is not a directory: {root}")

    docs, meta, tags = _read_workbook(xlsx_path)
    if not docs.rows:
        return res

    gold = _import_documents(docs.index, docs.rows, _doc_id_to_path(root), cfg, res)
    _import_metadata(meta.index, meta.rows, gold, cfg, res)
    _import_tags(tags.index, tags.rows, gold, cfg, res)
    _merge_gold(gold, res)

    if res.unresolved_doc_ids:
        log.warning("workbook rows whose doc_id is not under --root", extra={"count": len(res.unresolved_doc_ids)})
    for problem in res.problems:
        log.error("workbook uses names the config does not declare", extra={"detail": problem})
    return res


def _skip_not_gold(row: Row, index: dict[str, int], res: WorkbookImport) -> bool:
    """True, and counted, when the tab has IS_GOLD and this row does not set it."""
    if "IS_GOLD" in index and not _is_gold(_get(row, index, "IS_GOLD")):
        res.skipped_not_gold += 1
        return True
    return False


def _document_label(row: Row, doc_i: dict[str, int], doc_id: str, res: WorkbookImport) -> tuple[str, str] | None:
    """(verdict, label) of a Documents row, or None when it states no truth."""
    verdict = _verdict(_get(row, doc_i, "CLASS_VERDICT"), res, f"Documents/{doc_id}")
    if verdict == "ok":
        return verdict, _get(row, doc_i, "DOC_CLASS").casefold()
    if verdict in ("wrong", "missing"):
        label = _get(row, doc_i, "CLASS_CORRECTED").casefold()
        if not label:
            res.missing_correction.append(f"Documents/{doc_id}: {verdict} with no CLASS_CORRECTED")
            return None
        return verdict, label
    # No usable verdict means nobody has said what the truth is.
    return None


def _import_documents(
    doc_i: dict[str, int], doc_rows: list[Row], by_id: dict[str, str], cfg: Config, res: WorkbookImport
) -> dict[str, GoldRow]:
    """The Documents tab: one gold row per document the reviewer gave a class."""
    classes = {c.casefold() for c in cfg.class_names}
    unknown_label = str(cfg.classify_opts().get("unknown_label", "unknown")).casefold()

    gold: dict[str, GoldRow] = {}
    for row in doc_rows:
        doc_id = _get(row, doc_i, "DOC_ID")
        if not doc_id or _skip_not_gold(row, doc_i, res):
            continue
        got = _document_label(row, doc_i, doc_id, res)
        if got is None:
            continue
        verdict, label = got

        # Confirming an abstention says the pipeline was right not to answer.
        # It does not say the document is of class "unknown", and no such class
        # exists. There is no gold label here until a reviewer supplies the
        # real one, so the row is recorded and skipped rather than failing the
        # import: the pipeline abstains on a fifth of the corpus, and that is
        # an expected outcome rather than a broken sheet.
        if label == unknown_label:
            res.confirmed_abstentions.append(doc_id)
            continue
        if label not in classes:
            res.unknown_classes.append(label)
            continue

        path = by_id.get(doc_id)
        if path is None:
            res.unresolved_doc_ids.append(doc_id)

        res.counts[f"class_{verdict}"] += 1
        gold[doc_id] = {"doc_id": doc_id, "label": label, "metadata": [], "tags": []}
        if path:
            gold[doc_id]["path"] = path
        filename = _get(row, doc_i, "FILE_NAME")
        if filename:
            gold[doc_id]["filename"] = filename
    return gold


def _typed_value(
    row: Row, meta_i: dict[str, int], verdict: str, where: str, res: WorkbookImport
) -> tuple[str, dict[str, int]] | None:
    """The value a Metadata row stands behind, and its offsets when it has any."""
    if verdict == "ok":
        return _get(row, meta_i, "NORMALIZED_VALUE") or _get(row, meta_i, "VALUE"), _offsets(row, meta_i)
    typed = _get(row, meta_i, "CORRECTED_VALUE")
    if not typed:
        res.missing_correction.append(f"{where}: {verdict} with no CORRECTED_VALUE")
        return None
    # No span: the reviewer supplied an answer, not a place in the text.
    return typed, {}


def _normalised(normalizer: Any, typed: str, norm_opts: dict[str, Any], res: WorkbookImport) -> str:
    """Run the field's normalizer over a sheet value, keeping it raw if that fails.

    Both confirmed and corrected values normalise. A correction is raw by
    definition, and the sheet's NORMALIZED_VALUE column is only as normalised
    as whoever filled it in, so several confirmed rows carry a typed value
    too. The normalizers are idempotent, so a value the pipeline already
    produced passes through untouched.
    """
    value, _extras = normalize(normalizer, typed, NormCtx(full=typed, start=0, end=len(typed), opts=norm_opts))
    value = "" if value is None else str(value).strip()
    if not value:
        # The normalizer could not read it. The pipeline keeps such a field
        # with a null normalized_value rather than dropping it, and the
        # harness falls back to the raw value when scoring, so dropping it
        # here would lose a real answer and make the round trip lossy.
        res.counts["field_not_normalised"] += 1
        return typed
    if value != typed:
        res.counts["field_normalised"] += 1
    return value


def _field_verdict(row: Row, meta_i: dict[str, int], where: str, res: WorkbookImport) -> str | None:
    """Verdict of a Metadata row that can produce gold, else None."""
    verdict = _verdict(_get(row, meta_i, "VERDICT"), res, where)
    # A spurious row is a real statement: the truth is that this field is
    # not there. It contributes nothing to gold, which is the whole point.
    if verdict == "spurious":
        res.spurious_rows += 1
        return None
    if verdict not in ("ok", "wrong", "missing"):
        return None
    return verdict


def _import_metadata(
    meta_i: dict[str, int], meta_rows: list[Row], gold: dict[str, GoldRow], cfg: Config, res: WorkbookImport
) -> None:
    """The Metadata tab: confirmed and corrected field values, normalised."""
    normalizers = {str(f.get("name", "")).casefold(): f.get("normalizer") for f in cfg.field_defs}
    norm_opts = cfg.fields.get("normalizer_options", {}) or {}

    for row in meta_rows:
        doc_id = _get(row, meta_i, "DOC_ID")
        if doc_id not in gold:
            res.orphan_rows.append(f"Metadata/{doc_id or '<no doc_id>'}")
            continue
        if _skip_not_gold(row, meta_i, res):
            continue
        name = _get(row, meta_i, "FIELD").casefold()
        if not name:
            continue
        where = f"Metadata/{doc_id}/{name}"
        verdict = _field_verdict(row, meta_i, where, res)
        if verdict is None:
            continue
        if name not in normalizers:
            res.unknown_fields.append(name)
            continue
        got = _typed_value(row, meta_i, verdict, where, res)
        if got is None:
            continue
        typed, offsets = got
        value = _normalised(normalizers.get(name), typed, norm_opts, res)
        if not value:
            continue
        entry: dict[str, Any] = {"field": name, "normalized_value": value}
        entry.update(offsets)

        if _get(row, meta_i, "ROW_SOURCE").casefold() == "human":
            res.human_rows += 1
        res.counts[f"field_{verdict}"] += 1
        gold[doc_id]["metadata"].append(entry)


def _import_tags(
    tag_i: dict[str, int], tag_rows: list[Row], gold: dict[str, GoldRow], cfg: Config, res: WorkbookImport
) -> None:
    """The Tags tab: a tag is either on the document or it is not."""
    tags = {str(t.get("name", "")).casefold() for t in cfg.tag_defs}

    for row in tag_rows:
        doc_id = _get(row, tag_i, "DOC_ID")
        if doc_id not in gold:
            res.orphan_rows.append(f"Tags/{doc_id or '<no doc_id>'}")
            continue
        if _skip_not_gold(row, tag_i, res):
            continue

        name = _get(row, tag_i, "TAG").casefold()
        if not name:
            continue
        verdict = _verdict(_get(row, tag_i, "VERDICT"), res, f"Tags/{doc_id}/{name}")

        # A tag has no value to correct, so wrong and spurious say the same
        # thing: it does not belong on this document.
        if verdict in ("wrong", "spurious"):
            res.spurious_rows += 1
            continue
        if verdict not in ("ok", "missing"):
            continue
        if name not in tags:
            res.unknown_tags.append(name)
            continue
        if _get(row, tag_i, "ROW_SOURCE").casefold() == "human":
            res.human_rows += 1
        res.counts[f"tag_{verdict}"] += 1
        if name not in gold[doc_id]["tags"]:
            gold[doc_id]["tags"].append(name)
