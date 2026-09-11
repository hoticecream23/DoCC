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
from .columns import TRUTHY, VERDICTS, _cell, _get, _read_tab

log = get_logger(__name__)


@dataclass
class WorkbookImport:
    rows: list[dict[str, Any]] = field(default_factory=list)
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
    counts: Counter = field(default_factory=Counter)

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


def _offsets(row: tuple, index: dict[str, int]) -> dict[str, int]:
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


def import_workbook(xlsx_path: str | Path, root: str | Path, cfg: Config) -> WorkbookImport:
    """Read the reviewed workbook and emit one gold row per gold document."""
    import openpyxl

    res = WorkbookImport()
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"--root is not a directory: {root}")

    wb = openpyxl.load_workbook(xlsx_path, data_only=True, read_only=True)
    try:
        doc_i, doc_rows = _read_tab(wb, "Documents")
        meta_i, meta_rows = _read_tab(wb, "Metadata")
        tag_i, tag_rows = _read_tab(wb, "Tags")
    finally:
        wb.close()

    if not doc_rows:
        return res

    gold = _import_documents(doc_i, doc_rows, _doc_id_to_path(root), cfg, res)
    _import_metadata(meta_i, meta_rows, gold, cfg, res)
    _import_tags(tag_i, tag_rows, gold, cfg, res)

    for row in gold.values():
        row["tags"].sort()
        row["metadata"].sort(key=lambda e: (e["field"], e["normalized_value"]))
    res.rows = [gold[k] for k in sorted(gold)]

    if res.unresolved_doc_ids:
        log.warning("workbook rows whose doc_id is not under --root",
                    extra={"count": len(res.unresolved_doc_ids)})
    for problem in res.problems:
        log.error("workbook uses names the config does not declare", extra={"detail": problem})
    return res


def _import_documents(doc_i: dict[str, int], doc_rows: list[tuple], by_id: dict[str, str],
                      cfg: Config, res: WorkbookImport) -> dict[str, dict[str, Any]]:
    """The Documents tab: one gold row per document the reviewer gave a class."""
    classes = {c.casefold() for c in cfg.class_names}
    unknown_label = str(cfg.classify_opts().get("unknown_label", "unknown")).casefold()

    gold: dict[str, dict[str, Any]] = {}
    for row in doc_rows:
        doc_id = _get(row, doc_i, "DOC_ID")
        if not doc_id:
            continue
        if "IS_GOLD" in doc_i and not _is_gold(_get(row, doc_i, "IS_GOLD")):
            res.skipped_not_gold += 1
            continue

        verdict = _verdict(_get(row, doc_i, "CLASS_VERDICT"), res, f"Documents/{doc_id}")
        predicted = _get(row, doc_i, "DOC_CLASS").casefold()
        corrected = _get(row, doc_i, "CLASS_CORRECTED").casefold()

        if verdict == "ok":
            label = predicted
        elif verdict in ("wrong", "missing"):
            label = corrected
            if not label:
                res.missing_correction.append(
                    f"Documents/{doc_id}: {verdict} with no CLASS_CORRECTED")
                continue
        else:
            # No usable verdict means nobody has said what the truth is.
            continue

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


def _import_metadata(meta_i: dict[str, int], meta_rows: list[tuple],
                     gold: dict[str, dict[str, Any]], cfg: Config, res: WorkbookImport) -> None:
    """The Metadata tab: confirmed and corrected field values, normalised."""
    normalizers = {str(f.get("name", "")).casefold(): f.get("normalizer") for f in cfg.field_defs}
    fields = set(normalizers)
    norm_opts = cfg.fields.get("normalizer_options", {}) or {}

    for row in meta_rows:
        doc_id = _get(row, meta_i, "DOC_ID")
        if doc_id not in gold:
            res.orphan_rows.append(f"Metadata/{doc_id or '<no doc_id>'}")
            continue
        if "IS_GOLD" in meta_i and not _is_gold(_get(row, meta_i, "IS_GOLD")):
            res.skipped_not_gold += 1
            continue

        name = _get(row, meta_i, "FIELD").casefold()
        if not name:
            continue
        verdict = _verdict(_get(row, meta_i, "VERDICT"), res, f"Metadata/{doc_id}/{name}")

        # A spurious row is a real statement: the truth is that this field is
        # not there. It contributes nothing to gold, which is the whole point.
        if verdict == "spurious":
            res.spurious_rows += 1
            continue
        if verdict not in ("ok", "wrong", "missing"):
            continue
        if name not in fields:
            res.unknown_fields.append(name)
            continue

        if verdict == "ok":
            typed = _get(row, meta_i, "NORMALIZED_VALUE") or _get(row, meta_i, "VALUE")
            offsets = _offsets(row, meta_i)
        else:
            typed = _get(row, meta_i, "CORRECTED_VALUE")
            if not typed:
                res.missing_correction.append(
                    f"Metadata/{doc_id}/{name}: {verdict} with no CORRECTED_VALUE")
                continue
            # No span: the reviewer supplied an answer, not a place in the text.
            offsets = {}

        # Both branches normalise. A correction is raw by definition, and the
        # sheet's NORMALIZED_VALUE column is only as normalised as whoever
        # filled it in, so several confirmed rows carry a typed value too. The
        # normalizers are idempotent, so a value the pipeline already produced
        # passes through untouched.
        value, _extras = normalize(
            normalizers.get(name), typed,
            NormCtx(full=typed, start=0, end=len(typed), opts=norm_opts),
        )
        value = "" if value is None else str(value).strip()
        if not value:
            # The normalizer could not read it. The pipeline keeps such a field
            # with a null normalized_value rather than dropping it, and the
            # harness falls back to the raw value when scoring, so dropping it
            # here would lose a real answer and make the round trip lossy.
            value = typed
            res.counts["field_not_normalised"] += 1
        elif value != typed:
            res.counts["field_normalised"] += 1
        if not value:
            continue
        entry = {"field": name, "normalized_value": value}
        entry.update(offsets)

        if _get(row, meta_i, "ROW_SOURCE").casefold() == "human":
            res.human_rows += 1
        res.counts[f"field_{verdict}"] += 1
        gold[doc_id]["metadata"].append(entry)


def _import_tags(tag_i: dict[str, int], tag_rows: list[tuple],
                 gold: dict[str, dict[str, Any]], cfg: Config, res: WorkbookImport) -> None:
    """The Tags tab: a tag is either on the document or it is not."""
    tags = {str(t.get("name", "")).casefold() for t in cfg.tag_defs}

    for row in tag_rows:
        doc_id = _get(row, tag_i, "DOC_ID")
        if doc_id not in gold:
            res.orphan_rows.append(f"Tags/{doc_id or '<no doc_id>'}")
            continue
        if "IS_GOLD" in tag_i and not _is_gold(_get(row, tag_i, "IS_GOLD")):
            res.skipped_not_gold += 1
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


def render_summary(res: WorkbookImport, xlsx_path: str, root: str) -> str:
    """What this gold file can and cannot be used to claim."""
    with_spans = sum(
        1 for r in res.rows
        if any("char_start" in e for e in r["metadata"])
    )
    lines = [
        "# Imported workbook", "",
        f"Source: `{xlsx_path}`", f"Root: `{root}`", "",
        f"- gold rows written: {len(res.rows)}",
        f"- documents carrying at least one gold field: "
        f"{sum(1 for r in res.rows if r['metadata'])}",
        f"- documents carrying character offsets: {with_spans}",
        f"- gold fields: {sum(len(r['metadata']) for r in res.rows)}",
        f"- gold tags: {sum(len(r['tags']) for r in res.rows)}",
        f"- rows the reviewer marked as not belonging: {res.spurious_rows}",
        f"- rows a human added rather than corrected: {res.human_rows}",
        f"- confirmed abstentions, which carry no label: {len(res.confirmed_abstentions)}",
    ]
    if res.skipped_not_gold:
        lines.append(f"- rows skipped because IS_GOLD was not set: {res.skipped_not_gold}")

    verdicts = {k: v for k, v in sorted(res.counts.items())}
    if verdicts:
        lines += ["", "## Where the gold came from", "",
                  "| row | verdict | count |", "| --- | --- | --- |"]
        for key, count in verdicts.items():
            kind, _, verdict = key.partition("_")
            lines.append(f"| {kind} | `{verdict}` | {count} |")

    lines += [
        "", "## What this gold cannot measure", "",
        "The workbook is an export of what the pipeline produced, so a row",
        "exists only where the pipeline emitted something or a human added one",
        "by hand. A field or tag the pipeline never found leaves no trace in",
        "the sheet and therefore none in this gold.",
        "",
        f"**Precision over these documents is honest. Recall is not.** Only "
        f"{res.human_rows} row(s) here were added by a human rather than "
        "corrected, so a false negative is almost unrepresented. Treat the "
        "metadata and tagging recall numbers from this file as an upper bound, "
        "and do not quote them as the recall of the system.",
        "",
        "Classification is the exception. Every gold document carries exactly",
        "one label and the reviewer either confirmed it or replaced it, so the",
        "classification score over this file is a real score.",
    ]

    if res.unresolved_doc_ids:
        lines += ["", "## Rows pointing at content that is not here", "",
                  "The doc_id is the SHA-256 of the file's bytes, so these rows were",
                  "reviewed against a file that is not under `--root`. Same name is",
                  "not same file. They are still in the gold, without a path, and will",
                  "not join to any prediction until the content turns up.", ""]
        lines += [f"- `{d}`" for d in sorted(set(res.unresolved_doc_ids))]

    if res.confirmed_abstentions:
        lines += ["", "## Abstentions confirmed as correct", "",
                  "The reviewer agreed the pipeline was right not to answer. That is",
                  "useful, but it is not a label: these documents still have a real",
                  "class that nobody has written down, so they carry no classification",
                  "gold. Put the true class in CLASS_CORRECTED to score them.", "",
                  f"- {len(res.confirmed_abstentions)} document(s)"]

    if res.missing_correction:
        lines += ["", "## Rows marked wrong with nothing to replace them", "",
                  "Nobody has said what the right answer is, so these produce no gold.", ""]
        lines += [f"- {d}" for d in sorted(set(res.missing_correction))]

    if res.orphan_rows:
        counts = Counter(res.orphan_rows)
        lines += ["", "## Rows with no gold document", "",
                  "A Metadata or Tags row whose DOC_ID is absent from the Documents",
                  "tab, or whose document produced no gold row.", ""]
        lines += [f"- {k} ({v})" if v > 1 else f"- {k}" for k, v in sorted(counts.items())]

    if res.problems:
        lines += ["", "## Names the config does not declare", "",
                  "Nothing here was guessed at. Add the name to the relevant config",
                  "file, or correct it in the sheet.", ""]
        lines += [f"- {p}" for p in res.problems]

    return "\n".join(lines) + "\n"
