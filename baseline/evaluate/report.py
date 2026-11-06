"""Whole files scored end to end, compared run against run, and written up."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..logging_setup import get_logger
from .calibration import BUCKETS, covered_fields, eval_calibration, eval_covered
from .io import load_jsonl
from .metadata_scores import eval_metadata
from .scoring import eval_classification, eval_extraction, eval_tagging, join
from .table_scores import eval_tables

log = get_logger(__name__)

PathLike = str | Path
# Unmatched or unscored document keys listed in a result.
MAX_LISTED_KEYS = 20
# Metadata misses printed in the report.
MAX_REPORTED_MISSES = 25
# Error rates, in percent, shown in the coverage table.
COVERAGE_PCTS = (1, 5, 10)


def evaluate(
    gold_path: PathLike,
    pred_path: PathLike,
    unknown_label: str = "unknown",
    tables_path: PathLike | None = None,
) -> dict[str, Any]:
    """Score a predictions file against a gold file, every component.

    Args:
        gold_path: Gold JSONL.
        pred_path: Predictions JSONL.
        unknown_label: Label the system uses to abstain.
        tables_path: Optional tables JSONL to score against gold grids.

    Returns:
        Join counts and one result block per component.

    Raises:
        ValueError: An input line is not valid JSON.
    """
    gold = load_jsonl(gold_path)
    pred = load_jsonl(pred_path)
    pairs, missing, extra = join(gold, pred)
    covered, calibration_covered = eval_covered(pairs, covered_fields(gold))
    log.info(
        "joined",
        extra={
            "gold": len(gold),
            "pred": len(pred),
            "matched": len(pairs),
            "unmatched_gold": len(missing),
            "unscored_pred": len(extra),
        },
    )
    return {
        "gold_count": len(gold),
        "pred_count": len(pred),
        "matched": len(pairs),
        "unmatched_gold": missing[:MAX_LISTED_KEYS],
        "unscored_predictions": extra[:MAX_LISTED_KEYS],
        "extraction": eval_extraction(pairs),
        "classification": eval_classification(pairs, unknown_label),
        "metadata": eval_metadata(pairs),
        "tagging": eval_tagging(pairs),
        "tables": eval_tables(join(gold, load_jsonl(tables_path))[0] if tables_path else []),
        "calibration": eval_calibration(pairs, unknown_label),
        "covered": covered,
        "calibration_covered": calibration_covered,
    }


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


def render_report(res: dict[str, Any], gold_path: str, pred_path: str) -> str:
    """Render an `evaluate` result as markdown.

    Args:
        res: Result of `evaluate`.
        gold_path: Gold path, shown in the header.
        pred_path: Predictions path, shown in the header.

    Returns:
        The report text.
    """
    lines: list[str] = []
    a = lines.append
    a("# Evaluation report")
    a("")
    a(f"gold: `{gold_path}`")
    a(f"predictions: `{pred_path}`")
    a(f"matched {res['matched']} of {res['gold_count']} gold documents")
    a("")
    if res["unmatched_gold"]:
        a(f"Unmatched gold rows: {', '.join(res['unmatched_gold'])}")
        a("")

    lines += _report_extraction(res["extraction"])
    lines += _report_classification(res["classification"])
    lines += _report_metadata(res["metadata"])
    lines += _report_covered(res.get("covered") or {})
    lines += _report_calibration(res.get("calibration") or {}, res.get("calibration_covered"))
    lines += _report_tagging(res["tagging"])
    lines += _report_tables(res.get("tables") or {})
    return "\n".join(lines)


def _report_extraction(ex: dict[str, Any]) -> list[str]:
    """Extraction section."""
    lines: list[str] = []
    a = lines.append
    a("## Extraction")
    a("")
    if ex.get("scored"):
        a(f"Scored {ex['scored']} documents against reference text.")
        a("")
        a("| metric | value |")
        a("| --- | --- |")
        a(f"| CER | {ex['cer']:.4f} |")
        a(f"| WER | {ex['wer']:.4f} |")
    else:
        a("No gold text supplied, so extraction was not scored.")
    a("")
    return lines


def _report_classification(c: dict[str, Any]) -> list[str]:
    """Classification section."""
    lines: list[str] = []
    a = lines.append
    a("## Classification")
    a("")
    a("| metric | value |")
    a("| --- | --- |")
    a(f"| documents | {c['n']} |")
    a(f"| answered | {c['answered']} |")
    a(f"| abstained | {c['abstained']} |")
    a(f"| coverage | {c['coverage']:.3f} |")
    a(f"| accuracy overall | {c['accuracy_overall']:.3f} |")
    a(f"| accuracy when answered | {c['accuracy_when_answered']:.3f} |")
    a(f"| macro F1 | {c['macro_f1']:.3f} |")
    a("")
    cov = {k: v for k, v in c.items() if k.startswith("coverage_at_")}
    if cov:
        a("Coverage the system can reach while staying under a given error rate:")
        a("")
        a("| max error | coverage | confidence threshold |")
        a("| --- | --- | --- |")
        for pct in COVERAGE_PCTS:
            k, t = f"coverage_at_{pct}pct_error", f"threshold_at_{pct}pct_error"
            if k in c:
                a(f"| {pct}% | {c[k]:.3f} | {c[t]:.3f} |")
        a("")
    if c["per_label"]:
        a("| label | precision | recall | f1 | tp | fp | fn |")
        a("| --- | --- | --- | --- | --- | --- | --- |")
        for k, v in c["per_label"].items():
            a(
                f"| {k} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} "
                f"| {v['tp']} | {v['fp']} | {v['fn']} |"
            )
        a("")
    if c["confusion"]:
        a("Most common mistakes:")
        a("")
        for k, v in c["confusion"].items():
            a(f"- {k}: {v}")
        a("")
    return lines


def _report_metadata(m: dict[str, Any]) -> list[str]:
    """Metadata section."""
    lines: list[str] = []
    a = lines.append
    a("## Metadata")
    a("")
    a("Value scoring asks whether the answer was right. Span scoring asks")
    a("whether it pointed at the right place. They are different questions.")
    a("")
    mv = m["value"]["micro"]
    a(f"Value micro F1 **{mv['f1']:.3f}**, macro F1 **{m['value']['macro_f1']:.3f}**")
    a("")
    a("| field | precision | recall | f1 | tp | fp | fn |")
    a("| --- | --- | --- | --- | --- | --- | --- |")
    for k, v in m["value"]["per_field"].items():
        a(f"| {k} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} " f"| {v['tp']} | {v['fp']} | {v['fn']} |")
    a("")
    if m["span"].get("per_field"):
        a(
            f"Span micro F1 **{m['span']['micro']['f1']:.3f}** "
            f"over {m['span_scored_docs']} documents with gold offsets"
        )
    else:
        a("Span scoring skipped, the gold file carries no character offsets.")
    a("")
    vp = m["validated_precision"]
    if vp is not None:
        a(f"Precision of fields the pipeline marked `validated`: **{vp:.4f}** " f"({m['validated_wrong']} wrong)")
        a("")
        a("This is the number to watch. A validated field claims a checksum")
        a("passed, so anything less than 1.0 means a checksum is lying.")
        a("")
    if m["errors"]:
        a("### Misses")
        a("")
        for e in m["errors"][:MAX_REPORTED_MISSES]:
            a(f"- {e}")
        a("")
    return lines


def _bucket_table(stats: dict[str, Any], metric: str) -> list[str]:
    """Confidence bucket table with its ECE line."""
    lines = [f"| confidence | n | {metric} | mean confidence |", "| --- | --- | --- | --- |"]
    for name, b in stats["buckets"].items():
        lines.append(f"| {name} | {b['n']} | {b[metric]:.3f} | {b['mean_confidence']:.3f} |")
    ece = stats["ece"]
    lines += [
        "",
        (
            f"ECE **{ece:.4f}** over {stats['n']} predictions " f"({stats['below_0.5']} below 0.5 not bucketed)"
            if ece is not None
            else "ECE not computed, no prediction at confidence 0.5 or more."
        ),
        "",
    ]
    return lines


def _report_covered(c: dict[str, Any]) -> list[str]:
    """Gold-covered fields section, empty when not scored."""
    if not c:
        return []
    m = c["micro"]
    lines = [
        "## Gold-covered fields",
        "",
        "Scores only fields the gold file labels at least once. A predicted field",
        "the gold never labels is unscored here, not a false positive.",
        "",
        f"Covered fields: {', '.join(c['fields']) or 'none'}",
        "",
        "| metric | value |",
        "| --- | --- |",
        f"| micro precision | {m['precision']:.3f} |",
        f"| micro recall | {m['recall']:.3f} |",
        f"| micro F1 | {m['f1']:.3f} |",
        f"| tp / fp / fn | {m['tp']} / {m['fp']} / {m['fn']} |",
        f"| unscored predictions | {c['unscored_predictions']} |",
        "",
    ]
    if c["unscored_per_field"]:
        lines += ["| unscored field | predictions |", "| --- | --- |"]
        lines += [f"| {k} | {v} |" for k, v in c["unscored_per_field"].items()]
        lines.append("")
    return lines


def _report_calibration(cal: dict[str, Any], covered: dict[str, Any] | None = None) -> list[str]:
    """Calibration section, empty when not scored."""
    if not cal:
        return []
    lines: list[str] = [
        "## Calibration",
        "",
        "A well calibrated 0.9 is right nine times in ten. ECE is the",
        "count weighted gap between stated confidence and observed hit rate.",
        "",
        "### Field values",
        "",
    ]
    lines += _bucket_table(cal["fields"]["overall"], "precision")
    per = cal["fields"]["per_field"]
    if per:
        names = [b[0] for b in BUCKETS]
        lines += [
            "Per field, n / precision:",
            "",
            "| field | " + " | ".join(names) + " | ECE |",
            "| --- " * (len(names) + 2) + "|",
        ]
        for k, v in per.items():
            cells = [
                f"{v['buckets'][n]['n']} / {v['buckets'][n]['precision']:.2f}" if v["buckets"][n]["n"] else "-"
                for n in names
            ]
            ece = f"{v['ece']:.3f}" if v["ece"] is not None else "-"
            lines.append(f"| {k} | " + " | ".join(cells) + f" | {ece} |")
        lines.append("")
    if covered:
        lines += ["### Field values, gold-covered fields only", ""]
        lines += _bucket_table(covered["overall"], "precision")
    lines += ["### Classification (answered documents)", ""]
    lines += _bucket_table(cal["classification"], "accuracy")
    return lines


def _report_tagging(t: dict[str, Any]) -> list[str]:
    """Tagging section."""
    lines: list[str] = []
    a = lines.append
    a("## Tagging")
    a("")
    a(f"Micro F1 **{t['micro']['f1']:.3f}**, macro F1 **{t['macro_f1']:.3f}**")
    a("")
    a("| tag | precision | recall | f1 | tp | fp | fn |")
    a("| --- | --- | --- | --- | --- | --- | --- |")
    for k, v in t["per_tag"].items():
        a(f"| {k} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} " f"| {v['tp']} | {v['fp']} | {v['fn']} |")
    a("")
    return lines


def _report_tables(tb: dict[str, Any]) -> list[str]:
    """Tables section, empty when no tables were scored."""
    lines: list[str] = []
    a = lines.append
    if tb.get("scored"):
        a("## Tables")
        a("")
        a(f"Scored {tb['scored']} documents carrying gold tables.")
        a("")
        a("| metric | value |")
        a("| --- | --- |")
        a(f"| gold tables | {tb['gold_tables']} |")
        a(f"| predicted tables | {tb['pred_tables']} |")
        a(f"| detection recall | {tb['detection_recall']:.3f} |")
        a(f"| cell precision | {tb['cells']['precision']:.3f} |")
        a(f"| cell recall | {tb['cells']['recall']:.3f} |")
        a(f"| cell F1 | {tb['cells']['f1']:.3f} |")
        a(f"| adjacency precision | {tb['adjacency']['precision']:.3f} |")
        a(f"| adjacency recall | {tb['adjacency']['recall']:.3f} |")
        a(f"| adjacency F1 | {tb['adjacency']['f1']:.3f} |")
        a("")
        a("Cell F1 says whether the values came out. Adjacency F1 says whether")
        a("they came out next to the right neighbours, which is what separates")
        a("a real table from a bag of numbers.")
        a("")
    return lines
