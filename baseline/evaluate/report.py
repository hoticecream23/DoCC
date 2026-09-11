"""Whole files scored end to end, compared run against run, and written up."""

from __future__ import annotations

from ..logging_setup import get_logger
from .scoring import (
    eval_classification,
    eval_extraction,
    eval_metadata,
    eval_tagging,
    join,
    load_jsonl,
)
from .table_scores import eval_tables

log = get_logger(__name__)


def evaluate(
    gold_path, pred_path, unknown_label: str = "unknown", tables_path=None
) -> dict:
    gold = load_jsonl(gold_path)
    pred = load_jsonl(pred_path)
    pairs, missing, extra = join(gold, pred)
    log.info(
        "joined", extra={"gold": len(gold), "pred": len(pred), "matched": len(pairs),
                         "unmatched_gold": len(missing), "unscored_pred": len(extra)}
    )
    return {
        "gold_count": len(gold),
        "pred_count": len(pred),
        "matched": len(pairs),
        "unmatched_gold": missing[:20],
        "unscored_predictions": extra[:20],
        "extraction": eval_extraction(pairs),
        "classification": eval_classification(pairs, unknown_label),
        "metadata": eval_metadata(pairs),
        "tagging": eval_tagging(pairs),
        "tables": eval_tables(join(gold, load_jsonl(tables_path))[0] if tables_path else []),
    }


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


def render_report(res: dict, gold_path: str, pred_path: str) -> str:
    L: list[str] = []
    a = L.append
    a("# Evaluation report")
    a("")
    a(f"gold: `{gold_path}`")
    a(f"predictions: `{pred_path}`")
    a(f"matched {res['matched']} of {res['gold_count']} gold documents")
    a("")
    if res["unmatched_gold"]:
        a(f"Unmatched gold rows: {', '.join(res['unmatched_gold'])}")
        a("")

    ex = res["extraction"]
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

    c = res["classification"]
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
        for pct in (1, 5, 10):
            k, t = f"coverage_at_{pct}pct_error", f"threshold_at_{pct}pct_error"
            if k in c:
                a(f"| {pct}% | {c[k]:.3f} | {c[t]:.3f} |")
        a("")
    if c["per_label"]:
        a("| label | precision | recall | f1 | tp | fp | fn |")
        a("| --- | --- | --- | --- | --- | --- | --- |")
        for k, v in c["per_label"].items():
            a(f"| {k} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} "
              f"| {v['tp']} | {v['fp']} | {v['fn']} |")
        a("")
    if c["confusion"]:
        a("Most common mistakes:")
        a("")
        for k, v in c["confusion"].items():
            a(f"- {k}: {v}")
        a("")

    m = res["metadata"]
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
        a(f"| {k} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} "
          f"| {v['tp']} | {v['fp']} | {v['fn']} |")
    a("")
    if m["span"].get("per_field"):
        a(f"Span micro F1 **{m['span']['micro']['f1']:.3f}** "
          f"over {res['metadata']['span_scored_docs']} documents with gold offsets")
    else:
        a("Span scoring skipped, the gold file carries no character offsets.")
    a("")
    vp = m["validated_precision"]
    if vp is not None:
        a(f"Precision of fields the pipeline marked `validated`: **{vp:.4f}** "
          f"({m['validated_wrong']} wrong)")
        a("")
        a("This is the number to watch. A validated field claims a checksum")
        a("passed, so anything less than 1.0 means a checksum is lying.")
        a("")
    if m["errors"]:
        a("### Misses")
        a("")
        for e in m["errors"][:25]:
            a(f"- {e}")
        a("")

    t = res["tagging"]
    a("## Tagging")
    a("")
    a(f"Micro F1 **{t['micro']['f1']:.3f}**, macro F1 **{t['macro_f1']:.3f}**")
    a("")
    a("| tag | precision | recall | f1 | tp | fp | fn |")
    a("| --- | --- | --- | --- | --- | --- | --- |")
    for k, v in t["per_tag"].items():
        a(f"| {k} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} "
          f"| {v['tp']} | {v['fp']} | {v['fn']} |")
    a("")
    tb = res.get("tables") or {}
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
    return "\n".join(L)


# --------------------------------------------------------------------------
# compare two runs
# --------------------------------------------------------------------------


def compare(gold_path, a_path, b_path, unknown_label: str = "unknown",
            a_tables=None, b_tables=None) -> dict:
    """Score two prediction files against the same gold and diff them."""
    ra = evaluate(gold_path, a_path, unknown_label, a_tables)
    rb = evaluate(gold_path, b_path, unknown_label, b_tables)

    fields = sorted(
        set(ra["metadata"]["value"]["per_field"]) | set(rb["metadata"]["value"]["per_field"])
    )
    field_delta = {}
    for f in fields:
        fa = ra["metadata"]["value"]["per_field"].get(f, {}).get("f1", 0.0)
        fb = rb["metadata"]["value"]["per_field"].get(f, {}).get("f1", 0.0)
        field_delta[f] = {"a": fa, "b": fb, "delta": round(fb - fa, 4)}

    tags = sorted(set(ra["tagging"]["per_tag"]) | set(rb["tagging"]["per_tag"]))
    tag_delta = {}
    for t in tags:
        fa = ra["tagging"]["per_tag"].get(t, {}).get("f1", 0.0)
        fb = rb["tagging"]["per_tag"].get(t, {}).get("f1", 0.0)
        tag_delta[t] = {"a": fa, "b": fb, "delta": round(fb - fa, 4)}

    headline = {
        "classification_macro_f1": {
            "a": ra["classification"]["macro_f1"],
            "b": rb["classification"]["macro_f1"],
            "delta": round(rb["classification"]["macro_f1"] - ra["classification"]["macro_f1"], 4),
        },
        "classification_coverage": {
            "a": ra["classification"]["coverage"],
            "b": rb["classification"]["coverage"],
            "delta": round(rb["classification"]["coverage"] - ra["classification"]["coverage"], 4),
        },
        "metadata_micro_f1": {
            "a": ra["metadata"]["value"]["micro"]["f1"],
            "b": rb["metadata"]["value"]["micro"]["f1"],
            "delta": round(
                rb["metadata"]["value"]["micro"]["f1"] - ra["metadata"]["value"]["micro"]["f1"], 4
            ),
        },
        "tagging_micro_f1": {
            "a": ra["tagging"]["micro"]["f1"],
            "b": rb["tagging"]["micro"]["f1"],
            "delta": round(rb["tagging"]["micro"]["f1"] - ra["tagging"]["micro"]["f1"], 4),
        },
    }
    # Table metrics only join the gate when both runs actually scored tables,
    # so a run without a tables file is not read as a regression to zero.
    if ra["tables"].get("scored") and rb["tables"].get("scored"):
        for key, path in (("tables_cell_f1", ("cells", "f1")),
                          ("tables_adjacency_f1", ("adjacency", "f1")),
                          ("tables_detection_recall", ("detection_recall",))):
            va = ra["tables"][path[0]][path[1]] if len(path) > 1 else ra["tables"][path[0]]
            vb = rb["tables"][path[0]][path[1]] if len(path) > 1 else rb["tables"][path[0]]
            headline[key] = {"a": va, "b": vb, "delta": round(vb - va, 4)}

    regressions = (
        [f"metadata {k}: {v['a']:.3f} -> {v['b']:.3f}" for k, v in field_delta.items() if v["delta"] < 0]
        + [f"tag {k}: {v['a']:.3f} -> {v['b']:.3f}" for k, v in tag_delta.items() if v["delta"] < 0]
        + [f"{k}: {v['a']:.3f} -> {v['b']:.3f}" for k, v in headline.items() if v["delta"] < 0]
    )
    return {"headline": headline, "field_delta": field_delta,
            "tag_delta": tag_delta, "regressions": regressions}


def render_comparison(cmp: dict, a_path: str, b_path: str) -> str:
    L: list[str] = []
    a = L.append
    a("# Comparison")
    a("")
    a(f"A: `{a_path}`")
    a(f"B: `{b_path}`")
    a("")
    a("Positive delta means B is better.")
    a("")
    a("| metric | A | B | delta |")
    a("| --- | --- | --- | --- |")
    for k, v in cmp["headline"].items():
        a(f"| {k} | {v['a']:.3f} | {v['b']:.3f} | {v['delta']:+.3f} |")
    a("")

    a("## Per field")
    a("")
    a("| field | A | B | delta |")
    a("| --- | --- | --- | --- |")
    for k, v in sorted(cmp["field_delta"].items(), key=lambda kv: kv[1]["delta"]):
        a(f"| {k} | {v['a']:.3f} | {v['b']:.3f} | {v['delta']:+.3f} |")
    a("")

    a("## Per tag")
    a("")
    a("| tag | A | B | delta |")
    a("| --- | --- | --- | --- |")
    for k, v in sorted(cmp["tag_delta"].items(), key=lambda kv: kv[1]["delta"]):
        a(f"| {k} | {v['a']:.3f} | {v['b']:.3f} | {v['delta']:+.3f} |")
    a("")

    a("## Regressions")
    a("")
    if cmp["regressions"]:
        for r in cmp["regressions"]:
            a(f"- {r}")
    else:
        a("None. B is at least as good as A everywhere.")
    a("")
    return "\n".join(L)
