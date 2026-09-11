"""Scoring. Reads any JSONL that matches the output schema.

This is deliberately implementation blind. It does not import the pipeline
and it does not care how a record was produced, so the baseline and whatever
replaces it are scored by exactly the same code.
"""

from __future__ import annotations

import collections
import json
import statistics
from pathlib import Path
from typing import Any

from .logging_setup import get_logger

log = get_logger(__name__)

# Gold rows may key on either. doc_id is preferred, path is the fallback.
KEYS = ("doc_id", "source_path", "path")


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{i} is not valid JSON: {exc}") from exc
    return rows


def _keys_of(row: dict) -> list[str]:
    out = []
    for k in KEYS:
        v = row.get(k)
        if v:
            out.append(str(v))
            out.append(Path(str(v)).name)
    return out


def join(gold: list[dict], pred: list[dict]) -> tuple[list[tuple[dict, dict]], list[str], list[str]]:
    """Match gold to predictions on doc_id, then path, then filename."""
    index: dict[str, dict] = {}
    for p in pred:
        for k in _keys_of(p):
            index.setdefault(k, p)

    pairs, missing = [], []
    used = set()
    for g in gold:
        hit = None
        for k in _keys_of(g):
            if k in index:
                hit = index[k]
                break
        if hit is None:
            missing.append(_keys_of(g)[0] if _keys_of(g) else "?")
        else:
            pairs.append((g, hit))
            used.add(id(hit))
    extra = [_keys_of(p)[0] for p in pred if id(p) not in used]
    return pairs, missing, extra


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
            "tp": tp, "fp": fp, "fn": fn}


def levenshtein(a: str, b: str) -> int:
    """Two row edit distance. Fine for page sized strings."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


# --------------------------------------------------------------------------
# per component scoring
# --------------------------------------------------------------------------


def _word_error(ref: list[str], hyp: list[str]) -> int:
    """Levenshtein over word tokens rather than characters."""
    if ref == hyp:
        return 0
    prev = list(range(len(hyp) + 1))
    for i, a in enumerate(ref, 1):
        cur = [i]
        for j, b in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a != b)))
        prev = cur
    return prev[-1]


def eval_extraction(pairs: list[tuple[dict, dict]]) -> dict:
    """CER and WER, only over gold rows that carry reference text."""
    cers, wers, n = [], [], 0
    for g, p in pairs:
        ref = g.get("text")
        if isinstance(ref, dict):
            ref = ref.get("full")
        if not ref:
            continue
        hyp = (p.get("text") or {}).get("full", "")
        n += 1
        # Collapse whitespace on both sides. Layout is not a recognition error.
        ref_n = " ".join(ref.split())
        hyp_n = " ".join(hyp.split())
        cers.append(levenshtein(ref_n, hyp_n) / max(1, len(ref_n)))
        rw, hw = ref.split(), hyp.split()
        wers.append(_word_error(rw, hw) / max(1, len(rw)))
    if not n:
        return {"scored": 0}
    return {
        "scored": n,
        "cer": round(statistics.fmean(cers), 4),
        "wer": round(statistics.fmean(wers), 4),
    }


def coverage_at_error(items: list[tuple[float, bool]], unknown_flags: list[bool],
                      targets=(0.01, 0.05, 0.10)) -> dict:
    """Max coverage achievable while staying under each error rate.

    Coverage is the share of all documents answered. Error is measured only
    among answered ones. This is the metric that matters when the system is
    allowed to decline.
    """
    total = len(items)
    if not total:
        return {}
    # Only non unknown predictions count as answers.
    answers = [(c, ok) for (c, ok), unk in zip(items, unknown_flags) if not unk]
    answers.sort(key=lambda t: -t[0])

    out: dict[str, Any] = {}
    for target in targets:
        best_cov, best_thr = 0.0, 1.0
        tp = 0
        for i, (conf, ok) in enumerate(answers, 1):
            tp += 1 if ok else 0
            err = 1.0 - tp / i
            if err <= target:
                cov = i / total
                if cov > best_cov:
                    best_cov, best_thr = cov, conf
        out[f"coverage_at_{int(target * 100)}pct_error"] = round(best_cov, 4)
        out[f"threshold_at_{int(target * 100)}pct_error"] = round(best_thr, 4)
    return out


def eval_classification(pairs: list[tuple[dict, dict]], unknown_label: str = "unknown") -> dict:
    per: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0}
    )
    correct = answered = 0
    conf_items, unknown_flags = [], []
    confusion: collections.Counter = collections.Counter()

    for g, p in pairs:
        truth = str(g.get("label", "")).strip()
        got = str((p.get("classification") or {}).get("label", "")).strip()
        conf = float((p.get("classification") or {}).get("confidence", 0.0))
        is_unknown = got == unknown_label or not got
        ok = got == truth

        conf_items.append((conf, ok))
        unknown_flags.append(is_unknown)
        if not is_unknown:
            answered += 1
            correct += int(ok)
        if not ok:
            confusion[(truth, got or unknown_label)] += 1

        if ok:
            per[truth]["tp"] += 1
        else:
            per[truth]["fn"] += 1
            if not is_unknown:
                per[got]["fp"] += 1

    labels = sorted(per)
    per_label = {k: prf(per[k]["tp"], per[k]["fp"], per[k]["fn"]) for k in labels}
    macro = round(
        statistics.fmean([v["f1"] for v in per_label.values()]) if per_label else 0.0, 4
    )
    n = len(pairs)
    return {
        "n": n,
        "answered": answered,
        "abstained": n - answered,
        "coverage": round(answered / n, 4) if n else 0.0,
        "accuracy_overall": round(correct / n, 4) if n else 0.0,
        "accuracy_when_answered": round(correct / answered, 4) if answered else 0.0,
        "macro_f1": macro,
        "per_label": per_label,
        "confusion": {f"{t} -> {g}": c for (t, g), c in confusion.most_common(10)},
        **coverage_at_error(conf_items, unknown_flags),
    }


def _norm_of(item: dict) -> str:
    v = item.get("normalized_value")
    if v is None:
        v = item.get("value")
    return str(v).strip() if v is not None else ""


def eval_metadata(pairs: list[tuple[dict, dict]]) -> dict:
    """Scored two ways, because they answer different questions.

    value: did we get the right answer.
    span:  did we point at the right place in the text.
    Span needs gold offsets, so it is skipped where gold has none.
    """
    val: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0}
    )
    span: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0}
    )
    span_scored = 0
    validated_tp = validated_fp = 0
    errors: list[str] = []

    for g, p in pairs:
        gold_fields = g.get("metadata") or []
        pred_fields = p.get("metadata") or []

        gv: dict[str, set] = collections.defaultdict(set)
        gs: dict[str, set] = collections.defaultdict(set)
        has_spans = False
        for item in gold_fields:
            name = item.get("field")
            if not name:
                continue
            gv[name].add(_norm_of(item))
            if item.get("char_start") is not None and item.get("char_end") is not None:
                gs[name].add((int(item["char_start"]), int(item["char_end"])))
                has_spans = True

        pv: dict[str, set] = collections.defaultdict(set)
        ps: dict[str, set] = collections.defaultdict(set)
        for item in pred_fields:
            name = item.get("field")
            if not name:
                continue
            pv[name].add(_norm_of(item))
            ps[name].add((int(item.get("char_start", -1)), int(item.get("char_end", -1))))
            if item.get("validated"):
                truth = gv.get(name, set())
                if _norm_of(item) in truth:
                    validated_tp += 1
                elif truth:
                    validated_fp += 1
                    errors.append(
                        f"{_keys_of(g)[0]}: {name} validated but wrong, "
                        f"got {_norm_of(item)!r} want {sorted(truth)!r}"
                    )

        for name in set(gv) | set(pv):
            got, want = pv.get(name, set()), gv.get(name, set())
            val[name]["tp"] += len(got & want)
            val[name]["fp"] += len(got - want)
            val[name]["fn"] += len(want - got)
            if want and not (got & want):
                errors.append(
                    f"{_keys_of(g)[0]}: {name} want {sorted(want)!r} got {sorted(got) or 'nothing'!r}"
                )

        if has_spans:
            span_scored += 1
            for name in set(gs) | set(ps):
                got, want = ps.get(name, set()), gs.get(name, set())
                span[name]["tp"] += len(got & want)
                span[name]["fp"] += len(got - want)
                span[name]["fn"] += len(want - got)

    def _roll(d):
        per = {k: prf(v["tp"], v["fp"], v["fn"]) for k, v in sorted(d.items())}
        tp = sum(v["tp"] for v in d.values())
        fp = sum(v["fp"] for v in d.values())
        fn = sum(v["fn"] for v in d.values())
        micro = prf(tp, fp, fn)
        macro = round(statistics.fmean([v["f1"] for v in per.values()]) if per else 0.0, 4)
        return {"per_field": per, "micro": micro, "macro_f1": macro}

    return {
        "value": _roll(val),
        "span": _roll(span) if span_scored else {"per_field": {}, "note": "gold has no offsets"},
        "span_scored_docs": span_scored,
        "validated_precision": round(
            validated_tp / (validated_tp + validated_fp), 4
        ) if validated_tp + validated_fp else None,
        "validated_wrong": validated_fp,
        "errors": errors[:40],
    }


def eval_tagging(pairs: list[tuple[dict, dict]]) -> dict:
    per: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0}
    )
    for g, p in pairs:
        want = set(g.get("tags") or [])
        got = {t["tag"] for t in (p.get("tags") or []) if t.get("tag")}
        for t in want | got:
            if t in want and t in got:
                per[t]["tp"] += 1
            elif t in got:
                per[t]["fp"] += 1
            else:
                per[t]["fn"] += 1

    per_tag = {k: prf(v["tp"], v["fp"], v["fn"]) for k, v in sorted(per.items())}
    tp = sum(v["tp"] for v in per.values())
    fp = sum(v["fp"] for v in per.values())
    fn = sum(v["fn"] for v in per.values())
    return {
        "per_tag": per_tag,
        "micro": prf(tp, fp, fn),
        "macro_f1": round(
            statistics.fmean([v["f1"] for v in per_tag.values()]) if per_tag else 0.0, 4
        ),
    }


# --------------------------------------------------------------------------
# tables
#
# Two metrics, because a table can be wrong in two different ways. Content
# asks whether the values came out at all. Adjacency asks whether they came
# out in the right relation to each other, which is the question that matters:
# a figure that drifts one column left is still perfect content and a wrong
# table. Adjacency is the standard structural metric for exactly that reason,
# and it does not need the predicted grid to line up index for index with the
# gold one, so a missed header row does not cascade into every later score.
# --------------------------------------------------------------------------


def _norm_cell(text: str) -> str:
    return " ".join(str(text or "").split()).strip().lower()


def _grid_of_gold(table: dict) -> list[list[str]]:
    return [[_norm_cell(c) for c in row] for row in table.get("cells", []) or []]


def _grid_of_pred(table: dict) -> list[list[str]]:
    n_rows = int(table.get("n_rows", 0))
    n_cols = int(table.get("n_cols", 0))
    grid = [["" for _ in range(n_cols)] for _ in range(n_rows)]
    for c in table.get("cells", []) or []:
        r, k = int(c.get("row", 0)), int(c.get("col", 0))
        if 0 <= r < n_rows and 0 <= k < n_cols:
            grid[r][k] = _norm_cell(c.get("text", ""))
    return grid


def _cell_bag(grids: list[list[list[str]]]) -> collections.Counter:
    bag: collections.Counter = collections.Counter()
    for g in grids:
        for row in g:
            for v in row:
                if v:
                    bag[v] += 1
    return bag


def _adjacency_bag(grids: list[list[list[str]]]) -> collections.Counter:
    """Nearest non empty neighbour to the right and below, per cell.

    Blank cells are skipped rather than treated as content, so a sparse table
    still states that its two populated columns sit next to each other.
    """
    bag: collections.Counter = collections.Counter()
    for g in grids:
        for row in g:
            filled = [v for v in row if v]
            for a, b in zip(filled, filled[1:]):
                bag[(a, b, "h")] += 1
        width = max((len(r) for r in g), default=0)
        for c in range(width):
            col = [r[c] for r in g if c < len(r) and r[c]]
            for a, b in zip(col, col[1:]):
                bag[(a, b, "v")] += 1
    return bag


def _bag_prf(gold: collections.Counter, pred: collections.Counter) -> dict:
    tp = sum((gold & pred).values())
    return prf(tp, sum(pred.values()) - tp, sum(gold.values()) - tp)


def eval_tables(pairs: list[tuple[dict, dict]]) -> dict:
    """Score a tables file against gold grids. Silent when gold has none."""
    scored = 0
    gold_cells: collections.Counter = collections.Counter()
    pred_cells: collections.Counter = collections.Counter()
    gold_adj: collections.Counter = collections.Counter()
    pred_adj: collections.Counter = collections.Counter()
    n_gold_tables = n_pred_tables = detected = 0

    for g, p in pairs:
        gt = g.get("tables")
        if gt is None:
            continue
        scored += 1
        g_grids = [_grid_of_gold(t) for t in gt]
        p_grids = [_grid_of_pred(t) for t in (p.get("tables") or [])]
        n_gold_tables += len(g_grids)
        n_pred_tables += len(p_grids)

        for gg in g_grids:
            want = _cell_bag([gg])
            if not want:
                continue
            # A gold table counts as found when some predicted table carries
            # over half its content. Deliberately loose: this measures whether
            # the region was located, not whether the grid is right.
            for pg in p_grids:
                if sum((want & _cell_bag([pg])).values()) / sum(want.values()) > 0.5:
                    detected += 1
                    break

        gold_cells += _cell_bag(g_grids)
        pred_cells += _cell_bag(p_grids)
        gold_adj += _adjacency_bag(g_grids)
        pred_adj += _adjacency_bag(p_grids)

    return {
        "scored": scored,
        "gold_tables": n_gold_tables,
        "pred_tables": n_pred_tables,
        "detected_tables": detected,
        "detection_recall": round(detected / n_gold_tables, 4) if n_gold_tables else 0.0,
        "cells": _bag_prf(gold_cells, pred_cells),
        "adjacency": _bag_prf(gold_adj, pred_adj),
    }


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
