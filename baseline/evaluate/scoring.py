"""Joining gold to predictions, and scoring each component of a record."""

from __future__ import annotations

import collections
import json
import statistics
from pathlib import Path
from typing import Any

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
