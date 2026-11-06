"""Joining gold to predictions, and scoring each component of a record."""

from __future__ import annotations

import collections
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypedDict

# A gold or predicted record, as read from JSONL.
Record = dict[str, Any]
Pair = tuple[Record, Record]

# Gold rows may key on either. doc_id is preferred, path is the fallback.
KEYS = ("doc_id", "source_path", "path")

# Error rates the coverage curve is read at.
COVERAGE_TARGETS = (0.01, 0.05, 0.10)
# Most common classification mistakes kept in a result.
CONFUSION_TOP = 10


class PRF(TypedDict):
    """Precision, recall and F1 with the counts behind them."""

    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


def keys_of(row: Record) -> list[str]:
    """Every join key a row offers, full value then file name."""
    out = []
    for k in KEYS:
        v = row.get(k)
        if v:
            out.append(str(v))
            out.append(Path(str(v)).name)
    return out


def join(gold: list[Record], pred: list[Record]) -> tuple[list[Pair], list[str], list[str]]:
    """Match gold to predictions on doc_id, then path, then filename.

    Args:
        gold: Gold records.
        pred: Predicted records.

    Returns:
        Matched (gold, prediction) pairs, keys of gold rows with no
        prediction, and keys of predictions no gold row used.
    """
    index: dict[str, Record] = {}
    for p in pred:
        for k in keys_of(p):
            index.setdefault(k, p)

    pairs: list[Pair] = []
    missing: list[str] = []
    used = set()
    for g in gold:
        hit = None
        for k in keys_of(g):
            if k in index:
                hit = index[k]
                break
        if hit is None:
            missing.append(keys_of(g)[0] if keys_of(g) else "?")
        else:
            pairs.append((g, hit))
            used.add(id(hit))
    extra = [keys_of(p)[0] for p in pred if id(p) not in used]
    return pairs, missing, extra


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def prf(tp: int, fp: int, fn: int) -> PRF:
    """Precision, recall and F1 from raw counts.

    Args:
        tp: True positives.
        fp: False positives.
        fn: False negatives.

    Returns:
        The three rates rounded to four places, plus the counts.
    """
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4), "tp": tp, "fp": fp, "fn": fn}


def levenshtein(a: str, b: str) -> int:
    """Two row edit distance. Fine for page sized strings.

    Args:
        a: First string.
        b: Second string.

    Returns:
        Character edit distance between the two.
    """
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


def counts() -> dict[str, int]:
    """Fresh tp/fp/fn counter."""
    return {"tp": 0, "fp": 0, "fn": 0}


def macro_f1(per: dict[str, PRF]) -> float:
    """Mean F1 over the entries, zero when there are none."""
    return round(statistics.fmean([v["f1"] for v in per.values()]) if per else 0.0, 4)


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


def eval_extraction(pairs: list[Pair]) -> dict[str, Any]:
    """CER and WER, only over gold rows that carry reference text.

    Args:
        pairs: Matched (gold, prediction) records.

    Returns:
        `scored`, and `cer` and `wer` when anything was scored.
    """
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


def coverage_at_error(
    items: list[tuple[float, bool]],
    unknown_flags: list[bool],
    targets: Sequence[float] = COVERAGE_TARGETS,
) -> dict[str, float]:
    """Max coverage achievable while staying under each error rate.

    Coverage is the share of all documents answered. Error is measured only
    among answered ones. This is the metric that matters when the system is
    allowed to decline.

    Args:
        items: (confidence, correct) per document.
        unknown_flags: True where the document was not answered.
        targets: Error rates to read the curve at.

    Returns:
        `coverage_at_<n>pct_error` and `threshold_at_<n>pct_error` per target.
    """
    total = len(items)
    if not total:
        return {}
    # Only non unknown predictions count as answers.
    answers = [(c, ok) for (c, ok), unk in zip(items, unknown_flags, strict=False) if not unk]
    answers.sort(key=lambda t: -t[0])

    out: dict[str, float] = {}
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


def _count_label(per: dict[str, dict[str, int]], truth: str, got: str, ok: bool, is_unknown: bool) -> None:
    """Add one classification outcome to the per label counts."""
    if ok:
        per[truth]["tp"] += 1
    else:
        per[truth]["fn"] += 1
        if not is_unknown:
            per[got]["fp"] += 1


def eval_classification(pairs: list[Pair], unknown_label: str = "unknown") -> dict[str, Any]:
    """Accuracy, coverage, per label F1 and the most common mistakes.

    Args:
        pairs: Matched (gold, prediction) records.
        unknown_label: Label the system uses to abstain.

    Returns:
        Summary counts and rates, `per_label` scores, `confusion`, and the
        coverage at error figures.
    """
    per: dict[str, dict[str, int]] = collections.defaultdict(counts)
    correct = answered = 0
    conf_items: list[tuple[float, bool]] = []
    unknown_flags: list[bool] = []
    confusion: collections.Counter[tuple[str, str]] = collections.Counter()

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
        _count_label(per, truth, got, ok, is_unknown)

    per_label = {k: prf(per[k]["tp"], per[k]["fp"], per[k]["fn"]) for k in sorted(per)}
    n = len(pairs)
    return {
        "n": n,
        "answered": answered,
        "abstained": n - answered,
        "coverage": round(answered / n, 4) if n else 0.0,
        "accuracy_overall": round(correct / n, 4) if n else 0.0,
        "accuracy_when_answered": round(correct / answered, 4) if answered else 0.0,
        "macro_f1": macro_f1(per_label),
        "per_label": per_label,
        "confusion": {f"{t} -> {g}": c for (t, g), c in confusion.most_common(CONFUSION_TOP)},
        **coverage_at_error(conf_items, unknown_flags),
    }


def eval_tagging(pairs: list[Pair]) -> dict[str, Any]:
    """Per tag, micro and macro F1 over the tag sets of each document.

    Args:
        pairs: Matched (gold, prediction) records.

    Returns:
        `per_tag`, `micro` and `macro_f1`.
    """
    per: dict[str, dict[str, int]] = collections.defaultdict(counts)
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
    return {"per_tag": per_tag, "micro": prf(tp, fp, fn), "macro_f1": macro_f1(per_tag)}
