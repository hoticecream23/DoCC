"""Confidence calibration, and scoring restricted to gold-covered fields."""

from __future__ import annotations

import collections
import statistics
from typing import Any

from .metadata_scores import eval_metadata, norm_of
from .scoring import Pair, Record

# Lower bound inclusive, upper exclusive, except the top bucket keeps 1.0.
BUCKETS = (("0.5-0.7", 0.5, 0.7), ("0.7-0.9", 0.7, 0.9), ("0.9-1.0", 0.9, float("inf")))
# Confidence below the lowest bucket is counted but not bucketed.
BUCKET_FLOOR = BUCKETS[0][1]


def bucket_stats(items: list[tuple[float, bool]], metric: str = "precision") -> dict[str, Any]:
    """Per bucket count, hit rate and mean confidence, plus ECE.

    ECE = sum over buckets of (n_b / N) * |hit_rate_b - mean_confidence_b|,
    where N counts only predictions that fall in a bucket (confidence >= 0.5).

    Args:
        items: (confidence, correct) per prediction.
        metric: Name the hit rate is reported under.

    Returns:
        `buckets`, `n`, `below_0.5` and `ece` (None when nothing was bucketed).
    """
    buckets, n_total, ece_sum = {}, 0, 0.0
    for name, lo, hi in BUCKETS:
        hits = [(c, ok) for c, ok in items if lo <= c < hi]
        n = len(hits)
        rate = sum(ok for _, ok in hits) / n if n else 0.0
        mean_c = statistics.fmean(c for c, _ in hits) if n else 0.0
        buckets[name] = {"n": n, metric: round(rate, 4), "mean_confidence": round(mean_c, 4)}
        n_total += n
        ece_sum += n * abs(rate - mean_c)
    return {
        "buckets": buckets,
        "n": n_total,
        "below_0.5": sum(1 for c, _ in items if c < BUCKET_FLOOR),
        "ece": round(ece_sum / n_total, 4) if n_total else None,
    }


def _field_items(g: Record, p: Record) -> list[tuple[str, float, bool]]:
    """(field, best confidence, correct) per distinct predicted value."""
    gv: dict[str, set[str]] = collections.defaultdict(set)
    for item in g.get("metadata") or []:
        if item.get("field"):
            gv[item["field"]].add(norm_of(item))
    best: dict[tuple[str, str], float] = {}
    for item in p.get("metadata") or []:
        name, conf = item.get("field"), item.get("confidence")
        if not name or conf is None:
            continue
        key = (name, norm_of(item))
        best[key] = max(best.get(key, 0.0), float(conf))
    return [(name, conf, value in gv.get(name, set())) for (name, value), conf in best.items()]


def eval_calibration(pairs: list[Pair], unknown_label: str = "unknown") -> dict[str, Any]:
    """Bucket field values and classification answers by their stated confidence.

    A field value is right when it is in the gold set for that field, the
    same test as a value TP. Repeated values keep their highest confidence,
    so counts line up with tp + fp.

    Args:
        pairs: Matched (gold, prediction) records.
        unknown_label: Label the system uses to abstain.

    Returns:
        `fields` (overall and per field) and `classification` bucket stats.
    """
    per_field: dict[str, list[tuple[float, bool]]] = collections.defaultdict(list)
    cls_items: list[tuple[float, bool]] = []
    for g, p in pairs:
        for name, conf, ok in _field_items(g, p):
            per_field[name].append((conf, ok))

        c = p.get("classification") or {}
        got = str(c.get("label", "")).strip()
        if got and got != unknown_label and c.get("confidence") is not None:
            cls_items.append((float(c["confidence"]), got == str(g.get("label", "")).strip()))

    all_items = [x for v in per_field.values() for x in v]
    return {
        "fields": {
            "overall": bucket_stats(all_items),
            "per_field": {k: bucket_stats(v) for k, v in sorted(per_field.items())},
        },
        "classification": bucket_stats(cls_items, metric="accuracy"),
    }


# --------------------------------------------------------------------------
# gold-covered fields
# --------------------------------------------------------------------------


def covered_fields(gold: list[Record]) -> set[str]:
    """Field names the gold file labels at least once.

    Args:
        gold: Gold records.

    Returns:
        The set of field names.
    """
    return {i["field"] for g in gold for i in g.get("metadata") or [] if i.get("field")}


def eval_covered(pairs: list[Pair], covered: set[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Metadata and field calibration over gold-covered fields only.

    A predicted field the gold never labels cannot be judged, so it is
    "unscored" here instead of a false positive. Counted as distinct values
    per document, the same unit as a value FP, so the existing FP total
    equals the covered FP plus the unscored count.

    Args:
        pairs: Matched (gold, prediction) records.
        covered: Field names to score.

    Returns:
        The covered value scores with unscored counts, and the field
        calibration over the same covered fields.
    """
    unscored: collections.Counter[str] = collections.Counter()
    kept: list[Pair] = []
    for g, p in pairs:
        seen = set()
        for item in p.get("metadata") or []:
            name = item.get("field")
            if name and name not in covered and (name, norm_of(item)) not in seen:
                seen.add((name, norm_of(item)))
                unscored[name] += 1
        meta = [i for i in p.get("metadata") or [] if i.get("field") in covered]
        kept.append((g, {**p, "metadata": meta}))
    value = eval_metadata(kept)["value"]
    return (
        {
            "fields": sorted(covered),
            "micro": value["micro"],
            "macro_f1": value["macro_f1"],
            "unscored_predictions": sum(unscored.values()),
            "unscored_per_field": dict(sorted(unscored.items())),
        },
        eval_calibration(kept)["fields"],
    )
