"""Metadata scoring: values, spans, and `validated` claims."""

from __future__ import annotations

import collections
from collections.abc import Iterable
from typing import Any

from .scoring import Pair, Record, counts, keys_of, macro_f1, prf

# Field name to the set of values, or (start, end) spans, seen for it.
FieldSets = dict[str, set[Any]]

# Metadata miss messages kept in a result.
MAX_ERRORS = 40


def norm_of(item: Record) -> str:
    """Normalized value of a metadata item, falling back to the raw value.

    Args:
        item: One metadata entry.

    Returns:
        The stripped value, or an empty string when there is none.
    """
    v = item.get("normalized_value")
    if v is None:
        v = item.get("value")
    return str(v).strip() if v is not None else ""


class _MetaTally:
    """Running counts for `eval_metadata` across documents."""

    def __init__(self) -> None:
        self.val: dict[str, dict[str, int]] = collections.defaultdict(counts)
        self.span: dict[str, dict[str, int]] = collections.defaultdict(counts)
        self.span_scored = 0
        self.validated_tp = 0
        self.validated_fp = 0
        self.errors: list[str] = []


def _gold_sets(gold_fields: Iterable[Record]) -> tuple[FieldSets, FieldSets, bool]:
    """Gold values and spans per field, and whether any span was present."""
    gv: FieldSets = collections.defaultdict(set)
    gs: FieldSets = collections.defaultdict(set)
    has_spans = False
    for item in gold_fields:
        name = item.get("field")
        if not name:
            continue
        gv[name].add(norm_of(item))
        if item.get("char_start") is not None and item.get("char_end") is not None:
            gs[name].add((int(item["char_start"]), int(item["char_end"])))
            has_spans = True
    return gv, gs, has_spans


def _pred_sets(
    doc: str, pred_fields: Iterable[Record], gv: FieldSets, tally: _MetaTally
) -> tuple[FieldSets, FieldSets]:
    """Predicted values and spans per field. Also scores `validated` claims."""
    pv: FieldSets = collections.defaultdict(set)
    ps: FieldSets = collections.defaultdict(set)
    for item in pred_fields:
        name = item.get("field")
        if not name:
            continue
        pv[name].add(norm_of(item))
        ps[name].add((int(item.get("char_start", -1)), int(item.get("char_end", -1))))
        if item.get("validated"):
            truth = gv.get(name, set())
            if norm_of(item) in truth:
                tally.validated_tp += 1
            elif truth:
                tally.validated_fp += 1
                tally.errors.append(f"{doc}: {name} validated but wrong, got {norm_of(item)!r} want {sorted(truth)!r}")
    return pv, ps


def _count_sets(counts: dict[str, dict[str, int]], got_map: FieldSets, want_map: FieldSets) -> None:
    """Add tp/fp/fn per field from predicted and gold sets."""
    for name in set(want_map) | set(got_map):
        got, want = got_map.get(name, set()), want_map.get(name, set())
        counts[name]["tp"] += len(got & want)
        counts[name]["fp"] += len(got - want)
        counts[name]["fn"] += len(want - got)


def _roll(d: dict[str, dict[str, int]]) -> dict[str, Any]:
    """Per field, micro and macro scores from per field counts."""
    per = {k: prf(v["tp"], v["fp"], v["fn"]) for k, v in sorted(d.items())}
    tp = sum(v["tp"] for v in d.values())
    fp = sum(v["fp"] for v in d.values())
    fn = sum(v["fn"] for v in d.values())
    return {"per_field": per, "micro": prf(tp, fp, fn), "macro_f1": macro_f1(per)}


def _score_metadata_doc(g: Record, p: Record, tally: _MetaTally) -> None:
    """Add one document's value and span counts to the tally."""
    gv, gs, has_spans = _gold_sets(g.get("metadata") or [])
    keys = keys_of(g)
    doc = keys[0] if keys else "?"
    pv, ps = _pred_sets(doc, p.get("metadata") or [], gv, tally)
    for name in set(gv) | set(pv):
        got, want = pv.get(name, set()), gv.get(name, set())
        tally.val[name]["tp"] += len(got & want)
        tally.val[name]["fp"] += len(got - want)
        tally.val[name]["fn"] += len(want - got)
        if want and not (got & want):
            tally.errors.append(f"{doc}: {name} want {sorted(want)!r} got {sorted(got) or 'nothing'!r}")
    if has_spans:
        tally.span_scored += 1
        _count_sets(tally.span, ps, gs)


def eval_metadata(pairs: list[Pair]) -> dict[str, Any]:
    """Scored two ways, because they answer different questions.

    value: did we get the right answer.
    span:  did we point at the right place in the text.
    Span needs gold offsets, so it is skipped where gold has none.

    Args:
        pairs: Matched (gold, prediction) records.

    Returns:
        `value` and `span` scores, `span_scored_docs`, the precision of
        `validated` fields, and up to `MAX_ERRORS` miss messages.
    """
    tally = _MetaTally()
    for g, p in pairs:
        _score_metadata_doc(g, p, tally)
    checked = tally.validated_tp + tally.validated_fp
    return {
        "value": _roll(tally.val),
        "span": _roll(tally.span) if tally.span_scored else {"per_field": {}, "note": "gold has no offsets"},
        "span_scored_docs": tally.span_scored,
        "validated_precision": round(tally.validated_tp / checked, 4) if checked else None,
        "validated_wrong": tally.validated_fp,
        "errors": tally.errors[:MAX_ERRORS],
    }
