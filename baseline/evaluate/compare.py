"""Two runs scored against the same gold, diffed, and written up."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .report import PathLike, evaluate


def _f1_delta(per_a: dict[str, Any], per_b: dict[str, Any]) -> dict[str, dict[str, float]]:
    """F1 of A and B and their difference, per key present in either."""
    out = {}
    for k in sorted(set(per_a) | set(per_b)):
        fa = per_a.get(k, {}).get("f1", 0.0)
        fb = per_b.get(k, {}).get("f1", 0.0)
        out[k] = {"a": fa, "b": fb, "delta": round(fb - fa, 4)}
    return out


# Headline metric name to its key path inside an `evaluate` result.
_HEADLINE_PATHS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("classification_macro_f1", ("classification", "macro_f1")),
    ("classification_coverage", ("classification", "coverage")),
    ("metadata_micro_f1", ("metadata", "value", "micro", "f1")),
    ("tagging_micro_f1", ("tagging", "micro", "f1")),
)
_TABLE_PATHS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("tables_cell_f1", ("tables", "cells", "f1")),
    ("tables_adjacency_f1", ("tables", "adjacency", "f1")),
    ("tables_detection_recall", ("tables", "detection_recall")),
)


def _dig(res: dict[str, Any], path: tuple[str, ...]) -> Any:
    """Follow a key path into a nested result."""
    node: Any = res
    for key in path:
        node = node[key]
    return node


def _headline(ra: dict[str, Any], rb: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Headline metrics of both runs, tables only when both scored them."""
    paths = _HEADLINE_PATHS
    # Table metrics only join the gate when both runs actually scored tables,
    # so a run without a tables file is not read as a regression to zero.
    if ra["tables"].get("scored") and rb["tables"].get("scored"):
        paths += _TABLE_PATHS
    out = {}
    for key, path in paths:
        va, vb = _dig(ra, path), _dig(rb, path)
        out[key] = {"a": va, "b": vb, "delta": round(vb - va, 4)}
    return out


def compare(
    gold_path: PathLike,
    a_path: PathLike,
    b_path: PathLike,
    unknown_label: str = "unknown",
    a_tables: PathLike | None = None,
    b_tables: PathLike | None = None,
) -> dict[str, Any]:
    """Score two prediction files against the same gold and diff them.

    Args:
        gold_path: Gold JSONL.
        a_path: Predictions of run A.
        b_path: Predictions of run B.
        unknown_label: Label the system uses to abstain.
        a_tables: Optional tables JSONL of run A.
        b_tables: Optional tables JSONL of run B.

    Returns:
        `headline`, `field_delta`, `tag_delta` and `regressions`, where a
        positive delta means B is better.

    Raises:
        ValueError: An input line is not valid JSON.
    """
    ra = evaluate(gold_path, a_path, unknown_label, a_tables)
    rb = evaluate(gold_path, b_path, unknown_label, b_tables)
    field_delta = _f1_delta(ra["metadata"]["value"]["per_field"], rb["metadata"]["value"]["per_field"])
    tag_delta = _f1_delta(ra["tagging"]["per_tag"], rb["tagging"]["per_tag"])
    headline = _headline(ra, rb)
    regressions = (
        [f"metadata {k}: {v['a']:.3f} -> {v['b']:.3f}" for k, v in field_delta.items() if v["delta"] < 0]
        + [f"tag {k}: {v['a']:.3f} -> {v['b']:.3f}" for k, v in tag_delta.items() if v["delta"] < 0]
        + [f"{k}: {v['a']:.3f} -> {v['b']:.3f}" for k, v in headline.items() if v["delta"] < 0]
    )
    return {"headline": headline, "field_delta": field_delta, "tag_delta": tag_delta, "regressions": regressions}


def _by_delta(kv: tuple[str, dict[str, float]]) -> float:
    """Sort key: worst delta first."""
    return kv[1]["delta"]


def _delta_table(label: str, rows: Iterable[tuple[str, dict[str, float]]]) -> list[str]:
    """A / B / delta table with a trailing blank line."""
    lines = [f"| {label} | A | B | delta |", "| --- | --- | --- | --- |"]
    lines += [f"| {k} | {v['a']:.3f} | {v['b']:.3f} | {v['delta']:+.3f} |" for k, v in rows]
    lines.append("")
    return lines


def render_comparison(cmp: dict[str, Any], a_path: str, b_path: str) -> str:
    """Render a `compare` result as markdown.

    Args:
        cmp: Result of `compare`.
        a_path: Run A path, shown in the header.
        b_path: Run B path, shown in the header.

    Returns:
        The comparison text.
    """
    lines: list[str] = []
    a = lines.append
    a("# Comparison")
    a("")
    a(f"A: `{a_path}`")
    a(f"B: `{b_path}`")
    a("")
    a("Positive delta means B is better.")
    a("")
    lines += _delta_table("metric", cmp["headline"].items())
    a("## Per field")
    a("")
    lines += _delta_table("field", sorted(cmp["field_delta"].items(), key=_by_delta))
    a("## Per tag")
    a("")
    lines += _delta_table("tag", sorted(cmp["tag_delta"].items(), key=_by_delta))

    a("## Regressions")
    a("")
    if cmp["regressions"]:
        for r in cmp["regressions"]:
            a(f"- {r}")
    else:
        a("None. B is at least as good as A everywhere.")
    a("")
    return "\n".join(lines)
