"""Scoring a tables file against gold grids.

Two metrics, because a table can be wrong in two different ways. Content
asks whether the values came out at all. Adjacency asks whether they came
out in the right relation to each other, which is the question that matters:
a figure that drifts one column left is still perfect content and a wrong
table. Adjacency is the standard structural metric for exactly that reason,
and it does not need the predicted grid to line up index for index with the
gold one, so a missed header row does not cascade into every later score.
"""

from __future__ import annotations

import collections
from typing import Any

from .scoring import PRF, Pair, prf

# Share of a gold table's content a predicted table must carry to count as found.
DETECTION_OVERLAP = 0.5
# Bag of normalized cell texts, or of (a, b, direction) neighbour triples.
Bag = collections.Counter[Any]


def _norm_cell(text: str) -> str:
    """Collapse whitespace and lowercase."""
    return " ".join(str(text or "").split()).strip().lower()


def _grid_of_gold(table: dict[str, Any]) -> list[list[str]]:
    """Normalized grid of a gold table."""
    return [[_norm_cell(c) for c in row] for row in table.get("cells", []) or []]


def _grid_of_pred(table: dict[str, Any]) -> list[list[str]]:
    """Dense normalized grid of a predicted table, cells out of range dropped."""
    n_rows = int(table.get("n_rows", 0))
    n_cols = int(table.get("n_cols", 0))
    grid = [["" for _ in range(n_cols)] for _ in range(n_rows)]
    for c in table.get("cells", []) or []:
        r, k = int(c.get("row", 0)), int(c.get("col", 0))
        if 0 <= r < n_rows and 0 <= k < n_cols:
            grid[r][k] = _norm_cell(c.get("text", ""))
    return grid


def _cell_bag(grids: list[list[list[str]]]) -> Bag:
    """Count of each non empty cell text."""
    bag: Bag = collections.Counter()
    for g in grids:
        for row in g:
            for v in row:
                if v:
                    bag[v] += 1
    return bag


def _adjacency_bag(grids: list[list[list[str]]]) -> Bag:
    """Nearest non empty neighbour to the right and below, per cell.

    Blank cells are skipped rather than treated as content, so a sparse table
    still states that its two populated columns sit next to each other.
    """
    bag: Bag = collections.Counter()
    for g in grids:
        for row in g:
            filled = [v for v in row if v]
            for a, b in zip(filled, filled[1:], strict=False):
                bag[(a, b, "h")] += 1
        width = max((len(r) for r in g), default=0)
        for c in range(width):
            col = [r[c] for r in g if c < len(r) and r[c]]
            for a, b in zip(col, col[1:], strict=False):
                bag[(a, b, "v")] += 1
    return bag


def _bag_prf(gold: Bag, pred: Bag) -> PRF:
    """PRF of a predicted bag against a gold bag."""
    tp = sum((gold & pred).values())
    return prf(tp, sum(pred.values()) - tp, sum(gold.values()) - tp)


def _detected(g_grids: list[list[list[str]]], p_grids: list[list[list[str]]]) -> int:
    """Number of gold tables some predicted table locates."""
    detected = 0
    for gg in g_grids:
        want = _cell_bag([gg])
        if not want:
            continue
        # A gold table counts as found when some predicted table carries
        # over half its content. Deliberately loose: this measures whether
        # the region was located, not whether the grid is right.
        for pg in p_grids:
            if sum((want & _cell_bag([pg])).values()) / sum(want.values()) > DETECTION_OVERLAP:
                detected += 1
                break
    return detected


def eval_tables(pairs: list[Pair]) -> dict[str, Any]:
    """Score a tables file against gold grids. Silent when gold has none.

    Args:
        pairs: Matched (gold, tables record) pairs.

    Returns:
        Table counts, `detection_recall`, and `cells` and `adjacency` PRF.
    """
    scored = 0
    gold_cells: Bag = collections.Counter()
    pred_cells: Bag = collections.Counter()
    gold_adj: Bag = collections.Counter()
    pred_adj: Bag = collections.Counter()
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

        detected += _detected(g_grids, p_grids)

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
