"""The harness itself. If scoring is wrong, every decision after it is wrong."""

import json

import pytest

from baseline.evaluate import (
    compare,
    coverage_at_error,
    eval_classification,
    eval_metadata,
    eval_tagging,
    evaluate,
    join,
    levenshtein,
    prf,
    eval_tables,
)


def _pred(path, label, conf=0.9, fields=(), tags=(), text=""):
    return {
        "doc_id": path.replace("/", "_"),
        "source_path": path,
        "text": {"full": text},
        "classification": {"label": label, "confidence": conf, "method": "rule"},
        "metadata": [dict(f) for f in fields],
        "tags": [{"tag": t, "confidence": 0.9, "method": "keyword"} for t in tags],
    }


def _gold(path, label, fields=(), tags=(), text=None):
    row = {"path": path, "label": label, "metadata": [dict(f) for f in fields],
           "tags": list(tags)}
    if text is not None:
        row["text"] = text
    return row


# ---------------------------------------------------------------- primitives

def test_prf_is_arithmetically_right():
    assert prf(0, 0, 0)["f1"] == 0.0
    r = prf(8, 2, 2)
    assert r["precision"] == 0.8 and r["recall"] == 0.8 and r["f1"] == 0.8


@pytest.mark.parametrize(
    "a,b,want",
    [("", "", 0), ("abc", "abc", 0), ("abc", "abd", 1), ("abc", "", 3),
     ("kitten", "sitting", 3)],
)
def test_levenshtein(a, b, want):
    assert levenshtein(a, b) == want


# ---------------------------------------------------------------- joining

def test_join_matches_on_doc_id_then_path_then_filename():
    gold = [{"doc_id": "abc", "label": "x"}, {"path": "dir/b.pdf", "label": "y"}]
    pred = [{"doc_id": "abc"}, {"source_path": "other/b.pdf"}]
    pairs, missing, extra = join(gold, pred)
    assert len(pairs) == 2
    assert not missing and not extra


def test_join_reports_unmatched_gold():
    gold = [{"doc_id": "abc", "label": "x"}, {"doc_id": "nope", "label": "y"}]
    pred = [{"doc_id": "abc"}]
    pairs, missing, extra = join(gold, pred)
    assert len(pairs) == 1
    assert missing == ["nope"]


def test_join_reports_predictions_with_no_gold():
    gold = [{"doc_id": "abc", "label": "x"}]
    pred = [{"doc_id": "abc"}, {"doc_id": "spare"}]
    _, _, extra = join(gold, pred)
    assert extra == ["spare"]


# ---------------------------------------------------------------- classification

def test_perfect_classification_scores_one():
    pairs = [
        (_gold("a", "invoice"), _pred("a", "invoice")),
        (_gold("b", "receipt"), _pred("b", "receipt")),
    ]
    r = eval_classification(pairs)
    assert r["macro_f1"] == 1.0
    assert r["accuracy_overall"] == 1.0
    assert r["coverage"] == 1.0


def test_abstaining_lowers_coverage_but_not_answered_accuracy():
    """Declining to answer is not the same as answering wrongly."""
    pairs = [
        (_gold("a", "invoice"), _pred("a", "invoice")),
        (_gold("b", "receipt"), _pred("b", "unknown", conf=0.1)),
    ]
    r = eval_classification(pairs)
    assert r["coverage"] == 0.5
    assert r["accuracy_when_answered"] == 1.0
    assert r["accuracy_overall"] == 0.5


def test_wrong_answer_is_counted_against_both():
    pairs = [
        (_gold("a", "invoice"), _pred("a", "receipt")),
        (_gold("b", "receipt"), _pred("b", "receipt")),
    ]
    r = eval_classification(pairs)
    assert r["accuracy_when_answered"] == 0.5
    assert r["per_label"]["invoice"]["recall"] == 0.0
    assert r["per_label"]["receipt"]["precision"] == 0.5


def test_confusion_records_the_mistake():
    pairs = [(_gold("a", "invoice"), _pred("a", "receipt"))]
    r = eval_classification(pairs)
    assert "invoice -> receipt" in r["confusion"]


def test_coverage_at_error_prefers_confident_answers():
    """High confidence right, low confidence wrong. Thresholding should help."""
    items = [(0.99, True), (0.98, True), (0.20, False)]
    flags = [False, False, False]
    r = coverage_at_error(items, flags, targets=(0.01,))
    # scores are rounded to 4dp, so compare at that resolution
    assert r["coverage_at_1pct_error"] == pytest.approx(2 / 3, abs=1e-4)
    assert r["threshold_at_1pct_error"] >= 0.98


def test_coverage_at_error_is_zero_when_the_top_answer_is_wrong():
    items = [(0.99, False), (0.5, True)]
    r = coverage_at_error(items, [False, False], targets=(0.01,))
    assert r["coverage_at_1pct_error"] == 0.0


# ---------------------------------------------------------------- metadata

def test_metadata_value_scoring_counts_misses_and_spurious_hits():
    gold = _gold("a", "invoice", fields=[
        {"field": "pan", "normalized_value": "AAPFU0939F"},
        {"field": "ifsc", "normalized_value": "HDFC0001234"},
    ])
    pred = _pred("a", "invoice", fields=[
        {"field": "pan", "normalized_value": "AAPFU0939F", "char_start": 0, "char_end": 10},
        {"field": "email", "normalized_value": "x@y.z", "char_start": 0, "char_end": 5},
    ])
    r = eval_metadata([(gold, pred)])
    per = r["value"]["per_field"]
    assert per["pan"]["f1"] == 1.0
    assert per["ifsc"]["recall"] == 0.0     # missed
    assert per["email"]["precision"] == 0.0  # invented


def test_span_scoring_is_skipped_without_gold_offsets():
    gold = _gold("a", "invoice", fields=[{"field": "pan", "normalized_value": "X"}])
    pred = _pred("a", "invoice",
                 fields=[{"field": "pan", "normalized_value": "X",
                          "char_start": 3, "char_end": 4}])
    r = eval_metadata([(gold, pred)])
    assert r["span_scored_docs"] == 0
    assert "note" in r["span"]


def test_span_scoring_runs_when_gold_has_offsets():
    gold = _gold("a", "invoice", fields=[
        {"field": "pan", "normalized_value": "X", "char_start": 3, "char_end": 4}
    ])
    right = _pred("a", "invoice", fields=[
        {"field": "pan", "normalized_value": "X", "char_start": 3, "char_end": 4}
    ])
    wrong = _pred("a", "invoice", fields=[
        {"field": "pan", "normalized_value": "X", "char_start": 9, "char_end": 10}
    ])
    assert eval_metadata([(gold, right)])["span"]["micro"]["f1"] == 1.0
    # Right answer, wrong place. Value scoring passes, span scoring does not.
    r = eval_metadata([(gold, wrong)])
    assert r["value"]["micro"]["f1"] == 1.0
    assert r["span"]["micro"]["f1"] == 0.0


def test_validated_precision_catches_a_lying_checksum():
    gold = _gold("a", "invoice", fields=[{"field": "pan", "normalized_value": "RIGHT"}])
    pred = _pred("a", "invoice", fields=[
        {"field": "pan", "normalized_value": "WRONG", "char_start": 0,
         "char_end": 5, "validated": True}
    ])
    r = eval_metadata([(gold, pred)])
    assert r["validated_precision"] == 0.0
    assert r["validated_wrong"] == 1


def test_unvalidated_field_does_not_count_against_validated_precision():
    gold = _gold("a", "invoice", fields=[{"field": "pan", "normalized_value": "RIGHT"}])
    pred = _pred("a", "invoice", fields=[
        {"field": "pan", "normalized_value": "WRONG", "char_start": 0,
         "char_end": 5, "validated": False}
    ])
    assert eval_metadata([(gold, pred)])["validated_precision"] is None


# ---------------------------------------------------------------- tagging

def test_tagging_is_scored_per_tag_not_as_a_set():
    gold = _gold("a", "invoice", tags=["gst_applicable", "signed"])
    pred = _pred("a", "invoice", tags=["gst_applicable", "payment_made"])
    r = eval_tagging([(gold, pred)])
    assert r["per_tag"]["gst_applicable"]["f1"] == 1.0
    assert r["per_tag"]["signed"]["recall"] == 0.0
    assert r["per_tag"]["payment_made"]["precision"] == 0.0
    assert r["micro"]["f1"] == pytest.approx(0.5)


# ---------------------------------------------------------------- end to end

def _write(tmp_path, name, rows):
    p = tmp_path / name
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return p


def test_a_perfect_run_scores_one_everywhere(tmp_path):
    fields = [{"field": "pan", "normalized_value": "AAPFU0939F"}]
    gold = _write(tmp_path, "g.jsonl",
                  [_gold("a", "invoice", fields=fields, tags=["signed"])])
    pred = _write(tmp_path, "p.jsonl", [
        _pred("a", "invoice",
              fields=[{**fields[0], "char_start": 0, "char_end": 10}], tags=["signed"])
    ])
    res = evaluate(gold, pred)
    assert res["matched"] == 1
    assert res["classification"]["macro_f1"] == 1.0
    assert res["metadata"]["value"]["micro"]["f1"] == 1.0
    assert res["tagging"]["micro"]["f1"] == 1.0


def test_cer_is_scored_only_when_gold_carries_text(tmp_path):
    gold = _write(tmp_path, "g.jsonl", [_gold("a", "invoice", text="hello world")])
    pred = _write(tmp_path, "p.jsonl", [_pred("a", "invoice", text="hello worlds")])
    res = evaluate(gold, pred)
    assert res["extraction"]["scored"] == 1
    assert 0 < res["extraction"]["cer"] < 0.2


def test_cer_ignores_layout_whitespace(tmp_path):
    gold = _write(tmp_path, "g.jsonl", [_gold("a", "invoice", text="hello world")])
    pred = _write(tmp_path, "p.jsonl", [_pred("a", "invoice", text="hello\n\n   world")])
    assert evaluate(gold, pred)["extraction"]["cer"] == 0.0


def test_compare_flags_a_regression(tmp_path):
    fields = [{"field": "pan", "normalized_value": "AAPFU0939F"}]
    gold = _write(tmp_path, "g.jsonl", [_gold("a", "invoice", fields=fields)])
    good = _write(tmp_path, "a.jsonl", [
        _pred("a", "invoice", fields=[{**fields[0], "char_start": 0, "char_end": 10}])
    ])
    bad = _write(tmp_path, "b.jsonl", [_pred("a", "invoice", fields=[])])

    worse = compare(gold, good, bad)
    assert worse["regressions"]
    assert worse["headline"]["metadata_micro_f1"]["delta"] < 0

    better = compare(gold, bad, good)
    assert better["regressions"] == []
    assert better["headline"]["metadata_micro_f1"]["delta"] > 0


def test_harness_scores_the_real_baseline_output(tmp_path):
    """The harness must not depend on the pipeline, only on the schema."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    if not (root / "results.jsonl").exists() or not (root / "gold.jsonl").exists():
        pytest.skip("no run output on disk")
    res = evaluate(root / "gold.jsonl", root / "results.jsonl")
    assert res["matched"] == res["gold_count"]
    # Anything claiming a checksum passed must actually be right.
    assert res["metadata"]["validated_precision"] == 1.0


# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------


def _gold_table(cells, **kw):
    row = {"doc_id": "a" * 16, "label": "invoice", "tags": [], "metadata": [],
           "tables": [{"page": 0, "cells": cells}]}
    row.update(kw)
    return row


def _pred_table(grid):
    cells = [
        {"row": r, "col": c, "text": v, "page": 0}
        for r, row in enumerate(grid) for c, v in enumerate(row) if v
    ]
    return {"doc_id": "a" * 16, "source_path": "x.pdf", "filename": "x.pdf",
            "tables": [{"table_id": "t", "page": 0, "n_rows": len(grid),
                        "n_cols": max(len(r) for r in grid), "cells": cells}]}


GRID = [["Qty", "Rate", "Amount"], ["2", "10.00", "20.00"]]


def test_an_exact_table_scores_one():
    res = eval_tables([(_gold_table(GRID), _pred_table(GRID))])
    assert res["cells"]["f1"] == 1.0
    assert res["adjacency"]["f1"] == 1.0
    assert res["detection_recall"] == 1.0


def test_a_value_in_the_wrong_column_keeps_cell_f1_but_loses_adjacency():
    """The failure the content metric cannot see, which is why both exist."""
    shifted = [["Qty", "Rate", "Amount"], ["2", "20.00", "10.00"]]
    res = eval_tables([(_gold_table(GRID), _pred_table(shifted))])
    assert res["cells"]["f1"] == 1.0, "the same values are all present"
    assert res["adjacency"]["f1"] < 1.0, "but they sit beside the wrong neighbours"


def test_a_table_invented_where_gold_has_none_costs_precision():
    gold = {"doc_id": "a" * 16, "label": "contract", "tags": [], "metadata": [],
            "tables": []}
    res = eval_tables([(gold, _pred_table(GRID))])
    assert res["cells"]["precision"] == 0.0
    assert res["cells"]["recall"] == 0.0


def test_a_missed_table_costs_recall_not_precision():
    empty = {"doc_id": "a" * 16, "source_path": "x.pdf", "filename": "x.pdf", "tables": []}
    res = eval_tables([(_gold_table(GRID), empty)])
    assert res["cells"]["recall"] == 0.0
    assert res["detection_recall"] == 0.0
    assert res["cells"]["precision"] == 0.0
    assert res["cells"]["fp"] == 0, "nothing was claimed, so nothing is a false positive"


def test_gold_without_a_tables_key_is_not_scored():
    """Silence, not zeros. A gold file that never mentions tables says nothing."""
    gold = {"doc_id": "a" * 16, "label": "invoice", "tags": [], "metadata": []}
    res = eval_tables([(gold, _pred_table(GRID))])
    assert res["scored"] == 0


def test_empty_cells_do_not_break_adjacency():
    """A sparse row still asserts that its populated columns are neighbours."""
    sparse = [["Date", "Debit", "Credit"], ["01/03", "", "50.00"]]
    res = eval_tables([(_gold_table(sparse), _pred_table(sparse))])
    assert res["adjacency"]["f1"] == 1.0
