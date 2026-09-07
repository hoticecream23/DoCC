"""Importing the reviewed index workbook back into gold.

The interesting logic is not the file format, it is what each verdict means.
`ok` keeps the pipeline's own answer, `wrong` and `missing` take the
correction, and `spurious` has to produce nothing at all, because the gold it
records is an absence. Getting that last one backwards would turn every
correctly rejected field into a false negative and quietly flatter the score.
"""

from __future__ import annotations

import pytest

openpyxl = pytest.importorskip("openpyxl")

from baseline.workbook import import_workbook, render_summary  # noqa: E402

DOC_COLS = ["DOC_ID", "FILE_NAME", "DOC_CLASS", "CLASS_CORRECTED", "CLASS_VERDICT", "IS_GOLD"]
META_COLS = ["DOC_ID", "ROW_SOURCE", "FIELD", "VALUE", "NORMALIZED_VALUE", "PAGE",
             "CHAR_START", "CHAR_END", "CORRECTED_VALUE", "VERDICT", "IS_GOLD"]
TAG_COLS = ["DOC_ID", "TAG", "ROW_SOURCE", "VERDICT", "IS_GOLD"]


def _write(path, docs, meta=(), tags=()):
    wb = openpyxl.Workbook()
    for name, cols, rows in (
        ("Documents", DOC_COLS, docs),
        ("Metadata", META_COLS, meta),
        ("Tags", TAG_COLS, tags),
    ):
        ws = wb.create_sheet(name)
        ws.append(cols)
        for row in rows:
            ws.append(list(row))
    del wb["Sheet"]
    wb.save(path)
    return path


@pytest.fixture()
def corpus_root(tmp_path):
    """Two files, so doc_id resolution has something real to hit."""
    root = tmp_path / "docs"
    root.mkdir()
    (root / "a.txt").write_bytes(b"first document")
    (root / "b.txt").write_bytes(b"second document")
    from baseline.schema import compute_doc_id

    return root, compute_doc_id(root / "a.txt"), compute_doc_id(root / "b.txt")


def test_ok_keeps_the_pipeline_answer_and_wrong_takes_the_correction(tmp_path, corpus_root, cfg):
    root, a, b = corpus_root
    xlsx = _write(
        tmp_path / "wb.xlsx",
        docs=[
            [a, "a.txt", "receipt", None, "OK", "True"],
            # Casing varies between reviewers and must not matter.
            [b, "b.txt", "purchase_order", "Invoice", "wrong", "TRUE "],
        ],
    )
    res = import_workbook(xlsx, root, cfg)
    assert not res.problems
    labels = {r["doc_id"]: r["label"] for r in res.rows}
    assert labels == {a: "receipt", b: "invoice"}
    assert res.counts["class_ok"] == 1
    assert res.counts["class_wrong"] == 1


def test_spurious_row_produces_no_gold(tmp_path, corpus_root, cfg):
    """The gold for a spurious field is its absence, not a corrected value."""
    root, a, _ = corpus_root
    xlsx = _write(
        tmp_path / "wb.xlsx",
        docs=[[a, "a.txt", "invoice", None, "OK", "True"]],
        meta=[
            [a, "pipeline", "total_amount", "500", "500", 0, 10, 13, None, "OK", "True"],
            [None, "pipeline", "tax_amount", "99", "99", 0, 20, 22, None, "spurious", "True"],
        ],
    )
    res = import_workbook(xlsx, root, cfg)
    fields = {e["field"] for e in res.rows[0]["metadata"]}
    assert fields == {"total_amount"}, "a spurious field must not become gold"
    assert res.spurious_rows == 1


def test_doc_id_is_carried_down_grouped_rows(tmp_path, corpus_root, cfg):
    """DOC_ID is written once per group in the sheet. Without a forward fill
    every continuation row orphans, which is half the sheet."""
    root, a, _ = corpus_root
    xlsx = _write(
        tmp_path / "wb.xlsx",
        docs=[[a, "a.txt", "invoice", None, "OK", "True"]],
        meta=[
            [a, "pipeline", "total_amount", "500", "500", 0, 10, 13, None, "OK", "True"],
            [None, "pipeline", "tax_amount", "90", "90", 0, 20, 22, None, "OK", "True"],
            [None, "pipeline", "invoice_number", "INV-1", "INV-1", 0, 30, 35, None, "OK", "True"],
        ],
    )
    res = import_workbook(xlsx, root, cfg)
    assert len(res.rows[0]["metadata"]) == 3
    assert not res.orphan_rows


def test_only_confirmed_values_carry_offsets(tmp_path, corpus_root, cfg):
    """A corrected value has no span: the reviewer typed an answer, not a
    place in the text. Keeping the pipeline's offset would point gold at the
    wrong string."""
    root, a, _ = corpus_root
    xlsx = _write(
        tmp_path / "wb.xlsx",
        docs=[[a, "a.txt", "invoice", None, "OK", "True"]],
        meta=[
            [a, "pipeline", "total_amount", "500", "500", 0, 10, 13, None, "OK", "True"],
            [None, "pipeline", "tax_amount", "99", "99", 0, 20, 22, "45", "WRONG", "True"],
        ],
    )
    res = import_workbook(xlsx, root, cfg)
    by_field = {e["field"]: e for e in res.rows[0]["metadata"]}
    assert by_field["total_amount"]["char_start"] == 10
    assert by_field["total_amount"]["char_end"] == 13
    # Normalised on the way in, so gold and prediction are in one space.
    assert by_field["tax_amount"]["normalized_value"] == "45.00"
    assert "char_start" not in by_field["tax_amount"]


def test_a_typed_correction_is_normalised_like_a_prediction(tmp_path, corpus_root, cfg):
    """The reviewer types what is on the page. The pipeline emits a normalised
    value. Comparing the two raw marks the pipeline wrong for being right."""
    root, a, _ = corpus_root
    xlsx = _write(
        tmp_path / "wb.xlsx",
        docs=[[a, "a.txt", "invoice", None, "OK", "True"]],
        meta=[
            [a, "pipeline", "total_amount", "1", "1", 0, 1, 2, "742.7", "wrong", "True"],
            [None, "pipeline", "invoice_date", "x", "x", 0, 3, 4, "12/22/2025", "wrong", "True"],
        ],
    )
    res = import_workbook(xlsx, root, cfg)
    by_field = {e["field"]: e["normalized_value"] for e in res.rows[0]["metadata"]}
    assert by_field["total_amount"] == "742.70"
    assert by_field["invoice_date"] == "2025-12-22"
    assert res.counts["field_normalised"] == 2


def test_a_wrong_tag_is_dropped_rather_than_corrected(tmp_path, corpus_root, cfg):
    root, a, _ = corpus_root
    xlsx = _write(
        tmp_path / "wb.xlsx",
        docs=[[a, "a.txt", "invoice", None, "OK", "True"]],
        tags=[
            [a, "has_line_items", "pipeline", "OK", "True"],
            [None, "Foreign_currency", "pipeline", "ok", "True"],
            [None, "payment_pending", "pipeline", "WRONG", "True"],
            [None, "signed", "pipeline", "spurious", "True"],
        ],
    )
    res = import_workbook(xlsx, root, cfg)
    assert res.rows[0]["tags"] == ["foreign_currency", "has_line_items"]


def test_undeclared_names_are_reported_never_guessed(tmp_path, corpus_root, cfg):
    """`Aadhar_card` is not `aadhaar`. Deciding that it is belongs to a human."""
    root, a, _ = corpus_root
    xlsx = _write(
        tmp_path / "wb.xlsx",
        docs=[[a, "a.txt", "invoice", None, "OK", "True"]],
        meta=[[a, "HUMAN", "Aadhar_card", "1234", "1234", 0, 1, 5, None, "OK", "True"]],
        tags=[[a, "not_a_tag", "pipeline", "OK", "True"]],
    )
    res = import_workbook(xlsx, root, cfg)
    assert res.unknown_fields == ["aadhar_card"]
    assert res.unknown_tags == ["not_a_tag"]
    assert res.problems
    assert res.rows[0]["metadata"] == []


def test_a_doc_id_not_under_root_is_reported_and_carries_no_path(tmp_path, corpus_root, cfg):
    """Same file name is not the same file. A row reviewed against content
    nobody has must not be silently attached to a lookalike."""
    root, a, _ = corpus_root
    xlsx = _write(
        tmp_path / "wb.xlsx",
        docs=[
            [a, "a.txt", "receipt", None, "OK", "True"],
            ["deadbeefdeadbeef", "a.txt", "invoice", None, "OK", "True"],
        ],
    )
    res = import_workbook(xlsx, root, cfg)
    assert res.unresolved_doc_ids == ["deadbeefdeadbeef"]
    by_id = {r["doc_id"]: r for r in res.rows}
    assert "path" in by_id[a]
    assert "path" not in by_id["deadbeefdeadbeef"]
    assert "not here" in render_summary(res, "wb.xlsx", str(root))


def test_rows_not_marked_gold_are_skipped(tmp_path, corpus_root, cfg):
    root, a, b = corpus_root
    xlsx = _write(
        tmp_path / "wb.xlsx",
        docs=[
            [a, "a.txt", "receipt", None, "OK", "True"],
            [b, "b.txt", "invoice", None, "OK", "False"],
        ],
    )
    res = import_workbook(xlsx, root, cfg)
    assert [r["doc_id"] for r in res.rows] == [a]
    assert res.skipped_not_gold == 1


def test_wrong_with_no_correction_produces_nothing(tmp_path, corpus_root, cfg):
    """Nobody has said what the right answer is, so there is no gold to write."""
    root, a, _ = corpus_root
    xlsx = _write(
        tmp_path / "wb.xlsx",
        docs=[[a, "a.txt", "invoice", None, "Wrong", "True"]],
    )
    res = import_workbook(xlsx, root, cfg)
    assert res.rows == []
    assert res.missing_correction


def test_gold_scores_against_the_harness(tmp_path, corpus_root, cfg):
    """The point of the importer: what it writes has to be scoreable."""
    import json

    from baseline.evaluate import evaluate

    root, a, _ = corpus_root
    xlsx = _write(
        tmp_path / "wb.xlsx",
        docs=[[a, "a.txt", "invoice", None, "OK", "True"]],
        meta=[[a, "pipeline", "total_amount", "500.00", "500.00", 0, 10, 13, None, "OK", "True"]],
        tags=[[a, "has_line_items", "pipeline", "OK", "True"]],
    )
    res = import_workbook(xlsx, root, cfg)
    assert res.rows[0]["metadata"][0]["normalized_value"] == "500.00"
    gold = tmp_path / "gold.jsonl"
    gold.write_text(json.dumps(res.rows[0]) + "\n", encoding="utf-8")

    pred = tmp_path / "pred.jsonl"
    pred.write_text(json.dumps({
        "doc_id": a,
        "classification": {"label": "invoice", "confidence": 0.9},
        "metadata": [{"field": "total_amount", "normalized_value": "500.00",
                      "char_start": 10, "char_end": 13}],
        "tags": [{"tag": "has_line_items", "confidence": 0.8}],
    }) + "\n", encoding="utf-8")

    scores = evaluate(str(gold), str(pred))
    assert scores["classification"]["accuracy_overall"] == 1.0
    assert scores["metadata"]["value"]["micro"]["f1"] == 1.0
    assert scores["tagging"]["micro"]["f1"] == 1.0
