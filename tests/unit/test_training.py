"""Annotation reading for training."""

from baseline.training import read_annotations


def test_annotations_come_back_sorted_by_doc_id(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"doc_id": "b", "text": "y"}\n{"doc_id": "a", "text": "x"}\n', encoding="utf-8")
    assert [row["doc_id"] for row in read_annotations(path)] == ["a", "b"]
