"""Reading JSONL for the scorer."""

import pytest

from baseline.evaluate.io import load_jsonl


def test_blank_lines_are_skipped(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")
    assert load_jsonl(path) == [{"a": 1}, {"a": 2}]


def test_a_bad_line_is_reported_with_file_and_line_number(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"rows.jsonl:2 is not valid JSON"):
        load_jsonl(path)
