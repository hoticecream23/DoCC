"""Atomic writes and --dry-run on every CLI."""

import pytest

from baseline.io import atomic_open, atomic_write_text


def test_atomic_write_replaces_content_and_leaves_no_temp(tmp_path):
    target = tmp_path / "out.jsonl"
    target.write_text("old\n", encoding="utf-8")
    atomic_write_text(target, "new\n")
    assert target.read_text(encoding="utf-8") == "new\n"
    assert list(tmp_path.iterdir()) == [target]


def test_a_crash_mid_write_keeps_the_old_file(tmp_path):
    target = tmp_path / "out.jsonl"
    target.write_text("old\n", encoding="utf-8")
    with pytest.raises(RuntimeError):
        with atomic_open(target) as fh:
            fh.write("half")
            raise RuntimeError("crash")
    assert target.read_text(encoding="utf-8") == "old\n"
    assert list(tmp_path.iterdir()) == [target]


def test_a_crash_before_the_first_write_creates_nothing(tmp_path):
    target = tmp_path / "out.jsonl"
    with pytest.raises(RuntimeError):
        with atomic_open(target):
            raise RuntimeError("crash")
    assert list(tmp_path.iterdir()) == []


def test_resume_flag_is_positive_and_no_resume_still_works():
    from baseline.cli import build_parser as baseline_parser
    from cascade.cli import build_parser as cascade_parser

    for parser in (baseline_parser(), cascade_parser()):
        assert parser.parse_args(["run", "--input", "i", "--output", "o"]).resume is True
        assert parser.parse_args(["run", "--input", "i", "--output", "o", "--no-resume"]).resume is False


def test_binary_mode_writes_bytes_atomically(tmp_path):
    out = tmp_path / "model.bin"
    with atomic_open(out, mode="wb") as fh:
        fh.write(b"\x00\x01\r\n")
    assert out.read_bytes() == b"\x00\x01\r\n"
    assert list(tmp_path.iterdir()) == [out]
