"""Atomic writes and --dry-run on every CLI."""

import os
import subprocess
import sys

import pytest
from conftest import ROOT

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


def test_baseline_dry_run_writes_nothing(corpus, tmp_path):
    from baseline.cli import main

    out = tmp_path / "out" / "results.jsonl"
    assert (
        main(["--config", str(ROOT / "config"), "run", "--input", str(corpus), "--output", str(out), "--dry-run"]) == 0
    )
    assert not (tmp_path / "out").exists()


def test_baseline_profile_dry_run_writes_nothing(corpus, tmp_path):
    from baseline.cli import main

    out = tmp_path / "profile.md"
    assert main(["profile", "--input", str(corpus), "--output", str(out), "--dry-run"]) == 0
    assert list(tmp_path.iterdir()) == []


def test_cascade_dry_run_writes_nothing_and_needs_no_server(corpus, tmp_path):
    from cascade.cli import main

    out = tmp_path / "cascade.jsonl"
    assert main(["run", "--input", str(corpus), "--output", str(out), "--dry-run"]) == 0
    assert list(tmp_path.iterdir()) == []


def test_resume_flag_is_positive_and_no_resume_still_works():
    from baseline.cli import build_parser as baseline_parser
    from cascade.cli import build_parser as cascade_parser

    for parser in (baseline_parser(), cascade_parser()):
        assert parser.parse_args(["run", "--input", "i", "--output", "o"]).resume is True
        assert parser.parse_args(["run", "--input", "i", "--output", "o", "--no-resume"]).resume is False


@pytest.mark.parametrize(
    "script, args",
    [
        ("convert_sroie.py", ["--parquet", "{parquet}", "--split", "test", "--out", "{out}"]),
        ("cv_classifier.py", ["{results}", "{out}"]),
    ],
)
def test_script_dry_run_writes_nothing(tmp_path, script, args):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    parquet = tmp_path / "in.parquet"
    pq.write_table(pa.table({"key": ["a"]}), parquet)
    results = tmp_path / "results.jsonl"
    results.write_text('{"doc_id": "a"}\n', encoding="utf-8")
    out = tmp_path / "out"
    argv = [a.format(parquet=parquet, results=results, out=out) for a in args]
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *argv, "--dry-run"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )
    assert proc.returncode == 0, proc.stderr
    assert "dry run" in proc.stderr
    assert not out.exists()
