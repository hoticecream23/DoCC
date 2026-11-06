"""Every CLI and script honours --dry-run."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


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
