"""The pipeline mirrors its JSONL into SQLite, and db-import rebuilds the same store."""

import json
from contextlib import closing
from pathlib import Path

import pytest
import yaml

from baseline import db
from baseline.cli import main
from baseline.pipeline import run_batch

ROOT = Path(__file__).resolve().parents[2]


def _counts(path):
    with closing(db.connect(path)) as conn:
        return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("documents", "fields", "tags")}


@pytest.fixture(scope="module")
def cabinet_run(corpus, tmp_path_factory):
    out = tmp_path_factory.mktemp("dbrun") / "results.jsonl"
    argv = ["--config", str(ROOT / "config"), "run", "--input", str(corpus), "--output", str(out), "--cabinet", "c1"]
    assert main(argv) == 0
    return out


def _lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_pipeline_writes_one_db_row_per_document_with_matching_doc_types(cabinet_run):
    with closing(db.connect(cabinet_run.parent / "docc.sqlite")) as conn:
        rows = db.list_documents(conn, limit=1000)
    want = {r["doc_id"]: (r["classification"]["label"], r["cabinet_id"]) for r in _lines(cabinet_run)}
    assert {r["doc_id"]: (r["doc_type"], r["cabinet_id"]) for r in rows} == want
    assert {c for _, c in want.values()} == {"c1"}


def test_deleting_the_db_and_reimporting_gives_identical_counts(cabinet_run, tmp_path):
    before = _counts(cabinet_run.parent / "docc.sqlite")
    fresh = tmp_path / "fresh.sqlite"
    assert main(["db-import", str(cabinet_run), "--db", str(fresh)]) == 0
    assert _counts(fresh) == before


def test_db_export_writes_the_cabinet_back_out(cabinet_run, tmp_path):
    out = tmp_path / "export.jsonl"
    assert main(["db-export", str(out), "--db", str(cabinet_run.parent / "docc.sqlite"), "--cabinet", "c1"]) == 0
    assert sorted(r["doc_id"] for r in _lines(out)) == sorted(r["doc_id"] for r in _lines(cabinet_run))


@pytest.mark.parametrize("command", ["db-import", "db-export"])
def test_db_commands_dry_run_writes_nothing(cabinet_run, tmp_path, command):
    target = cabinet_run if command == "db-import" else tmp_path / "out.jsonl"
    assert main([command, str(target), "--db", str(tmp_path / "x.sqlite"), "--dry-run"]) == 0
    assert list(tmp_path.iterdir()) == []


def _config_with_db(cfg, tmp_path, db_path):
    cfgdir = tmp_path / "cfg"
    cfgdir.mkdir()
    for name in ("classes.yaml", "fields.yaml", "tags.yaml"):
        (cfgdir / name).write_text((cfg.config_dir / name).read_text(encoding="utf-8"), encoding="utf-8")
    pipe = dict(cfg.pipeline, db={"path": db_path})
    (cfgdir / "pipeline.yaml").write_text(yaml.safe_dump(pipe), encoding="utf-8")
    return str(cfgdir)


def test_a_broken_db_never_blocks_the_jsonl(corpus, cfg, tmp_path):
    cfgdir = _config_with_db(cfg, tmp_path, "blocked")
    out = tmp_path / "results.jsonl"
    (tmp_path / "blocked").mkdir()  # a directory where the DB file should go
    stats = run_batch(corpus, out, config_dir=cfgdir)
    assert stats["written"] == len(_lines(out)) > 0


def test_null_db_path_disables_the_store(corpus, cfg, tmp_path):
    cfgdir = _config_with_db(cfg, tmp_path, None)
    out = tmp_path / "out" / "results.jsonl"
    run_batch(corpus, out, config_dir=cfgdir)
    assert sorted(p.name for p in out.parent.iterdir()) == ["results.jsonl", "results.jsonl.bboxes"]


def _review_sheet(path, fixes):
    """A reviewed workbook: each document confirmed, one field per document corrected."""
    import openpyxl

    wb = openpyxl.Workbook()
    tabs = {
        "Documents": ["DOC_ID", "DOC_CLASS", "CLASS_VERDICT", "IS_GOLD"],
        "Metadata": ["DOC_ID", "FIELD", "VALUE", "NORMALIZED_VALUE", "CORRECTED_VALUE", "VERDICT", "IS_GOLD"],
        "Tags": ["DOC_ID", "TAG", "VERDICT", "IS_GOLD"],
    }
    for name, cols in tabs.items():
        wb.create_sheet(name).append(cols)
    for doc_id, label, field, value, new in fixes:
        wb["Documents"].append([doc_id, label, "ok", "TRUE"])
        wb["Metadata"].append([doc_id, field, value, value, new, "wrong", "TRUE"])
    del wb["Sheet"]
    wb.save(path)
    return path


def test_workbook_corrections_become_review_events_once(cabinet_run, corpus, tmp_path):
    store = tmp_path / "store.sqlite"
    assert main(["db-import", str(cabinet_run), "--db", str(store)]) == 0
    with_ids = [r for r in _lines(cabinet_run) if any(m["field"] == "invoice_number" for m in r["metadata"])]
    fixes = []
    for i, r in enumerate(with_ids[:2]):
        m = next(m for m in r["metadata"] if m["field"] == "invoice_number")
        fixes.append((r["doc_id"], r["classification"]["label"], "invoice_number", m["normalized_value"], f"FIX-{i}"))
    assert len(fixes) == 2
    xlsx = _review_sheet(tmp_path / "review.xlsx", fixes)
    argv = ["--config", str(ROOT / "config"), "import-workbook", "--xlsx", str(xlsx), "--root", str(corpus)]
    argv += ["--output", str(tmp_path / "gold.jsonl"), "--report", str(tmp_path / "r.md"), "--db", str(store)]
    assert main(argv) == 0
    assert main(argv) == 0  # the same sheet again adds nothing
    with closing(db.connect(store)) as conn:
        events = db.list_review_events(conn)
    got = sorted((e["doc_id"], e["field"], e["old_value"], e["new_value"]) for e in events)
    assert got == sorted((d, f, old, new) for d, _, f, old, new in fixes)
    assert {e["source"] for e in events} == {f"{xlsx.resolve().as_posix()}#Metadata"}
