"""client-eval scores class and key field against a client pack, and nothing else."""

from __future__ import annotations

from pathlib import Path

import pytest

from baseline.cli import main
from baseline.clientpack import load_pack
from baseline.clientpack_build import build_pack
from baseline.evaluate.client_eval import render_client_report, score_cabinet
from baseline.schema import (
    Classification,
    ClassifyMethod,
    Diagnostics,
    DocumentRecord,
    ExtractionMethod,
    MetadataField,
    MetadataMethod,
    TextRecord,
)

INVENTORY = """cabinet,folder,name,ext,size_kb,pages,in_zip,kind
Acme,Payment Voucher,PV-14380.pdf,.pdf,1,1,,document
Acme,Payment Voucher,6390_PV-200.pdf,.pdf,2,1,,document
Acme,LGRN,LG-77.pdf,.pdf,3,1,,document
Acme,Misc,notes.pdf,.pdf,4,1,,document
Acme,Claims/C-1,a.pdf,.pdf,5,1,,document
Acme,Claims/C-1,b.pdf,.pdf,6,1,,document
Acme,Claims/C-2,c.pdf,.pdf,7,1,,document
Acme,Claims/C-2,d.pdf,.pdf,8,1,,document
Acme,Claims/C-2,e.pdf,.pdf,9,1,,document
"""
SCHEMA = """workbook,sheet,document_type,index_field,data_type,field_type,source
Acme/t.xlsx,S,Payment Voucher,PV No,Text,Key,
Acme/t.xlsx,S,LGRN (Local GRN),GRN No,Text,Key,
Acme/t.xlsx,S,Claims,Claim No,Text,Key,
"""
# file name -> (predicted class, [(field, value)], errors)
PREDICTIONS = {
    "PV-14380.pdf": ("Payment Voucher", [("pv_no", "PV 14380")], []),
    "6390_PV-200.pdf": ("unknown", [("reference", "Ref PV-200")], []),
    "LG-77.pdf": ("Payment Voucher", [], ["ocr: timeout"]),
    "notes.pdf": ("Payment Voucher", [], []),
    "a.pdf": ("Claims", [], []),
    "b.pdf": ("Claims", [], []),
    "c.pdf": ("Claims", [], []),
    "d.pdf": ("other", [], []),
    "e.pdf": ("other", [], []),
}


def _record(doc_id: str, label: str, fields: list[tuple[str, str]], errors: list[str]) -> DocumentRecord:
    text = " ".join(v for _, v in fields) or "x"
    metadata, at = [], 0
    for name, value in fields:
        start = text.index(value, at)
        at = start + len(value)
        metadata.append(
            MetadataField(
                field=name,
                value=value,
                normalized_value=value,
                page=0,
                char_start=start,
                char_end=at,
                confidence=0.9,
                method=MetadataMethod.ANCHOR,
                validated=False,
            )
        )
    return DocumentRecord(
        doc_id=doc_id,
        source_path=f"work/acme/{doc_id}.pdf",
        filename=f"{doc_id}.pdf",
        text=TextRecord(
            full=text, pages=[text], method=ExtractionMethod.NATIVE, char_count=len(text), extraction_confidence=0.9
        ),
        classification=Classification(label=label, confidence=0.8, method=ClassifyMethod.RULE),
        metadata=metadata,
        diagnostics=Diagnostics(page_count=1, is_scanned=False, has_native_text=True, errors=errors),
    )


@pytest.fixture
def scored(tmp_path: Path) -> tuple[Path, Path]:
    (tmp_path / "inv.csv").write_text(INVENTORY, encoding="utf-8")
    (tmp_path / "schema.csv").write_text(SCHEMA, encoding="utf-8")
    pack = tmp_path / "pack"
    build_pack(tmp_path / "inv.csv", tmp_path / "schema.csv", pack)
    lines = []
    for row in load_pack(pack, "Acme").manifest:
        label, fields, errors = PREDICTIONS[Path(row.rel_path).name]
        lines.append(_record(row.doc_id, label, fields, errors).model_dump_json())
    results = tmp_path / "results.jsonl"
    results.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return pack, results


def test_classification_scores_only_labeled_single_documents(scored: tuple[Path, Path]):
    score = score_cabinet(scored[0], "Acme", scored[1])
    assert (score.scored, score.correct, score.unlabeled) == (3, 1, 1)


def test_per_class_precision_and_recall(scored: tuple[Path, Path]):
    pv = score_cabinet(scored[0], "Acme", scored[1]).per_class["Payment Voucher"]
    assert (pv.precision, pv.recall, pv.support) == (0.5, 0.5, 2)


def test_key_exact_and_contained(scored: tuple[Path, Path]):
    score = score_cabinet(scored[0], "Acme", scored[1])
    assert (score.key_scored, score.key_exact, score.key_contained) == (3, 1, 2)


def test_cases_scored_by_majority_and_any_hit(scored: tuple[Path, Path]):
    score = score_cabinet(scored[0], "Acme", scored[1])
    assert (score.cases, score.cases_majority, score.cases_any) == (2, 1, 2)


def test_failed_and_abstained_are_counted(scored: tuple[Path, Path]):
    score = score_cabinet(scored[0], "Acme", scored[1])
    assert (score.failed, score.abstained, score.missing) == (1, 1, 0)


def test_missing_results_are_counted(scored: tuple[Path, Path], tmp_path: Path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    assert score_cabinet(scored[0], "Acme", empty).missing == 9


def test_report_says_no_other_field_is_scored(scored: tuple[Path, Path]):
    text = render_client_report(score_cabinet(scored[0], "Acme", scored[1]), "results.jsonl")
    assert "no other field is scored" in text


def test_cli_client_eval_writes_report(scored: tuple[Path, Path], tmp_path: Path):
    out = tmp_path / "client_report.md"
    assert (
        main(
            [
                "client-eval",
                "--pack",
                str(scored[0]),
                "--cabinet",
                "Acme",
                "--results",
                str(scored[1]),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    assert "## Case bundles" in out.read_text()


def test_cli_client_eval_dry_run_writes_nothing(scored: tuple[Path, Path], tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "dry_report.md"
    argv = ["client-eval", "--pack", str(scored[0]), "--cabinet", "Acme", "--results", str(scored[1])]
    assert main([*argv, "--out", str(out), "--dry-run"]) == 0
    assert not out.exists()
