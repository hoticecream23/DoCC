"""client-eval scores class and key field against a client pack, and nothing else."""

from __future__ import annotations

from pathlib import Path

import pytest

from baseline.cli import main
from baseline.clientpack import load_pack
from baseline.clientpack_build import build_pack
from baseline.evaluate.client_eval import (
    ABSTAINED,
    NO_RESULT,
    CabinetScore,
    _pct,
    _score_cases,
    canon_key,
    contained,
    loose_key,
    render_client_report,
    score_cabinet,
)
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


def _record(
    doc_id: str, label: str, fields: list[tuple[str, str]], errors: list[str], normalized: str | None = None
) -> DocumentRecord:
    text = " ".join(v for _, v in fields) or "x"
    metadata, at = [], 0
    for name, value in fields:
        start = text.index(value, at)
        at = start + len(value)
        metadata.append(
            MetadataField(
                field=name,
                value=value,
                normalized_value=value if normalized is None else normalized,
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


def test_key_exact_loose_and_contained(scored: tuple[Path, Path]):
    score = score_cabinet(scored[0], "Acme", scored[1])
    assert (score.key_scored, score.key_exact, score.key_loose, score.key_contained, score.key_too_short) == (
        3,
        0,
        1,
        1,
        2,
    )


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


# --- regressions from the scoring review -------------------------------------

Prediction = tuple[str, list[tuple[str, str]], list[str]]


def _small_pack(tmp_path: Path, files: list[tuple[str, str]]) -> Path:
    rows = "".join(f"Acme,{folder},{name},.pdf,1,1,,document\n" for folder, name in files)
    (tmp_path / "inv.csv").write_text(INVENTORY.splitlines()[0] + "\n" + rows, encoding="utf-8")
    (tmp_path / "schema.csv").write_text(SCHEMA, encoding="utf-8")
    build_pack(tmp_path / "inv.csv", tmp_path / "schema.csv", tmp_path / "pack")
    return tmp_path / "pack"


def _write_results(pack: Path, out: Path, predictions: dict[str, Prediction], normalized: str | None = None) -> Path:
    lines = []
    for row in load_pack(pack, "Acme").manifest:
        if Path(row.rel_path).name in predictions:
            label, fields, errors = predictions[Path(row.rel_path).name]
            lines.append(_record(row.doc_id, label, fields, errors, normalized).model_dump_json())
    out.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return out


def _score_one(tmp_path: Path, name: str, prediction: Prediction, normalized: str | None = None) -> CabinetScore:
    pack = _small_pack(tmp_path, [("Payment Voucher", name)])
    return score_cabinet(pack, "Acme", _write_results(pack, tmp_path / "r.jsonl", {name: prediction}, normalized))


FOUR = [("Payment Voucher", f"PV-10000{i}.pdf") for i in range(4)]
TWO_RIGHT: dict[str, Prediction] = {n: ("Payment Voucher", [], []) for _, n in FOUR[:2]}


def test_missing_documents_count_as_wrong(tmp_path: Path):
    pack = _small_pack(tmp_path, FOUR)
    score = score_cabinet(pack, "Acme", _write_results(pack, tmp_path / "r.jsonl", TWO_RIGHT))
    assert (score.correct, score.scored, score.scored_missing) == (2, 4, 2)


def test_headline_names_documents_with_no_result(tmp_path: Path):
    pack = _small_pack(tmp_path, FOUR)
    score = score_cabinet(pack, "Acme", _write_results(pack, tmp_path / "r.jsonl", TWO_RIGHT))
    assert "Accuracy 2/4 (50.0%) \u2014 2 documents produced no result" in render_client_report(score)


def test_missing_documents_stay_in_the_key_denominator(tmp_path: Path):
    pack = _small_pack(tmp_path, FOUR)
    score = score_cabinet(pack, "Acme", _write_results(pack, tmp_path / "r.jsonl", TWO_RIGHT))
    assert score.key_scored == 4


def test_a_case_with_every_file_missing_is_scored_wrong(tmp_path: Path):
    pack = _small_pack(tmp_path, [("Claims/C-9", "a.pdf"), ("Claims/C-9", "b.pdf")])
    score = score_cabinet(pack, "Acme", _write_results(pack, tmp_path / "r.jsonl", {}))
    assert (score.cases, score.cases_majority) == (1, 0)


def test_missing_documents_are_listed_under_no_result(tmp_path: Path):
    pack = _small_pack(tmp_path, FOUR)
    score = score_cabinet(pack, "Acme", _write_results(pack, tmp_path / "r.jsonl", TWO_RIGHT))
    assert sorted(Path(p).name for p in score.no_result) == ["PV-100002.pdf", "PV-100003.pdf"]


def test_exact_key_keeps_separators_that_differ():
    assert canon_key("1-2345") != canon_key("12-345")


def test_loose_key_ignores_separators():
    assert loose_key("1-2345") == loose_key("12-345")


def test_exact_key_maps_unicode_dashes_and_spacing():
    assert canon_key("  pv\u2013 14380 ") == canon_key("PV- 14380")


def test_moved_separator_is_loose_not_exact(tmp_path: Path):
    score = _score_one(tmp_path, "1-2345.pdf", ("Payment Voucher", [("pv_no", "12-345")], []))
    assert (score.key_exact, score.key_loose) == (0, 1)


def test_float_artefact_is_an_exact_match(tmp_path: Path):
    score = _score_one(tmp_path, "4502424845.pdf", ("Payment Voucher", [("pv_no", "4502424845.0")], []))
    assert score.key_exact == 1


def test_normalized_value_match_counts(tmp_path: Path):
    prediction = ("Payment Voucher", [("pv_no", "No 7712345")], [])
    assert _score_one(tmp_path, "PV-7712345.pdf", prediction, normalized="PV-7712345").key_exact == 1


def test_contained_skips_a_short_gold(tmp_path: Path):
    score = _score_one(tmp_path, "11.pdf", ("Payment Voucher", [("amount", "1,115.00")], []))
    assert (score.key_contained, score.key_too_short) == (0, 1)


def test_contained_needs_a_whole_token_run():
    assert not contained("E30270120230089", "E302701202300891")


def test_contained_finds_a_whole_run_inside_text():
    assert contained("INV-100001", "ref: inv 100001, paid")


def test_contained_never_matches_an_empty_gold():
    assert not contained("---", "anything")


def test_key_is_read_from_the_gold_class_field(tmp_path: Path):
    score = _score_one(tmp_path, "PV-14380.pdf", ("Claims", [("pv_no", "PV-14380")], []))
    assert (score.key_exact, score.key_class_ok, score.key_exact_class_ok) == (1, 0, 0)


def test_report_explains_the_key_lookup(scored: tuple[Path, Path]):
    text = render_client_report(score_cabinet(scored[0], "Acme", scored[1]))
    assert "read from the folder's (gold) type" in text


def test_per_class_has_a_row_for_a_predicted_only_class(scored: tuple[Path, Path]):
    other = score_cabinet(scored[0], "Acme", scored[1]).per_class[ABSTAINED]
    assert (other.support, other.predicted) == (0, 1)


def test_per_class_rows_sum_to_the_totals(scored: tuple[Path, Path]):
    score = score_cabinet(scored[0], "Acme", scored[1])
    rows = score.per_class.values()
    assert (sum(c.support for c in rows), sum(c.predicted for c in rows)) == (score.scored, score.scored)


def test_labels_match_ignoring_case_and_spacing(tmp_path: Path):
    assert _score_one(tmp_path, "PV-1.pdf", ("  payment   VOUCHER ", [], [])).correct == 1


def test_an_alias_of_the_gold_class_counts_as_correct(tmp_path: Path):
    pack = _small_pack(tmp_path, [("LGRN", "LG-1.pdf")])
    results = _write_results(pack, tmp_path / "r.jsonl", {"LG-1.pdf": ("lgrn", [], [])})
    assert score_cabinet(pack, "Acme", results).correct == 1


def test_case_gold_is_the_majority_label():
    score = CabinetScore(cabinet="Acme", documents=3)
    _score_cases(score, {"C-1": [("A", "A"), ("A", "A"), ("B", "B")]})
    assert (score.cases_majority, score.mixed_cases) == (1, ["C-1"])


def test_a_case_with_only_abstentions_is_wrong():
    score = CabinetScore(cabinet="Acme", documents=2)
    _score_cases(score, {"C-1": [("A", ABSTAINED), ("A", NO_RESULT)]})
    assert (score.cases_majority, score.cases_any) == (0, 0)


def test_a_correct_record_with_errors_stays_correct(tmp_path: Path):
    score = _score_one(tmp_path, "PV-1.pdf", ("Payment Voucher", [], ["ocr: timeout"]))
    assert (score.correct, score.failed, score.missing) == (1, 1, 0)


def test_records_with_errors_are_listed_apart_from_no_result(scored: tuple[Path, Path]):
    score = score_cabinet(scored[0], "Acme", scored[1])
    assert ([Path(p).name for p in score.with_errors], score.no_result) == (["LG-77.pdf"], [])


def test_pct_never_rounds_up_to_a_full_score():
    assert _pct(9996, 10000) == "9996/10000 (99.9%)"


def test_pct_of_nothing_is_not_applicable():
    assert _pct(0, 0) == "n/a"


def test_a_duplicated_stem_uses_the_last_line(tmp_path: Path):
    pack = _small_pack(tmp_path, [("Payment Voucher", "PV-1.pdf")])
    first = _write_results(pack, tmp_path / "a.jsonl", {"PV-1.pdf": ("Claims", [], [])}).read_text()
    last = _write_results(pack, tmp_path / "b.jsonl", {"PV-1.pdf": ("Payment Voucher", [], [])}).read_text()
    (tmp_path / "r.jsonl").write_text(first + last, encoding="utf-8")
    score = score_cabinet(pack, "Acme", tmp_path / "r.jsonl")
    assert (score.correct, score.duplicates) == (1, 1)


def test_report_has_a_how_to_read_section(scored: tuple[Path, Path]):
    assert "## How to read this" in render_client_report(score_cabinet(scored[0], "Acme", scored[1]))


def test_cli_client_eval_fails_when_scoring_fails(scored: tuple[Path, Path], tmp_path: Path):
    argv = ["client-eval", "--pack", str(scored[0]), "--cabinet", "Acme", "--results", str(tmp_path / "none.jsonl")]
    assert main([*argv, "--out", str(tmp_path / "r.md")]) != 0
