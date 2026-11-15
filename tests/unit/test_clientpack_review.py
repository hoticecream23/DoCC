"""Client pack regressions found in review: dedup, wrappers, cases, labels, fields, slugs, label_map, materialize."""

from __future__ import annotations

import csv
import zipfile
from pathlib import Path

import pytest
import yaml

from baseline.clientpack import load_pack, materialize
from baseline.clientpack_build import build_pack
from baseline.clientpack_rules import child_roles, field_origin, key_gold, match_label, parse_dtype

HEADER = "cabinet,folder,name,ext,size_kb,pages,in_zip,kind\n"
SCHEMA_HEADER = "workbook,sheet,document_type,index_field,data_type,field_type,source\n"


def _build(tmp_path: Path, inventory: str, schema: str = "", **kwargs: object) -> Path:
    inv, sch, out = tmp_path / "inv.csv", tmp_path / "schema.csv", tmp_path / "pack"
    inv.write_text(HEADER + inventory, encoding="utf-8")
    sch.write_text(SCHEMA_HEADER + schema, encoding="utf-8")
    build_pack(inv, sch, out, **kwargs)  # type: ignore[arg-type]
    return out


def _manifest(pack: Path, slug: str) -> dict[str, dict[str, str]]:
    with open(pack / slug / "manifest.csv", encoding="utf-8", newline="") as fh:
        return {r["rel_path"]: r for r in csv.DictReader(fh)}


def _excluded(pack: Path, slug: str) -> dict[str, str]:
    with open(pack / slug / "excluded.csv", encoding="utf-8", newline="") as fh:
        return {r["rel_path"]: r["reason"] for r in csv.DictReader(fh)}


# 1, 2: dedup -----------------------------------------------------------------


def test_same_name_in_two_case_folders_is_not_a_duplicate(tmp_path: Path):
    pack = _build(
        tmp_path, "Acme,Inv/c1,Invoice.pdf,.pdf,100,1,,document\nAcme,Inv/c2,Invoice.pdf,.pdf,100,1,,document\n"
    )
    assert set(_manifest(pack, "acme")) == {"Acme/Inv/c1/Invoice.pdf", "Acme/Inv/c2/Invoice.pdf"}


def test_same_member_name_in_two_bundle_zips_is_not_a_duplicate(tmp_path: Path):
    rows = "".join(
        f"Acme,Bundle/{n},ExportedDocuments.zip,.zip,9,,,archive\n"
        f"Acme,Bundle/{n}/ExportedDocuments,z.pdf,.pdf,9,1,ExportedDocuments.zip,document\n"
        for n in (7, 8)
    )
    assert len(_manifest(_build(tmp_path, rows), "acme")) == 2


def test_copy_under_another_cabinets_name_loses_to_the_owner(tmp_path: Path):
    rows = "Acme,Borealis/Samples/P,1.pdf,.pdf,5,1,,document\nBorealis,Borealis/Samples/P,1.pdf,.pdf,5,1,,document\n"
    pack = _build(tmp_path, rows)
    assert _excluded(pack, "acme")["Acme/Borealis/Samples/P/1.pdf"] == "duplicate"
    assert "Borealis/Borealis/Samples/P/1.pdf" in _manifest(pack, "borealis")


# 3, 4, 10: wrappers, cases, keys ----------------------------------------------

KEYED = "Acme/types.xlsx,S,Purchase Invoice,Invoice No,Text,Key,\nAcme/types.xlsx,S,Invoice,Invoice No,Text,Key,\n"


def test_samples_wrapper_is_skipped_for_the_label(tmp_path: Path):
    rows = "Acme,Acme/Samples/Purchase Invoice/100,100.pdf,.pdf,5,1,,document\n"
    rows += "Acme,Acme/Samples/Purchase Invoice/101,101.pdf,.pdf,6,1,,document\n"
    row = _manifest(_build(tmp_path, rows, KEYED), "acme")["Acme/Acme/Samples/Purchase Invoice/100/100.pdf"]
    assert (row["label"], row["case_id"], row["key_gold"]) == ("Purchase Invoice", "", "100")


def test_case_folders_with_one_document_are_still_cases(tmp_path: Path):
    rows = "Acme,Invoice/11344 MONIA,11344 MONIA 1.pdf,.pdf,5,1,,document\n"
    rows += "Acme,Invoice/11344 MONIA,11344 MONIA 2.pdf,.pdf,6,1,,document\n"
    rows += "Acme,Invoice/55 X,55 X 1.pdf,.pdf,7,1,,document\n"
    row = _manifest(_build(tmp_path, rows, KEYED), "acme")["Acme/Invoice/11344 MONIA/11344 MONIA 1.pdf"]
    assert (row["case_id"], row["key_gold"]) == ("Invoice/11344 MONIA", "")


def test_one_file_named_like_its_number_folder_is_not_a_case():
    assert child_roles("100", ["100"]) == ["document"]


def test_key_gold_needs_a_folder_label(tmp_path: Path):
    schema = "Acme/types.xlsx,S,Form 16,Form No,Text,Key,\n"
    row = _manifest(_build(tmp_path, "Acme,,Form 16.tif,.tif,5,1,,document\n", schema), "acme")["Acme/Form 16.tif"]
    assert (row["label"], row["key_gold"]) == ("Form 16", "")


def test_key_gold_drops_copy_suffixes():
    assert [key_gold(n) for n in ("PV-1 - Copy (2).pdf", "PV-2 copy.pdf", "PV-3 (1).pdf")] == ["PV-1", "PV-2", "PV-3"]


# 5: slugs ----------------------------------------------------------------------


def test_cabinets_with_the_same_slug_get_a_suffix(tmp_path: Path):
    rows = "Acme Insurance,,a1.pdf,.pdf,5,1,,document\nAcme-Insurance,,b2.pdf,.pdf,6,1,,document\n"
    pack = _build(tmp_path, rows)
    assert load_pack(pack, "Acme-Insurance").cabinet.slug == "acme_insurance_2"
    assert [r.rel_path for r in load_pack(pack, "acme_insurance").manifest] == ["Acme Insurance/a1.pdf"]


# 7: origin and field overrides ----------------------------------------------------


@pytest.mark.parametrize(
    ("name", "origin"),
    [
        ("Model Year", "document"),
        ("Policy Year", "document"),
        ("P.O. Box", "document"),
        ("Post Box No", "document"),
        ("Driver Name", "document"),
        ("Year", "manual"),
        ("Month", "manual"),
        ("File Number", "manual"),
        ("Box and Pouch No", "manual"),
    ],
)
def test_field_origin_keywords_match_whole_names_or_phrases(name: str, origin: str):
    assert field_origin(name, "Required", {}, "required") == origin


def test_field_overrides_file_wins_and_is_kept(tmp_path: Path):
    (tmp_path / "pack").mkdir()
    overrides = tmp_path / "pack" / "field_overrides.yaml"
    overrides.write_text(yaml.safe_dump({"Acme": {"Invoice": {"Invoice No": {"origin": "erp"}}}}), encoding="utf-8")
    pack = _build(tmp_path, "Acme,Invoice,I-1.pdf,.pdf,5,1,,document\n", KEYED)
    assert yaml.safe_load((pack / "acme" / "fields.yaml").read_text())["Invoice"][0]["origin"] == "erp"
    assert overrides.exists()


# 8, 13: schema reading ------------------------------------------------------------


def test_type_spellings_differing_in_case_and_space_merge(tmp_path: Path):
    schema = "Acme/t.xlsx,S,PAYMENT VOUCHER,PV No,Text,Key,\nAcme/t.xlsx,S,Payment Voucher ,PV Date,Date,Required,\n"
    pack = _build(tmp_path, "Acme,Payment Voucher,p1.pdf,.pdf,5,1,,document\n", schema)
    assert _manifest(pack, "acme")["Acme/Payment Voucher/p1.pdf"]["label"] == "PAYMENT VOUCHER"
    assert "`Payment Voucher` merged into `PAYMENT VOUCHER`" in (pack / "REPORT.md").read_text()


def test_loose_prefix_match_is_rejected():
    found = match_label("Payment Voucher", ["Payment Voucher Transfer"])
    assert (found.target, found.loose) == ("", "Payment Voucher Transfer")


def test_prefix_match_with_a_dash_qualifier_is_accepted():
    assert match_label("Sales Invoice", ["Sales Invoice - Yellow Copy"]).target == "Sales Invoice - Yellow Copy"


def test_workbook_in_a_copied_cabinet_belongs_to_the_named_cabinet(tmp_path: Path):
    rows = "Acme,,a.pdf,.pdf,5,1,,document\nBorealis,Claims,c.pdf,.pdf,6,1,,document\n"
    pack = _build(tmp_path, rows, "Acme/Borealis/x.xlsx,S,Claims,Claim No,Text,Key,\n")
    assert _manifest(pack, "borealis")["Borealis/Claims/c.pdf"]["label"] == "Claims"


def test_workbook_under_no_cabinet_is_reported(tmp_path: Path):
    pack = _build(tmp_path, "Acme,,a.pdf,.pdf,5,1,,document\n", "loose.xlsx,S,Claims,Claim No,Text,Key,\n")
    assert "- `loose.xlsx`" in (pack / "REPORT.md").read_text()


def test_schema_type_named_other_is_renamed(tmp_path: Path):
    pack = _build(tmp_path, "Acme,OTHER,o.pdf,.pdf,5,1,,document\n", "Acme/t.xlsx,S,OTHER,Ref,Text,Key,\n")
    assert _manifest(pack, "acme")["Acme/OTHER/o.pdf"]["label"] == "Other (client)"


# 9: data types --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "parsed"),
    [
        ("Date(dd\\MM\\YYYY)", ("date", "DD/MM/YYYY")),
        ("Date (DD/MM/YYYY HH:MM)", ("date", "DD/MM/YYYY HH:MM")),
        ("Updated", ("text", None)),
    ],
)
def test_parse_dtype_review_cases(text: str, parsed: tuple[str, str | None]):
    assert parse_dtype(text) == parsed


# 11: label_map sections -------------------------------------------------------------


def test_auto_label_is_not_frozen_as_an_override(tmp_path: Path):
    rows = "Acme,LGRN,g1.pdf,.pdf,5,1,,document\n"
    _build(tmp_path, rows, "Acme/t.xlsx,S,LGRN (Local GRN),GRN No,Text,Key,\n")
    pack = _build(tmp_path, rows, "Acme/t.xlsx,S,LGRN (Local GRN Note),GRN No,Text,Key,\n")
    assert _manifest(pack, "acme")["Acme/LGRN/g1.pdf"]["label"] == "LGRN (Local GRN Note)"


def test_old_flat_label_map_keeps_only_differing_entries_as_overrides(tmp_path: Path):
    (tmp_path / "pack").mkdir()
    flat = {"Acme": {"LGRN": "LGRN (Local GRN)", "Notes": "LGRN (Local GRN)"}}
    (tmp_path / "pack" / "label_map.yaml").write_text(yaml.safe_dump(flat), encoding="utf-8")
    rows = "Acme,LGRN,g1.pdf,.pdf,5,1,,document\nAcme,Notes,n1.pdf,.pdf,6,1,,document\n"
    pack = _build(tmp_path, rows, "Acme/t.xlsx,S,LGRN (Local GRN),GRN No,Text,Key,\n")
    assert yaml.safe_load((pack / "label_map.yaml").read_text())["Acme"]["overrides"] == {"Notes": "LGRN (Local GRN)"}


# 12: duplicate zip members and materialize --------------------------------------------


@pytest.fixture
def twin_share(tmp_path: Path) -> tuple[Path, Path]:
    share = tmp_path / "share"
    (share / "Acme").mkdir(parents=True)
    with pytest.warns(UserWarning), zipfile.ZipFile(share / "Acme" / "b.zip", "w") as archive:
        archive.writestr("x.pdf", b"first")
        archive.writestr("x.pdf", b"second")
    inv = tmp_path / "inv.csv"
    inv.write_text(
        HEADER.replace("kind", "kind,member")
        + "Acme,,b.zip,.zip,1,,,archive,\n"
        + "Acme,b,x.pdf,.pdf,1,1,b.zip,document,0\n"
        + "Acme,b,x.pdf,.pdf,1,1,b.zip,document,1\n"
        + "Acme,,gone.pdf,.pdf,1,1,,document,\n",
        encoding="utf-8",
    )
    schema = tmp_path / "schema.csv"
    schema.write_text(SCHEMA_HEADER, encoding="utf-8")
    build_pack(inv, schema, tmp_path / "pack")
    return share, tmp_path / "pack"


def test_zip_entries_with_one_name_are_both_extracted(tmp_path: Path, twin_share: tuple[Path, Path]):
    share, pack = twin_share
    paths = materialize(pack, "Acme", share, tmp_path / "work")
    assert sorted(p.read_bytes() for p in paths) == [b"first", b"second"]


def test_materialize_reports_a_missing_file_and_goes_on(tmp_path: Path, twin_share: tuple[Path, Path]):
    share, pack = twin_share
    result = materialize(pack, "Acme", share, tmp_path / "work")
    gone = next(r.doc_id for r in load_pack(pack, "Acme").manifest if r.rel_path == "Acme/gone.pdf")
    assert (result.missing, len(result.copied)) == ([gone], 2)
