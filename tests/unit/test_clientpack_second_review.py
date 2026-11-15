"""Client pack regressions from the second review: cases, slugs, field merges, duplicates, keys, aliases, wrappers."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import yaml

from baseline.clientpack import load_pack
from baseline.clientpack_build import build_pack
from baseline.clientpack_rules import key_gold

HEADER = "cabinet,folder,name,ext,size_kb,pages,in_zip,kind\n"
SCHEMA_HEADER = "workbook,sheet,document_type,index_field,data_type,field_type,source\n"
INVOICE = "Acme/t.xlsx,S,Invoice,Invoice No,Text,Key,\n"


def _build(tmp_path: Path, inventory: str, schema: str = "", **kwargs: object) -> Path:
    inv, sch, out = tmp_path / "inv.csv", tmp_path / "schema.csv", tmp_path / "pack"
    inv.write_text(HEADER + inventory, encoding="utf-8")
    sch.write_text(SCHEMA_HEADER + schema, encoding="utf-8")
    build_pack(inv, sch, out, **kwargs)  # type: ignore[arg-type]
    return out


def _manifest(pack: Path, slug: str = "acme") -> dict[str, dict[str, str]]:
    with open(pack / slug / "manifest.csv", encoding="utf-8", newline="") as fh:
        return {r["rel_path"]: r for r in csv.DictReader(fh)}


def _report(pack: Path) -> str:
    return (pack / "REPORT.md").read_text(encoding="utf-8")


# 1: cases are decided per child folder ---------------------------------------------

NUMBERED = "".join(f"Acme,Inv/{n},{n}.pdf,.pdf,{n},1,,document\n" for n in range(1, 6))
NUMBERED += "Acme,Inv/3,3-annex.pdf,.pdf,9,1,,document\n"


def test_an_annex_beside_a_numbered_document_is_an_attachment(tmp_path: Path):
    row = _manifest(_build(tmp_path, NUMBERED, "Acme/t.xlsx,S,Inv,Inv No,Text,Key,\n"))["Acme/Inv/3/3-annex.pdf"]
    assert (row["role"], row["case_id"], row["key_gold"]) == ("attachment", "", "")


def test_an_annex_does_not_turn_the_type_folder_into_cases(tmp_path: Path):
    rows = _manifest(_build(tmp_path, NUMBERED, "Acme/t.xlsx,S,Inv,Inv No,Text,Key,\n"))
    assert [(rows[f"Acme/Inv/{n}/{n}.pdf"]["case_id"], rows[f"Acme/Inv/{n}/{n}.pdf"]["key_gold"]) for n in (1, 3)] == [
        ("", "1"),
        ("", "3"),
    ]


def test_a_type_folder_may_mix_cases_and_single_documents(tmp_path: Path):
    rows = "Acme,Invoice/C-1,a.pdf,.pdf,1,1,,document\nAcme,Invoice/C-1,b.pdf,.pdf,2,1,,document\n"
    rows += "Acme,Invoice/INV-7,INV-7.pdf,.pdf,3,1,,document\n"
    manifest = _manifest(_build(tmp_path, rows, INVOICE))
    assert (manifest["Acme/Invoice/C-1/a.pdf"]["case_id"], manifest["Acme/Invoice/INV-7/INV-7.pdf"]["key_gold"]) == (
        "C-1",
        "INV-7",
    )


# 2: slugs recorded in pack.yaml are kept --------------------------------------------


def test_a_later_colliding_cabinet_does_not_take_an_existing_slug(tmp_path: Path):
    _build(tmp_path, "Dubai-Insurance,,a1.pdf,.pdf,5,1,,document\n")
    rows = "Dubai-Insurance,,a1.pdf,.pdf,5,1,,document\nDubai Insurance,,b2.pdf,.pdf,6,1,,document\n"
    pack = _build(tmp_path, rows)
    slugs = {n: load_pack(pack, n).cabinet.slug for n in ("Dubai-Insurance", "Dubai Insurance")}
    assert slugs == {"Dubai-Insurance": "dubai_insurance", "Dubai Insurance": "dubai_insurance_2"}


def test_a_removed_cabinet_keeps_its_slug_reserved(tmp_path: Path):
    _build(tmp_path, "Dubai-Insurance,,a1.pdf,.pdf,5,1,,document\n")
    _build(tmp_path, "Acme,,x.pdf,.pdf,5,1,,document\n")
    pack = _build(tmp_path, "Dubai Insurance,,b2.pdf,.pdf,6,1,,document\n")
    assert load_pack(pack, "Dubai Insurance").cabinet.slug == "dubai_insurance_2"


# 3: merged type spellings ----------------------------------------------------------

SPELLINGS = "Acme/t.xlsx,S,Payment Voucher,PV No,Text,Optional,\nAcme/t.xlsx,S,PAYMENT VOUCHER,PV No,Text,Key,\n"


def test_the_stronger_role_wins_a_merge(tmp_path: Path):
    pack = _build(tmp_path, "Acme,Payment Voucher,PV-14380.pdf,.pdf,5,1,,document\n", SPELLINGS)
    assert _manifest(pack)["Acme/Payment Voucher/PV-14380.pdf"]["key_gold"] == "PV-14380"


def test_a_merge_conflict_is_reported(tmp_path: Path):
    pack = _build(tmp_path, "Acme,Payment Voucher,PV-14380.pdf,.pdf,5,1,,document\n", SPELLINGS)
    assert "`Payment Voucher` / `PV No`: role `key` and `optional`; used `key`" in _report(pack)


def test_a_filled_date_format_wins_a_merge(tmp_path: Path):
    schema = "Acme/t.xlsx,S,Memo,Date,Date,Required,\nAcme/t.xlsx,S,MEMO,Date,Date(DD/MM/YYYY),Required,\n"
    pack = _build(tmp_path, "Acme,Memo,m1.pdf,.pdf,5,1,,document\n", schema)
    assert load_pack(pack, "Acme").fields["Memo"][0].date_format == "DD/MM/YYYY"


# 4: possible duplicates ------------------------------------------------------------

COPIES = "Acme,Inv/Old,x.pdf,.pdf,5,1,,document\nAcme,Inv/New,x.pdf,.pdf,5,1,,document\n"


def test_same_name_and_size_in_two_folders_share_a_dup_group(tmp_path: Path):
    rows = _manifest(_build(tmp_path, COPIES))
    groups = {rows["Acme/Inv/Old/x.pdf"]["dup_group"], rows["Acme/Inv/New/x.pdf"]["dup_group"]}
    assert len(groups) == 1 and "" not in groups


def test_possible_duplicates_are_reported(tmp_path: Path):
    assert "### Possible duplicates (same name and size, different folder)" in _report(_build(tmp_path, COPIES))


def test_a_content_hash_column_decides_the_group(tmp_path: Path):
    inv = (tmp_path / "inv.csv", tmp_path / "schema.csv")
    inv[0].write_text(
        HEADER.replace("kind", "kind,sha256")
        + "Acme,Inv/Old,x.pdf,.pdf,5,1,,document,aa\nAcme,Inv/New,y.pdf,.pdf,6,1,,document,aa\n",
        encoding="utf-8",
    )
    inv[1].write_text(SCHEMA_HEADER, encoding="utf-8")
    build_pack(inv[0], inv[1], tmp_path / "pack")
    assert _manifest(tmp_path / "pack")["Acme/Inv/New/y.pdf"]["dup_group"] != ""


# 6: copy markers -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "key"),
    [
        ("Copy of PV-14380.pdf", "PV-14380"),
        ("INV(2024).pdf", "INV(2024)"),
        ("Form 16 (2023).pdf", "Form 16 (2023)"),
        ("PV-9 - Copy (2).pdf", "PV-9"),
        ("PV-8 copy (3).pdf", "PV-8"),
        ("PV-7 (1).pdf", "PV-7"),
    ],
)
def test_key_gold_strips_only_real_copy_markers(name: str, key: str):
    assert key_gold(name) == key


# 7: an alias equal to another class name -------------------------------------------


def test_an_alias_equal_to_another_class_is_dropped_and_reported(tmp_path: Path):
    (tmp_path / "pack").mkdir()
    (tmp_path / "pack" / "label_map.yaml").write_text(
        yaml.safe_dump({"Acme": {"overrides": {"Beta": "Alpha"}}}), encoding="utf-8"
    )
    schema = "Acme/t.xlsx,S,Alpha,A No,Text,Key,\nAcme/t.xlsx,S,Beta,B No,Text,Key,\n"
    pack = _build(tmp_path, "Acme,Beta,b1.pdf,.pdf,5,1,,document\n", schema, all_types=True)
    alpha = next(c for c in load_pack(pack, "Acme").classes if c.name == "Alpha")
    assert (alpha.aliases, "`Beta` of `Alpha`" in _report(pack)) == ([], True)


# 8: wrappers -----------------------------------------------------------------------


def test_a_samples_folder_with_cases_and_no_matching_type_is_the_label(tmp_path: Path):
    rows = "Acme,Samples/C-1,a.pdf,.pdf,1,1,,document\nAcme,Samples/C-1,b.pdf,.pdf,2,1,,document\n"
    row = _manifest(_build(tmp_path, rows, INVOICE))["Acme/Samples/C-1/a.pdf"]
    assert (row["case_id"], row["label"]) == ("C-1", "")


def test_without_a_schema_a_lone_folder_chain_is_not_skipped(tmp_path: Path):
    pack = _build(tmp_path, "Acme,Invoices/2023/Jan,i1.pdf,.pdf,1,1,,document\n")
    assert "- `Invoices`: no schema type matched" in _report(pack)


def test_skipped_wrappers_are_reported(tmp_path: Path):
    pack = _build(tmp_path, "Acme,Docs/Invoice,INV-1.pdf,.pdf,1,1,,document\n", INVOICE)
    assert "### Wrapper folders skipped\n\n- `Docs`" in _report(pack)


# 10: unknown overrides and duplicate details --------------------------------------


def test_unknown_field_override_keys_are_reported(tmp_path: Path):
    (tmp_path / "pack").mkdir()
    overrides = {"Acme": {"Invoice": {"Nope": {"origin": "erp"}}, "Ghost": {}}, "Nobody": {}}
    (tmp_path / "pack" / "field_overrides.yaml").write_text(yaml.safe_dump(overrides), encoding="utf-8")
    text = _report(_build(tmp_path, "Acme,Invoice,INV-1.pdf,.pdf,1,1,,document\n", INVOICE))
    assert all(s in text for s in ("field `Invoice` / `Nope`", "type `Ghost`", "- `Nobody`"))


def test_a_loose_file_and_its_zip_copy_are_told_apart(tmp_path: Path):
    rows = (
        "Acme,,Inv.zip,.zip,9,,,archive\nAcme,Inv,x.pdf,.pdf,5,1,,document\nAcme,Inv,x.pdf,.pdf,5,1,Inv.zip,document\n"
    )
    text = _report(_build(tmp_path, rows))
    assert "(zip member `x.pdf` of `Inv.zip`); kept `Acme/Inv/x.pdf` (loose file)" in (
        tmp_path / "pack" / "acme" / "excluded.csv"
    ).read_text(encoding="utf-8")
    assert "dropped `Acme/Inv/x.pdf` (zip member" in text


# 11: old manifests -----------------------------------------------------------------


def test_a_manifest_without_role_and_dup_group_still_loads(tmp_path: Path):
    pack = _build(tmp_path, "Acme,Invoice,INV-1.pdf,.pdf,1,1,,document\n", INVOICE)
    path = pack / "acme" / "manifest.csv"
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    old = ["doc_id", "rel_path", "in_zip", "case_id", "label", "label_source", "key_gold", "pages", "size_kb"]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=old, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    row = load_pack(pack, "Acme").manifest[0]
    assert (row.role, row.dup_group, row.member) == ("document", "", "")
