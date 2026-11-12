"""Client packs: label mapping, screening, dedup, cases, keys, fields, load, materialize, CLI."""

from __future__ import annotations

import csv
import zipfile
from pathlib import Path

import pytest
import yaml

from baseline.cli import main
from baseline.clientpack import ClientPackError, load_pack, materialize
from baseline.clientpack_build import build_pack
from baseline.clientpack_rules import field_origin, key_gold, match_label, parse_dtype, parse_role

INVENTORY = """cabinet,folder,name,ext,size_kb,pages,in_zip,kind
Acme,Acme/Payment Voucher,PV-14380.pdf,.pdf,120,1,,document
Acme,Acme/Payment Voucher,6390_PV-200.pdf,.pdf,90,1,,document
Acme,Acme/Payment Voucher,Payment Voucher.png,.png,30,1,,document
Acme,Acme/Payment Voucher,big.pdf,.pdf,99999,1,,document
Acme,Acme/Payment Voucher,more.zip,.zip,70,,,archive
Acme,Acme/Payment Voucher/more,PV-555.pdf,.pdf,60,1,more.zip,document
Acme,Acme,opex- Index fields.jpg,.jpg,40,1,,document
Acme,Acme,Acme types.xlsx,.xlsx,10,,,reference
Acme,Acme,Thumbs.db,.db,1,,,other
Acme,Acme/LGRN,LG-77.tif,.tif,200,2,,document
Acme,Acme/Delivery Note,dn_scan.pdf,.pdf,50,1,,document
Acme,Acme/ENDORSEMENT,e1.pdf,.pdf,51,1,,document
Acme,Acme/Misc,notes.pdf,.pdf,10,1,,document
Borealis,,Purchase Daily Invoice.tif,.tif,300,3,,document
Borealis,,Scan 0001.tif,.tif,100,1,,document
Corvid,Claims/C-1,a.pdf,.pdf,11,1,,document
Corvid,Claims/C-1,b.pdf,.pdf,12,1,,document
Corvid,Claims/C-2,c.pdf,.pdf,13,1,,document
Corvid,Claims/C-2,d.pdf,.pdf,14,1,,document
Corvid,Claims/C-2,e.pdf,.pdf,15,1,,document
Delta,Corvid/Claims/C-1,a.pdf,.pdf,11,1,,document
Delta,Corvid/Claims/C-1,b.pdf,.pdf,12,1,,document
Delta,Invoices,old.zip,.zip,30,,,archive
Delta,Invoices,new.zip,.zip,60,,,archive
Delta,Invoices/old,INV-1.pdf,.pdf,20,1,old.zip,document
Delta,Invoices/new/old,INV-1.pdf,.pdf,20,1,new.zip,document
"""

SCHEMA = """workbook,sheet,document_type,index_field,data_type,field_type,source
Acme\\Acme\\Acme types.xlsx,S,Payment Voucher,PV No,Text (Alpha Numeric),Key,indexing method=OCR
Acme\\Acme\\Acme types.xlsx,S,Payment Voucher,Voucher Date,Date(DD/MM/YYYY),Required,
Acme\\Acme\\Acme types.xlsx,S,Payment Voucher,Amount,Numeric (3 Decimal Place),optional,
Acme\\Acme\\Acme types.xlsx,S,Payment Voucher,Vendor Code,Text,Required,indexing method=Autofil(DB); db_view=v_vendor
Acme\\Acme\\Acme types.xlsx,S,Payment Voucher,Box No,Text,Required,indexing method=Manual
Acme\\Acme\\Acme types.xlsx,S,Payment Voucher,Company,Text,Fixed,
Acme\\Acme\\Acme types.xlsx,S,Payment Voucher,Status,Selection,Maybe,
Acme\\Acme\\Acme types.xlsx,S,Delivery Note - Yellow,DN No,Text,Key,
Acme\\Acme\\Acme types.xlsx,S,Delivery Note - Pink,DN No,Text,Key,
Acme\\Acme\\Acme types.xlsx,S,LGRN (Local GRN),GRN No,Text,Key,
Acme\\Acme\\Acme types.xlsx,S,Bank ENDORSEMENT,Ref,Text,Key,
Acme\\Acme\\Acme types.xlsx,S,Cheque ENDORSEMENT,Ref,Text,Key,
Borealis/types.xlsx,S,Purchase Daily Invoice,Invoice No,Text,Key,
Corvid/types.xlsx,S,Claims,Claim No,Text,Key,
"""


@pytest.fixture
def csvs(tmp_path: Path) -> tuple[Path, Path]:
    inv, schema = tmp_path / "inventory.csv", tmp_path / "schema.csv"
    inv.write_text(INVENTORY, encoding="utf-8-sig")
    schema.write_text(SCHEMA, encoding="utf-8")
    return inv, schema


@pytest.fixture
def pack(tmp_path: Path, csvs: tuple[Path, Path]) -> Path:
    out = tmp_path / "client_pack"
    build_pack(*csvs, out)
    return out


def _manifest(pack_dir: Path, cabinet: str) -> dict[str, dict[str, str]]:
    with open(pack_dir / cabinet / "manifest.csv", encoding="utf-8", newline="") as fh:
        return {r["rel_path"]: r for r in csv.DictReader(fh)}


def _excluded(pack_dir: Path, cabinet: str) -> dict[str, str]:
    with open(pack_dir / cabinet / "excluded.csv", encoding="utf-8", newline="") as fh:
        return {r["rel_path"]: r["reason"] for r in csv.DictReader(fh)}


# label mapping ------------------------------------------------------------

TYPES = ["Payment Voucher", "LGRN (Local GRN)", "Delivery Note - Yellow", "Bank ENDORSEMENT", "Cheque ENDORSEMENT"]


def test_label_exact_match_ignores_case_and_punctuation():
    assert match_label("payment-voucher", TYPES).target == "Payment Voucher"


def test_label_prefix_matches_alias():
    assert match_label("LGRN", TYPES).target == "LGRN (Local GRN)"


def test_label_token_subset_matches():
    assert match_label("Yellow Note", TYPES).target == "Delivery Note - Yellow"


def test_label_matching_several_types_is_ambiguous():
    found = match_label("ENDORSEMENT", TYPES)
    assert (found.target, found.candidates) == ("", ("Bank ENDORSEMENT", "Cheque ENDORSEMENT"))


def test_ambiguous_label_is_unmapped_and_reported(pack: Path):
    assert _manifest(pack, "acme")["Acme/Acme/ENDORSEMENT/e1.pdf"]["label"] == ""
    assert "`ENDORSEMENT`: candidates `Bank ENDORSEMENT`, `Cheque ENDORSEMENT`" in (pack / "REPORT.md").read_text()


def test_label_map_override_wins_and_survives_rebuild(pack: Path, csvs: tuple[Path, Path]):
    path = pack / "label_map.yaml"
    data = yaml.safe_load(path.read_text())
    data["Acme"]["Delivery Note"] = "Delivery Note - Pink"
    path.write_text(yaml.safe_dump(data))
    build_pack(*csvs, pack)
    assert _manifest(pack, "acme")["Acme/Acme/Delivery Note/dn_scan.pdf"]["label"] == "Delivery Note - Pink"
    assert yaml.safe_load(path.read_text())["Acme"]["Delivery Note"] == "Delivery Note - Pink"


def test_file_name_is_the_label_without_type_folders(pack: Path):
    row = _manifest(pack, "borealis")["Borealis/Purchase Daily Invoice.tif"]
    assert (row["label"], row["label_source"]) == ("Purchase Daily Invoice", "filename")


# screening and dedup ------------------------------------------------------


def test_copied_cabinet_is_dropped_as_duplicate(pack: Path):
    assert _excluded(pack, "delta")["Delta/Corvid/Claims/C-1/a.pdf"] == "duplicate"
    assert "Corvid/Claims/C-1/a.pdf" in _manifest(pack, "corvid")


def test_zip_copied_inside_a_sibling_zip_is_dropped(pack: Path):
    assert _excluded(pack, "delta")["Delta/Invoices/new/old/INV-1.pdf"] == "duplicate"
    assert "Delta/Invoices/old/INV-1.pdf" in _manifest(pack, "delta")


def test_screenshots_beside_samples_are_references(pack: Path):
    excluded = _excluded(pack, "acme")
    assert excluded["Acme/Acme/Payment Voucher/Payment Voucher.png"] == "reference"
    assert excluded["Acme/Acme/opex- Index fields.jpg"] == "reference"


def test_archives_other_files_and_oversize_are_excluded(pack: Path):
    excluded = _excluded(pack, "acme")
    assert excluded["Acme/Acme/Payment Voucher/more.zip"] == "archive"
    assert excluded["Acme/Acme/Thumbs.db"] == "unsupported"
    assert excluded["Acme/Acme/Payment Voucher/big.pdf"] == "oversize"


# cases and keys -----------------------------------------------------------


def test_case_bundles_get_a_case_id_and_no_key(pack: Path):
    row = _manifest(pack, "corvid")["Corvid/Claims/C-2/e.pdf"]
    assert (row["case_id"], row["label"], row["key_gold"]) == ("C-2", "Claims", "")


def test_key_gold_strips_numeric_prefix():
    assert key_gold("6390_C111322.pdf") == "C111322"


def test_key_gold_needs_a_digit():
    assert key_gold("Purchase Daily Invoice.tif") == ""


def test_manifest_key_gold_comes_from_the_stem(pack: Path):
    rows = _manifest(pack, "acme")
    assert rows["Acme/Acme/Payment Voucher/6390_PV-200.pdf"]["key_gold"] == "PV-200"
    assert rows["Acme/Acme/Payment Voucher/more/PV-555.pdf"]["key_gold"] == "PV-555"


# fields -------------------------------------------------------------------


def test_dtype_and_date_format_parse():
    assert parse_dtype("Date(DD/MM/YYYY)") == ("date", "DD/MM/YYYY")
    assert parse_dtype("Text (Alpha Numeric)") == ("text", None)
    assert parse_dtype("Numeric (3 Decimal Place)") == ("number", None)


def test_unrecognised_field_type_has_no_role():
    assert parse_role("Maybe") is None


def test_field_origin_rules():
    assert field_origin("Company", "Fixed", {}, "required") == "fixed"
    assert field_origin("Vendor Code", "Required", {"indexing method": "Autofil(DB)"}, "required") == "erp"
    assert field_origin("Vendor Name", "Required", {"db_field": "NAME"}, "required") == "erp"
    assert field_origin("PV No", "Key", {"db_field": "PVNO"}, "key") == "document"
    assert field_origin("Box No", "Required", {}, "required") == "manual"


def test_fields_yaml_records_roles_and_origins(pack: Path):
    fields = {f["name"]: f for f in yaml.safe_load((pack / "acme" / "fields.yaml").read_text())["Payment Voucher"]}
    assert (fields["PV No"]["role"], fields["PV No"]["origin"]) == ("key", "document")
    assert fields["Voucher Date"]["date_format"] == "DD/MM/YYYY"
    assert (fields["Status"]["role"], fields["Company"]["origin"]) == ("optional", "fixed")
    assert "`Payment Voucher` / `Status`: `Maybe`" in (pack / "REPORT.md").read_text()


# taxonomy, idempotence, load ----------------------------------------------


def test_taxonomy_holds_types_with_samples_plus_other(pack: Path):
    classes = yaml.safe_load((pack / "acme" / "taxonomy.yaml").read_text())["classes"]
    assert [c["name"] for c in classes] == ["LGRN (Local GRN)", "Payment Voucher", "other"]
    assert classes[0]["aliases"] == ["LGRN"]


def test_all_types_writes_every_schema_type(tmp_path: Path, csvs: tuple[Path, Path]):
    build_pack(*csvs, tmp_path / "p", all_types=True)
    classes = yaml.safe_load((tmp_path / "p" / "acme" / "taxonomy.yaml").read_text())["classes"]
    assert len(classes) == 7  # six Acme types plus other


def test_build_is_idempotent(pack: Path, csvs: tuple[Path, Path]):
    names = ["label_map.yaml", "REPORT.md", "acme/manifest.csv", "acme/fields.yaml", "delta/excluded.csv"]
    before = {n: (pack / n).read_bytes() for n in names}
    build_pack(*csvs, pack)
    assert {n: (pack / n).read_bytes() for n in names} == before


def test_load_pack_round_trip(pack: Path):
    loaded = load_pack(pack, "Acme")
    assert loaded.cabinet.slug == "acme"
    assert loaded.key_field("Payment Voucher") == "PV No"
    assert {r.rel_path for r in loaded.manifest} == set(_manifest(pack, "acme"))
    assert loaded.manifest[0].pages is not None


def test_load_pack_unknown_cabinet_raises(pack: Path):
    with pytest.raises(ClientPackError):
        load_pack(pack, "Nobody")


# materialize --------------------------------------------------------------


@pytest.fixture
def share_pack(tmp_path: Path) -> tuple[Path, Path]:
    share = tmp_path / "share"
    folder = share / "Acme" / "Payment Voucher"
    folder.mkdir(parents=True)
    (folder / "PV-1.pdf").write_bytes(b"%PDF loose")
    with zipfile.ZipFile(folder / "more.zip", "w") as archive:
        archive.writestr("inner/PV-2.pdf", b"%PDF zipped")
    inv = tmp_path / "inv.csv"
    inv.write_text(
        "cabinet,folder,name,ext,size_kb,pages,in_zip,kind\n"
        "Acme,Payment Voucher,PV-1.pdf,.pdf,1,1,,document\n"
        "Acme,Payment Voucher,more.zip,.zip,1,,,archive\n"
        "Acme,Payment Voucher/more/inner,PV-2.pdf,.pdf,1,1,more.zip,document\n",
        encoding="utf-8",
    )
    schema = tmp_path / "schema.csv"
    schema.write_text("workbook,sheet,document_type,index_field,data_type,field_type,source\n", encoding="utf-8")
    build_pack(inv, schema, tmp_path / "pack")
    return share, tmp_path / "pack"


def test_materialize_copies_loose_files_and_zip_members(tmp_path: Path, share_pack: tuple[Path, Path]):
    share, pack_dir = share_pack
    paths = materialize(pack_dir, "Acme", share, tmp_path / "work")
    assert sorted(p.read_bytes() for p in paths) == [b"%PDF loose", b"%PDF zipped"]


def test_materialize_is_idempotent(tmp_path: Path, share_pack: tuple[Path, Path]):
    share, pack_dir = share_pack
    first = materialize(pack_dir, "Acme", share, tmp_path / "work")
    assert materialize(pack_dir, "Acme", share, tmp_path / "work") == first


def test_materialize_refuses_a_work_dir_inside_the_share(share_pack: tuple[Path, Path]):
    share, pack_dir = share_pack
    with pytest.raises(ClientPackError):
        materialize(pack_dir, "Acme", share, share / "work")


# CLI ----------------------------------------------------------------------


def test_cli_build_dry_run_writes_nothing(tmp_path: Path, csvs: tuple[Path, Path]):
    out = tmp_path / "dry"
    assert main(["client-pack", "build", str(csvs[0]), str(csvs[1]), "--out", str(out), "--dry-run"]) == 0
    assert not out.exists()


def test_cli_materialize_dry_run_writes_nothing(tmp_path: Path, share_pack: tuple[Path, Path]):
    share, pack_dir = share_pack
    work = tmp_path / "work"
    argv = ["client-pack", "materialize", "--pack", str(pack_dir), "--cabinet", "Acme", "--share", str(share)]
    assert main([*argv, "--work", str(work), "--dry-run"]) == 0
    assert not work.exists()


def test_cli_build_then_materialize(tmp_path: Path, share_pack: tuple[Path, Path]):
    share, pack_dir = share_pack
    argv = ["client-pack", "materialize", "--pack", str(pack_dir), "--cabinet", "acme", "--share", str(share)]
    assert main([*argv, "--work", str(tmp_path / "work")]) == 0
    assert len(list((tmp_path / "work" / "acme").iterdir())) == 2


def test_cli_build_writes_the_pack(tmp_path: Path, csvs: tuple[Path, Path]):
    out = tmp_path / "cli_pack"
    assert main(["client-pack", "build", str(csvs[0]), str(csvs[1]), "--out", str(out), "--share", "S:/x"]) == 0
    assert yaml.safe_load((out / "pack.yaml").read_text())["share_root"] == "S:/x"
