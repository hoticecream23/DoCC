"""scripts/inventory_share.py lists a share without reading document content."""

from __future__ import annotations

import csv
import importlib.util
import sys
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts" / "inventory_share.py"
spec = importlib.util.spec_from_file_location("inventory_share", SCRIPT)
assert spec and spec.loader
inv = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = inv  # dataclasses look the module up by name
spec.loader.exec_module(inv)


@pytest.fixture
def share(tmp_path: Path) -> Path:
    client = tmp_path / "Acme" / "Acme"
    client.mkdir(parents=True)
    (client / "Acme- Claims document types.xlsx").write_bytes(b"x")
    (client / "Screenshot 2026-10-08 121432.png").write_bytes(b"x")
    with zipfile.ZipFile(client / "Claims Payable.zip", "w") as archive:
        archive.writestr("claim_001.pdf", b"not a real pdf")
    (tmp_path / "Borealis").mkdir()
    (tmp_path / "Borealis" / "Purchase Daily Invoice.tif").write_bytes(b"x" * 2048)
    return tmp_path


def _by_name(root: Path) -> dict[str, object]:
    return {row.name: row for row in inv.inventory(root)}


def test_first_level_folder_is_the_cabinet(share: Path):
    assert _by_name(share)["Purchase Daily Invoice.tif"].cabinet == "Borealis"


def test_workbooks_and_screenshots_are_reference_not_documents(share: Path):
    rows = _by_name(share)
    assert rows["Acme- Claims document types.xlsx"].kind == "reference"
    assert rows["Screenshot 2026-10-08 121432.png"].kind == "reference"


def test_zip_members_are_listed_under_the_zip_name(share: Path):
    member = _by_name(share)["claim_001.pdf"]
    assert (member.in_zip, member.folder, member.kind) == ("Claims Payable.zip", "Acme/Claims Payable", "document")


def test_unreadable_document_has_empty_page_count(share: Path):
    assert _by_name(share)["claim_001.pdf"].pages == ""


def test_main_writes_csv_and_dry_run_writes_nothing(share: Path, tmp_path: Path):
    out = tmp_path / "inventory.csv"
    assert inv.main([str(share), "--out", str(out), "--dry-run"]) == 0
    assert not out.exists()
    assert inv.main([str(share), "--out", str(out)]) == 0
    with out.open(encoding="utf-8-sig", newline="") as fh:
        names = [r["name"] for r in csv.DictReader(fh)]
    assert "claim_001.pdf" in names and "Claims Payable.zip" in names


def test_office_lock_files_are_skipped(share: Path):
    (share / "Acme" / "~$Acme- Claims document types.xlsx").write_bytes(b"x")
    assert not any(name.startswith("~$") for name in _by_name(share))


def test_schema_finds_header_below_a_title_and_fills_merged_types(tmp_path: Path):
    openpyxl = pytest.importorskip("openpyxl")
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.title = "Corvid"
    for row in (
        ["CORVID MARKET"],
        ["Document Type", "Index Name", "Data Type", "Field Type", "Indexing Method", "DB_VIEW", "Remarks"],
        ["Purchase Invoice", "Invoice No", "Text", "Mandatory", "OCR", "Y", "free text stays out"],
        [None, "Invoice Date", "Date", "Optional", "Manual", None, None],
    ):
        sheet.append(row)
    (tmp_path / "Delta").mkdir()
    book.save(tmp_path / "Delta" / "types.xlsx")
    rows = [r[2:] for r in inv.workbook_schema(tmp_path)]
    assert rows == [
        ("Purchase Invoice", "Invoice No", "Text", "Mandatory", "indexing method=OCR; db_view=Y"),
        ("Purchase Invoice", "Invoice Date", "Date", "Optional", "indexing method=Manual"),
    ]


def test_headers_show_first_rows_of_a_sheet_with_no_recognised_header(tmp_path: Path):
    openpyxl = pytest.importorskip("openpyxl")
    book = openpyxl.Workbook()
    for row in (["CORVID MARKET"], ["Purchase Invoice"], ["Invoice No", "Text"]):
        book.active.append(row)
    book.save(tmp_path / "types.xlsx")
    assert [r[2] for r in inv.workbook_headers(tmp_path)] == ["CORVID MARKET", "Purchase Invoice", "Invoice No | Text"]
