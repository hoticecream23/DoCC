"""Client pack regressions from the confirmation review: case ids, marked copies, attachments, slugs."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from baseline.clientpack import assign_slugs, load_pack, materialize, slugify
from baseline.clientpack_build import build_pack
from baseline.evaluate.client_eval import CabinetScore, score_cabinet

HEADER = "cabinet,folder,name,ext,size_kb,pages,in_zip,kind\n"
SCHEMA = "workbook,sheet,document_type,index_field,data_type,field_type,source\n"
SCHEMA += "Acme/t.xlsx,S,Inv,Inv No,Text,Key,\nAcme/t.xlsx,S,Rec,Rec No,Text,Key,\n"
SLUG_RE = r"[a-z0-9_-]{1,64}"


def _build(tmp_path: Path, files: list[tuple[str, str]]) -> Path:
    rows = "".join(f"Acme,{folder},{name},.pdf,{i + 1},1,,document\n" for i, (folder, name) in enumerate(files))
    (tmp_path / "inv.csv").write_text(HEADER + rows, encoding="utf-8")
    (tmp_path / "schema.csv").write_text(SCHEMA, encoding="utf-8")
    build_pack(tmp_path / "inv.csv", tmp_path / "schema.csv", tmp_path / "pack")
    return tmp_path / "pack"


def _perfect_score(tmp_path: Path, files: list[tuple[str, str]]) -> CabinetScore:
    pack = _build(tmp_path, files)
    lines = [
        json.dumps({"filename": f"{r.doc_id}.pdf", "classification": {"label": r.label}})
        for r in load_pack(pack, "Acme").manifest
    ]
    results = tmp_path / "results.jsonl"
    results.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return score_cabinet(pack, "Acme", results)


def _manifest(pack: Path) -> dict[str, dict[str, str]]:
    with open(pack / "acme" / "manifest.csv", encoding="utf-8", newline="") as fh:
        return {r["rel_path"]: r for r in csv.DictReader(fh)}


# 1: a case is its full folder path -------------------------------------------------

SAME_NAME = [("Inv/A1", "a.pdf"), ("Inv/A1", "b.pdf"), ("Rec/A1", "c.pdf"), ("Rec/A1", "d.pdf")]
NESTED = [("Inv/2025/A1", "a.pdf"), ("Inv/2025/A1", "b.pdf"), ("Inv/2026/A1", "c.pdf"), ("Inv/2026/A1", "d.pdf")]


def test_same_named_case_folders_under_two_types_stay_two_right_cases(tmp_path: Path):
    score = _perfect_score(tmp_path, SAME_NAME)
    assert (score.cases, score.cases_all, score.mixed_cases) == (2, 2, [])


def test_nested_case_folders_are_two_cases(tmp_path: Path):
    assert _perfect_score(tmp_path, NESTED).cases == 2


def test_a_nested_case_id_is_the_path_to_the_folder_holding_the_files(tmp_path: Path):
    ids = {r.case_id for r in load_pack(_build(tmp_path, NESTED), "Acme").manifest}
    assert ids == {"Inv/2025/A1", "Inv/2026/A1"}


# 2: a marked copy beside its document ------------------------------------------------

COPY = [("Inv/3", "3.pdf"), ("Inv/3", "3 (1).pdf")]


def test_the_unmarked_file_is_the_document_with_its_key(tmp_path: Path):
    row = _manifest(_build(tmp_path, COPY))["Acme/Inv/3/3.pdf"]
    assert (row["role"], row["case_id"], row["key_gold"]) == ("document", "", "3")


def test_a_marked_copy_is_an_attachment_in_the_documents_dup_group(tmp_path: Path):
    rows = _manifest(_build(tmp_path, COPY))
    doc, copy = rows["Acme/Inv/3/3.pdf"], rows["Acme/Inv/3/3 (1).pdf"]
    assert (copy["role"], copy["dup_group"] == doc["dup_group"] != "") == ("attachment", True)


def test_a_marked_copy_is_listed_as_a_possible_duplicate(tmp_path: Path):
    text = (_build(tmp_path, COPY) / "REPORT.md").read_text(encoding="utf-8")
    assert "`Acme/Inv/3/3 (1).pdf`, `Acme/Inv/3/3.pdf`" in text


# 3: attachments are not materialized by default ------------------------------------

ANNEX = [("Inv/3", "3.pdf"), ("Inv/3", "annex.pdf")]


def _share(tmp_path: Path) -> Path:
    share = tmp_path / "share"
    (share / "Acme" / "Inv" / "3").mkdir(parents=True)
    for _, name in ANNEX:
        (share / "Acme" / "Inv" / "3" / name).write_bytes(b"%PDF")
    return share


def test_materialize_skips_attachments_by_default(tmp_path: Path):
    pack = _build(tmp_path, ANNEX)
    assert len(materialize(pack, "Acme", _share(tmp_path), tmp_path / "work")) == 1


def test_materialize_copies_attachments_when_asked(tmp_path: Path):
    pack = _build(tmp_path, ANNEX)
    assert len(materialize(pack, "Acme", _share(tmp_path), tmp_path / "work", with_attachments=True)) == 2


def test_an_unprocessed_attachment_is_not_a_missing_result(tmp_path: Path):
    score = _perfect_score(tmp_path, ANNEX)
    assert (score.attachments, score.missing, score.no_result) == (1, 0, [])


# 4: slug length and characters -----------------------------------------------------


def test_a_77_character_name_gives_a_slug_the_api_accepts():
    name = ("Acme Records " * 6)[:77]
    assert re.fullmatch(SLUG_RE, slugify(name))


def test_a_name_with_no_ascii_letters_gives_a_non_empty_ascii_slug():
    slug = slugify("\u0645\u0633\u062a\u0646\u062f\u0627\u062a")
    assert slug.startswith("cabinet_") and slug.isascii() and len(slug) > len("cabinet_")


def test_two_long_names_sharing_64_characters_get_distinct_capped_slugs():
    names = ["a" * 70 + " one", "a" * 70 + " two"]
    slugs = assign_slugs(names)
    assert (len(set(slugs.values())), max(len(s) for s in slugs.values())) == (2, 64)


def test_a_recorded_long_slug_is_kept():
    old = "b" * 80
    assert assign_slugs(["B"], {"B": old}) == {"B": old}
