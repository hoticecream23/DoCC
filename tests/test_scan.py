"""What counts as a document, and what is only shaped like one.

Every rule here came from a real file in the client corpus. The macOS one is
the important one: those files carry a supported extension, so they were being
processed as documents rather than skipped, and a corpus count that silently
includes junk is as wrong as one that silently drops input.
"""

from __future__ import annotations

import pytest

from baseline.scan import scan_documents


def _corpus(tmp_path):
    root = tmp_path / "corpus"
    (root / "__MACOSX" / "Jindal").mkdir(parents=True)
    (root / "real").mkdir()

    (root / "real" / "invoice.pdf").write_bytes(b"%PDF-1.4 pretend")
    (root / "real" / "scan.jpg").write_bytes(b"\xff\xd8\xff pretend")
    (root / "real" / "notes.docx").write_bytes(b"PK pretend")

    # AppleDouble stubs. Same names, same extensions, not documents.
    (root / "__MACOSX" / "Jindal" / "._invoice.pdf").write_bytes(b"\x00\x05\x16\x07stub")
    (root / "real" / "._scan.jpg").write_bytes(b"\x00\x05\x16\x07stub")

    (root / "real" / "questions.xlsx").write_bytes(b"PK pretend")
    (root / "real" / "bundle.zip").write_bytes(b"PK pretend")
    (root / "real" / "readme.rtf").write_bytes(b"unsupported")
    return root


def test_scan_accounts_for_every_file(tmp_path):
    """Kept plus excluded must equal what is on disk, or the total is a guess."""
    root = _corpus(tmp_path)
    kept, excluded = scan_documents(root)
    on_disk = sum(1 for p in root.rglob("*") if p.is_file())
    assert len(kept) + sum(len(v) for v in excluded.values()) == on_disk


def test_macos_stubs_are_not_documents(tmp_path):
    """They carry .pdf and .docx, which is exactly why they slipped through."""
    root = _corpus(tmp_path)
    kept, excluded = scan_documents(root)
    names = {p.name for p in kept}
    assert not any(n.startswith("._") for n in names)
    assert "__MACOSX" not in {part for p in kept for part in p.parts}
    assert len(excluded["macos_resource_fork"]) == 2


def test_spreadsheets_and_archives_are_excluded_with_a_reason(tmp_path):
    """Not silently: the reason is what stops the next person re-adding them."""
    root = _corpus(tmp_path)
    kept, excluded = scan_documents(root)
    assert "questions.xlsx" not in {p.name for p in kept}
    assert "bundle.zip" not in {p.name for p in kept}
    assert excluded["evaluation_set"] and excluded["archive"]
    assert any("readme.rtf" in p for p in excluded["unsupported_extension"])


def test_the_real_documents_survive(tmp_path):
    root = _corpus(tmp_path)
    kept, _ = scan_documents(root)
    assert {p.name for p in kept} == {"invoice.pdf", "scan.jpg", "notes.docx"}


def test_a_single_file_root_is_taken_as_given(tmp_path):
    root = _corpus(tmp_path)
    one = root / "real" / "invoice.pdf"
    kept, excluded = scan_documents(one)
    assert kept == [one]
    assert excluded == {}


def test_a_missing_root_is_an_error_not_an_empty_corpus(tmp_path):
    """A mistyped --input used to walk nothing and report a clean run of zero."""
    with pytest.raises(FileNotFoundError):
        scan_documents(tmp_path / "does_not_exist")
