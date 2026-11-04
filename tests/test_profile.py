"""`baseline profile`: corpus counts and the markdown report, no models."""

from conftest import ROOT

from baseline.profile import run_profile


def test_profile_counts_the_synthetic_corpus(corpus, tmp_path):
    out = tmp_path / "profile.md"
    stats = run_profile(corpus, out, str(ROOT / "config"))
    assert stats["n_documents"] == 10
    assert stats["by_extension"] == {".pdf": 6, ".txt": 3, ".docx": 1}
    assert stats["scanned"] == 1 and stats["empty"] == 1  # receipt_scanned, broken.pdf
    assert stats["errors"]["extract"] == 1
    assert [d[0] for d in stats["hybrid_deltas"]] == ["invoice_hybrid.pdf"]
    md = out.read_text(encoding="utf-8")
    assert "Documents found: **10**" in md


def test_profile_of_an_empty_folder_writes_a_report_not_a_crash(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    out = tmp_path / "out" / "profile.md"
    stats = run_profile(empty, out, str(ROOT / "config"))
    assert stats["n_documents"] == 0
    md = out.read_text(encoding="utf-8")
    assert "Documents found: **0**" in md
    assert "n/a" in md  # percentages of zero documents, not a ZeroDivisionError
