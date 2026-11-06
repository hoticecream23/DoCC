"""Court document fields, against headers shaped like the real indiankanoon ones."""

from baseline.metadata import extract_fields

KARNATAKA = """Mr. P S Manjunath vs The State Of Karnataka on 29 January,
2026
Author: Suraj Govindaraj
Bench: Suraj Govindaraj
                                                  NC: 2026:KHC:5069
                   IN THE HIGH COURT OF KARNATAKA AT BENGALURU
                        WRIT PETITION NO. 24060 OF 2024 (LB-RES)
The petitioner relied on the view the Bombay High Court took
in the decision to a civil suit No. 5 of 1919.
"""

OLD_SUPREME_COURT = """Supra Dye Chem vs Union Of India on 1 January, 1800
Equivalent citations: 1990(49)ELT196(BOM), 1991 (2) SCC 1
PETITIONER:
Supra Dye Chem
BENCH:
ACT:
THE HIGH COURT FELL INTO GRAVE ERROR AND ACTED
"""


def _fields(cfg, text):
    fields, errors = extract_fields(text, [text], cfg, doc_class="court_document")
    assert not errors
    for f in fields:
        assert text[f.char_start : f.char_end] == f.value
    return {f.field: f.normalized_value for f in fields}


def test_a_modern_high_court_header(cfg):
    got = _fields(cfg, KARNATAKA)
    assert got["petitioner"] == "Mr. P S Manjunath"
    assert got["respondent"] == "The State Of Karnataka"
    assert got["judgment_date"] == "2026-01-29"  # the title wraps inside the date
    assert got["bench"] == "Suraj Govindaraj"
    assert got["court"] == "HIGH COURT OF KARNATAKA AT BENGALURU"
    assert got["citation"] == "2026:KHC:5069"
    assert got["case_number"] == "WRIT PETITION NO. 24060 OF 2024"


def test_placeholders_and_running_text_are_not_values(cfg):
    got = _fields(cfg, OLD_SUPREME_COURT)
    assert got["petitioner"] == "Supra Dye Chem"
    assert "judgment_date" not in got  # 1 January 1800 means no date
    assert "bench" not in got  # "ACT:" is the next heading, not a judge
    assert "court" not in got  # a sentence, not a header
    assert got["citation"] == "1990(49)ELT196(BOM)"
