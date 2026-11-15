"""Pack-mode label matching, value layouts, digits and number kinds, from the second-pass review inputs."""

from dataclasses import replace
from pathlib import Path

import pytest

from baseline import packmode
from baseline.clientpack import FieldSpec
from baseline.config import load_config
from baseline.metadata.fields import extract_fields
from baseline.metadata.layouts import PHRASE_CONFIDENCE

CONFIG = Path(__file__).resolve().parents[2] / "config"
CLASS = "Voucher"
ARABIC_LABEL = "\u0631\u0642\u0645 \u0627\u0644\u0633\u0646\u062f"  # "voucher number"
ARABIC_12345 = "\u0661\u0662\u0663\u0664\u0665"
RLM = "\u200f"


@pytest.fixture(scope="module")
def cfg():
    return load_config(CONFIG)


def _read(cfg, name, text, dtype="text", role="optional"):
    """(raw, normalised, confidence) of one pack field over text, or None."""
    by_name = {f["name"]: f for f in cfg.field_defs}
    fdef = packmode._field_def(FieldSpec(name=name, dtype=dtype, role=role, origin="document"), CLASS, by_name)
    one = replace(cfg, fields={**cfg.fields, "fields": [fdef], "templates": []})
    found = extract_fields(text, [text], one, CLASS)[0]
    return (found[0].value, found[0].normalized_value, found[0].confidence) if found else None


def test_an_empty_text_field_gives_nothing(cfg):
    assert _read(cfg, "Remarks", "Remarks:\nsecond line") is None


def test_a_label_followed_by_another_word_is_skipped(cfg):
    assert _read(cfg, "Total", "Total Tax 5.00\nTotal 100.00", "number")[0] == "100.00"


def test_a_label_inside_a_longer_phrase_after_it_gives_nothing(cfg):
    assert _read(cfg, "Date", "Date of birth 01/02/2026", "date") is None


def test_a_complete_label_beats_an_earlier_longer_phrase(cfg):
    text = "Supplier Invoice No: S-99\nInvoice No: I-5"
    assert _read(cfg, "Invoice No", text, role="key")[0] == "I-5"


def test_a_phrase_only_match_is_kept_at_review_confidence(cfg):
    assert _read(cfg, "Invoice No", "Supplier Invoice No: S-99", role="key")[1:] == ("S-99", PHRASE_CONFIDENCE)


def test_a_value_on_the_next_line_is_read(cfg):
    assert _read(cfg, "PV No", "PV No.\nPV-14380")[0] == "PV-14380"


@pytest.mark.parametrize("sep", [" | ", "\t", "   "])
def test_a_header_row_value_is_read_from_the_same_column(cfg, sep):
    text = sep.join(["PV No", "Date", "Amount"]) + "\n" + sep.join(["PV-14380", "01-02-2026", "100.00"])
    assert _read(cfg, "Amount", text, "number")[0] == "100.00"


def test_a_header_row_key_is_read_from_the_row_below(cfg):
    assert _read(cfg, "PV No", "PV No | Date | Amount\nPV-14380 | 01-02-2026 | 100.00")[0] == "PV-14380"


def test_a_combined_label_gives_the_first_field_its_value(cfg):
    assert _read(cfg, "Invoice No", "Invoice No. & Date: INV-77 / 01-02-2026")[0] == "INV-77"


@pytest.mark.parametrize("name", ["Invoice Date", "Date"])
def test_a_combined_label_gives_the_date_field_its_value(cfg, name):
    assert _read(cfg, name, "Invoice No. & Date: INV-77 / 01-02-2026", "date")[1] == "2026-02-01"


def test_an_abbreviated_label_matches_the_full_field_name(cfg):
    assert _read(cfg, "Invoice Number", "Inv. No: INV-9")[0] == "INV-9"


def test_a_po_number_is_not_read_after_against_invoice(cfg):
    assert _read(cfg, "PO No", "Against Invoice 333444\nPO No: 111222", role="key")[0] == "111222"


def test_an_invoice_number_does_not_take_the_po_number_below_it(cfg):
    assert _read(cfg, "Invoice No", "Invoice No:\nPO No: 555", role="key") is None


def test_a_bare_date_does_not_take_a_due_date_on_the_next_line(cfg):
    assert _read(cfg, "Date", "Date:\nDue Date: 01/02/2024", "date") is None


def test_arabic_indic_digits_keep_the_raw_value_and_normalise_to_ascii(cfg):
    assert _read(cfg, "PV No", f"PV No: {ARABIC_12345}")[:2] == (ARABIC_12345, "12345")


def test_an_arabic_label_with_a_direction_mark_reads_arabic_digits(cfg):
    assert _read(cfg, ARABIC_LABEL, f"{ARABIC_LABEL}: {RLM}{ARABIC_12345}")[1] == "12345"


def test_an_amount_with_arabic_separators_normalises(cfg):
    text = "Total: \u0661\u066c\u0662\u0663\u0664\u066b\u0665\u0660"
    assert _read(cfg, "Total", text, "number")[1] == "1234.50"


def test_a_mixed_line_reads_the_arabic_value_after_a_latin_label(cfg):
    assert _read(cfg, "PV No", f"{ARABIC_LABEL} / PV No: {ARABIC_12345} {RLM}")[1] == "12345"


@pytest.mark.parametrize(
    ("name", "text", "want"), [("Year", "Year: 2026", "2026"), ("Total Qty", "Total Qty: 12", "12")]
)
def test_a_counting_number_normalises_to_an_integer(cfg, name, text, want):
    assert _read(cfg, name, text, "number")[1] == want


def test_a_non_integral_counting_number_stays_text(cfg):
    assert _read(cfg, "Quantity", "Quantity: 2.5", "number")[1] == "2.5"


def test_number_of_pages_is_integer_like_not_an_identifier():
    spec = FieldSpec(name="No of Pages", dtype="number", role="optional", origin="document")
    assert (packmode.is_integer_like(spec), packmode.is_identifier(spec)) == (True, False)


def test_a_built_in_field_never_uses_the_complete_label_rule(cfg):
    assert not any((f.get("anchor") or {}).get("label_rule") for f in cfg.field_defs)
