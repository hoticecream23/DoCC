"""Checksum validators, known good against known bad.

The generic property tests at the bottom matter more than the fixed vectors:
they assert the error detection guarantee each algorithm actually promises.
"""

import pytest

from baseline.metadata import (
    aadhaar_check,
    amount_sane_check,
    date_sane_check,
    email_check,
    gstin_check,
    ifsc_check,
    invoice_number_check,
    luhn_check,
    pan_check,
    phone_in_check,
    run_validator,
    verhoeff_check,
    verhoeff_digit,
)

# ---------------------------------------------------------------- PAN

GOOD_PAN = ["AAPFU0939F", "AAAPZ1234C", "AABCU9603R", "ABCPD1234E"]
BAD_PAN = [
    "ABCDE1234F",   # D is not a holder type code
    "AAPFU0939",    # too short
    "AAPFU09399",   # last char must be a letter
    "AAPF10939F",   # digit inside the letter block
    "",
]


@pytest.mark.parametrize("v", GOOD_PAN)
def test_pan_good(v):
    assert pan_check(v) is True


@pytest.mark.parametrize("v", BAD_PAN)
def test_pan_bad(v):
    assert pan_check(v) is False


def test_validators_are_case_insensitive():
    """The regex layer is case sensitive, the validator is not. Deliberate."""
    assert pan_check("aapfu0939f") is True
    assert ifsc_check("hdfc0001234") is True
    assert gstin_check("27aapfu0939f1zv") is True


def test_pan_entity_types_are_configurable():
    assert pan_check("ABCDE1234F") is False
    assert pan_check("ABCDE1234F", {"entity_types": "ABCDEFGHJKLPT"}) is True


# ---------------------------------------------------------------- Aadhaar / Verhoeff

GOOD_AADHAAR = ["269837654322", "2698 3765 4322", "2698-3765-4322"]
BAD_AADHAAR = [
    "269837654321",   # wrong check digit
    "269837654",      # too short
    "126983765432",   # Aadhaar never starts with 0 or 1
    "069837654322",
    "abcdefghijkl",
    "",
]


@pytest.mark.parametrize("v", GOOD_AADHAAR)
def test_aadhaar_good(v):
    assert aadhaar_check(v) is True


@pytest.mark.parametrize("v", BAD_AADHAAR)
def test_aadhaar_bad(v):
    assert aadhaar_check(v) is False


def test_verhoeff_digit_round_trips():
    for payload in ["26983765432", "99999999999", "20000000000", "31415926535"]:
        assert verhoeff_check(payload + verhoeff_digit(payload)) is True


def test_verhoeff_catches_every_single_digit_error():
    """Verhoeff's core guarantee. If this fails the tables are wrong."""
    base = "26983765432" + verhoeff_digit("26983765432")
    for i in range(len(base)):
        for d in "0123456789":
            if d != base[i]:
                assert verhoeff_check(base[:i] + d + base[i + 1:]) is False


def test_verhoeff_catches_every_adjacent_transposition():
    base = "26983765432" + verhoeff_digit("26983765432")
    for i in range(len(base) - 1):
        if base[i] != base[i + 1]:
            swapped = base[:i] + base[i + 1] + base[i] + base[i + 2:]
            assert verhoeff_check(swapped) is False


# ---------------------------------------------------------------- GSTIN

GOOD_GSTIN = ["27AAPFU0939F1ZV", "29AAGCB7383J1Z4"]
BAD_GSTIN = [
    "27AAPFU0939F1ZX",   # check digit wrong
    "00AAPFU0939F1ZV",   # state code 00 does not exist
    "39AAPFU0939F1ZV",   # state code out of range
    "27AAPFU0939F1AV",   # 14th char must be Z
    "27ABCDE0939F1ZV",   # embedded PAN is structurally invalid
    "27AAPFU0939F1Z",    # too short
    "",
]


@pytest.mark.parametrize("v", GOOD_GSTIN)
def test_gstin_good(v):
    assert gstin_check(v) is True


@pytest.mark.parametrize("v", BAD_GSTIN)
def test_gstin_bad(v):
    assert gstin_check(v) is False


def test_gstin_rejects_every_wrong_check_digit():
    good = "27AAPFU0939F1ZV"
    alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    wrong = [c for c in alphabet if c != good[14]]
    assert all(gstin_check(good[:14] + c) is False for c in wrong)


# ---------------------------------------------------------------- Luhn

GOOD_CARDS = ["4539578763621486", "4111111111111111", "5500005555555559",
              "378282246310005"]
BAD_CARDS = ["4539578763621487", "4111111111111112", "1234567890123456",
             "411111111111", "", "abcd"]


@pytest.mark.parametrize("v", GOOD_CARDS)
def test_luhn_good(v):
    assert luhn_check(v) is True


@pytest.mark.parametrize("v", BAD_CARDS)
def test_luhn_bad(v):
    assert luhn_check(v) is False


def test_luhn_catches_every_single_digit_error():
    base = "4539578763621486"
    for i in range(len(base)):
        for d in "0123456789":
            if d != base[i]:
                assert luhn_check(base[:i] + d + base[i + 1:]) is False


def test_luhn_accepts_separators():
    assert luhn_check("4539 5787 6362 1486") is True
    assert luhn_check("4539-5787-6362-1486") is True


# ---------------------------------------------------------------- IFSC

@pytest.mark.parametrize("v", ["HDFC0001234", "SBIN0000456", "UTIB0000123"])
def test_ifsc_good(v):
    assert ifsc_check(v) is True


@pytest.mark.parametrize("v", ["HDFC1001234", "HDF00001234", "HDFC000123",
                               "HDFC00012345", "HD1C0001234", ""])
def test_ifsc_bad(v):
    assert ifsc_check(v) is False


# ---------------------------------------------------------------- soft validators

@pytest.mark.parametrize("v", ["a@b.co", "billing@acme.example", "x.y+z@sub.domain.in"])
def test_email_good(v):
    assert email_check(v) is True


@pytest.mark.parametrize("v", ["a@@b.co", "a@b", "@b.co", "a@.co", "a..b@c.co", ""])
def test_email_bad(v):
    assert email_check(v) is False


@pytest.mark.parametrize("v", ["9876543210", "+91 9876543210", "919876543210"])
def test_phone_good(v):
    assert phone_in_check(v) is True


@pytest.mark.parametrize("v", ["1234567890", "5876543210", "98765432", ""])
def test_phone_bad(v):
    assert phone_in_check(v) is False


@pytest.mark.parametrize("v", ["14/03/2024", "2024-03-14", "14 Mar 2024", "01-01-2020"])
def test_date_sane_good(v):
    assert date_sane_check(v) is True


@pytest.mark.parametrize("v", ["32/03/2024", "14/13/2024", "14/03/1850", "notadate", ""])
def test_date_sane_bad(v):
    assert date_sane_check(v) is False


# Every one of these is printed on a SROIE receipt, and none of them used to
# parse, so 15 of 361 gold dates had to be dropped from scoring.
@pytest.mark.parametrize("v, iso", [
    ("27/MAR/2018", "2018-03-27"),
    ("28-FEB-2018", "2018-02-28"),
    ("22 MAR 18", "2018-03-22"),
    ("2018/03/27", "2018-03-27"),
    ("23.03.18", "2018-03-23"),
    ("5 Sept. 2020", "2020-09-05"),
])
def test_receipt_date_formats_parse(v, iso):
    from baseline.metadata.normalizers import _parse_date

    assert str(_parse_date(v)) == iso


@pytest.mark.parametrize("v", ["1/2/345", "12.50.00", "999/12/2020", "12 items 2020"])
def test_near_dates_still_do_not_parse(v):
    from baseline.metadata.normalizers import _parse_date

    assert _parse_date(v) is None


@pytest.mark.parametrize("line, iso", [
    ("Date: 27/MAR/2018 12:01", "2018-03-27"),
    ("Date : 22 MAR 18", "2018-03-22"),
    ("DATE 23.03.18", "2018-03-23"),
    ("Date: 2018/03/27", "2018-03-27"),
])
def test_receipt_dates_are_extracted(line, iso, cfg):
    """Parsing them is not enough. The field regexes have to reach them too."""
    from baseline.metadata import extract_fields

    text = f"SOME SHOP SDN BHD\n{line}\nTOTAL 9.00"
    fields, _ = extract_fields(text, [text], cfg, doc_class="receipt")
    assert [f.normalized_value for f in fields if f.field == "invoice_date"] == [iso]


def test_date_range_is_configurable():
    assert date_sane_check("14/03/1985") is False
    assert date_sane_check("14/03/1985", {"min_year": 1900}) is True


@pytest.mark.parametrize("v", ["0", "11800.00", "11,800.00", "999999"])
def test_amount_sane_good(v):
    assert amount_sane_check(v) is True


@pytest.mark.parametrize("v", ["-5", "abc", ""])
def test_amount_sane_bad(v):
    assert amount_sane_check(v) is False


@pytest.mark.parametrize("v", ["INV-2024-0042", "A1", "PO-7781"])
def test_invoice_number(v):
    assert invoice_number_check(v) is (len(v) >= 3)


def test_invoice_number_needs_a_digit():
    assert invoice_number_check("INVOICE") is False


# ---------------------------------------------------------------- dispatch

def test_run_validator_dispatches_by_name():
    assert run_validator("pan", "AAPFU0939F") is True
    assert run_validator("gstin", "27AAPFU0939F1ZV") is True


def test_run_validator_is_safe_on_junk():
    assert run_validator("nope_not_a_validator", "x") is False
    assert run_validator(None, "x") is False
    assert run_validator("luhn", None) is False


# ------------------------------------------------- invoice_number pattern

@pytest.mark.parametrize("word", [
    # Every one of these matched before a digit was required, and the first is
    # printed on nearly every invoice in the corpus.
    "INVOICE", "INVOICES", "POLICY", "PORTAL", "POSTAL", "PORTION",
    "POWER", "BILLING", "BILLABLE", "PORTNON",
])
def test_invoice_number_pattern_ignores_ordinary_words(word, cfg):
    """The pattern layer emitted seven English words for every real reference.

    A field that fires on the word INVOICE fires on every invoice, which is
    why the hit rate looked like 69% and the validation rate like 1%.
    """

    rx = _invoice_number_regex(cfg)
    assert rx.search(word) is None, f"{word} still reads as an invoice number"


@pytest.mark.parametrize("ref", [
    "INV-2024-0042", "PO-1234", "PO-001234", "INV/2024/7", "BILL-99", "INV2024",
])
def test_invoice_number_pattern_still_reads_real_references(ref, cfg):
    rx = _invoice_number_regex(cfg)
    m = rx.search(ref)
    assert m is not None and m.group(1) == ref


def _invoice_number_regex(cfg):
    import re

    for f in cfg.field_defs:
        if f["name"] == "invoice_number":
            return re.compile(f["pattern"]["regex"])
    raise AssertionError("invoice_number is not in fields.yaml")


# ------------------------------------- a failed checksum emits nothing

def test_a_failed_checksum_emits_no_candidate(cfg):
    """On the client corpus every failing Aadhaar and card was OCR noise.

    Emitting them at low confidence still asserted PII the document does not
    hold, and sent each one to a reviewer to mark spurious by hand.
    """
    from baseline.metadata import extract_fields

    text = "Aadhaar 2698 3765 4321 and card 4111 1111 1111 1112"
    fields, _ = extract_fields(text, [text], cfg)
    assert not [f for f in fields if f.field in ("aadhaar", "card_number")]


def test_a_passing_checksum_is_still_emitted(cfg):
    from baseline.metadata import extract_fields

    text = "Aadhaar 2698 3765 4322 and card 4111 1111 1111 1111"
    fields, _ = extract_fields(text, [text], cfg)
    got = {f.field: f for f in fields if f.field in ("aadhaar", "card_number")}
    assert set(got) == {"aadhaar", "card_number"}
    assert all(f.validated and f.method.value == "regex_checksum" for f in got.values())


@pytest.mark.parametrize("line, ref", [
    ("TAX INVOICE ~ CASH # 13333469", "13333469"),
    ("TAX INVOICE - CREDIT # 13333481", "13333481"),
])
def test_yaumi_cash_and_credit_references_are_read(line, ref, cfg):
    """The corpus's dominant layout. Before these labels 2 of 140 were read."""
    from baseline.metadata import extract_fields

    text = f"YAUMI INTERNATIONAL BAKERIES\n{line}\nDate: 08/04/2021"
    fields, _ = extract_fields(text, [text], cfg, doc_class="invoice")
    assert [f.normalized_value for f in fields if f.field == "invoice_number"] == [ref]


@pytest.mark.parametrize("v, want", [
    ("11800.00", "11800.00"),
    ("1,18,000.50", "118000.50"),
    ("Rs. 11,800.00", "11800.00"),
    ("Rs.11800", "11800"),
    ("AED 2,500.75", "2500.75"),
    ("11,800/-", "11800"),
    ("-500.00", "-500.00"),
    ("11 800.00", "11800.00"),
    ("500 2024", "500"),
    ("Rs.", None),
])
def test_amounts_parse_past_a_currency_prefix(v, want):
    """A VLM answers with the printed amount, symbol and all."""
    from baseline.metadata.normalizers import _to_decimal

    got = _to_decimal(v)
    assert (str(got) if got is not None else None) == want
