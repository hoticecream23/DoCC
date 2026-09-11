"""Offset correctness on synthetic documents.

The whole contract rests on text.full[char_start:char_end] == value. If these
fail, every downstream score is measuring the wrong span.
"""

import pytest

from baseline.metadata import OffsetMismatch, extract_fields
from baseline.schema import (
    PAGE_SEP,
    build_canonical_text,
    page_for_offset,
    page_spans,
)

PAGE_A = (
    "TAX INVOICE\n"
    "Invoice No: INV-2024-0042\n"
    "Invoice Date: 14/03/2024\n"
    "GSTIN: 27AAPFU0939F1ZV\n"
    "PAN: AAPFU0939F\n"
)
PAGE_B = (
    "Grand Total: Rs. 11,800.00\n"
    "IFSC: HDFC0001234\n"
    "Aadhaar: 2698 3765 4322\n"
)
PAGE_C = "Account Number: 502010123456789\nEmail: pay@acme.example\n"


# ---------------------------------------------------------------- canonical text

def test_canonical_text_is_pages_joined_by_the_separator():
    pages = [PAGE_A, PAGE_B, PAGE_C]
    full = build_canonical_text(pages)
    assert full == PAGE_SEP.join(pages)
    assert full.count(PAGE_SEP) == len(pages) - 1


def test_page_spans_slice_back_to_the_pages():
    pages = [PAGE_A, PAGE_B, PAGE_C]
    full = build_canonical_text(pages)
    for i, (s, e) in enumerate(page_spans(pages)):
        assert full[s:e] == pages[i]


def test_page_spans_handle_empty_pages():
    pages = ["", "abc", "", "de"]
    full = build_canonical_text(pages)
    for i, (s, e) in enumerate(page_spans(pages)):
        assert full[s:e] == pages[i]


def test_page_for_offset_maps_into_the_right_page():
    pages = [PAGE_A, PAGE_B, PAGE_C]
    spans = page_spans(pages)
    for i, (s, e) in enumerate(spans):
        assert page_for_offset(spans, s) == i
        assert page_for_offset(spans, max(s, e - 1)) == i


def test_single_page_canonical_text_has_no_separator():
    assert build_canonical_text([PAGE_A]) == PAGE_A


# ---------------------------------------------------------------- extraction offsets

@pytest.mark.parametrize(
    "pages",
    [
        [PAGE_A],
        [PAGE_A, PAGE_B],
        [PAGE_A, PAGE_B, PAGE_C],
        ["", PAGE_A, "", PAGE_B],          # empty pages must not shift anything
        [PAGE_C, PAGE_A, PAGE_B],          # order must not matter
    ],
)
def test_every_field_offset_slices_back_to_its_value(cfg, pages):
    full = build_canonical_text(pages)
    fields, errors = extract_fields(full, pages, cfg, doc_class="invoice")
    assert fields, "expected at least one field"
    for f in fields:
        assert full[f.char_start:f.char_end] == f.value


def test_offsets_stay_correct_past_a_page_break(cfg):
    """Fields on page 2 are the ones a naive implementation gets wrong."""
    pages = [PAGE_A, PAGE_B]
    full = build_canonical_text(pages)
    fields, _ = extract_fields(full, pages, cfg, doc_class="invoice")
    by_name = {f.field: f for f in fields}

    assert by_name["ifsc"].page == 1
    assert by_name["total_amount"].page == 1
    assert by_name["gstin"].page == 0
    for f in fields:
        assert full[f.char_start:f.char_end] == f.value


def test_page_attribution_matches_the_span_table(cfg):
    pages = [PAGE_A, PAGE_B, PAGE_C]
    full = build_canonical_text(pages)
    spans = page_spans(pages)
    fields, _ = extract_fields(full, pages, cfg, doc_class="invoice")
    for f in fields:
        s, e = spans[f.page]
        assert s <= f.char_start <= e, f"{f.field} attributed to the wrong page"


def test_leading_page_shifts_offsets_by_exactly_the_separator(cfg):
    """Prepending a page must move every offset by len(page) + len(PAGE_SEP)."""
    base_pages = [PAGE_A]
    base_full = build_canonical_text(base_pages)
    base, _ = extract_fields(base_full, base_pages, cfg, doc_class="invoice")

    prefix = "COVER SHEET"
    pages = [prefix, PAGE_A]
    full = build_canonical_text(pages)
    shifted, _ = extract_fields(full, pages, cfg, doc_class="invoice")

    delta = len(prefix) + len(PAGE_SEP)
    a = {f.field: f.char_start for f in base}
    b = {f.field: f.char_start for f in shifted}
    for name in set(a) & set(b):
        assert b[name] == a[name] + delta, name


def test_no_field_offset_runs_past_the_text(cfg):
    pages = [PAGE_A, PAGE_B]
    full = build_canonical_text(pages)
    fields, _ = extract_fields(full, pages, cfg, doc_class="invoice")
    for f in fields:
        assert 0 <= f.char_start <= f.char_end <= len(full)


def test_empty_document_yields_no_fields_and_no_errors(cfg):
    fields, errors = extract_fields("", [""], cfg, doc_class="invoice")
    assert fields == []
    assert errors == []


# ---------------------------------------------------------------- normalisation

def test_normalized_value_is_separate_from_the_raw_span(cfg):
    """Offsets point at raw text. Normalisation must never move them."""
    pages = [PAGE_B]
    full = build_canonical_text(pages)
    fields, _ = extract_fields(full, pages, cfg, doc_class="invoice")
    by_name = {f.field: f for f in fields}

    aadhaar = by_name["aadhaar"]
    assert aadhaar.value == "2698 3765 4322"        # raw, with spaces
    assert aadhaar.normalized_value == "269837654322"
    assert full[aadhaar.char_start:aadhaar.char_end] == aadhaar.value

    total = by_name["total_amount"]
    assert total.value == "11,800.00"
    assert total.normalized_value == "11800.00"
    assert total.currency == "INR"


def test_dates_normalise_to_iso_without_moving_offsets(cfg):
    pages = [PAGE_A]
    full = build_canonical_text(pages)
    fields, _ = extract_fields(full, pages, cfg, doc_class="invoice")
    d = {f.field: f for f in fields}["invoice_date"]
    assert d.value == "14/03/2024"
    assert d.normalized_value == "2024-03-14"
    assert full[d.char_start:d.char_end] == "14/03/2024"


def test_normalizer_version_is_recorded(cfg):
    pages = [PAGE_A]
    fields, _ = extract_fields(build_canonical_text(pages), pages, cfg, doc_class="invoice")
    assert all(f.normalizer_version == cfg.normalizer_version for f in fields)


# ---------------------------------------------------------------- strict mode

def test_strict_mode_raises_on_a_deliberately_broken_offset(cfg, monkeypatch):
    """Prove the guard fires rather than silently passing bad spans through."""
    import baseline.metadata.fields as md

    real = md._pattern_candidates

    def sabotage(full, fdef, limit):
        cands = real(full, fdef, limit)
        for c in cands:
            c.start += 1  # shift so the slice no longer matches
        return cands

    monkeypatch.setattr(md, "_pattern_candidates", sabotage)
    strict = dict(cfg.pipeline)
    strict["metadata"] = {**strict.get("metadata", {}), "strict_offsets": True}
    broken = type(cfg)(
        config_dir=cfg.config_dir, pipeline=strict,
        classes=cfg.classes, fields=cfg.fields, tags=cfg.tags,
    )
    pages = [PAGE_A]
    with pytest.raises(OffsetMismatch):
        extract_fields(build_canonical_text(pages), pages, broken, doc_class="invoice")
