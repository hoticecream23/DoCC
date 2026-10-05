from baseline.metadata import Candidate
from baseline.schema import PAGE_SEP, MetadataMethod


def _fdef(cfg, name):
    return next(f for f in cfg.field_defs if f["name"] == name)


def _opts(cfg):
    return cfg.fields.get("normalizer_options", {}) or {}


def _ground(cfg, field, value, pages, rule_cands=()):
    from cascade.ground import TextIndex, ground

    index = TextIndex(PAGE_SEP.join(pages), pages)
    return index.full, ground(value, index, _fdef(cfg, field), list(rule_cands), _opts(cfg))


def _cand(full, value, start):
    assert full[start : start + len(value)] == value
    return Candidate(value=value, start=start, end=start + len(value),
                     confidence=0.9, method=MetadataMethod.ANCHOR, validated=False)


def test_exact_unique_match(cfg):
    full, g = _ground(cfg, "invoice_number", "INV-2024-0042", ["Invoice No: INV-2024-0042\n"])
    assert (g.outcome, g.n_matches, g.rule_agrees) == ("exact", 1, False)
    assert full[g.start : g.end] == "INV-2024-0042"


def test_exact_ambiguous_without_a_rule_takes_the_first(cfg):
    text = "Acme Ltd\nSold By: Acme Ltd\n"
    full, g = _ground(cfg, "vendor_name", "Acme Ltd", [text])
    assert (g.outcome, g.n_matches, g.rule_agrees, g.start) == ("exact", 2, False, 0)


def test_exact_ambiguous_prefers_the_span_a_rule_also_produced(cfg):
    text = "Acme Ltd\nSold By: Acme Ltd\n"
    second = text.index("Acme Ltd", 1)
    full, g = _ground(cfg, "vendor_name", "Acme Ltd", [text], [_cand(text, "Acme Ltd", second)])
    assert (g.outcome, g.n_matches, g.rule_agrees, g.start) == ("exact", 2, True, second)


def test_loose_match_spans_the_documents_own_whitespace(cfg):
    full, g = _ground(cfg, "vendor_name", "Acme Traders Ltd", ["Sold By: Acme   Traders\nLtd\n"])
    assert g.outcome == "loose"
    assert full[g.start : g.end] == "Acme   Traders\nLtd"


def test_loose_match_ignores_case(cfg):
    full, g = _ground(cfg, "buyer_name", "Globex Corporation", ["Bill To: GLOBEX CORPORATION\n"])
    assert g.outcome == "loose"
    assert full[g.start : g.end] == "GLOBEX CORPORATION"


def test_loose_match_survives_a_casefold_that_changes_length(cfg):
    full, g = _ground(cfg, "vendor_name", "Hauptstraße GmbH", ["From: HAUPTSTRASSE GMBH\n"])
    assert g.outcome == "loose"
    assert full[g.start : g.end] == "HAUPTSTRASSE GMBH"


def test_normalised_date_grounds_a_reformatted_value(cfg):
    full, g = _ground(cfg, "invoice_date", "2024-03-14", ["Invoice Date: 14/03/2024\n"])
    assert g.outcome == "normalised"
    assert full[g.start : g.end] == "14/03/2024"


def test_normalised_amount_grounds_without_the_thousands_separator(cfg):
    full, g = _ground(cfg, "total_amount", "11800", ["Grand Total: Rs. 11,800.00\n"])
    assert g.outcome == "normalised"
    assert full[g.start : g.end] == "11,800.00"


def test_a_value_inside_a_longer_number_is_not_grounded_there(cfg):
    # "11,800" is printed only as part of "11,800.00". It must not be located
    # as a shorter span inside it, which would cost span F1 for no reason.
    full, g = _ground(cfg, "total_amount", "11,800", ["Grand Total: 11,800.00\n"])
    assert g.outcome == "normalised"
    assert full[g.start : g.end] == "11,800.00"


def test_a_value_inside_a_longer_token_is_ungrounded(cfg):
    _, g = _ground(cfg, "invoice_number", "100", ["Ref 10042\n"])
    assert g.outcome == "ungrounded"
    assert not g.grounded


def test_an_invented_value_is_ungrounded(cfg):
    _, g = _ground(cfg, "buyer_name", "Hallucinated Holdings", ["Bill To: Globex Corporation\n"])
    assert (g.outcome, g.start, g.end) == ("ungrounded", None, None)


def test_a_match_never_crosses_a_page_boundary(cfg):
    _, g = _ground(cfg, "vendor_name", "Acme Traders", ["Sold By: Acme", "Traders and more"])
    assert g.outcome == "ungrounded"


def test_an_empty_value_is_ungrounded(cfg):
    _, g = _ground(cfg, "vendor_name", "   ", ["Acme Ltd\n"])
    assert g.outcome == "ungrounded"


def test_every_grounded_span_slices_back_to_the_document_text(cfg):
    cases = [
        ("vendor_name", "Acme Traders Ltd", ["Sold By: Acme   Traders\nLtd\n"]),
        ("invoice_date", "2024-03-14", ["Invoice Date: 14/03/2024\n"]),
        ("total_amount", "11800", ["Grand Total: Rs. 11,800.00\n"]),
    ]
    for field, value, pages in cases:
        full, g = _ground(cfg, field, value, pages)
        assert g.grounded
        assert 0 <= g.start < g.end <= len(full)
        assert PAGE_SEP not in full[g.start : g.end]
