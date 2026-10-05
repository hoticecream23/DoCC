from baseline.metadata.fields import candidates_for, extract_fields, scan_spans
from baseline.schema import MetadataMethod


def _fdef(cfg, name):
    return next(f for f in cfg.field_defs if f["name"] == name)


TEXT = "Invoice No: INV-2024-0042\nRef INV-2024-0099\n"


def test_candidates_for_returns_every_strategy_not_just_the_first_hit(cfg):
    fdef = _fdef(cfg, "invoice_number")  # strategies: [anchor, pattern]
    cands = candidates_for(TEXT, [TEXT], fdef, cfg, "invoice")
    got = {(c.value, c.method) for c in cands}
    assert ("INV-2024-0042", MetadataMethod.ANCHOR) in got
    assert ("INV-2024-0099", MetadataMethod.REGEX_FORMAT) in got
    # extract_fields stops at the first strategy that hits, so it never sees 0099.
    fields, _ = extract_fields(TEXT, [TEXT], cfg, "invoice")
    assert [m.value for m in fields if m.field == "invoice_number"] == ["INV-2024-0042"]


def test_candidates_for_offsets_slice_back_to_their_values(cfg):
    fdef = _fdef(cfg, "invoice_number")
    for c in candidates_for(TEXT, [TEXT], fdef, cfg, "invoice"):
        assert TEXT[c.start : c.end] == c.value


def test_scan_spans_finds_values_with_no_anchor_label_near_them(cfg):
    text = "Tax 1800.00\nGrand Total: Rs. 11,800.00\n"
    spans = scan_spans(text, _fdef(cfg, "total_amount"))
    raws = [raw for _, _, raw in spans]
    assert "1800.00" in raws
    assert "11,800.00" in raws
    for s, e, raw in spans:
        assert text[s:e] == raw


def test_scan_spans_is_empty_for_a_field_with_no_value_patterns():
    assert scan_spans("anything at all", {"name": "x", "strategies": []}) == []
