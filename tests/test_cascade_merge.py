from baseline.classify import rule_scores
from baseline.metadata import extract_fields
from baseline.schema import ClassifyMethod, MetadataMethod

TEXT = (
    "TAX INVOICE\n"
    "Sold By: Acme Ltd\n"
    "PAN: AAPFU0939F\n"
    "Bad PAN: ABCDE1234F\n"
    "Email: billing@acme.example\n"
    "Invoice No: INV-1\n"
    "Ref INV-2\n"
    "Grand Total: Rs. 11,800.00\n"
)


def _tiers():
    from cascade.config import ConfidenceTiers

    return ConfidenceTiers(agree=0.9, grounded=0.75, ambiguous=0.5, vlm_class=0.8)


def _decide(cfg, fields, label="invoice", text=TEXT, rules=None, baseline_meta=None):
    from cascade.merge import decide
    from cascade.prompt import VLMAnswer

    pages = [text]
    base = extract_fields(text, pages, cfg, label)[0] if baseline_meta is None else baseline_meta
    rules = rule_scores(text, pages, cfg) if rules is None else rules
    return decide(text, pages, cfg, _tiers(), rules, VLMAnswer(label=label, fields=fields), base)


def _by_field(decision):
    out = {}
    for m in decision.metadata:
        out.setdefault(m.field, []).append(m)
    return out


def _prov(decision, field):
    return [p for p in decision.field_provenance if p["field"] == field]


def test_a_proven_id_survives_a_model_that_omits_it(cfg):
    d = _decide(cfg, {})
    pan = _by_field(d)["pan"]
    assert [(m.value, m.method, m.validated) for m in pan] == [
        ("AAPFU0939F", MetadataMethod.REGEX_CHECKSUM, True)
    ]


def test_a_model_id_proven_by_its_own_pattern_is_emitted_as_validated(cfg):
    d = _decide(cfg, {"pan": ["AAPFU0939F"]}, baseline_meta=[])  # nothing pre-proven
    (pan,) = _by_field(d)["pan"]
    assert (pan.value, pan.method, pan.validated, pan.confidence) == (
        "AAPFU0939F", MetadataMethod.REGEX_CHECKSUM, True, 1.0)
    assert [p["emitted"] for p in _prov(d, "pan")] == [True]


def test_a_model_id_that_fails_its_checksum_is_dropped(cfg):
    d = _decide(cfg, {"pan": ["ABCDE1234F"]})
    assert [m.value for m in _by_field(d)["pan"]] == ["AAPFU0939F"]
    assert [p["outcome"] for p in _prov(d, "pan")] == ["checksum_failed"]


def test_validated_is_never_true_without_a_checksum(cfg):
    d = _decide(cfg, {"email": ["billing@acme.example"], "invoice_number": ["INV-1"],
                      "total_amount": ["11800"]})
    for m in d.metadata:
        assert m.validated == (m.method is MetadataMethod.REGEX_CHECKSUM)
    assert _by_field(d)["email"][0].method in (MetadataMethod.VLM, MetadataMethod.VLM_RULE_AGREE)


def test_a_value_a_rule_also_found_is_marked_as_agreement(cfg):
    d = _decide(cfg, {"invoice_number": ["INV-1"]})
    (m,) = _by_field(d)["invoice_number"]
    assert (m.value, m.method, m.confidence) == ("INV-1", MetadataMethod.VLM_RULE_AGREE, 0.9)


def test_a_non_multi_field_prefers_the_value_a_rule_agrees_with(cfg):
    d = _decide(cfg, {"invoice_number": ["NOT-THERE-9", "INV-2", "INV-1"]})
    assert [m.value for m in _by_field(d)["invoice_number"]] == ["INV-1"]
    assert [(p["vlm_value"], p["rule_agrees"]) for p in _prov(d, "invoice_number")] == [("INV-1", True)]


def test_without_rule_agreement_the_strongest_grounding_wins(cfg):
    text = TEXT + "Ref ab-7\n"
    d = _decide(cfg, {"invoice_number": ["NOT-THERE-9", "AB-7", "INV-2"]}, text=text, baseline_meta=[],
                rules={})
    assert [m.value for m in _by_field(d)["invoice_number"]] == ["INV-2"]  # exact beats loose "ab-7"


def test_a_reformatted_amount_is_emitted_as_printed_and_normalised(cfg):
    d = _decide(cfg, {"total_amount": ["11800"]})
    (m,) = _by_field(d)["total_amount"]
    assert m.value == "11,800.00" and m.normalized_value == "11800.00"
    assert TEXT[m.char_start : m.char_end] == m.value


def test_a_rule_value_the_model_did_not_return_is_not_emitted_and_is_recorded(cfg):
    d = _decide(cfg, {})
    assert "invoice_number" not in _by_field(d)
    (p,) = _prov(d, "invoice_number")
    assert p["outcome"] == "rule_only" and p["baseline_values"] == ["INV-1"]


def test_an_invented_value_is_not_emitted(cfg):
    d = _decide(cfg, {"buyer_name": ["Hallucinated Holdings"]})
    assert "buyer_name" not in _by_field(d)
    assert [p["outcome"] for p in _prov(d, "buyer_name")] == ["ungrounded"]


def test_a_field_out_of_scope_for_the_decided_class_is_not_emitted(cfg):
    text = TEXT + "Dated: 14/03/2024\n"
    d = _decide(cfg, {"judgment_date": ["14/03/2024"]}, text=text)
    assert d.classification.label == "invoice"
    assert "judgment_date" not in _by_field(d)


def test_a_short_circuit_rule_decides_the_class_over_the_model(cfg):
    d = _decide(cfg, {}, label="receipt", rules={"invoice": 0.95})
    c = d.classification
    assert (c.label, c.method, c.confidence) == ("invoice", ClassifyMethod.RULE, 0.95)
    assert d.class_provenance == {"source": "rule", "label": "invoice",
                                  "vlm_label": "receipt", "disagrees": True}


def test_below_the_short_circuit_the_model_decides(cfg):
    d = _decide(cfg, {}, label="receipt", rules={"invoice": 0.6})
    c = d.classification
    assert (c.label, c.method, c.confidence) == ("receipt", ClassifyMethod.VLM, 0.8)
    assert c.all_scores == {"invoice": 0.6}


def test_a_model_abstention_is_unknown_with_zero_confidence(cfg):
    d = _decide(cfg, {}, label="unknown", rules={})
    c = d.classification
    assert (c.label, c.method, c.confidence) == ("unknown", ClassifyMethod.VLM, 0.0)


def test_metadata_is_sorted_and_every_offset_slices_back(cfg):
    d = _decide(cfg, {"email": ["billing@acme.example"], "invoice_number": ["INV-1"],
                      "total_amount": ["11800"], "vendor_name": ["Acme Ltd"]})
    keys = [(m.char_start, m.field) for m in d.metadata]
    assert keys == sorted(keys)
    for m in d.metadata:
        assert TEXT[m.char_start : m.char_end] == m.value


def test_a_model_id_that_passes_its_checksum_but_not_its_pattern_is_dropped(cfg):
    text = "TAX INVOICE\nAccount No: 4222222222222\n"
    d = _decide(cfg, {"card_number": ["4222222222222"]}, text=text)
    assert "card_number" not in _by_field(d)
    assert [p["outcome"] for p in _prov(d, "card_number")] == ["checksum_failed"]


def test_a_proven_id_is_kept_even_when_its_field_is_out_of_scope(cfg):
    import dataclasses

    # Scope the PAN field to court_document only
    scoped_fields = [
        dict(f, classes=["court_document"]) if f["name"] == "pan" else f
        for f in cfg.field_defs
    ]
    scoped_cfg = dataclasses.replace(cfg, fields={**cfg.fields, "fields": scoped_fields})

    pages = [TEXT]
    # Extract baseline with "unknown" label so PAN is found
    base, _ = extract_fields(TEXT, pages, cfg, "unknown")
    rules = rule_scores(TEXT, pages, scoped_cfg)

    from cascade.merge import decide
    from cascade.prompt import VLMAnswer

    # Decide with "invoice" label (so PAN is out of scope), no model answer
    d = decide(TEXT, pages, scoped_cfg, _tiers(), rules, VLMAnswer(label="invoice", fields={}), base)

    # PAN should still be in metadata as REGEX_CHECKSUM, validated
    pan = _by_field(d)["pan"]
    assert [(m.value, m.method, m.validated) for m in pan] == [
        ("AAPFU0939F", MetadataMethod.REGEX_CHECKSUM, True)
    ]


def test_an_out_of_scope_field_leaves_no_rule_only_record(cfg):
    import dataclasses

    # Scope the invoice_number field to invoice only
    scoped_fields = [
        dict(f, classes=["invoice"]) if f["name"] == "invoice_number" else f
        for f in cfg.field_defs
    ]
    scoped_cfg = dataclasses.replace(cfg, fields={**cfg.fields, "fields": scoped_fields})

    pages = [TEXT]
    # Extract baseline with "invoice" label so invoice_number is found
    base, _ = extract_fields(TEXT, pages, cfg, "invoice")
    # Decide with "court_document" label (so invoice_number is out of scope)
    rules = {"court_document": 0.95}

    from cascade.merge import decide
    from cascade.prompt import VLMAnswer

    d = decide(TEXT, pages, scoped_cfg, _tiers(), rules, VLMAnswer(label="court_document", fields={}), base)

    # invoice_number should not be in metadata (out of scope)
    assert "invoice_number" not in _by_field(d)
    # invoice_number should have no provenance record (out of scope produces no record)
    assert [p for p in d.field_provenance if p["field"] == "invoice_number"] == []


SHAPES = TEXT + "Account No: 1234-5678-901\nCust ID 40044448\nTel 9876543210\nOrder: ABC\n"


def test_a_model_only_account_number_must_be_9_to_18_digits(cfg):
    d = _decide(cfg, {"account_number": ["1234-5678-901"]}, text=SHAPES, baseline_meta=[], rules={})
    assert [m.value for m in _by_field(d)["account_number"]] == ["1234-5678-901"]
    d = _decide(cfg, {"account_number": ["40044448"]}, text=SHAPES, baseline_meta=[], rules={})
    assert "account_number" not in _by_field(d)
    assert [p["outcome"] for p in _prov(d, "account_number")] == ["shape_failed"]


def test_a_model_only_account_number_equal_to_another_id_is_rejected():
    from cascade.merge import _shape_ok

    assert not _shape_ok("account_number", "98765 43210", {"9876543210"})
    assert _shape_ok("account_number", "98765 43210", {"1111111111"})
    assert not _shape_ok("invoice_number", "A1" * 16, set())  # over 30 chars


def test_a_model_only_invoice_number_needs_a_digit(cfg):
    d = _decide(cfg, {"invoice_number": ["ABC"]}, text=SHAPES, baseline_meta=[], rules={})
    assert "invoice_number" not in _by_field(d)
    assert [p["outcome"] for p in _prov(d, "invoice_number")] == ["shape_failed"]


def test_an_empty_model_value_is_not_present(cfg):
    from cascade.prompt import parse_answer

    d = _decide(cfg, {"invoice_number": ["", "  "]})
    assert "invoice_number" not in _by_field(d)
    assert all(p["outcome"] in ("ungrounded", "rule_only") for p in _prov(d, "invoice_number"))
    import json

    fields = {f["name"]: [] for f in cfg.field_defs}
    fields["total_amount"] = [""]
    assert "total_amount" not in parse_answer(json.dumps({"class": "invoice", "fields": fields}), cfg).fields
