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


def _decide(cfg, fields, label="invoice", text=TEXT, rules=None):
    from cascade.merge import decide
    from cascade.prompt import VLMAnswer

    pages = [text]
    base, _ = extract_fields(text, pages, cfg, label)
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


def test_a_non_multi_field_keeps_only_the_first_grounded_value(cfg):
    d = _decide(cfg, {"invoice_number": ["NOT-THERE-9", "INV-2", "INV-1"]})
    assert [m.value for m in _by_field(d)["invoice_number"]] == ["INV-2"]
    assert [p["outcome"] for p in _prov(d, "invoice_number")] == ["ungrounded", "exact"]


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
