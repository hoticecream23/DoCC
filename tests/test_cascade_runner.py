import json

import pytest

from baseline.pipeline import process_document as baseline_process
from baseline.schema import ClassifyMethod, MetadataMethod

INVOICE_ANSWER = {
    "class": "invoice",
    "fields": {
        "invoice_number": ["INV-2024-0042"],
        "invoice_date": ["2024-03-14"],  # model reformatted it
        "total_amount": ["11800.00"],  # printed as 11,800.00
        "vendor_name": ["Acme Traders Private Limited"],  # printed twice
        "buyer_name": ["Hallucinated Holdings"],  # not on the document
        "gstin": ["27AAPFU0939F1ZV"],
        "pan": ["AAPFU0939F"],
    },
}
NOTHING = {"class": "unknown", "fields": {}}


class FakeBackend:
    """Answers from the page text it is sent. No GPU, no network."""

    def __init__(self, answer_for):
        self.answer_for = answer_for
        self.calls = 0

    def complete(self, messages, schema):
        from cascade.vlm import VLMReply

        self.calls += 1
        text = "".join(p.get("text", "") for p in messages[1]["content"])
        answer = self.answer_for(text)
        # the schema requires every declared field, so a real server always sends them all
        names = schema["properties"]["fields"]["required"]
        answer = {**answer, "fields": {n: answer["fields"].get(n, []) for n in names}}
        return VLMReply(content=json.dumps(answer), model="fake", finish_reason="stop")


class DownBackend:
    def complete(self, messages, schema):
        from cascade.vlm import VLMError

        raise VLMError("server down")


def _invoice_or_nothing(text):
    return INVOICE_ANSWER if "INV-2024-0042" in text else NOTHING


@pytest.fixture(scope="module")
def ccfg():
    from conftest import ROOT

    from cascade.config import load_cascade_config

    return load_cascade_config(ROOT / "config")


def test_invoice_end_to_end(corpus, cfg, ccfg):
    from cascade.runner import process_document

    r = process_document(corpus / "invoice_native.pdf", cfg, ccfg, FakeBackend(_invoice_or_nothing))
    rec, prov = r.record, r.provenance
    assert rec.verify_offsets() == []
    assert rec.classification.label == "invoice" and rec.classification.method is ClassifyMethod.RULE
    by = {m.field: m for m in rec.metadata}

    assert (by["invoice_number"].value, by["invoice_number"].method) == ("INV-2024-0042", MetadataMethod.VLM_RULE_AGREE)
    assert (by["invoice_date"].value, by["invoice_date"].normalized_value) == ("14/03/2024", "2024-03-14")
    assert (by["total_amount"].value, by["total_amount"].normalized_value) == ("11,800.00", "11800.00")

    base, _ = baseline_process(corpus / "invoice_native.pdf", cfg)
    base_vendor = next(m for m in base.metadata if m.field == "vendor_name")
    assert by["vendor_name"].char_start == base_vendor.char_start  # the rule's occurrence

    for proven in ("gstin", "pan", "ifsc"):  # ifsc kept although the model omitted it
        assert by[proven].method is MetadataMethod.REGEX_CHECKSUM and by[proven].validated
    assert "buyer_name" not in by
    assert "tax_amount" not in by  # rule only, the model did not confirm it
    for m in rec.metadata:
        assert m.validated == (m.method is MetadataMethod.REGEX_CHECKSUM)

    outcomes = {(p["field"], p["outcome"]) for p in prov["fields"]}
    assert ("buyer_name", "ungrounded") in outcomes
    assert ("tax_amount", "rule_only") in outcomes
    assert prov["fallback"] is False and prov["model"] == "fake"
    assert prov["class"]["source"] == "rule"
    assert rec.schema_version == "1.2.0"


def test_a_down_server_falls_back_to_the_baseline_answer(corpus, cfg, ccfg):
    from cascade.runner import process_document

    r = process_document(corpus / "invoice_native.pdf", cfg, ccfg, DownBackend())
    base, _ = baseline_process(corpus / "invoice_native.pdf", cfg)
    assert r.record.classification == base.classification
    assert r.record.metadata == base.metadata
    assert r.record.tags == base.tags
    assert any(e.startswith("cascade: VLMError: server down") for e in r.record.diagnostics.errors)
    assert r.provenance["fallback"] is True


def test_a_malformed_answer_falls_back_too(corpus, cfg, ccfg):
    from cascade.runner import process_document

    r = process_document(
        corpus / "invoice_native.pdf", cfg, ccfg, FakeBackend(lambda text: {"class": "not_a_class", "fields": {}})
    )
    assert r.provenance["fallback"] is True
    assert any("VLMOutputError" in e for e in r.record.diagnostics.errors)


def test_a_document_with_no_image_is_sent_as_text_only(corpus, cfg, ccfg):
    from cascade.runner import process_document

    backend = FakeBackend(lambda text: NOTHING)
    r = process_document(corpus / "contract.txt", cfg, ccfg, backend)
    assert backend.calls == 1
    assert r.provenance["fallback"] is False
    assert r.provenance["images_sent"] == 0 and r.provenance["pages_sent"] == 1


def test_batch_writes_one_record_and_one_provenance_line_per_document(corpus, cfg, ccfg, tmp_path):
    from cascade.runner import run_batch

    out = tmp_path / "cascade.jsonl"
    stats = run_batch(corpus, out, cfg, ccfg, FakeBackend(_invoice_or_nothing))
    assert stats["written"] + stats["failed"] == stats["found"]
    assert stats["failed"] == 0
    assert stats["fallback"] >= 1  # broken.pdf cannot be rendered
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    provs = [
        json.loads(line) for line in (tmp_path / "cascade.jsonl.provenance").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == stats["written"] == len(provs)
    assert [r["doc_id"] for r in records] == [p["doc_id"] for p in provs]

    from baseline.schema import DocumentRecord

    for r in records:
        assert DocumentRecord.model_validate(r).verify_offsets() == []

    again = run_batch(corpus, out, cfg, ccfg, FakeBackend(_invoice_or_nothing))
    assert again["written"] == 0 and again["skipped"] == stats["written"]


def test_a_tagging_failure_falls_back_instead_of_losing_the_document(corpus, cfg, ccfg, monkeypatch):
    from cascade import runner

    def boom(*a, **k):
        raise RuntimeError("tagger broke")

    monkeypatch.setattr(runner, "apply_tags", boom)
    r = runner.process_document(corpus / "invoice_native.pdf", cfg, ccfg, FakeBackend(_invoice_or_nothing))
    base, _ = baseline_process(corpus / "invoice_native.pdf", cfg)
    assert r.provenance["fallback"] is True
    assert r.record.classification == base.classification
    assert r.record.metadata == base.metadata
    assert r.record.tags == base.tags
    assert any("RuntimeError: tagger broke" in e for e in r.record.diagnostics.errors)


def test_the_fallback_also_drops_a_bad_offset(corpus, cfg, ccfg, monkeypatch):
    from cascade import runner

    real = baseline_process

    def shifted(path, cfg, models=None):
        rec, side = real(path, cfg, models)
        first = rec.metadata[0]
        bad = first.model_copy(update={"char_start": first.char_start + 1, "char_end": first.char_end + 1})
        return rec.model_copy(update={"metadata": [bad, *rec.metadata[1:]]}), side

    monkeypatch.setattr(runner.baseline_pipeline, "process_document", shifted)
    r = runner.process_document(corpus / "invoice_native.pdf", cfg, ccfg, DownBackend())
    bad_field = shifted(corpus / "invoice_native.pdf", cfg)[0].metadata[0]
    assert bad_field not in r.record.metadata
    assert r.record.verify_offsets() == []
    assert any("dropped" in e for e in r.record.diagnostics.errors)


def _stub_client(url):
    from cascade.config import VLMSettings
    from cascade.vlm import VLMClient

    return VLMClient(
        VLMSettings(base_url=url, model="m", timeout_s=2, retries=0, backoff_s=0, max_tokens=100), sleep=lambda s: None
    )


def test_a_400_is_retried_once_with_half_the_pages(corpus, cfg, ccfg):
    from vlm_stub import chat, serve

    from cascade.prompt import response_schema
    from cascade.runner import process_document

    names = response_schema(cfg)["properties"]["fields"]["required"]
    ok = chat(json.dumps({"class": "unknown", "fields": {n: [] for n in names}}))
    with serve([(400, '{"error": "context"}', 0, None), (200, ok, 0, None)]) as (url, reqs):
        r = process_document(corpus / "invoice_native.pdf", cfg, ccfg, _stub_client(url))
    assert len(reqs) == 2
    assert r.provenance["fallback"] is False
    assert r.provenance["reduced_pages"] == max(1, ccfg.render.max_pages // 2)


def test_a_second_400_falls_back(corpus, cfg, ccfg):
    from vlm_stub import serve

    from cascade.runner import process_document

    with serve([(400, '{"error": "context"}', 0, None)]) as (url, reqs):
        r = process_document(corpus / "invoice_native.pdf", cfg, ccfg, _stub_client(url))
    assert len(reqs) == 2
    assert r.provenance["fallback"] is True
    assert any(e.startswith("cascade: VLMBadRequest: HTTP 400") for e in r.record.diagnostics.errors)
