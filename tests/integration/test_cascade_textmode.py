"""Text mode (vlm.input: text): what page text is sent, what is recorded, and what may ground."""

import json
from pathlib import Path

import pytest

from cascade.prompt import plan_text

FILLER = "Terms and conditions apply to this document in full. " * 6  # past the native-text threshold
PAGE_COUNT = 5


@pytest.fixture(scope="module")
def ccfg():
    from cascade.config import load_cascade_config

    base = load_cascade_config(Path(__file__).resolve().parents[2] / "config")
    vlm = base.vlm.model_copy(update={"input": "text", "text_page_chars": 200, "text_total_chars": 450})
    return base.model_copy(update={"vlm": vlm})


@pytest.fixture(scope="module")
def five_pages(tmp_path_factory):
    import fitz

    path = tmp_path_factory.mktemp("text") / "five.pdf"
    with fitz.open() as doc:
        for i in range(1, PAGE_COUNT + 1):
            page = doc.new_page()
            page.insert_textbox(fitz.Rect(40, 40, 560, 800), f"Invoice No: INV-{i}{i}{i}{i}\n{FILLER}", fontsize=9)
        doc.save(path)
    return path


class Backend:
    """Answers with fixed field values and keeps every request's page text."""

    def __init__(self, fields):
        self.fields, self.texts = fields, []

    def complete(self, messages, schema):
        from cascade.vlm import VLMReply

        self.texts.append("".join(p.get("text", "") for p in messages[1]["content"]))
        names = schema["properties"]["fields"]["required"]
        answer = {"class": "invoice", "fields": {n: self.fields.get(n, []) for n in names}}
        return VLMReply(content=json.dumps(answer), model="fake", finish_reason="stop")


def test_every_page_gets_its_head_when_one_page_is_long():
    plan = plan_text(["A" * 14000, "B" * 100, "C" * 100], 4000, 12000)
    assert (plan.included, plan.kept, plan.truncated) == ([0, 1, 2], [4000, 100, 100], True)


def test_the_middle_is_dropped_first_when_the_total_runs_out():
    plan = plan_text(["x" * 5000] * 5, 4000, 10000)
    assert plan.included == [0, 1, 4] and plan.chars == 10000


def test_nothing_is_cut_when_the_text_fits():
    plan = plan_text(["ab", "cd"], 10, 10)
    assert (plan.truncated, plan.chars) == (False, 4)


def test_provenance_records_the_pages_and_characters_sent(five_pages, cfg, ccfg):
    from cascade.runner import process_document

    prov = process_document(five_pages, cfg, ccfg, Backend({})).provenance
    assert (prov["pages_included"], prov["pages_sent"], prov["text_truncated"]) == ([1, 2, 5], 3, True)
    assert prov["text_chars_sent"] == 450


def test_omitted_pages_are_noted_in_diagnostics(five_pages, cfg, ccfg):
    from cascade.runner import process_document

    errors = process_document(five_pages, cfg, ccfg, Backend({})).record.diagnostics.errors
    assert any("text mode sent 3 of 5 pages" in e for e in errors)


def test_a_value_only_on_an_unsent_page_is_not_accepted(five_pages, cfg, ccfg):
    from cascade.runner import process_document

    r = process_document(five_pages, cfg, ccfg, Backend({"invoice_number": ["INV-3333"]}))
    assert "INV-3333" not in {m.value for m in r.record.metadata}


def test_a_value_on_a_sent_page_is_accepted(five_pages, cfg, ccfg):
    from cascade.runner import process_document

    r = process_document(five_pages, cfg, ccfg, Backend({"invoice_number": ["INV-5555"]}))
    assert "INV-5555" in {m.value for m in r.record.metadata if m.field == "invoice_number"}


def test_a_400_in_text_mode_retries_with_half_the_text(five_pages, cfg, ccfg):
    from vlm_stub import chat, serve

    from cascade.config import VLMSettings
    from cascade.prompt import response_schema
    from cascade.runner import process_document
    from cascade.vlm import VLMClient

    names = response_schema(cfg)["properties"]["fields"]["required"]
    ok = chat(json.dumps({"class": "unknown", "fields": {n: [] for n in names}}))
    with serve([(400, '{"error": "context"}', 0, None), (200, ok, 0, None)]) as (url, reqs):
        settings = VLMSettings(base_url=url, model="m", timeout_s=2, retries=0, backoff_s=0, max_tokens=100)
        r = process_document(five_pages, cfg, ccfg, VLMClient(settings, sleep=lambda s: None))
    assert len(reqs) == 2 and r.provenance["reduced_text_chars"] == 225
    assert r.provenance["text_chars_sent"] == 225
