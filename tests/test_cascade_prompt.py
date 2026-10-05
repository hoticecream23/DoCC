import base64
import dataclasses
import json

import pytest


def _names(cfg):
    return [f["name"] for f in cfg.field_defs]


def test_every_class_and_field_in_the_repo_config_has_a_description(cfg):
    from cascade.prompt import check_descriptions

    check_descriptions(cfg)


def test_a_missing_description_is_named(cfg):
    from cascade.prompt import check_descriptions

    bad = dataclasses.replace(cfg, fields={"fields": [{"name": "orphan_field"}]})
    with pytest.raises(ValueError, match="orphan_field"):
        check_descriptions(bad)


def test_schema_enumerates_classes_and_requires_every_field(cfg):
    from cascade.prompt import response_schema

    s = response_schema(cfg)
    assert s["additionalProperties"] is False
    assert set(s["properties"]["class"]["enum"]) == set(cfg.class_names) | {"unknown"}
    fields = s["properties"]["fields"]
    assert fields["additionalProperties"] is False
    assert sorted(fields["required"]) == sorted(_names(cfg))
    assert fields["properties"]["total_amount"] == {"type": "array", "items": {"type": "string"}}


def test_messages_carry_descriptions_text_and_images(cfg):
    from cascade.prompt import build_messages

    png = b"\x89PNG fake"
    msgs = build_messages(cfg, ["page one text", "page two text"], [png, png], max_text_chars=1000)
    system, user = msgs
    assert system["role"] == "system" and user["role"] == "user"
    for name in cfg.class_names + _names(cfg):
        assert name in system["content"]
    texts = [p["text"] for p in user["content"] if p["type"] == "text"]
    images = [p["image_url"]["url"] for p in user["content"] if p["type"] == "image_url"]
    assert "page one text" in texts[0] and "page two text" in texts[0]
    assert images == ["data:image/png;base64," + base64.b64encode(png).decode("ascii")] * 2


def test_page_text_is_capped_in_total(cfg):
    from cascade.prompt import build_messages

    # Q and Z appear nowhere in the header text, so the counts are the pages'.
    msgs = build_messages(cfg, ["Q" * 80, "Z" * 80], [], max_text_chars=100)
    text = msgs[1]["content"][0]["text"]
    assert text.count("Q") == 80
    assert text.count("Z") == 20


def test_no_text_part_when_the_budget_is_zero(cfg):
    from cascade.prompt import build_messages

    msgs = build_messages(cfg, ["some text"], [b"x"], max_text_chars=0)
    assert [p["type"] for p in msgs[1]["content"]] == ["image_url"]


def test_parse_answer_strips_drops_empties_and_deduplicates(cfg):
    from cascade.prompt import parse_answer

    content = json.dumps({"class": "invoice", "fields": {
        "invoice_number": [" INV-1 ", "INV-1", ""], "total_amount": [], "pan": ["  "],
    }})
    a = parse_answer(content, cfg)
    assert a.label == "invoice"
    assert a.fields == {"invoice_number": ["INV-1"]}


@pytest.mark.parametrize("content, message", [
    ("not json", "not JSON"),
    ("[]", "not an object"),
    (json.dumps({"class": "invoice", "fields": {}, "extra": 1}), "unexpected keys"),
    (json.dumps({"class": "made_up", "fields": {}}), "not a declared class"),
    (json.dumps({"class": ["invoice"], "fields": {}}), "not a declared class"),
    (json.dumps({"fields": {}}), "not a declared class"),
    (json.dumps({"class": "invoice", "fields": {"made_up": ["x"]}}), "undeclared fields"),
    (json.dumps({"class": "invoice", "fields": {"pan": "ABCDE1234F"}}), "list of strings"),
    (json.dumps({"class": "invoice", "fields": {"pan": [1]}}), "list of strings"),
    (json.dumps({"class": "invoice", "fields": []}), "fields is not an object"),
])
def test_parse_answer_rejects_anything_off_schema(cfg, content, message):
    from cascade.prompt import VLMOutputError, parse_answer

    with pytest.raises(VLMOutputError, match=message):
        parse_answer(content, cfg)
