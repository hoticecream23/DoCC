"""The keyword prior and the pack-mode config, without running the pipeline."""

from dataclasses import replace
from pathlib import Path

import make_pack
import pytest

from baseline import packmode
from baseline.clientpack import ClientPackError
from baseline.config import load_config

CONFIG = Path(__file__).resolve().parents[2] / "config"


@pytest.fixture(scope="module")
def base_cfg():
    return load_config(CONFIG)


@pytest.fixture(scope="module")
def pack_cfg(base_cfg, tmp_path_factory):
    return packmode.pack_config(base_cfg, make_pack.build(tmp_path_factory.mktemp("pack")), make_pack.CABINET)


def _fdef(cfg, name):
    return next(f for f in cfg.field_defs if f["name"] == name)


def test_a_class_alias_in_the_header_wins(pack_cfg):
    text = "PURCHASE ORDER\nPO No: PO-1"
    assert packmode.classify(text, [text], pack_cfg).label == "Buying Order"


def test_the_prior_abstains_to_other_on_a_tie(pack_cfg):
    text = "Tax Invoice and Payment Receipt"
    assert packmode.classify(text, [text], pack_cfg).label == "other"


def test_the_prior_abstains_to_other_with_no_evidence(pack_cfg):
    assert packmode.classify("nothing here", ["nothing here"], pack_cfg).label == "other"


def test_evidence_names_the_term_and_where_it_was_found(pack_cfg):
    text = "TAX INVOICE"
    assert "Tax Invoice @ header" in packmode.evidence(text, [text], pack_cfg)["Supplier Bill"]


def test_a_key_field_is_an_identifier_with_the_text_normaliser(pack_cfg):
    assert _fdef(pack_cfg, "Invoice No")["normalizer"] == "text_norm"


def test_a_field_is_anchored_on_its_own_label_and_its_spellings_only(pack_cfg):
    labels = {item["text"] for item in _fdef(pack_cfg, "Receipt No")["anchor"]["labels"]}
    assert labels == {"receipt no", "receipt no.", "receipt number", "receipt num", "receipt #", "receipt#"}


def test_the_key_field_comes_first(pack_cfg):
    names = [f["name"] for f in pack_cfg.field_defs if f["classes"] == ["Supplier Bill"]]
    assert names == ["Invoice No", "Invoice Date"]


def test_a_selection_field_has_no_rule_extractor(base_cfg):
    spec = packmode.FieldSpec(name="Branch", dtype="selection", role="optional", origin="document")
    assert packmode._field_def(spec, "Supplier Bill", {})["strategies"] == []


def test_the_default_config_is_not_pack_mode(base_cfg):
    assert packmode.resolve(base_cfg) is base_cfg


def test_a_pack_without_a_cabinet_is_refused(base_cfg, tmp_path):
    with pytest.raises(ClientPackError):
        packmode.resolve(base_cfg, (str(tmp_path), ""))


def test_resolve_leaves_a_pack_config_alone(pack_cfg):
    assert packmode.resolve(pack_cfg, ("elsewhere", "x")) is pack_cfg


# ------------------------------------------------- review regressions: a pack of made-up names

_F = make_pack._field
ARABIC_INVOICE = "\u0641\u0627\u062a\u0648\u0631\u0629"
REVIEW_CLASSES = [
    {"name": "Purchase Order", "aliases": [], "has_samples": True},
    {"name": "Payment Voucher", "aliases": [], "has_samples": True},
    {"name": "Payment Voucher Transfer", "aliases": [], "has_samples": True},
    {"name": "Delivery Note", "aliases": [], "has_samples": True},
    {"name": ARABIC_INVOICE, "aliases": [], "has_samples": True},
    {"name": "Other", "aliases": [], "has_samples": False},
]
REVIEW_FIELDS = {
    "Purchase Order": [
        _F("PO No", "text", "key"),
        _F("Invoice No", "text", "optional"),
        _F("Date", "date", "optional"),
        _F("Cheque No", "number", "optional"),
        _F("Total", "number", "optional"),
    ]
}


@pytest.fixture(scope="module")
def review_cfg(base_cfg, tmp_path_factory):
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(make_pack, "CLASSES", REVIEW_CLASSES)
        mp.setattr(make_pack, "FIELDS", REVIEW_FIELDS)
        mp.setattr(make_pack, "MANIFEST", [])
        pack = make_pack.build(tmp_path_factory.mktemp("review"))
    return packmode.pack_config(base_cfg, pack, make_pack.CABINET)


def _values(cfg, text):
    from baseline.metadata.fields import extract_fields

    return {m.field: (m.value, m.normalized_value) for m in extract_fields(text, [text], cfg, "Purchase Order")[0]}


def test_a_key_is_read_after_its_own_label_not_another_documents_number(review_cfg):
    assert _values(review_cfg, "Against Invoice 333444\nPO No: 111222")["PO No"][0] == "111222"


def test_a_field_whose_label_is_absent_yields_nothing(review_cfg):
    assert "Invoice No" not in _values(review_cfg, "PO NO: 4455\nDoc No: 9001")


def test_a_bare_date_skips_due_order_and_posting_dates(review_cfg):
    text = "Due Date: 01/02/2024\nOrder Date: 02/02/2024\nPosting Date: 03/02/2024\nDate: 05/02/2024"
    assert _values(review_cfg, text)["Date"][0] == "05/02/2024"


def test_an_identifier_keeps_its_leading_zeros(review_cfg):
    assert _values(review_cfg, "Cheque No: 00123456")["Cheque No"] == ("00123456", "00123456")


def test_an_identifier_keeps_its_separators(review_cfg):
    assert _values(review_cfg, "Cheque No: 1,234")["Cheque No"][1] == "1,234"


def test_a_non_identifier_number_is_an_amount(review_cfg):
    assert _values(review_cfg, "Total: 1,234.50")["Total"][1] == "1234.50"


def test_a_longer_class_name_wins_over_the_name_it_contains(review_cfg):
    text = "Payment Voucher Transfer"
    assert packmode.classify(text, [text], review_cfg).label == "Payment Voucher Transfer"


def test_a_first_page_hit_beats_an_anywhere_hit_despite_float_error(review_cfg):
    pages = ["x " * 400 + "Purchase Order", "Delivery Note"]  # past the header: 0.7 against 0.6
    assert packmode.classify("\n\f\n".join(pages), pages, review_cfg).label == "Purchase Order"


def test_a_class_name_with_no_ascii_letters_matches(review_cfg):
    text = f"{ARABIC_INVOICE} \u0631\u0642\u0645 5"
    assert packmode.classify(text, [text], review_cfg).label == ARABIC_INVOICE


def test_a_class_named_other_in_any_case_is_not_a_class(review_cfg):
    assert "Other" not in review_cfg.class_names


def test_pack_mode_reads_every_frame_and_caps_pages_by_default(review_cfg):
    opts = review_cfg.extract_opts()
    assert (opts["multi_frame_images"], opts["max_pages"]) == (True, packmode.PACK_MAX_PAGES)


def test_an_explicit_page_limit_survives_pack_mode(base_cfg, tmp_path):
    cfg = replace(base_cfg, pipeline={**base_cfg.pipeline, "extraction": {"max_pages": None}})
    pcfg = packmode.pack_config(cfg, make_pack.build(tmp_path), make_pack.CABINET)
    assert pcfg.extract_opts()["max_pages"] is None


def test_a_configured_relative_pack_dir_resolves_against_the_config_dir(base_cfg, tmp_path):
    make_pack.build(tmp_path / "packs")
    spec = {"dir": "packs", "cabinet": make_pack.CABINET}
    cfg = replace(base_cfg, config_dir=tmp_path, pipeline={**base_cfg.pipeline, "client_pack": spec})
    assert packmode.is_active(packmode.resolve(cfg))


def test_a_cabinet_alone_picks_the_cabinet_of_the_configured_pack(base_cfg, tmp_path):
    spec = {"dir": str(make_pack.build(tmp_path / "packs")), "cabinet": "no such cabinet"}
    cfg = replace(base_cfg, pipeline={**base_cfg.pipeline, "client_pack": spec})
    assert packmode.resolve(cfg, (None, make_pack.CABINET)).cabinet_id == make_pack.SLUG


def test_a_cabinet_with_no_pack_anywhere_leaves_normal_mode(base_cfg):
    assert packmode.resolve(base_cfg, (None, "c1")) is base_cfg


def test_pack_mode_resumes_by_file_stem(pack_cfg, tmp_path):
    assert packmode.resume_key(tmp_path / "a000000000000001.pdf", pack_cfg) == "a000000000000001"


def test_an_output_of_another_cabinet_is_refused_on_resume(pack_cfg, tmp_path):
    out = tmp_path / "out.jsonl"
    out.write_text('{"doc_id": "x", "cabinet_id": "elsewhere", "filename": "a.pdf"}\n', encoding="utf-8")
    with pytest.raises(ClientPackError, match="elsewhere"):
        packmode.done_keys(out, pack_cfg)


def test_done_keys_are_file_stems_in_pack_mode(pack_cfg, tmp_path):
    out = tmp_path / "out.jsonl"
    line = f'{{"doc_id": "x", "cabinet_id": "{make_pack.SLUG}", "filename": "a01.pdf"}}\n'
    out.write_text(line + "not json\n", encoding="utf-8")
    assert packmode.done_keys(out, pack_cfg) == {"a01"}


def test_done_keys_are_doc_ids_in_normal_mode(base_cfg, tmp_path):
    out = tmp_path / "out.jsonl"
    out.write_text('{"doc_id": "x", "filename": "a.pdf"}\n', encoding="utf-8")
    assert packmode.done_keys(out, base_cfg) == {"x"}


def test_done_keys_of_a_missing_file_are_empty(base_cfg, tmp_path):
    assert packmode.done_keys(tmp_path / "none.jsonl", base_cfg) == set()
