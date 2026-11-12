"""The keyword prior and the pack-mode config, without running the pipeline."""

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


def test_a_mapped_field_reuses_the_existing_extractor(pack_cfg):
    assert _fdef(pack_cfg, "Invoice No")["normalizer"] == "ref_upper"


def test_an_unmapped_field_is_anchored_on_its_own_label(pack_cfg):
    assert _fdef(pack_cfg, "Receipt No")["anchor"]["labels"] == [{"text": "receipt no", "specificity": 1.0}]


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
