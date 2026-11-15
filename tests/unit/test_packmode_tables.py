"""Pack-mode header-row pairing, phrase qualifiers and the integer normaliser, from the confirmation review."""

from dataclasses import replace
from pathlib import Path

import pytest

from baseline import packmode
from baseline.clientpack import FieldSpec
from baseline.config import load_config
from baseline.metadata import NormCtx, normalize
from baseline.metadata.fields import extract_fields
from baseline.metadata.layouts import PHRASE_CONFIDENCE

CONFIG = Path(__file__).resolve().parents[2] / "config"
CLASS = "Voucher"
# The class's fields: name -> (dtype, role).
CLASS_FIELDS = {
    "Invoice No": ("text", "key"),
    "PV No": ("text", "optional"),
    "Date": ("date", "optional"),
    "Amount": ("number", "optional"),
    "Vendor Name": ("text", "optional"),
}


@pytest.fixture(scope="module")
def cfg():
    return load_config(CONFIG)


def _read(cfg, name, text, fields=None):
    """Raw values found for one field of the class over text, best first."""
    fields = fields or CLASS_FIELDS
    specs = [FieldSpec(name=n, dtype=d, role=r, origin="document") for n, (d, r) in fields.items()]
    by_name = {f["name"]: f for f in cfg.field_defs}
    fdef = next(d for d in packmode.class_field_defs(specs, CLASS, by_name) if d["name"] == name)
    one = replace(cfg, fields={**cfg.fields, "fields": [fdef], "templates": []})
    return [c.value for c in extract_fields(text, [text], one, CLASS)[0]]


def _integer(value):
    return normalize("integer", value, NormCtx(full=value, start=0, end=len(value), opts={}))[0]


def test_a_value_row_without_the_labels_column_gives_nothing(cfg):
    assert _read(cfg, "PV No", "Invoice No   Date   PV No\nINV-1   01-01-2026") == []


def test_a_value_row_shorter_than_the_header_still_fills_its_columns(cfg):
    assert _read(cfg, "Invoice No", "Invoice No   Date   PV No\nINV-1   01-01-2026") == ["INV-1"]


def test_an_empty_header_cell_keeps_its_position(cfg):
    assert _read(cfg, "PV No", "Invoice No |  | PV No\nINV-1 | x | PV-2") == ["PV-2"]


def test_a_trailing_pipe_does_not_shift_the_columns(cfg):
    assert _read(cfg, "PV No", "Invoice No | PV No |\nINV-1 | PV-2 |") == ["PV-2"]


def test_a_single_spaced_header_splits_into_known_labels(cfg):
    assert _read(cfg, "PV No", "Invoice No. PV No.\nINV-1 PV-2") == ["PV-2"]


def test_a_single_spaced_header_with_unknown_words_gives_nothing(cfg):
    assert _read(cfg, "PV No", "Invoice No. Remark PV No.\nINV-1 PV-2") == []


def test_a_single_spaced_value_row_with_another_count_gives_nothing(cfg):
    assert _read(cfg, "PV No", "Invoice No. PV No.\nINV-1 PV-2 extra") == []


def test_stacked_tables_pair_each_label_with_its_own_row(cfg):
    text = "Invoice No | Date\nINV-1 | 01-01-2026\nPV No | Amount\nPV-2 | 100.00"
    assert _read(cfg, "PV No", text) == ["PV-2"]


def test_a_multi_word_cell_is_read_whole_in_a_text_column(cfg):
    assert _read(cfg, "Vendor Name", "PV No | Vendor Name\nPV-2 | Acme Trading LLC") == ["Acme Trading LLC"]


def test_a_paired_cell_that_does_not_fit_gives_nothing_not_another_cell(cfg):
    assert _read(cfg, "PV No", "Invoice No | PV No\nINV-1 | pending") == []


def test_a_label_line_below_a_lone_label_gives_nothing(cfg):
    assert _read(cfg, "PV No", "PV No.\nDate: 01-02-2026") == []


def test_a_lone_label_still_reads_the_next_line(cfg):
    assert _read(cfg, "PV No", "PV No.\nPV-14380") == ["PV-14380"]


@pytest.mark.parametrize(
    ("name", "text"), [("PV No", "Old PV No: 5"), ("Invoice No", "Against Invoice No: 5"), ("PV No", "Orig. PV No: 5")]
)
def test_a_label_after_an_other_document_qualifier_is_refused(cfg, name, text):
    assert _read(cfg, name, text) == []


@pytest.mark.parametrize(("name", "text"), [("Invoice No", "Tax Invoice No: 5"), ("Order No", "Sales Order No: 5")])
def test_a_same_concept_phrase_is_kept(cfg, name, text):
    assert _read(cfg, name, text, {name: ("text", "key")}) == ["5"]


def test_a_complete_match_beats_a_qualified_phrase(cfg):
    assert _read(cfg, "PV No", "Supplier's PV No: 7\nPV No: 8") == ["8"]


def test_a_phrase_only_match_scores_review_confidence(cfg):
    specs = [FieldSpec(name="Invoice No", dtype="text", role="key", origin="document")]
    fdef = packmode.class_field_defs(specs, CLASS, {f["name"]: f for f in cfg.field_defs})[0]
    one = replace(cfg, fields={**cfg.fields, "fields": [fdef], "templates": []})
    text = "Tax Invoice No: 5"
    assert extract_fields(text, [text], one, CLASS)[0][0].confidence == PHRASE_CONFIDENCE


@pytest.mark.parametrize(
    ("raw", "want"),
    [
        ("1,200", "1200"),
        ("12,345,678", "12345678"),
        ("2026.0", "2026"),
        ("-3", "-3"),
        ("12,50", "12,50"),
        ("1,5", "1,5"),
        ("12.50", "12.50"),
        ("١٬٢٠٠", "1200"),
    ],
)
def test_the_integer_normaliser_reads_only_standard_groups(raw, want):
    assert _integer(raw) == want
