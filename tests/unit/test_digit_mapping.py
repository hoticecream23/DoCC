"""Unicode digits and direction marks: mapped for comparison only, and a no-op on ASCII text."""

import string
from pathlib import Path

import pytest

from baseline.config import load_config
from baseline.metadata import NormCtx, normalize
from baseline.metadata.normalizers import NORMALIZERS, ascii_digits
from cascade.ground import TextIndex, _fold, ground

CONFIG = Path(__file__).resolve().parents[2] / "config"
ASCII_SAMPLES = [string.printable, "INV-0042 / 1,234.50 Rs. 27 MAR 2018", "  Due Date:\t01/02/2024\n"]


@pytest.mark.parametrize("text", ASCII_SAMPLES)
def test_ascii_text_is_unchanged_by_digit_mapping(text):
    assert ascii_digits(text) is text


@pytest.mark.parametrize("name", sorted(NORMALIZERS))
def test_every_normaliser_gives_the_same_output_for_ascii_input(name):
    ctx = NormCtx(full=ASCII_SAMPLES[1], start=0, end=len(ASCII_SAMPLES[1]), opts={})
    assert normalize(name, ASCII_SAMPLES[1], ctx) == NORMALIZERS[name](ASCII_SAMPLES[1], ctx)


def _fold_before_digit_mapping(text):
    """The grounding fold as it was: casefold and collapse whitespace runs, with a source map."""
    out, src, in_space = [], [], False
    for i, ch in enumerate(text):
        if ch.isspace():
            if not in_space:
                out.append(" ")
                src.append(i)
            in_space = True
            continue
        in_space = False
        out += list(ch.casefold())
        src += [i] * len(ch.casefold())
    return "".join(out), src


@pytest.mark.parametrize("text", ASCII_SAMPLES)
def test_grounding_fold_of_ascii_text_is_unchanged(text):
    assert _fold(text) == _fold_before_digit_mapping(text)


def test_arabic_indic_and_eastern_arabic_digits_map_to_ascii():
    assert ascii_digits("\u0661\u0662\u06f3\u200f") == "123"


def test_a_model_value_in_ascii_grounds_on_arabic_indic_digits_in_the_text():
    cfg = load_config(CONFIG)
    page = "Voucher No: \u200f\u0661\u0662\u0663\u0664\u0665\n"
    fdef = next(f for f in cfg.field_defs if f["name"] == "invoice_number")
    g = ground("12345", TextIndex(page, [page]), fdef, [], {})
    assert (g.outcome, page[g.start : g.end]) == ("loose", "\u0661\u0662\u0663\u0664\u0665")
