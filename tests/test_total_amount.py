"""total_amount picks the grand total, not a subtotal, a tax line or a count."""

import pytest

from baseline.metadata import extract_fields


def _total(cfg, text):
    fields, errors = extract_fields(text, [text], cfg, doc_class="receipt")
    assert not errors
    got = [f for f in fields if f.field == "total_amount"]
    for f in got:
        assert text[f.char_start:f.char_end] == f.value
    return got[0].value if got else None


@pytest.mark.parametrize(
    "text, want",
    [
        ("SUB TOTAL (RM) : 411.50\nTOTAL GST(RM) : 24.69\nTOTAL(RM) : 436.20\nCASH : 436.20\n", "436.20"),
        ("TOTAL QTY: 3\nTOTAL : 12.90\n", "12.90"),
        ("SUBTOTAL: 25.94\nGST @6% 1.56\nAMOUNT DUE 30.25\n", "30.25"),
        ("TOTAL INCL. GST @6%: 6.36\n", "6.36"),
    ],
)
def test_grand_total_wins(cfg, text, want):
    assert _total(cfg, text) == want


def test_a_subtotal_alone_is_not_a_total(cfg):
    assert _total(cfg, "Subtotal $350.00\nTax Rate 8%\n") is None
