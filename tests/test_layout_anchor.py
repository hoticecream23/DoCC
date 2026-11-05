"""The layout fallback pairs a label with the value printed by it on the page."""

from baseline.metadata.fields import _anchor_candidates
from baseline.schema import WordBox


def _doc(rows):
    """rows: [(text, x0, y0)] in reading order. Words are 10 high, 8 per char."""
    full, words = "", []
    for text, x0, y0 in rows:
        if full:
            full += "\n"
        words.append(WordBox(page=0, text=text, x0=x0, y0=y0, x1=x0 + 8 * len(text),
                             y1=y0 + 10, char_start=len(full), char_end=len(full) + len(text)))
        full += text
    return full, words


def _fdef(how, value_regex=r"([A-Z0-9]*[0-9][A-Z0-9]*)"):
    # The text rule reads only the same line, so the scrambled order defeats it.
    return {"anchor": {"labels": [{"text": "invoice no", "specificity": 1.0}],
                       "value_regex": "^[ :]*" + value_regex, "window": 8,
                       "direction": "after", "layout_direction": how,
                       "reject_value": "CS999"}}


def test_right_of_label():
    # OCR order puts the value after an unrelated line, on the label's row.
    full, words = _doc([("INVOICE NO", 10, 100), ("CASHIER", 10, 120), ("CS123", 200, 101)])
    [c] = _anchor_candidates(full, _fdef("right_of_label"), 120, 25, words)
    assert c.value == "CS123" and full[c.start:c.end] == c.value


def test_below_label():
    full, words = _doc([("INVOICE NO", 10, 100), ("DATE", 200, 100),
                        ("01/02", 200, 115), ("CS999", 20, 115), ("CS42", 40, 115)])
    [c] = _anchor_candidates(full, _fdef("below_label"), 120, 25, words)
    assert c.value == "CS42" and full[c.start:c.end] == c.value  # CS999 rejected


def test_empty_sidecar_or_text_hit_unchanged():
    full, words = _doc([("INVOICE NO", 10, 100), ("CASHIER", 10, 120), ("CS123", 200, 101)])
    assert _anchor_candidates(full, _fdef("right_of_label"), 120, 25, []) == []
    full2 = "INVOICE NO: A77\nCS123"
    [c] = _anchor_candidates(full2, _fdef("right_of_label"), 120, 25, words)
    assert c.value == "A77"  # text rule wins, boxes never consulted
