"""Normalizers. Each returns (normalized_value, extras).

Raw matched text is what the document says; the normalized value is what gets
compared. Both the pipeline and the workbook importer go through these, so a
value a reviewer types and a value the pipeline emits land in the same form.
"""

from __future__ import annotations

import datetime
import re
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from ..errors import summarize_exception
from ..logging_setup import get_logger

log = get_logger(__name__)

# Two digit years below this are read as 20xx, the rest as 19xx.
CENTURY_PIVOT = 70
# A year below this is a two digit year.
TWO_DIGIT_YEAR_LIMIT = 100
MONTHS_PER_YEAR = 12

# (normalized value or None, extras such as the currency code).
NormResult = tuple[str | None, dict[str, Any]]

_MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


def _full_year(y: int) -> int:
    if y < TWO_DIGIT_YEAR_LIMIT:
        return y + (2000 if y < CENTURY_PIVOT else 1900)
    return y


def _parse_date(value: str, dayfirst: bool = True) -> datetime.date | None:
    """Deterministic date parsing. No fuzzy guessing, no locale surprises."""
    v = value.strip().replace(",", " ")
    v = re.sub(r"\s+", " ", v)

    # Year first is unambiguous: 2018-03-27, 2018/03/27, 2018.03.27.
    m = re.fullmatch(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", v)
    if m:
        y, mo, d = (int(x) for x in m.groups())
        return _safe_date(y, mo, d)

    m = re.fullmatch(r"(\d{1,2})[/.-](\d{1,2})[/.-](\d{2}|\d{4})", v)
    if m:
        a, b, y = (int(x) for x in m.groups())
        d, mo = (a, b) if dayfirst else (b, a)
        # Fall back to the other order when the first reading is impossible.
        if mo > MONTHS_PER_YEAR and d <= MONTHS_PER_YEAR:
            d, mo = mo, d
        return _safe_date(_full_year(y), mo, d)

    # A month name in the middle: 27 MAR 2018, 27/MAR/2018, 28-FEB-18.
    m = re.fullmatch(r"(\d{1,2})[ /.-]?([A-Za-z]{3,9})\.?[ /.-]?(\d{2}|\d{4})", v)
    if m:
        d, name, y = int(m.group(1)), m.group(2).lower()[:3], int(m.group(3))
        if name in _MONTHS:
            return _safe_date(_full_year(y), _MONTHS[name], d)

    m = re.fullmatch(r"([A-Za-z]{3,9})\.? (\d{1,2}) (\d{4})", v)
    if m:
        name, d, y = m.group(1).lower()[:3], int(m.group(2)), int(m.group(3))
        if name in _MONTHS:
            return _safe_date(y, _MONTHS[name], d)
    return None


def _safe_date(y: int, mo: int, d: int) -> datetime.date | None:
    try:
        return datetime.date(y, mo, d)
    except ValueError:
        return None


def _to_decimal(value: str) -> Decimal | None:
    # The first number in the value, so the dot in "Rs." and the "/-" in
    # "11,800/-" are not read as part of it. A space before three digits is
    # a thousands separator, as OCR often prints "11 800.00".
    m = re.search(r"-?\d[\d,]*(?: \d{3}(?!\d))*(?:\.\d+)?", value)
    if not m:
        return None
    try:
        return Decimal(re.sub(r"[, ]", "", m.group()))
    except InvalidOperation:
        return None


@dataclass
class NormCtx:
    """Context a normalizer may read beyond the value itself.

    Attributes:
        full: Canonical document text.
        start: Start offset of the value in ``full``.
        end: End offset of the value in ``full``.
        opts: The ``normalizer_options`` block from fields.yaml.
    """

    full: str
    start: int
    end: int
    opts: dict[str, Any]


def _n_id_upper(value: str, ctx: NormCtx) -> NormResult:
    return re.sub(r"[\s-]", "", value).upper(), {}


def _n_digits_only(value: str, ctx: NormCtx) -> NormResult:
    return re.sub(r"\D", "", value), {}


def _n_lower_strip(value: str, ctx: NormCtx) -> NormResult:
    return value.strip().lower(), {}


def _n_upper_strip(value: str, ctx: NormCtx) -> NormResult:
    return re.sub(r"\s+", " ", value).strip().upper(), {}


def _n_ref_upper(value: str, ctx: NormCtx) -> NormResult:
    # Reference numbers: OCR spacing around separators is noise, "02 - 140305"
    # and "02-140305" are the same number.
    v = re.sub(r"\s+", " ", value).strip().upper()
    return re.sub(r" ?([-/.:]) ?", r"\1", v), {}


def _n_text_norm(value: str, ctx: NormCtx) -> NormResult:
    return re.sub(r"\s+", " ", value).strip(), {}


def _n_date_iso(value: str, ctx: NormCtx) -> NormResult:
    o = ctx.opts.get("date_iso", {})
    parsed = _parse_date(value, bool(o.get("dayfirst", True)))
    return (parsed.isoformat() if parsed else None), {}


def _n_currency(value: str, ctx: NormCtx) -> NormResult:
    """Decimal string plus a separately reported currency code."""
    o = ctx.opts.get("currency", {})
    dec = _to_decimal(value)
    norm = str(dec.quantize(Decimal("0.01"))) if dec is not None else None

    symbols: dict[str, str] = o.get("symbols", {}) or {}
    look = int(o.get("lookbehind", 20))
    before = ctx.full[max(0, ctx.start - look) : ctx.start]
    code = None
    # Longest symbol first so "Rs." beats "Rs".
    for sym in sorted(symbols, key=len, reverse=True):
        if sym.lower() in before.lower():
            code = symbols[sym]
            break
    return norm, {"currency": code or o.get("default")}


NORMALIZERS: dict[str, Callable[[str, NormCtx], NormResult]] = {
    "id_upper": _n_id_upper,
    "digits_only": _n_digits_only,
    "lower_strip": _n_lower_strip,
    "upper_strip": _n_upper_strip,
    "ref_upper": _n_ref_upper,
    "text_norm": _n_text_norm,
    "date_iso": _n_date_iso,
    "currency": _n_currency,
}


def normalize(name: str | None, value: str, ctx: NormCtx) -> NormResult:
    """Run the named normalizer on a value.

    Args:
        name: Normalizer name from fields.yaml, or None for no normalizer.
        value: Raw matched text.
        ctx: Where the value sits in the document, plus options.

    Returns:
        (normalized value, extras). An empty or unknown name returns the value
        unchanged; a normalizer that raises returns (None, {}).
    """
    if not name:
        return value, {}
    fn = NORMALIZERS.get(name)
    if fn is None:
        log.warning("unknown normalizer", extra={"normalizer": name})
        return value, {}
    try:
        return fn(value, ctx)
    except (ValueError, TypeError, KeyError, IndexError, ArithmeticError, AttributeError) as exc:  # bad value
        log.warning("normalizer raised", extra={"normalizer": name, "error": summarize_exception(exc)})
        return None, {}
