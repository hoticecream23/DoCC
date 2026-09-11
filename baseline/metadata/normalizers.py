"""Normalizers. Each returns (normalized_value, extras).

Raw matched text is what the document says; the normalized value is what gets
compared. Both the pipeline and the workbook importer go through these, so a
value a reviewer types and a value the pipeline emits land in the same form.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Callable

from ..logging_setup import get_logger

log = get_logger(__name__)

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _parse_date(value: str, dayfirst: bool = True):
    """Deterministic date parsing. No fuzzy guessing, no locale surprises."""
    v = value.strip().replace(",", " ")
    v = re.sub(r"\s+", " ", v)

    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", v)
    if m:
        y, mo, d = (int(x) for x in m.groups())
        return _safe_date(y, mo, d)

    m = re.fullmatch(r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})", v)
    if m:
        a, b, y = (int(x) for x in m.groups())
        if y < 100:
            y += 2000 if y < 70 else 1900
        d, mo = (a, b) if dayfirst else (b, a)
        # Fall back to the other order when the first reading is impossible.
        if mo > 12 and d <= 12:
            d, mo = mo, d
        return _safe_date(y, mo, d)

    m = re.fullmatch(r"(\d{1,2}) ([A-Za-z]{3,9}) (\d{4})", v)
    if m:
        d, name, y = int(m.group(1)), m.group(2).lower()[:3], int(m.group(3))
        if name in _MONTHS:
            return _safe_date(y, _MONTHS[name], d)

    m = re.fullmatch(r"([A-Za-z]{3,9}) (\d{1,2}) (\d{4})", v)
    if m:
        name, d, y = m.group(1).lower()[:3], int(m.group(2)), int(m.group(3))
        if name in _MONTHS:
            return _safe_date(y, _MONTHS[name], d)
    return None


def _safe_date(y: int, mo: int, d: int):
    import datetime

    try:
        return datetime.date(y, mo, d)
    except ValueError:
        return None


def _to_decimal(value: str) -> Decimal | None:
    v = re.sub(r"[^\d.\-]", "", value.replace(",", ""))
    if not v or v in {"-", "."}:
        return None
    try:
        return Decimal(v)
    except InvalidOperation:
        return None


@dataclass
class NormCtx:
    full: str
    start: int
    end: int
    opts: dict[str, Any]


def _n_id_upper(value: str, ctx: NormCtx):
    return re.sub(r"[\s-]", "", value).upper(), {}


def _n_digits_only(value: str, ctx: NormCtx):
    return re.sub(r"\D", "", value), {}


def _n_lower_strip(value: str, ctx: NormCtx):
    return value.strip().lower(), {}


def _n_upper_strip(value: str, ctx: NormCtx):
    return re.sub(r"\s+", " ", value).strip().upper(), {}


def _n_text_norm(value: str, ctx: NormCtx):
    return re.sub(r"\s+", " ", value).strip(), {}


def _n_date_iso(value: str, ctx: NormCtx):
    o = ctx.opts.get("date_iso", {})
    parsed = _parse_date(value, bool(o.get("dayfirst", True)))
    return (parsed.isoformat() if parsed else None), {}


def _n_currency(value: str, ctx: NormCtx):
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


NORMALIZERS: dict[str, Callable[[str, NormCtx], tuple]] = {
    "id_upper": _n_id_upper,
    "digits_only": _n_digits_only,
    "lower_strip": _n_lower_strip,
    "upper_strip": _n_upper_strip,
    "text_norm": _n_text_norm,
    "date_iso": _n_date_iso,
    "currency": _n_currency,
}


def normalize(name: str | None, value: str, ctx: NormCtx) -> tuple:
    if not name:
        return value, {}
    fn = NORMALIZERS.get(name)
    if fn is None:
        log.warning("unknown normalizer", extra={"normalizer": name})
        return value, {}
    try:
        return fn(value, ctx)
    except Exception as exc:
        log.warning("normalizer raised", extra={"normalizer": name, "error": str(exc)})
        return None, {}
