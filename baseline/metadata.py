"""Typed field extraction with character offsets.

Three strategies per field, applied in the order listed in fields.yaml,
first hit wins. Offsets are always against the canonical text.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Callable

from .config import Config
from .logging_setup import get_logger
from .schema import MetadataField, MetadataMethod, WordBox, page_for_offset, page_spans

log = get_logger(__name__)


class OffsetMismatch(ValueError):
    """Raised in strict mode when an offset does not slice back to its value."""


# --------------------------------------------------------------------------
# validators. each takes the raw matched string, returns True if it checks out
# --------------------------------------------------------------------------

# Verhoeff dihedral group tables, used by Aadhaar.
_VERHOEFF_D = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)
_VERHOEFF_P = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)
_VERHOEFF_INV = (0, 4, 3, 2, 1, 5, 6, 7, 8, 9)

# 4th char of a PAN encodes holder type. Anything else is not a real PAN.
PAN_ENTITY_TYPES = set("ABCEFGHJKLPT")

# Base 36 alphabet for the GSTIN check digit.
_B36 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Valid GST state codes: 01-38 plus the two special ones.
_GST_STATES = {f"{i:02d}" for i in range(1, 39)} | {"97", "99"}


def verhoeff_check(digits: str) -> bool:
    """True if the trailing digit is a valid Verhoeff checksum for the rest."""
    if not digits.isdigit():
        return False
    c = 0
    for i, ch in enumerate(reversed(digits)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][int(ch)]]
    return c == 0


def verhoeff_digit(payload: str) -> str:
    """Check digit for a payload that does not yet carry one. Test helper."""
    if not payload.isdigit():
        raise ValueError("payload must be digits")
    c = 0
    for i, ch in enumerate(reversed(payload)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[(i + 1) % 8][int(ch)]]
    return str(_VERHOEFF_INV[c])


def luhn_check(digits: str) -> bool:
    d = re.sub(r"\D", "", digits)
    if len(d) < 12 or len(d) > 19:
        return False
    total = 0
    for i, ch in enumerate(reversed(d)):
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def gstin_check(value: str) -> bool:
    """15 char GSTIN: state code, PAN, entity number, Z, base 36 check digit."""
    v = value.strip().upper()
    if len(v) != 15 or not re.fullmatch(r"[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][0-9A-Z]Z[0-9A-Z]", v):
        return False
    if v[:2] not in _GST_STATES:
        return False
    if not pan_check(v[2:12]):
        return False
    total = 0
    factor = 2
    for ch in reversed(v[:14]):
        prod = factor * _B36.index(ch)
        factor = 1 if factor == 2 else 2
        total += prod // 36 + prod % 36
    return _B36[(36 - total % 36) % 36] == v[14]


def pan_check(value: str, args: dict | None = None) -> bool:
    """PAN has no public checksum, so this is format plus structural rules.

    The 4th char must be a real holder type code. Override the set from
    fields.yaml via validator_args.entity_types if your corpus needs it.
    """
    v = value.strip().upper()
    if not re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]", v):
        return False
    allowed = set((args or {}).get("entity_types") or PAN_ENTITY_TYPES)
    return v[3] in allowed


def aadhaar_check(value: str) -> bool:
    d = re.sub(r"\D", "", value)
    if len(d) != 12 or d[0] in "01":
        return False
    return verhoeff_check(d)


def ifsc_check(value: str) -> bool:
    """No checksum exists. Format plus the mandatory zero in position five."""
    v = value.strip().upper()
    return bool(re.fullmatch(r"[A-Z]{4}0[A-Z0-9]{6}", v))


def email_check(value: str) -> bool:
    v = value.strip()
    if v.count("@") != 1 or ".." in v:
        return False
    local, _, domain = v.partition("@")
    return bool(local) and "." in domain and not domain.startswith(".")


def phone_in_check(value: str) -> bool:
    d = re.sub(r"\D", "", value)
    if len(d) == 12 and d.startswith("91"):
        d = d[2:]
    return len(d) == 10 and d[0] in "6789"


def date_sane_check(value: str, args: dict | None = None) -> bool:
    args = args or {}
    parsed = _parse_date(value, bool(args.get("dayfirst", True)))
    if parsed is None:
        return False
    return int(args.get("min_year", 1990)) <= parsed.year <= int(args.get("max_year", 2035))


def amount_sane_check(value: str, args: dict | None = None) -> bool:
    args = args or {}
    dec = _to_decimal(value)
    if dec is None:
        return False
    return Decimal(str(args.get("min", 0))) <= dec <= Decimal(str(args.get("max", 10**12)))


def invoice_number_check(value: str, args: dict | None = None) -> bool:
    v = value.strip()
    # An invoice number with no digit in it is almost always a bad capture.
    return 3 <= len(v) <= 30 and any(c.isdigit() for c in v)


# Validators whose format alone proves the value. Everything else is a
# sanity check: it can pass and still be wrong, so it reports regex_format.
CHECKSUM_VALIDATORS = frozenset({"pan", "aadhaar", "gstin", "ifsc", "luhn"})

VALIDATORS: dict[str, Callable[..., bool]] = {
    "pan": pan_check,
    "aadhaar": aadhaar_check,
    "gstin": gstin_check,
    "ifsc": ifsc_check,
    "luhn": luhn_check,
    "email": email_check,
    "phone_in": phone_in_check,
    "date_sane": date_sane_check,
    "amount_sane": amount_sane_check,
    "invoice_number": invoice_number_check,
}


def run_validator(name: str | None, value: str, args: dict | None = None) -> bool:
    if not name:
        return False
    fn = VALIDATORS.get(name)
    if fn is None:
        log.warning("unknown validator", extra={"validator": name})
        return False
    try:
        try:
            return bool(fn(value, args))
        except TypeError:
            return bool(fn(value))
    except Exception as exc:
        log.warning("validator raised", extra={"validator": name, "error": str(exc)})
        return False


# --------------------------------------------------------------------------
# normalizers. return (normalized_value, extras)
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# candidate generation
# --------------------------------------------------------------------------


@dataclass
class Candidate:
    value: str
    start: int
    end: int
    confidence: float
    method: MetadataMethod
    validated: bool


@functools.lru_cache(maxsize=512)
def _compile(pattern: str, ignorecase: bool = True) -> re.Pattern:
    flags = re.IGNORECASE if ignorecase else 0
    return re.compile(pattern, flags)


def _span_of(m: re.Match) -> tuple[int, int, str]:
    """Prefer group 1 as the value. Falls back to the whole match."""
    if m.lastindex:
        return m.start(1), m.end(1), m.group(1)
    return m.start(), m.end(), m.group(0)


def _pattern_candidates(full: str, fdef: dict, limit: int) -> list[Candidate]:
    spec = fdef.get("pattern") or {}
    rx = spec.get("regex")
    if not rx:
        return []
    validator = spec.get("validator")
    args = spec.get("validator_args") or {}
    hard = validator in CHECKSUM_VALIDATORS
    c_ok = float(spec.get("confidence_validated", 1.0))
    c_no = float(spec.get("confidence_unvalidated", 0.5))
    # IDs are uppercase by convention, so patterns are case sensitive by default.
    ignorecase = bool(spec.get("ignorecase", False))

    out: list[Candidate] = []
    for m in _compile(rx, ignorecase).finditer(full):
        s, e, val = _span_of(m)
        ok = run_validator(validator, val, args)
        out.append(
            Candidate(
                value=val,
                start=s,
                end=e,
                confidence=c_ok if ok else c_no,
                # Checksum backed and format only are different things.
                method=(
                    MetadataMethod.REGEX_CHECKSUM
                    if ok and hard
                    else MetadataMethod.REGEX_FORMAT
                ),
                validated=ok,
            )
        )
        if len(out) >= limit:
            break
    return out


def _anchor_candidates(full: str, fdef: dict, default_window: int, limit: int) -> list[Candidate]:
    spec = fdef.get("anchor") or {}
    labels = spec.get("labels") or []
    value_rx = spec.get("value_regex")
    if not labels or not value_rx:
        return []

    window = int(spec.get("window", default_window))
    direction = spec.get("direction", "after")
    validator = spec.get("validator")
    args = spec.get("validator_args") or {}
    vrx = _compile(value_rx, bool(spec.get("ignorecase", False)))
    low = full.lower()

    out: list[Candidate] = []
    # Most specific labels first so a weak label never shadows a strong one.
    for lab in sorted(labels, key=lambda l: -float(l.get("specificity", 0.5))):
        text = str(lab.get("text", "")).lower()
        if not text:
            continue
        spec_score = float(lab.get("specificity", 0.5))
        pos = low.find(text)
        while pos >= 0:
            label_end = pos + len(text)
            hits = []
            if direction in ("after", "both"):
                seg = full[label_end : label_end + window]
                m = vrx.search(seg)
                if m:
                    s, e, val = _span_of(m)
                    hits.append((label_end + s, label_end + e, val, s))
            if direction in ("before", "both"):
                seg_start = max(0, pos - window)
                seg = full[seg_start:pos]
                matches = list(vrx.finditer(seg))
                if matches:
                    m = matches[-1]  # nearest to the label
                    s, e, val = _span_of(m)
                    hits.append((seg_start + s, seg_start + e, val, pos - (seg_start + e)))

            for start, end, val, dist in hits:
                ok = run_validator(validator, val, args) if validator else False
                # Specificity sets the ceiling, distance erodes it.
                decay = 1.0 - 0.4 * min(1.0, dist / max(1, window))
                conf = min(0.95, spec_score * decay)
                if ok:
                    conf = min(1.0, conf + 0.05)
                out.append(
                    Candidate(
                        value=val,
                        start=start,
                        end=end,
                        confidence=round(conf, 4),
                        method=MetadataMethod.ANCHOR,
                        validated=ok,
                    )
                )
            if len(out) >= limit:
                return out
            pos = low.find(text, pos + 1)
    return out


def _positional_candidates(
    full: str,
    fdef: dict,
    templates: list[dict],
    doc_class: str,
    words: list[WordBox],
    spans: list[tuple[int, int]],
    page_sizes: list[tuple[float, float]],
) -> list[Candidate]:
    """Read a field from a fixed region of a registered template.

    Needs bboxes, so it only fires for extraction methods that give geometry.
    """
    if not templates or not words:
        return []
    name = fdef.get("name")
    for tpl in templates:
        if tpl.get("class") and tpl["class"] != doc_class:
            continue
        match = tpl.get("match") or {}
        rx = match.get("regex")
        if rx and not _compile(rx).search(full):
            continue
        region = (tpl.get("fields") or {}).get(name)
        if not region:
            continue

        pno = int(region.get("page", 0))
        if pno >= len(page_sizes):
            continue
        pw, ph = page_sizes[pno]
        if pw <= 0 or ph <= 0:
            continue
        x0, y0, x1, y1 = (float(v) for v in region["region"])

        inside = [
            w
            for w in words
            if w.page == pno
            and w.char_start is not None
            and w.char_end is not None
            and x0 <= (w.x0 / pw) <= x1
            and y0 <= (w.y0 / ph) <= y1
        ]
        if not inside:
            continue
        start = min(w.char_start for w in inside)
        end = max(w.char_end for w in inside)
        if start >= end or end > len(full):
            continue
        # Slice the canonical text so the offsets are correct by construction.
        return [
            Candidate(
                value=full[start:end],
                start=start,
                end=end,
                confidence=float(tpl.get("confidence", 0.85)),
                method=MetadataMethod.POSITIONAL,
                validated=False,
            )
        ]
    return []


def _rank(c: Candidate) -> tuple:
    # Validated wins, then confidence, then earliest position for stability.
    return (0 if c.validated else 1, -c.confidence, c.start, c.end)


def extract_fields(
    full: str,
    pages: list[str],
    cfg: Config,
    doc_class: str = "",
    words: list[WordBox] | None = None,
    page_sizes: list[tuple[float, float]] | None = None,
) -> tuple[list[MetadataField], list[str]]:
    """Run every configured field. Returns (fields, errors)."""
    opts = cfg.metadata_opts()
    limit = int(opts.get("max_candidates_per_field", 25))
    default_window = int(opts.get("default_anchor_window", 120))
    strict = bool(opts.get("strict_offsets", False))
    unknown_label = str(cfg.classify_opts().get("unknown_label", "unknown"))
    run_all_when_unknown = bool(opts.get("run_scoped_fields_when_unknown", True))
    drop_bad = bool(opts.get("drop_on_offset_mismatch", True))
    norm_opts = cfg.fields.get("normalizer_options", {}) or {}
    nver = cfg.normalizer_version
    spans = page_spans(pages)
    words = words or []
    page_sizes = page_sizes or []

    results: list[MetadataField] = []
    errors: list[str] = []

    for fdef in cfg.field_defs:
        name = fdef.get("name")
        if not name:
            continue
        # A field may be scoped to certain classes. Unknown docs still get
        # everything, otherwise we lose recall exactly where we need it.
        scope = fdef.get("classes")
        if scope and doc_class not in scope:
            if not (doc_class in ("", unknown_label) and run_all_when_unknown):
                continue
        cands: list[Candidate] = []
        for strategy in fdef.get("strategies", []):
            if strategy == "pattern":
                cands = _pattern_candidates(full, fdef, limit)
            elif strategy == "anchor":
                cands = _anchor_candidates(full, fdef, default_window, limit)
            elif strategy == "positional":
                cands = _positional_candidates(
                    full, fdef, cfg.templates, doc_class, words, spans, page_sizes
                )
            else:
                errors.append(f"{name}: unknown strategy {strategy!r}")
                continue
            if cands:
                break  # first strategy that hits wins

        if not cands:
            continue

        cands.sort(key=_rank)
        keep = cands if fdef.get("multi") else cands[:1]

        seen: set[str] = set()
        for c in keep:
            ctx = NormCtx(full=full, start=c.start, end=c.end, opts=norm_opts)
            normalized, extras = normalize(fdef.get("normalizer"), c.value, ctx)

            # Contract check: the offsets must slice back to the raw value.
            if full[c.start : c.end] != c.value:
                msg = f"{name}: offset mismatch at {c.start}:{c.end}"
                if strict:
                    raise OffsetMismatch(msg)
                log.error("offset mismatch", extra={"field": name, "start": c.start})
                errors.append(msg)
                if drop_bad:
                    continue

            key = normalized if normalized is not None else c.value
            if fdef.get("multi") and key in seen:
                continue
            seen.add(key)

            results.append(
                MetadataField(
                    field=name,
                    value=c.value,
                    normalized_value=normalized,
                    page=page_for_offset(spans, c.start),
                    char_start=c.start,
                    char_end=c.end,
                    confidence=round(c.confidence, 4),
                    method=c.method,
                    validated=c.validated,
                    normalizer_version=nver,
                    currency=extras.get("currency"),
                )
            )

    results.sort(key=lambda m: (m.char_start, m.field))
    return results, errors

