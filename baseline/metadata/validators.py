"""Validators. Each takes the raw matched string and returns True if it checks out.

Only some of them prove anything. A checksum that passes is strong evidence;
a format or sanity check that passes can still be wrong. CHECKSUM_VALIDATORS
is the line between the two, and it is what decides regex_checksum against
regex_format in the output.
"""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Callable

from ..logging_setup import get_logger
from .normalizers import _parse_date, _to_decimal

log = get_logger(__name__)

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
