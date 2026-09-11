"""Typed field extraction with character offsets.

Three strategies per field, applied in the order listed in fields.yaml,
first hit wins. Offsets are always against the canonical text.

- `validators.py`   checksums and sanity checks, one per validator name
- `normalizers.py`  raw matched text into a comparable value
- `fields.py`       candidate generation and `extract_fields`
"""

from .fields import Candidate, OffsetMismatch, extract_fields
from .normalizers import NORMALIZERS, NormCtx, normalize
from .validators import (
    CHECKSUM_VALIDATORS,
    PAN_ENTITY_TYPES,
    VALIDATORS,
    aadhaar_check,
    amount_sane_check,
    date_sane_check,
    email_check,
    gstin_check,
    ifsc_check,
    invoice_number_check,
    luhn_check,
    pan_check,
    phone_in_check,
    run_validator,
    verhoeff_check,
    verhoeff_digit,
)

__all__ = [
    "CHECKSUM_VALIDATORS", "Candidate", "NORMALIZERS", "NormCtx", "OffsetMismatch",
    "PAN_ENTITY_TYPES", "VALIDATORS", "aadhaar_check", "amount_sane_check",
    "date_sane_check", "email_check", "extract_fields", "gstin_check", "ifsc_check",
    "invoice_number_check", "luhn_check", "normalize", "pan_check", "phone_in_check",
    "run_validator", "verhoeff_check", "verhoeff_digit",
]
