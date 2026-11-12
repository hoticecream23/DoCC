"""Pure rules behind a client pack: label matching, dedup, references, cases, keys and fields.

No file IO here. ``clientpack`` reads the inventory and schema CSVs, applies
these rules and writes the pack. Rows are the plain dicts ``csv.DictReader``
yields from the inventory and schema files that ``scripts/inventory_share.py``
writes.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import PurePosixPath

# Real samples on a client share. Images beside them are screenshots of the index screen.
SAMPLE_EXT = frozenset({".pdf", ".tif", ".tiff"})
IMAGE_EXT = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".webp"})
INDEX_FIELD_MARK = "index field"  # "opex- Index fields.jpg" is a screenshot of the schema
KEY_PREFIX = re.compile(r"^\d+_")  # "6390_C111322" -> "C111322"
DATE_FORMAT = re.compile(r"\(\s*([DMY][DMY/\-. ]*[DMY])\s*\)", re.IGNORECASE)
MIN_CASE_DOCS = 2  # a case folder bundles at least this many supporting documents

# Words in a free-text data type, checked in this order. "Text (Alpha Numeric)" is text, not a number.
DTYPE_WORDS: tuple[tuple[str, str], ...] = (
    ("date", "date"),
    ("text", "text"),
    ("alpha", "text"),
    ("selection", "selection"),
    ("dropdown", "selection"),
    ("list", "selection"),
    ("numeric", "number"),
    ("number", "number"),
    ("decimal", "number"),
    ("integer", "number"),
    ("amount", "number"),
)
# Normalised field_type -> role. Fixed fields are always present, so they count as required.
ROLE_BY_FIELD_TYPE = {
    "key": "key",
    "required": "required",
    "mandatory": "required",
    "optional": "optional",
    "not required": "optional",
    "fixed": "required",
}
FIXED_FIELD_TYPE = "fixed"
# Values staff type in while indexing. They are not printed on the paper.
OPERATIONAL_KEYWORDS = (
    "box",
    "pouch",
    "file number",
    "file no",
    "indexing person",
    "indexed by",
    "year",
    "month",
    "verification",
    "verified",
    "remark",
    "remarks",
    "posted by",
    "driver",
    "bundle",
)
OPERATIONAL_NAME = re.compile(r"\b(" + "|".join(re.escape(k) for k in OPERATIONAL_KEYWORDS) + r")\b")
ERP_METHOD_MARK = "autofil"  # "Autofil(DB)", the client's spelling
INDEXING_METHOD_KEY = "indexing method"
DB_FIELD_KEY = "db_field"


def norm(text: str) -> str:
    """Lower-case, punctuation to spaces, whitespace collapsed.

    Args:
        text: Any label, type name or field name.

    Returns:
        The normalised form; "Delivery Note - Yellow" -> "delivery note yellow".
    """
    return " ".join(re.sub(r"[^0-9a-z]+", " ", text.lower()).split())


def compact(text: str) -> str:
    """Letters and digits only, lower-cased, for value comparison.

    Args:
        text: A key value.

    Returns:
        "PV-14380 " -> "pv14380".
    """
    return re.sub(r"[^0-9a-z]", "", text.lower())


@dataclass(frozen=True)
class LabelMatch:
    """Outcome of matching one folder label to a cabinet's schema types."""

    target: str
    candidates: tuple[str, ...] = ()

    @property
    def ambiguous(self) -> bool:
        """True when several types matched and none was chosen."""
        return bool(self.candidates)


def match_label(label: str, types: Iterable[str]) -> LabelMatch:
    """Match a folder or file label to one schema type name.

    Three steps, first decisive one wins: exact match of the normalised
    names, then a type name that starts with the label ("LGRN" ->
    "LGRN (Local GRN)"), then a type whose words include all the label's
    words. A step that hits several types stops the search with no target.

    Args:
        label: The folder name or file stem.
        types: The cabinet's schema type names.

    Returns:
        The target type, or "" with the candidates when ambiguous.
    """
    n = norm(label)
    if not n:
        return LabelMatch("")
    words = set(n.split())
    normed = {t: norm(t) for t in types}
    steps = (
        lambda t: t == n,
        lambda t: t.startswith(n + " "),
        lambda t: words <= set(t.split()),
    )
    for step in steps:
        hits = sorted(t for t, nt in normed.items() if step(nt))
        if len(hits) == 1:
            return LabelMatch(hits[0])
        if hits:
            return LabelMatch("", tuple(hits))
    return LabelMatch("")


def key_gold(name: str) -> str:
    """The key value a file name carries, or "".

    Args:
        name: File name, with or without extension.

    Returns:
        The stem with a leading "<digits>_" removed, kept only when it holds a digit.
    """
    stem = KEY_PREFIX.sub("", PurePosixPath(name).stem).strip()
    return stem if any(c.isdigit() for c in stem) else ""


def parse_dtype(data_type: str) -> tuple[str, str | None]:
    """Data type and date format from the client's free text.

    Args:
        data_type: For example "Date(DD/MM/YYYY)" or "Numeric (3 Decimal Place)".

    Returns:
        (text|date|number|selection, the date format upper-cased or None).
    """
    low = data_type.lower()
    dtype = next((out for word, out in DTYPE_WORDS if word in low), "text")
    found = DATE_FORMAT.search(data_type) if dtype == "date" else None
    return dtype, found.group(1).upper().replace(" ", "") if found else None


def parse_role(field_type: str) -> str | None:
    """Role from the field type, or None when the text is not one we know.

    Args:
        field_type: Key / Required / Optional / Not Required / Fixed, any case.

    Returns:
        key, required or optional; None for blank or unknown text.
    """
    return ROLE_BY_FIELD_TYPE.get(norm(field_type))


def parse_source(source: str) -> dict[str, str]:
    """Split "k=v; k=v" into a dict with lower-case keys.

    Args:
        source: The schema CSV's source column.

    Returns:
        Keys to values; parts without "=" are skipped.
    """
    pairs = (part.split("=", 1) for part in source.split(";") if "=" in part)
    return {k.strip().lower(): v.strip() for k, v in pairs}


def field_origin(name: str, field_type: str, source: dict[str, str], role: str) -> str:
    """Where a field's value comes from.

    Args:
        name: Index field name.
        field_type: The raw field type.
        source: Parsed source column.
        role: The field's role.

    Returns:
        fixed for Fixed fields; erp when filled from an ERP view (Autofil(DB),
        or a db_field on a non-key row); manual for operational names; else document.
    """
    if norm(field_type) == FIXED_FIELD_TYPE:
        return "fixed"
    erp_method = ERP_METHOD_MARK in source.get(INDEXING_METHOD_KEY, "").lower()
    if erp_method or (source.get(DB_FIELD_KEY) and role != "key"):
        return "erp"
    return "manual" if OPERATIONAL_NAME.search(norm(name)) else "document"


def is_reference_image(name: str, folder_has_samples: bool, type_labels: set[str]) -> bool:
    """True for a screenshot lying beside the samples rather than a sample.

    Args:
        name: File name.
        folder_has_samples: The same folder also holds .pdf or .tif files.
        type_labels: Normalised folder labels and schema type names of the cabinet.

    Returns:
        True for an image in a sample folder, named after a type, or naming index fields.
    """
    path = PurePosixPath(name)
    if path.suffix.lower() not in IMAGE_EXT:
        return False
    stem = norm(path.stem)
    return folder_has_samples or stem in type_labels or INDEX_FIELD_MARK in stem


def case_folders(parts: Iterable[tuple[str, ...]]) -> set[str]:
    """Type folders that hold case bundles.

    A type folder is a case folder when it has no documents of its own and
    every child folder holds at least two documents.

    Args:
        parts: For each document, its folder parts below the cabinet.

    Returns:
        The first-level folder names that are case folders.
    """
    children: dict[str, Counter[str]] = defaultdict(Counter)
    loose: set[str] = set()
    for p in parts:
        if len(p) == 1:
            loose.add(p[0])
        elif p:
            children[p[0]][p[1]] += 1
    return {top for top, kids in children.items() if top not in loose and min(kids.values()) >= MIN_CASE_DOCS}


def dedup_rank(cabinet: str, folder: str, in_zip: str) -> tuple[bool, int, str]:
    """Sort key for picking which copy of a duplicate to keep: lowest wins.

    Loose files beat zip members, shallow paths beat deep ones (a cabinet
    copied inside another sits one level deeper), then path order.

    Args:
        cabinet: Cabinet name.
        folder: Folder below the cabinet.
        in_zip: Zip name, or "".

    Returns:
        The sort key.
    """
    depth = len([p for p in folder.split("/") if p])
    return bool(in_zip), depth, f"{cabinet}/{folder}"
