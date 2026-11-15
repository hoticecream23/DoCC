"""Pure rules behind a client pack: label matching, dedup, references, cases, keys and fields.

No file IO here. ``clientpack`` reads the inventory and schema CSVs, applies
these rules and writes the pack. Rows are the plain dicts ``csv.DictReader``
yields from the inventory and schema files that ``scripts/inventory_share.py``
writes.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import PurePosixPath

# Real samples on a client share. Images beside them are screenshots of the index screen.
SAMPLE_EXT = frozenset({".pdf", ".tif", ".tiff"})
IMAGE_EXT = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".webp"})
INDEX_FIELD_MARK = "index field"  # "opex- Index fields.jpg" is a screenshot of the schema
KEY_PREFIX = re.compile(r"^\d+_")  # "6390_C111322" -> "C111322"
# Copy markers: "x - Copy", "x - Copy (2)", "x copy", "x copy (2)", "x (1)". A space must precede them and
# the number has one or two digits, so "INV(2024)" and "Form 16 (2023)" keep their brackets.
COPY_SUFFIX = re.compile(r"(?:\s+-\s+copy|\s+copy)(?:\s+\(\d{1,2}\))?$|\s+\(\d{1,2}\)$", re.IGNORECASE)
COPY_PREFIX = re.compile(r"^copy of\s+", re.IGNORECASE)  # "Copy of PV-14380"
# "(DD/MM/YYYY)", "(dd\MM\YYYY)" once backslashes are slashes, "(DD/MM/YYYY HH:MM)".
DATE_FORMAT = re.compile(r"\(\s*([DMY][DMY/\-. ]*[DMY])(?:\s+(H[HMS:.]*))?\s*\)", re.IGNORECASE)
DATE_WORD = re.compile(r"^\s*date\b", re.IGNORECASE)  # "Date(...)" is a date; "Updated" is not
# A prefix or word-subset match must cover more than two of a type's three words,
# so "Payment Voucher" does not become "Payment Voucher Transfer".
MIN_LABEL_COVERAGE = 0.75
TYPE_QUALIFIER = re.compile(r"\([^)]*\)|\s[-–—]+.*$")  # "(Local GRN)" and " - Yellow" qualify a type
# Folders that only wrap the type folders: "Acme/Samples/Purchase Invoice/...".
WRAPPER_NAMES = frozenset({"samples", "sample", "documents", "docs", "exporteddocuments"})

# Words in a free-text data type, checked in this order after DATE_WORD. "Text (Alpha Numeric)" is text.
DTYPE_WORDS: tuple[tuple[str, str], ...] = (
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
# Whole normalised field names: "Year" is a filing field, "Model Year" is printed on the document.
OPERATIONAL_NAMES = frozenset({"year", "month", "box", "pouch", "bundle", "remark", "remarks", "verified"})
# Phrases that are operational wherever they occur in the name.
OPERATIONAL_PHRASES = (
    "box and pouch no",
    "box no",
    "box number",
    "pouch no",
    "file number",
    "file no",
    "indexing person",
    "indexed by",
    "verification status",
    "posted by",
    "bundle no",
    "remarks",
)
# A postal box ("P.O. Box No", "Post Box No") is printed on the document.
OPERATIONAL_PHRASE = re.compile(
    r"(?<!\bpost )(?<!\bpo )(?<!\bp o )\b(" + "|".join(re.escape(k) for k in OPERATIONAL_PHRASES) + r")\b"
)
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
    loose: str = ""  # the one type a prefix or word step found but the label does not cover well enough

    @property
    def ambiguous(self) -> bool:
        """True when several types matched and none was chosen."""
        return bool(self.candidates)


def _covers(label_words: set[str], type_name: str) -> bool:
    """True when a label is close enough to a type it only partly matched.

    The label covers at least MIN_LABEL_COVERAGE of the type's words, or all
    words outside the type's qualifier ("(Local GRN)", " - Yellow").
    """
    words = norm(type_name).split()
    covered = sum(w in label_words for w in words)
    core = set(norm(TYPE_QUALIFIER.sub(" ", type_name)).split())
    return covered >= MIN_LABEL_COVERAGE * len(words) or (bool(core) and core <= label_words)


def match_label(label: str, types: Iterable[str]) -> LabelMatch:
    """Match a folder or file label to one schema type name.

    Three steps, first decisive one wins: exact match of the normalised
    names, then a type name that starts with the label ("LGRN" ->
    "LGRN (Local GRN)"), then a type whose words include all the label's
    words. A step that hits several types stops the search with no target.
    A single prefix or word hit counts only when ``_covers`` holds; else it
    comes back as ``loose`` with no target.

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
    for exact, step in zip((True, False, False), steps, strict=True):
        hits = sorted(t for t, nt in normed.items() if step(nt))
        if len(hits) > 1:
            return LabelMatch("", tuple(hits))
        if hits:
            return LabelMatch(hits[0]) if exact or _covers(words, hits[0]) else LabelMatch("", loose=hits[0])
    return LabelMatch("")


def clean_stem(name: str) -> str:
    """A file stem without a leading "Copy of " and trailing copy markers.

    Args:
        name: File name, with or without extension.

    Returns:
        "PV-1 - Copy (2).pdf" -> "PV-1"; "Scan (1).tif" -> "Scan"; "INV(2024).pdf" unchanged.
    """
    stem = COPY_PREFIX.sub("", PurePosixPath(name).stem.strip())
    while (cut := COPY_SUFFIX.sub("", stem).strip()) and cut != stem:
        stem = cut
    return stem


def key_gold(name: str) -> str:
    """The key value a file name carries, or "".

    Args:
        name: File name, with or without extension.

    Returns:
        The stem without copy suffixes and a leading "<digits>_", kept only when it holds a digit.
    """
    stem = KEY_PREFIX.sub("", clean_stem(name)).strip()
    return stem if any(c.isdigit() for c in stem) else ""


def parse_dtype(data_type: str) -> tuple[str, str | None]:
    """Data type and date format from the client's free text.

    Args:
        data_type: For example "Date(DD/MM/YYYY)" or "Numeric (3 Decimal Place)".

    Returns:
        (text|date|number|selection, the date format upper-cased or None).
        A time part follows the date after one space: "DD/MM/YYYY HH:MM".
    """
    if not DATE_WORD.match(data_type):
        low = data_type.lower()
        return next((out for word, out in DTYPE_WORDS if word in low), "text"), None
    found = DATE_FORMAT.search(data_type.replace("\\", "/"))
    if not found:
        return "date", None
    date, time = found.group(1).upper().replace(" ", ""), found.group(2)
    return "date", f"{date} {time.upper()}" if time else date


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
        or a db_field on a non-key row); manual for an OPERATIONAL_NAMES name
        or a name holding an OPERATIONAL_PHRASES phrase; else document.
    """
    if norm(field_type) == FIXED_FIELD_TYPE:
        return "fixed"
    erp_method = ERP_METHOD_MARK in source.get(INDEXING_METHOD_KEY, "").lower()
    if erp_method or (source.get(DB_FIELD_KEY) and role != "key"):
        return "erp"
    n = norm(name)
    return "manual" if n in OPERATIONAL_NAMES or OPERATIONAL_PHRASE.search(n) else "document"


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


Folder = tuple[str, ...]


def folder_tree(parts: Iterable[Folder]) -> tuple[dict[Folder, set[str]], set[Folder]]:
    """Child folder names of every folder, and the folders that hold documents.

    Args:
        parts: For each document, its folder parts below the cabinet.

    Returns:
        (folder -> child folder names, folders with documents of their own).
    """
    children: dict[Folder, set[str]] = defaultdict(set)
    holds: set[Folder] = set()
    for p in parts:
        holds.add(p)
        for i in range(len(p)):
            children[p[:i]].add(p[i])
    return children, holds


def wrapper_folders(parts: Iterable[Folder], matches: Callable[[str], bool], has_schema: bool) -> set[Folder]:
    """Folders that only wrap the type folders, such as "Samples".

    A wrapper holds subfolders and no documents of its own, and its name
    matches no schema type. With a schema, it is a wrapper only when skipping
    it yields a child that matches a type, directly or through another
    wrapper; its name does not matter. With no schema, only a folder named in
    WRAPPER_NAMES is a wrapper.

    Args:
        parts: For each document, its folder parts below the cabinet.
        matches: True when a folder name matches a schema type or an override.
        has_schema: The cabinet has schema types.

    Returns:
        The folders that are wrappers.
    """
    children, holds = folder_tree(parts)
    seen: dict[Folder, bool] = {}

    def wrapper(prefix: Folder) -> bool:
        if prefix not in seen:
            name, kids = prefix[-1], children[prefix]
            if prefix in holds or not kids or matches(name):
                seen[prefix] = False
            elif not has_schema:
                seen[prefix] = norm(name) in WRAPPER_NAMES
            else:
                seen[prefix] = any(matches(k) or wrapper(prefix + (k,)) for k in kids)
        return seen[prefix]

    return {prefix for prefix in list(children) if prefix and wrapper(prefix)}


def label_depth(parts: Folder, wrappers: set[Folder]) -> int:
    """Index of the label folder in a document's folder parts, past any wrappers.

    Args:
        parts: Folder parts below the cabinet.
        wrappers: From ``wrapper_folders``.

    Returns:
        The index; it equals len(parts) only when parts is empty.
    """
    depth = 0
    while parts[: depth + 1] in wrappers:
        depth += 1
    return depth


def child_roles(child: str, stems: list[str]) -> list[str] | None:
    """Role of each document below one child folder of a label folder, or None for a case.

    A folder with one document holds that document. A folder where exactly one
    stem equals the folder name holds that document, and the others are its
    attachments. Any other folder is a case bundle. So "100/100.pdf" plus
    "100/100-annex.pdf" is one document with an attachment, not a case.

    Args:
        child: The child folder name.
        stems: The clean stems of the documents below it, in order.

    Returns:
        "document" or "attachment" per stem; None when the folder is a case.
    """
    if len(stems) == 1:
        return ["document"]
    named = [norm(stem) == norm(child) for stem in stems]
    if sum(named) != 1:
        return None
    return ["document" if n else "attachment" for n in named]


def dedup_rank(foreign: bool, folder: str, in_zip: str, rel_path: str) -> tuple[bool, bool, int, str]:
    """Sort key for picking which copy of a duplicate to keep: lowest wins.

    The row in the cabinet that owns it beats a copy filed under another
    cabinet's name. Then loose files beat zip members, shallow paths beat
    deep ones, then path order.

    Args:
        foreign: The row's first folder names another cabinet.
        folder: Folder below the cabinet.
        in_zip: Zip name, or "".
        rel_path: Path relative to the share root.

    Returns:
        The sort key.
    """
    depth = len([p for p in folder.split("/") if p])
    return foreign, bool(in_zip), depth, rel_path
