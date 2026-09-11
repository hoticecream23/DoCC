"""Output schema. This is the contract an evaluation harness scores against.

Anything that replaces a baseline component later must emit exactly this.
Adding fields is a versioned change, not a casual one.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_VERSION = "1.1.0"

# The one and only page joiner. Every char offset in the record is measured
# against pages joined by this. Changing it invalidates every stored offset.
PAGE_SEP = "\n\f\n"

DOC_ID_LEN = 16

Confidence = Annotated[float, Field(ge=0.0, le=1.0)]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ExtractionMethod(str, Enum):
    NATIVE = "native"
    OCR = "ocr"
    HYBRID = "hybrid"
    NONE = "none"
    # The page had a text layer, that layer failed the quality gate, and OCR
    # of the rendered page scored cleaner. Distinct from OCR so a reader can
    # tell a page that never had text from one whose text we chose to discard.
    # Added in schema 1.1.0.
    OCR_RESCUED = "ocr_rescued"


class ClassifyMethod(str, Enum):
    TFIDF_SVM = "tfidf_svm"
    RULE = "rule"
    FALLBACK = "fallback"


class MetadataMethod(str, Enum):
    REGEX_CHECKSUM = "regex_checksum"
    REGEX_FORMAT = "regex_format"
    ANCHOR = "anchor"
    POSITIONAL = "positional"


class TagMethod(str, Enum):
    KEYWORD = "keyword"
    TFIDF_SVM = "tfidf_svm"
    DERIVED = "derived"


class TextRecord(Strict):
    full: str
    pages: list[str]
    method: ExtractionMethod
    char_count: int = Field(ge=0)
    extraction_confidence: Confidence
    # Per-page routing decision. Lets you audit hybrid docs without re-running.
    page_methods: list[ExtractionMethod] = Field(default_factory=list)

    @field_validator("pages")
    @classmethod
    def _no_sep_in_pages(cls, v: list[str]) -> list[str]:
        # A page containing the separator would make offsets ambiguous.
        for i, p in enumerate(v):
            if PAGE_SEP in p:
                raise ValueError(f"page {i} contains the page separator")
        return v

    def check_consistent(self) -> None:
        """Full text must be exactly the pages joined by PAGE_SEP."""
        expected = PAGE_SEP.join(self.pages)
        if self.full != expected:
            raise ValueError("text.full is not pages joined by PAGE_SEP")
        if self.char_count != len(self.full):
            raise ValueError("char_count does not match len(text.full)")


class Classification(Strict):
    label: str
    confidence: Confidence
    all_scores: dict[str, Confidence] = Field(default_factory=dict)
    method: ClassifyMethod


class MetadataField(Strict):
    field: str
    value: str
    normalized_value: str | None
    page: int = Field(ge=0)
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    confidence: Confidence
    method: MetadataMethod
    validated: bool
    # Bump when a normalizer changes so old runs can be re-scored, not re-extracted.
    normalizer_version: str = "1.0.0"
    # Set only for amount fields. Kept apart from the decimal, as specified.
    currency: str | None = None

    @field_validator("char_end")
    @classmethod
    def _end_after_start(cls, v: int, info: Any) -> int:
        start = info.data.get("char_start")
        if start is not None and v < start:
            raise ValueError("char_end precedes char_start")
        return v


class Tag(Strict):
    tag: str
    confidence: Confidence
    method: TagMethod


class Diagnostics(Strict):
    page_count: int = Field(ge=0)
    is_scanned: bool
    has_native_text: bool
    timings_ms: dict[str, float] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


class DocumentRecord(Strict):
    doc_id: str = Field(min_length=DOC_ID_LEN, max_length=DOC_ID_LEN)
    source_path: str
    # Kept separate from source_path so a moved corpus stays comparable.
    filename: str
    schema_version: str = SCHEMA_VERSION
    text: TextRecord
    classification: Classification
    metadata: list[MetadataField] = Field(default_factory=list)
    tags: list[Tag] = Field(default_factory=list)
    diagnostics: Diagnostics

    @field_validator("doc_id")
    @classmethod
    def _hex(cls, v: str) -> str:
        int(v, 16)  # raises if not hex
        return v

    def verify_offsets(self) -> list[str]:
        """Every offset must slice the canonical text back to its raw value.

        Returns a list of problems. Empty means clean.
        """
        problems = []
        full = self.text.full
        for m in self.metadata:
            if m.char_end > len(full):
                problems.append(f"{m.field}: offset {m.char_end} past end {len(full)}")
                continue
            got = full[m.char_start : m.char_end]
            if got != m.value:
                problems.append(
                    f"{m.field}: text[{m.char_start}:{m.char_end}]={got!r} != {m.value!r}"
                )
        return problems


class WordBox(Strict):
    """Word-level geometry. Stored in a sidecar, not in the record.

    Baseline ignores these. Layout models will not, and re-extracting is slow.
    """

    page: int = Field(ge=0)
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    conf: float | None = None
    # Offset of this word in the canonical text, when the extractor can map it.
    char_start: int | None = None
    char_end: int | None = None


class BBoxSidecar(Strict):
    doc_id: str
    page_sizes: list[tuple[float, float]] = Field(default_factory=list)
    words: list[WordBox] = Field(default_factory=list)


def compute_doc_id(path: str | Path) -> str:
    """SHA-256 of file bytes, first 16 hex chars. Content addressed."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:DOC_ID_LEN]


def build_canonical_text(pages: list[str]) -> str:
    return PAGE_SEP.join(pages)


def page_spans(pages: list[str]) -> list[tuple[int, int]]:
    """Start/end offset of each page inside the canonical text."""
    spans = []
    cursor = 0
    for p in pages:
        spans.append((cursor, cursor + len(p)))
        cursor += len(p) + len(PAGE_SEP)
    return spans


def page_for_offset(spans: list[tuple[int, int]], offset: int) -> int:
    """Which page an offset falls in. Separators bind to the preceding page."""
    for i, (s, e) in enumerate(spans):
        if s <= offset <= e:
            return i
    return max(0, len(spans) - 1)


# --------------------------------------------------------------------------
# tables
#
# Tables live in their own sidecar, built by a separate pass, exactly like the
# knowledge graph. DocumentRecord is not touched, so the record contract above
# stays frozen and tables can be rebuilt from any results file at any time.
# --------------------------------------------------------------------------

TABLES_SCHEMA_VERSION = "1.0.0"


class TableMethod(str, Enum):
    # Column structure recovered from word bounding boxes.
    GEOMETRY = "geometry"
    # Column structure recovered from whitespace columns in the canonical text.
    TEXT_GRID = "text_grid"


class TableCell(Strict):
    row: int = Field(ge=0)
    col: int = Field(ge=0)
    # v0 never merges cells, so both are always 1. They are here so that a
    # reader can tell a genuine 1x1 cell from one whose span was simply not
    # represented, and so that adding span detection later is not a change to
    # the shape of every record already written.
    row_span: int = Field(default=1, ge=1)
    col_span: int = Field(default=1, ge=1)
    text: str
    page: int = Field(ge=0)
    # Offsets into the canonical text, same space as metadata. Null when the
    # extractor could not place the words, never guessed.
    char_start: int | None = None
    char_end: int | None = None
    # Page geometry, when the cell was built from boxes or could be matched
    # back onto them. Null for documents that carry no boxes at all.
    x0: float | None = None
    y0: float | None = None
    x1: float | None = None
    y1: float | None = None
    is_header: bool = False


class Table(Strict):
    table_id: str
    page: int = Field(ge=0)
    n_rows: int = Field(ge=0)
    n_cols: int = Field(ge=0)
    method: TableMethod
    # An ordering signal only. Not calibrated against accuracy.
    confidence: Confidence
    # Empty when no header row was identified. Never invented.
    header: list[str] = Field(default_factory=list)
    cells: list[TableCell] = Field(default_factory=list)
    # Span of the whole table in the canonical text, when every cell placed.
    char_start: int | None = None
    char_end: int | None = None

    @property
    def fill(self) -> float:
        n = self.n_rows * self.n_cols
        return round(len(self.cells) / n, 4) if n else 0.0


class TableDiagnostics(Strict):
    # Bands that looked like a table but failed a structural check. Each entry
    # says why, because an abstention is a finding, not a silence.
    abstained: list[str] = Field(default_factory=list)
    strategies_tried: list[str] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


class TableRecord(Strict):
    """One line of the tables sidecar. Joins to DocumentRecord on doc_id."""

    doc_id: str = Field(min_length=DOC_ID_LEN, max_length=DOC_ID_LEN)
    # Carried over from the record so a gold file keyed on a path joins to
    # this sidecar by exactly the same rules it joins to the record.
    source_path: str
    filename: str
    schema_version: str = TABLES_SCHEMA_VERSION
    tables: list[Table] = Field(default_factory=list)
    diagnostics: TableDiagnostics

    def verify_offsets(self, full: str) -> list[str]:
        """Every placed cell must slice the canonical text back to its text.

        Same contract as DocumentRecord.verify_offsets. A cell that cannot
        honour it is a bug, not a rounding error.
        """
        problems = []
        for t in self.tables:
            for c in t.cells:
                if c.char_start is None or c.char_end is None:
                    continue
                if c.char_end > len(full):
                    problems.append(
                        f"{t.table_id} r{c.row}c{c.col}: offset {c.char_end} past end {len(full)}"
                    )
                    continue
                got = full[c.char_start : c.char_end]
                if got != c.text:
                    problems.append(
                        f"{t.table_id} r{c.row}c{c.col}: "
                        f"text[{c.char_start}:{c.char_end}]={got!r} != {c.text!r}"
                    )
        return problems


def read_jsonl(path: str | Path) -> list[dict]:
    """One record per line. Iterating the handle rather than splitlines() is
    deliberate: splitlines() also breaks on the form feed inside PAGE_SEP and
    on U+2028, either of which would cut a record in half.
    """
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{i} is not valid JSON: {exc}") from exc
    return rows
