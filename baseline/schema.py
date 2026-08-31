"""Output schema. This is the contract an evaluation harness scores against.

Anything that replaces a baseline component later must emit exactly this.
Adding fields is a versioned change, not a casual one.
"""

from __future__ import annotations

import hashlib
from enum import Enum
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_VERSION = "1.0.0"

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
