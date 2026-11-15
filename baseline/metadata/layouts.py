"""Complete-label matching and value layouts for the anchor strategy.

A field opts in with ``anchor.label_rule: complete`` (pack mode sets it; the
built-in fields never do, so their output is unchanged). Then a label matches
only as a complete label phrase, and the value may sit on the label's line, on
the next line, in the row below a header cell, or after a combined label
("Invoice No. & Date: INV-77 / 01-02-2026"). See docs/parts/pack-mode.md.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

COMPLETE_RULE = "complete"
# A label at the start of its line beats one later on a line; both beat a phrase-only match.
LINE_START_CONFIDENCE = 0.9
COMPLETE_CONFIDENCE = 0.85
# Below the review workbook's HIGH-priority edge (0.7), so a phrase-only match is always reviewed.
PHRASE_CONFIDENCE = 0.5
# Right-to-left and left-to-right marks and embeddings; blanked (same length) before matching.
MIN_HEADER_CELLS = 2  # a header row has at least two cells
DIRECTION_MARKS = re.compile("[\u200e\u200f\u202a-\u202e]")

_WORDS = re.compile(r"#|[^\W_]+")
# After a label: a separator, two or more spaces, a tab, or the end of the line.
_SEPARATOR = re.compile(r"[ \t]*[:\-.#=|]|\t|[ \t]{2,}|[ \t]*$")
# After a label: another alphabetic word, which makes the label part of a longer phrase.
_WORD_AFTER = re.compile(r"[ \t]?[^\W\d_]+(?=[ \t,;:]|$)")
# Before a label: another word on the same line segment.
_WORD_BEFORE = re.compile(r"[^\W\d_] ?$")
_EMPTY_REST = re.compile(r"[ \t:\-.#=|]*$")
_BLANK_LINES = re.compile(r"(?:[ \t\r]*\n)*")
_LINES = re.compile(r"[^\n]+")
_CELL_SPLIT = re.compile(r"[ \t]*\|[ \t]*|\t+|[ \t]{2,}")
_LABEL_JOIN = re.compile(r"\s*[&/]\s*|\s+and\s+", re.IGNORECASE)
# Value splitters for a combined label, tried in order until the counts agree.
_VALUE_SPLITS = (re.compile(r"\s+/\s+|\s{2,}|\s*,\s*"), re.compile(r"\s*/\s*"))
# Label word separator: a dot and/or spaces, or nothing before "#".
_WORD_GAP = r"(?:[ \t]*\.[ \t]*|[ \t]+|(?=#))"

Span = tuple[int, int]


class Hit(NamedTuple):
    """One value found for a field: offsets into the text and a confidence."""

    start: int
    end: int
    confidence: float


@dataclass(frozen=True)
class _Ctx:
    view: str  # the text with direction marks blanked, offsets unchanged
    value_rx: re.Pattern[str]
    window: int
    next_line: bool


def _canon(word: str, variants: dict[str, list[str]]) -> str:
    return min(variants.get(word, [word]))


def _label_regex(words: list[str], variants: dict[str, list[str]]) -> str:
    parts = []
    for i, word in enumerate(words):
        alts = sorted({re.escape(v) for v in variants.get(word, [word])}, key=len, reverse=True)
        parts.append((_WORD_GAP if i else "") + "(?:" + "|".join(alts) + ")")
    return r"(?<![^\W_])" + "".join(parts) + r"(?:(?<=#)|(?![^\W_]))"


def label_patterns(spec: dict[str, Any]) -> list[re.Pattern[str]]:
    """One case-insensitive pattern per distinct label, each word allowing its variants.

    Args:
        spec: The field's anchor spec: labels and word_variants.

    Returns:
        The compiled label patterns.
    """
    variants = spec.get("word_variants") or {}
    found: dict[str, None] = {}
    for label in spec.get("labels") or []:
        words = _WORDS.findall(str(label.get("text", "")).lower())
        if words:
            found[_label_regex(words, variants)] = None
    return [re.compile(p, re.IGNORECASE) for p in found]


def _value_span(ctx: _Ctx, start: int, end: int) -> Span | None:
    """Span of the value pattern's first match inside view[start:end] (group 1 when it has one)."""
    m = ctx.value_rx.search(ctx.view[start:end])
    if not m:
        return None
    group = 1 if m.lastindex else 0
    return start + m.start(group), start + m.end(group)


def _line_end(view: str, pos: int) -> int:
    end = view.find("\n", pos)
    return len(view) if end < 0 else end


def _next_line(view: str, line_end: int) -> Span:
    start = min(len(view), line_end + 1)
    blank = _BLANK_LINES.match(view, start)
    start = blank.end() if blank else start
    return start, _line_end(view, start)


def _cells(view: str, start: int, end: int, splitter: re.Pattern[str] = _CELL_SPLIT) -> list[Span]:
    out, pos = [], start
    for sm in splitter.finditer(view, start, end):
        if sm.start() > pos:
            out.append((pos, sm.start()))
        pos = sm.end()
    if pos < end:
        out.append((pos, end))
    return out


def _header_hit(ctx: _Ctx, label: re.Match[str], line_start: int) -> Span | None:
    """The cell in the same column of the next row, when the label is a whole header cell."""
    line_end = _line_end(ctx.view, label.end())
    cells = _cells(ctx.view, line_start, line_end)
    col = next(
        (i for i, (s, e) in enumerate(cells) if s == label.start() and not ctx.view[label.end() : e].strip(" .#")),
        None,
    )
    if len(cells) < MIN_HEADER_CELLS or col is None:
        return None
    row = _cells(ctx.view, *_next_line(ctx.view, line_end))
    return _value_span(ctx, *row[col]) if len(row) == len(cells) else None


def _next_line_hit(ctx: _Ctx, line_end: int) -> Span | None:
    start, end = _next_line(ctx.view, line_end)
    span = _value_span(ctx, start, min(end, start + ctx.window))
    return span if span and not ctx.view[start : span[0]].strip() else None


def _line_hit(ctx: _Ctx, label: re.Match[str]) -> list[Span] | None:
    """The value after a label on its line, or on the next line when the line is empty; None if incomplete."""
    line_end = _line_end(ctx.view, label.end())
    rest = ctx.view[label.end() : line_end]
    if _EMPTY_REST.match(rest):
        span = _next_line_hit(ctx, line_end) if ctx.next_line else None
        return [span] if span else []
    span = _value_span(ctx, label.end(), min(line_end, label.end() + ctx.window))
    direct = span is not None and span[0] - label.end() <= 1 and not _WORD_AFTER.match(rest)
    if not (_SEPARATOR.match(rest) or direct):
        return None
    return [span] if span else []


def _part_index(parts: list[str], labels: set[tuple[str, ...]], variants: dict[str, list[str]]) -> int | None:
    """Which part of a combined label names this field; a later part may inherit the first part's prefix."""
    canon = [tuple(_canon(w, variants) for w in _WORDS.findall(p.lower())) for p in parts]
    prefix = canon[0][:-1]
    return next((k for k, words in enumerate(canon) if words in labels or (k and prefix + words in labels)), None)


def _combined_hits(ctx: _Ctx, spec: dict[str, Any]) -> list[Span]:
    variants = spec.get("word_variants") or {}
    labels = {
        tuple(_canon(w, variants) for w in _WORDS.findall(str(lab.get("text", "")).lower()))
        for lab in spec.get("labels") or []
    }
    out = []
    for line in _LINES.finditer(ctx.view):
        colon = line.group().find(":")
        parts = _LABEL_JOIN.split(line.group()[:colon]) if colon > 0 else []
        k = _part_index(parts, labels, variants) if len(parts) > 1 else None
        if k is None:
            continue
        start = line.start() + colon + 1
        for splitter in _VALUE_SPLITS:
            values = _cells(ctx.view, start, line.end(), splitter)
            if len(values) == len(parts):
                span = _value_span(ctx, *values[k])
                out += [span] if span else []
                break
    return out


def _label_hits(ctx: _Ctx, spec: dict[str, Any], rejected: Callable[[int, int], bool]) -> tuple[list[Hit], list[Hit]]:
    """(complete hits, phrase-only hits) for every label match."""
    complete: list[Hit] = []
    phrase: list[Hit] = []
    for rx in label_patterns(spec):
        for label in rx.finditer(ctx.view):
            if rejected(label.start(), label.end()):
                continue
            line_start = ctx.view.rfind("\n", 0, label.start()) + 1
            header = _header_hit(ctx, label, line_start)
            spans = [header] if header else _line_hit(ctx, label)
            before = ctx.view[line_start : label.start()]
            if _WORD_BEFORE.search(before):
                phrase += [Hit(s, e, PHRASE_CONFIDENCE) for s, e in spans or []]
            else:
                conf = COMPLETE_CONFIDENCE if before.strip() else LINE_START_CONFIDENCE
                complete += [Hit(s, e, conf) for s, e in spans or []]
    return complete, phrase


def complete_hits(
    full: str, spec: dict[str, Any], value_rx: re.Pattern[str], window: int, rejected: Callable[[int, int], bool]
) -> list[Hit]:
    """Values for a field under the complete-label rule, best first.

    Phrase-only matches ("Supplier Invoice No" for "Invoice No") count only
    when no complete match exists, at PHRASE_CONFIDENCE.

    Args:
        full: Canonical document text.
        spec: The field's anchor spec.
        value_rx: The compiled value pattern.
        window: Characters searched for a value on one line.
        rejected: Whether a label at (start, end) is refused by the spec's reject rules.

    Returns:
        Unique hits, highest confidence then earliest first.
    """
    ctx = _Ctx(DIRECTION_MARKS.sub(" ", full), value_rx, window, bool(spec.get("next_line", False)))
    complete, phrase = _label_hits(ctx, spec, rejected)
    if spec.get("combined_labels", False):
        complete += [Hit(s, e, LINE_START_CONFIDENCE) for s, e in _combined_hits(ctx, spec)]
    best: dict[Span, float] = {}
    for hit in complete or phrase:
        if hit.end > hit.start:
            best[(hit.start, hit.end)] = max(hit.confidence, best.get((hit.start, hit.end), 0.0))
    return sorted((Hit(s, e, c) for (s, e), c in best.items()), key=lambda h: (-h.confidence, h.start))
