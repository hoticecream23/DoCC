"""Locate a value the model returned inside the canonical text.

A value that cannot be located is not emitted. The record contract requires
full[char_start:char_end] == value, and a value the document does not contain
is exactly the value a model invented, so this step is both the offset step
and the guard against invention.

Three steps, and the first that matches wins:

1. exact: the value occurs verbatim;
2. loose: it occurs once whitespace runs are collapsed and case is folded;
3. normalised: the field's own normalizer gives the same value for it and for
   a place a value of the field's shape is printed (`scan_spans`) or a rule
   candidate.

A match never crosses a page boundary and never starts or ends inside a
longer word or number. The span reported is always the document's own text.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from baseline.metadata import Candidate, NormCtx, normalize
from baseline.metadata.fields import scan_spans
from baseline.schema import page_spans

EXACT = "exact"
LOOSE = "loose"
NORMALISED = "normalised"
UNGROUNDED = "ungrounded"
# Characters needed before a match to see a joined token: one token character and the joiner.
JOIN_LOOKBEHIND = 2
JOINERS = "-/,."  # Separators: -, /, , always join; . joins only between digits


@dataclass(frozen=True)
class Grounding:
    """Where a value was found, and how.

    Attributes:
        outcome: EXACT, LOOSE, NORMALISED or UNGROUNDED.
        start: Span start in the full text; None when ungrounded.
        end: Span end in the full text; None when ungrounded.
        n_matches: How many places matched at this step.
        rule_agrees: A rule candidate is the same span or the same value.
    """

    outcome: str
    start: int | None = None
    end: int | None = None
    n_matches: int = 0
    rule_agrees: bool = False

    @property
    def grounded(self) -> bool:
        """True when the value was found in the text."""
        return self.outcome != UNGROUNDED


def _fold(text: str) -> tuple[str, list[int]]:
    """Casefold, and collapse each whitespace run to one space.

    Returns the folded string and, for each of its characters, the index in
    `text` it came from. Casefolding can lengthen a character (ß to ss), which
    is why this is a map and not a constant shift.
    """
    out: list[str] = []
    src: list[int] = []
    in_space = False
    for i, ch in enumerate(text):
        if ch.isspace():
            if not in_space:
                out.append(" ")
                src.append(i)
            in_space = True
            continue
        in_space = False
        for f in ch.casefold():
            out.append(f)
            src.append(i)
    return "".join(out), src


def _folded(text: str) -> str:
    return _fold(text.strip())[0]


def _joins(left: str, sep: str, right: str) -> bool:
    """Check if sep joins left and right into a single token.

    - and / and , always join alphanumerics.
    - . only joins digits (decimal point).
    """
    if sep not in JOINERS:
        return False
    if sep == ".":
        return left.isdigit() and right.isdigit()
    return left.isalnum() and right.isalnum()


class TextIndex:
    """Per document lookups, built once and shared by every value."""

    def __init__(self, full: str, pages: list[str]) -> None:
        self.full = full
        self.spans = page_spans(pages)
        self.folded, self.src = _fold(full)

    def _on_one_page(self, s: int, e: int) -> bool:
        return any(ps <= s and e <= pe for ps, pe in self.spans)

    def _on_boundary(self, s: int, e: int) -> bool:
        full = self.full
        first, last = full[s], full[e - 1]
        # Reject if starting or ending adjacent to an alphanumeric
        if first.isalnum() and s > 0 and full[s - 1].isalnum():
            return False
        if last.isalnum() and e < len(full) and full[e].isalnum():
            return False
        # Reject if starting inside a joined token
        if s >= JOIN_LOOKBEHIND and first.isalnum() and _joins(full[s - 2], full[s - 1], first):
            return False
        # Reject if ending inside a joined token
        if e + 1 < len(full) and last.isalnum() and _joins(last, full[e], full[e + 1]):
            return False
        return True

    def usable(self, s: int, e: int) -> bool:
        """True when text[s:e] is in range, on one page and not part of a longer token."""
        return 0 <= s < e <= len(self.full) and self._on_one_page(s, e) and self._on_boundary(s, e)


def _norm(fdef: dict[str, Any], value: str, full: str, s: int, e: int, opts: dict[str, Any]) -> str | None:
    normalized: str | None  # normalize() returns a bare tuple
    normalized, _ = normalize(fdef.get("normalizer"), value, NormCtx(full=full, start=s, end=e, opts=opts))
    return normalized


def _exact(index: TextIndex, value: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    i = index.full.find(value)
    while i >= 0:
        if index.usable(i, i + len(value)):
            out.append((i, i + len(value)))
        i = index.full.find(value, i + 1)
    return out


def _loose(index: TextIndex, value: str) -> list[tuple[int, int]]:
    target = _folded(value)
    if not target:
        return []
    out: set[tuple[int, int]] = set()
    i = index.folded.find(target)
    while i >= 0:
        s = index.src[i]
        e = index.src[i + len(target) - 1] + 1
        # Re-check on the original slice. A match that starts or ends partway
        # through a character's casefold expansion fails here.
        if _folded(index.full[s:e]) == target and index.usable(s, e):
            out.add((s, e))
        i = index.folded.find(target, i + 1)
    return sorted(out)


def _normalised(
    index: TextIndex, value: str, fdef: dict[str, Any], rule_cands: list[Candidate], opts: dict[str, Any]
) -> list[tuple[int, int]]:
    target = _norm(fdef, value, value, 0, len(value), opts)
    if not target:
        return []
    places = scan_spans(index.full, fdef) + [(c.start, c.end, c.value) for c in rule_cands]
    return sorted(
        {(s, e) for s, e, raw in places if index.usable(s, e) and _norm(fdef, raw, index.full, s, e, opts) == target}
    )


def _agrees(c: Candidate, s: int, e: int, index: TextIndex, fdef: dict[str, Any], opts: dict[str, Any]) -> bool:
    """The rule candidate is this span, or overlaps it with the same value."""
    if not (c.start < e and s < c.end):
        return False
    if (c.start, c.end) == (s, e):
        return True
    mine = _norm(fdef, index.full[s:e], index.full, s, e, opts)
    return mine is not None and _norm(fdef, c.value, index.full, c.start, c.end, opts) == mine


def ground(
    value: str, index: TextIndex, fdef: dict[str, Any], rule_cands: list[Candidate], norm_opts: dict[str, Any]
) -> Grounding:
    """Find a model value in the text: exact, then loose, then normalised.

    Args:
        value: The value the model returned.
        index: The document's text index.
        fdef: The field definition from fields.yaml.
        rule_cands: Rule candidates for the field, best first.
        norm_opts: Normalizer options from fields.yaml.

    Returns:
        The first step that matched, with the best span, or UNGROUNDED.
    """
    value = value.strip()
    if not value:
        return Grounding(UNGROUNDED)
    steps = (
        (EXACT, lambda: _exact(index, value)),
        (LOOSE, lambda: _loose(index, value)),
        (NORMALISED, lambda: _normalised(index, value, fdef, rule_cands, norm_opts)),
    )
    for outcome, find in steps:
        matches = find()
        if not matches:
            continue
        # rule_cands arrive ranked best first (candidates_for sorts them), so
        # a value printed twice lands where the strongest rule put it, which
        # is the occurrence the baseline itself would have emitted.
        for c in rule_cands:
            for s, e in matches:
                if _agrees(c, s, e, index, fdef, norm_opts):
                    return Grounding(outcome, s, e, len(matches), True)
        s, e = matches[0]
        return Grounding(outcome, s, e, len(matches), False)
    return Grounding(UNGROUNDED)
