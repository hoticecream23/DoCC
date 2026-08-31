"""Shared marker matching for the classification and tagging rule layers.

Same machinery, different vocabulary. Both read their patterns from YAML.
"""

from __future__ import annotations

import functools
import re

# Scope names a marker may declare.
SCOPE_FIRST_PAGE = "first_page"
SCOPE_HEADER = "header"
SCOPE_ANYWHERE = "anywhere"


@functools.lru_cache(maxsize=1024)
def _compile(pattern: str, ignorecase: bool = True) -> re.Pattern:
    return re.compile(pattern, re.IGNORECASE if ignorecase else 0)


@functools.lru_cache(maxsize=4096)
def _word_pattern(term: str) -> re.Pattern:
    """Whole word match. Falls back to substring for multi word phrases."""
    return re.compile(r"(?<!\w)" + re.escape(term) + r"(?!\w)", re.IGNORECASE)


def scope_text(scope: str, full: str, pages: list[str], header_chars: int) -> str:
    """The slice of the document a marker is allowed to look at."""
    if scope == SCOPE_FIRST_PAGE:
        return pages[0] if pages else ""
    if scope == SCOPE_HEADER:
        return (pages[0] if pages else "")[:header_chars]
    return full


def match_marker(marker: dict, full: str, pages: list[str], header_chars: int) -> bool:
    """True if this marker fires. Unknown kinds simply never fire."""
    text = scope_text(marker.get("scope", SCOPE_ANYWHERE), full, pages, header_chars)
    if not text:
        return False
    kind = marker.get("kind")

    if kind == "regex":
        pat = marker.get("pattern")
        if not pat:
            return False
        return bool(_compile(pat, not marker.get("case_sensitive", False)).search(text))

    if kind == "keywords":
        need_all = marker.get("all") or []
        need_any = marker.get("any") or []
        if not need_all and not need_any:
            return False
        if any(not _word_pattern(str(t)).search(text) for t in need_all):
            return False
        if need_any and not any(_word_pattern(str(t)).search(text) for t in need_any):
            return False
        return True

    return False


def best_marker_score(markers: list, full: str, pages: list[str], header_chars: int) -> float:
    """Highest specificity among the markers that fire. Zero if none do."""
    best = 0.0
    for m in markers or []:
        if match_marker(m, full, pages, header_chars):
            best = max(best, float(m.get("specificity", 0.5)))
    return best
