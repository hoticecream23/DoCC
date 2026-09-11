"""Candidate generation, and `extract_fields`, which runs every configured field.

Three strategies per field, applied in the order listed in fields.yaml,
first hit wins. Offsets are always against the canonical text.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass

from ..config import Config
from ..logging_setup import get_logger
from ..schema import MetadataField, MetadataMethod, WordBox, page_for_offset, page_spans
from .normalizers import NormCtx, normalize
from .validators import CHECKSUM_VALIDATORS, run_validator

log = get_logger(__name__)


class OffsetMismatch(ValueError):
    """Raised in strict mode when an offset does not slice back to its value."""


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
