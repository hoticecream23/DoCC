"""Candidate generation, and `extract_fields`, which runs every configured field.

Three strategies per field, applied in the order listed in fields.yaml,
first hit wins. Offsets are always against the canonical text.
"""

# Over 400 lines: the candidate strategies and their dispatcher must agree on offsets, so they stay together.
from __future__ import annotations

import functools
import re
from dataclasses import dataclass
from typing import Any

from ..config import Config
from ..errors import OffsetMismatch  # noqa: F401 - re-exported
from ..logging_setup import get_logger
from ..schema import MetadataField, MetadataMethod, WordBox, page_for_offset, page_spans
from .normalizers import NormCtx, normalize
from .validators import CHECKSUM_VALIDATORS, run_validator

log = get_logger(__name__)

# Defaults used when fields.yaml or the metadata options leave a value out.
DEFAULT_MAX_CANDIDATES = 25
DEFAULT_ANCHOR_WINDOW = 120
DEFAULT_SPECIFICITY = 0.5
DEFAULT_POSITIONAL_CONFIDENCE = 0.85
# Distance from the label erodes anchor confidence by up to this fraction.
ANCHOR_DISTANCE_DECAY = 0.4
# Anchor confidence never exceeds this without a passing validator.
ANCHOR_MAX_CONFIDENCE = 0.95
# A passing validator lifts anchor confidence by this much.
ANCHOR_VALIDATED_BONUS = 0.05

# (start, end, value, distance from the label) for one anchor value match.
_AnchorHit = tuple[int, int, str, int]
# (words, page spans, page sizes) for positional templates.
_Geometry = tuple[list[WordBox], list[tuple[int, int]], list[tuple[float, float]]]


@dataclass
class Candidate:
    """One possible value for a field, with offsets into the canonical text.

    Attributes:
        value: Raw matched text, equal to ``full[start:end]``.
        start: Start offset in the canonical text.
        end: End offset in the canonical text.
        confidence: Score in [0, 1].
        method: Strategy that produced the candidate.
        validated: True if a validator passed on the value.
    """

    value: str
    start: int
    end: int
    confidence: float
    method: MetadataMethod
    validated: bool


@dataclass
class _ExtractOpts:
    """Options `extract_fields` reads once from the config."""

    limit: int
    default_window: int
    strict: bool
    unknown_label: str
    run_all_when_unknown: bool
    drop_bad: bool
    norm_opts: dict[str, Any]
    nver: str


@functools.lru_cache(maxsize=512)
def _compile(pattern: str, ignorecase: bool = True) -> re.Pattern[str]:
    flags = re.IGNORECASE if ignorecase else 0
    return re.compile(pattern, flags)


def _span_of(match: re.Match[str]) -> tuple[int, int, str]:
    """Prefer group 1 as the value. Falls back to the whole match."""
    if match.lastindex:
        return match.start(1), match.end(1), match.group(1)
    return match.start(), match.end(), match.group(0)


def _pattern_candidates(full: str, fdef: dict[str, Any], limit: int) -> list[Candidate]:
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
    for match in _compile(rx, ignorecase).finditer(full):
        s, e, val = _span_of(match)
        ok = run_validator(validator, val, args)
        if hard and not ok:
            # A failed checksum is strong evidence, not weak. On the client
            # corpus every Aadhaar and card candidate that failed one was OCR
            # noise, and emitting it asserted PII the text does not hold.
            continue
        out.append(
            Candidate(
                value=val,
                start=s,
                end=e,
                confidence=c_ok if ok else c_no,
                # Checksum backed and format only are different things.
                method=(MetadataMethod.REGEX_CHECKSUM if ok and hard else MetadataMethod.REGEX_FORMAT),
                validated=ok,
            )
        )
        if len(out) >= limit:
            break
    return out


def _label_rejected(full: str, low: str, pos: int, label_end: int, spec: dict[str, Any]) -> bool:
    """True if the label's own line says it is a different quantity.

    Typical rejections are "sub" before "total" and "qty" after it.
    """
    rej_before = spec.get("reject_before")
    rej_after = spec.get("reject_after")
    line_start = full.rfind("\n", 0, pos) + 1
    line_end = full.find("\n", label_end)
    if line_end < 0:
        line_end = len(full)
    return bool(
        (rej_before and _compile(rej_before).search(low[line_start:pos]))
        or (rej_after and _compile(rej_after).match(low[label_end:line_end]))
    )


def _anchor_hits(
    full: str, value_rx: re.Pattern[str], pos: int, label_end: int, window: int, direction: str
) -> list[_AnchorHit]:
    """Value matches near one label hit, after it and/or before it."""
    hits: list[_AnchorHit] = []
    if direction in ("after", "both"):
        match = value_rx.search(full[label_end : label_end + window])
        if match:
            s, e, val = _span_of(match)
            hits.append((label_end + s, label_end + e, val, s))
    if direction in ("before", "both"):
        seg_start = max(0, pos - window)
        matches = list(value_rx.finditer(full[seg_start:pos]))
        if matches:
            s, e, val = _span_of(matches[-1])  # nearest to the label
            hits.append((seg_start + s, seg_start + e, val, pos - (seg_start + e)))
    return hits


def _anchor_candidate(hit: _AnchorHit, spec: dict[str, Any], spec_score: float, window: int) -> Candidate | None:
    """Score one anchor hit, or None if its value is another field's value."""
    start, end, val, dist = hit
    # A value that is wholly another field's value (a date under a blank
    # invoice number) is refused.
    rej_value = spec.get("reject_value")
    if rej_value and _compile(rej_value, False).fullmatch(val):
        return None
    validator = spec.get("validator")
    ok = run_validator(validator, val, spec.get("validator_args") or {}) if validator else False
    # Specificity sets the ceiling, distance erodes it.
    decay = 1.0 - ANCHOR_DISTANCE_DECAY * min(1.0, dist / max(1, window))
    conf = min(ANCHOR_MAX_CONFIDENCE, spec_score * decay)
    if ok:
        conf = min(1.0, conf + ANCHOR_VALIDATED_BONUS)
    return Candidate(
        value=val, start=start, end=end, confidence=round(conf, 4), method=MetadataMethod.ANCHOR, validated=ok
    )


def _anchor_candidates(full: str, fdef: dict[str, Any], default_window: int, limit: int) -> list[Candidate]:
    spec = fdef.get("anchor") or {}
    labels = spec.get("labels") or []
    value_regex = spec.get("value_regex")
    if not labels or not value_regex:
        return []
    window = int(spec.get("window", default_window))
    direction = spec.get("direction", "after")
    value_rx = _compile(value_regex, bool(spec.get("ignorecase", False)))
    low = full.lower()

    out: list[Candidate] = []
    # Most specific labels first so a weak label never shadows a strong one.
    for label in sorted(labels, key=lambda item: -float(item.get("specificity", DEFAULT_SPECIFICITY))):
        text = str(label.get("text", "")).lower()
        if not text:
            continue
        spec_score = float(label.get("specificity", DEFAULT_SPECIFICITY))
        pos = low.find(text)
        while pos >= 0:
            label_end = pos + len(text)
            if not _label_rejected(full, low, pos, label_end, spec):
                for hit in _anchor_hits(full, value_rx, pos, label_end, window, direction):
                    cand = _anchor_candidate(hit, spec, spec_score, window)
                    if cand is not None:
                        out.append(cand)
                if len(out) >= limit:
                    return out
            pos = low.find(text, pos + 1)
    return out


def _region_offsets(
    full: str, words: list[WordBox], region: dict[str, Any], page_sizes: list[tuple[float, float]]
) -> tuple[int, int] | None:
    """Character span of the words inside a template region, or None."""
    pno = int(region.get("page", 0))
    if pno >= len(page_sizes):
        return None
    pw, ph = page_sizes[pno]
    if pw <= 0 or ph <= 0:
        return None
    x0, y0, x1, y1 = (float(v) for v in region["region"])
    inside = [
        (w.char_start, w.char_end)
        for w in words
        if w.page == pno
        and w.char_start is not None
        and w.char_end is not None
        and x0 <= (w.x0 / pw) <= x1
        and y0 <= (w.y0 / ph) <= y1
    ]
    if not inside:
        return None
    start = min(s for s, _ in inside)
    end = max(e for _, e in inside)
    if start >= end or end > len(full):
        return None
    return start, end


def _positional_candidates(
    full: str,
    fdef: dict[str, Any],
    templates: list[dict[str, Any]],
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
        rx = (tpl.get("match") or {}).get("regex")
        if rx and not _compile(rx).search(full):
            continue
        region = (tpl.get("fields") or {}).get(name)
        if not region:
            continue
        offsets = _region_offsets(full, words, region, page_sizes)
        if offsets is None:
            continue
        start, end = offsets
        # Slice the canonical text so the offsets are correct by construction.
        return [
            Candidate(
                value=full[start:end],
                start=start,
                end=end,
                confidence=float(tpl.get("confidence", DEFAULT_POSITIONAL_CONFIDENCE)),
                method=MetadataMethod.POSITIONAL,
                validated=False,
            )
        ]
    return []


def _rank(cand: Candidate) -> tuple[int, float, int, int]:
    # Validated wins, then confidence, then earliest position for stability.
    return (0 if cand.validated else 1, -cand.confidence, cand.start, cand.end)


def candidates_for(
    full: str,
    pages: list[str],
    fdef: dict[str, Any],
    cfg: Config,
    doc_class: str = "",
    words: list[WordBox] | None = None,
    page_sizes: list[tuple[float, float]] | None = None,
) -> list[Candidate]:
    """Every strategy's candidates for one field, ranked.

    extract_fields stops at the first strategy that hits. Cascade asks a
    different question, whether any rule would have produced a given span,
    and that needs all of them. extract_fields does not call this, so the
    baseline's own output cannot change because of it.

    Args:
        full: Canonical document text.
        pages: Per-page text, used for page spans.
        fdef: One field definition from fields.yaml.
        cfg: Loaded configuration.
        doc_class: Document class, used by positional templates.
        words: Word boxes with offsets, for positional templates.
        page_sizes: (width, height) per page, for positional templates.

    Returns:
        All candidates from every listed strategy, best first.
    """
    opts = cfg.metadata_opts()
    limit = int(opts.get("max_candidates_per_field", DEFAULT_MAX_CANDIDATES))
    default_window = int(opts.get("default_anchor_window", DEFAULT_ANCHOR_WINDOW))
    out: list[Candidate] = []
    for strategy in fdef.get("strategies", []):
        if strategy == "pattern":
            out.extend(_pattern_candidates(full, fdef, limit))
        elif strategy == "anchor":
            out.extend(_anchor_candidates(full, fdef, default_window, limit))
        elif strategy == "positional":
            spans = page_spans(pages)
            out += _positional_candidates(full, fdef, cfg.templates, doc_class, words or [], spans, page_sizes or [])
    out.sort(key=_rank)
    return out


def scan_spans(full: str, fdef: dict[str, Any]) -> list[tuple[int, int, str]]:
    """Every place a value of this field's shape is printed, label or not.

    Runs the field's own `pattern.regex` and `anchor.value_regex` over the
    whole text, with no anchor label and no validator. These are not claims
    that a match is this field. Cascade uses them only to locate a value the
    model has already chosen, when the model wrote it in another format.

    Args:
        full: Canonical document text.
        fdef: One field definition from fields.yaml.

    Returns:
        Sorted unique (start, end, value) spans.
    """
    out: set[tuple[int, int, str]] = set()
    for spec, key in ((fdef.get("pattern") or {}, "regex"), (fdef.get("anchor") or {}, "value_regex")):
        rx = spec.get(key)
        if not rx:
            continue
        for match in _compile(rx, bool(spec.get("ignorecase", False))).finditer(full):
            s, e, val = _span_of(match)
            if 0 <= s < e:
                out.add((s, e, val))
    return sorted(out)


def _extract_opts(cfg: Config) -> _ExtractOpts:
    opts = cfg.metadata_opts()
    return _ExtractOpts(
        limit=int(opts.get("max_candidates_per_field", DEFAULT_MAX_CANDIDATES)),
        default_window=int(opts.get("default_anchor_window", DEFAULT_ANCHOR_WINDOW)),
        strict=bool(opts.get("strict_offsets", False)),
        unknown_label=str(cfg.classify_opts().get("unknown_label", "unknown")),
        run_all_when_unknown=bool(opts.get("run_scoped_fields_when_unknown", True)),
        drop_bad=bool(opts.get("drop_on_offset_mismatch", True)),
        norm_opts=cfg.fields.get("normalizer_options", {}) or {},
        nver=cfg.normalizer_version,
    )


def _field_in_scope(fdef: dict[str, Any], doc_class: str, eopts: _ExtractOpts) -> bool:
    # A field may be scoped to certain classes. Unknown docs still get
    # everything, otherwise we lose recall exactly where we need it.
    scope = fdef.get("classes")
    if scope and doc_class not in scope:
        return doc_class in ("", eopts.unknown_label) and eopts.run_all_when_unknown
    return True


def _first_hit_candidates(
    full: str,
    fdef: dict[str, Any],
    cfg: Config,
    doc_class: str,
    geometry: _Geometry,
    eopts: _ExtractOpts,
    errors: list[str],
) -> list[Candidate]:
    """Candidates from the first strategy that hits. Unknown strategies go to errors."""
    words, spans, page_sizes = geometry
    name = fdef.get("name")
    cands: list[Candidate] = []
    for strategy in fdef.get("strategies", []):
        if strategy == "pattern":
            cands = _pattern_candidates(full, fdef, eopts.limit)
        elif strategy == "anchor":
            cands = _anchor_candidates(full, fdef, eopts.default_window, eopts.limit)
        elif strategy == "positional":
            cands = _positional_candidates(full, fdef, cfg.templates, doc_class, words, spans, page_sizes)
        else:
            errors.append(f"{name}: unknown strategy {strategy!r}")
            continue
        if cands:
            break  # first strategy that hits wins
    return cands


def _emit_fields(
    full: str,
    fdef: dict[str, Any],
    cands: list[Candidate],
    spans: list[tuple[int, int]],
    eopts: _ExtractOpts,
    errors: list[str],
) -> list[MetadataField]:
    """Normalize and offset-check the kept candidates of one field.

    Raises:
        OffsetMismatch: If offsets do not slice back to the value and strict offsets are on.
    """
    name: str = fdef["name"]  # extract_fields skips nameless fields
    cands.sort(key=_rank)
    keep = cands if fdef.get("multi") else cands[:1]
    results: list[MetadataField] = []
    seen: set[str] = set()
    for cand in keep:
        ctx = NormCtx(full=full, start=cand.start, end=cand.end, opts=eopts.norm_opts)
        normalized, extras = normalize(fdef.get("normalizer"), cand.value, ctx)
        # Contract check: the offsets must slice back to the raw value.
        if full[cand.start : cand.end] != cand.value:
            msg = f"{name}: offset mismatch at {cand.start}:{cand.end}"
            if eopts.strict:
                raise OffsetMismatch(msg)
            log.error("offset mismatch", extra={"field": name, "start": cand.start})
            errors.append(msg)
            if eopts.drop_bad:
                continue
        key = normalized if normalized is not None else cand.value
        if fdef.get("multi") and key in seen:
            continue
        seen.add(key)
        results.append(
            MetadataField(
                field=name,
                value=cand.value,
                normalized_value=normalized,
                page=page_for_offset(spans, cand.start),
                char_start=cand.start,
                char_end=cand.end,
                confidence=round(cand.confidence, 4),
                method=cand.method,
                validated=cand.validated,
                normalizer_version=eopts.nver,
                currency=extras.get("currency"),
            )
        )
    return results


def extract_fields(
    full: str,
    pages: list[str],
    cfg: Config,
    doc_class: str = "",
    words: list[WordBox] | None = None,
    page_sizes: list[tuple[float, float]] | None = None,
) -> tuple[list[MetadataField], list[str]]:
    """Run every configured field.

    Args:
        full: Canonical document text.
        pages: Per-page text, used for page spans.
        cfg: Loaded configuration.
        doc_class: Document class, used for field scoping and templates.
        words: Word boxes with offsets, for positional templates.
        page_sizes: (width, height) per page, for positional templates.

    Returns:
        (fields sorted by offset then name, error messages).

    Raises:
        OffsetMismatch: If a candidate's offsets are wrong and strict offsets are on.
    """
    eopts = _extract_opts(cfg)
    spans = page_spans(pages)
    geometry: _Geometry = (words or [], spans, page_sizes or [])
    results: list[MetadataField] = []
    errors: list[str] = []
    for fdef in cfg.field_defs:
        if not fdef.get("name") or not _field_in_scope(fdef, doc_class, eopts):
            continue
        cands = _first_hit_candidates(full, fdef, cfg, doc_class, geometry, eopts, errors)
        if cands:
            results.extend(_emit_fields(full, fdef, cands, spans, eopts, errors))
    results.sort(key=lambda field: (field.char_start, field.field))
    return results, errors
