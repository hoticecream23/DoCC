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
from ..tables.geometry import _group_lines, _overlap_ratio
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


def _layout_value(
    full: str,
    words: list[WordBox],
    pos: int,
    label_end: int,
    how: str,
    vrx: re.Pattern,
    rej_value: str | None,
) -> tuple[int, int, str] | None:
    """The value printed right of, or on the line below, the label at [pos, label_end).

    Only words whose offsets slice back to their own text are used, so a
    mis-mapped native PDF word can never give a wrong span.
    """
    good = [
        w for w in words
        if w.char_start is not None and w.char_end is not None
        and full[w.char_start : w.char_end] == w.text
    ]
    lab = [w for w in good if w.char_end > pos and w.char_start < label_end]
    if not lab:
        return None
    page = lab[0].page
    lab = [w for w in lab if w.page == page]
    lx0, lx1 = min(w.x0 for w in lab), max(w.x1 for w in lab)
    ly0, ly1 = min(w.y0 for w in lab), max(w.y1 for w in lab)
    page_words = [w for w in good if w.page == page and w.char_start >= label_end]

    if how == "right_of_label":
        # ponytail: 0.5 y overlap and a 2pt slack on x; per-font tolerance if
        # tight receipts mis-pair rows.
        cands = sorted(
            (w for w in page_words
             if w.x0 >= lx1 - 2 and _overlap_ratio(ly0, ly1, w.y0, w.y1) >= 0.5),
            key=lambda w: w.x0,
        )
    elif how == "below_label":
        dicts = [
            {"x0": w.x0, "y0": w.y0, "x1": w.x1, "y1": w.y1, "text": w.text, "w": w}
            for w in good if w.page == page
        ]
        # ponytail: 0.5 overlap groups lines, as the table code does.
        lines = _group_lines(dicts, 0.5)
        idx = next(
            (i for i, ln in enumerate(lines) if any(d["w"] is lab[0] for d in ln.words)),
            None,
        )
        if idx is None or idx + 1 >= len(lines):
            return None
        # Only the next line, and only words under the label's x span.
        cands = [
            d["w"] for d in lines[idx + 1].words
            if _overlap_ratio(lx0, lx1, d["x0"], d["x1"]) > 0
            and d["w"].char_start >= label_end
        ]
    else:
        return None

    # ponytail: one word, not a run; join neighbours if values split on spaces.
    for w in cands:
        m = vrx.search(w.text)
        if not m:
            continue
        s, e, val = _span_of(m)
        if s >= e or (rej_value and _compile(rej_value, False).fullmatch(val)):
            continue
        start, end = w.char_start + s, w.char_start + e
        return start, end, full[start:end]
    return None


def _anchor_candidates(
    full: str,
    fdef: dict,
    default_window: int,
    limit: int,
    words: list[WordBox] | None = None,
) -> list[Candidate]:
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
    # A label hit is refused when its own line says it is a different
    # quantity: "sub" before "total", "qty" after it.
    rej_before = spec.get("reject_before")
    rej_after = spec.get("reject_after")
    # A value that is wholly another field's value (a date under a blank
    # invoice number) is refused.
    rej_value = spec.get("reject_value")
    low = full.lower()
    layout = spec.get("layout_direction")

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
            line_start = full.rfind("\n", 0, pos) + 1
            line_end = full.find("\n", label_end)
            if line_end < 0:
                line_end = len(full)
            if (rej_before and _compile(rej_before).search(low[line_start:pos])) or (
                rej_after and _compile(rej_after).match(low[label_end:line_end])
            ):
                pos = low.find(text, pos + 1)
                continue
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
                if rej_value and _compile(rej_value, False).fullmatch(val):
                    continue
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
    if out or not layout or not words:
        return out

    # The text rule found nothing: pair each label with the value printed by
    # it on the page. Same reject rules, validator and confidence as above,
    # with the distance taken as half the window.
    for lab in sorted(labels, key=lambda l: -float(l.get("specificity", 0.5))):
        text = str(lab.get("text", "")).lower()
        if not text:
            continue
        spec_score = float(lab.get("specificity", 0.5))
        pos = low.find(text)
        while pos >= 0:
            label_end = pos + len(text)
            line_start = full.rfind("\n", 0, pos) + 1
            line_end = full.find("\n", label_end)
            if line_end < 0:
                line_end = len(full)
            rejected = (rej_before and _compile(rej_before).search(low[line_start:pos])) or (
                rej_after and _compile(rej_after).match(low[label_end:line_end])
            )
            hit = None if rejected else _layout_value(
                full, words, pos, label_end, layout, vrx, rej_value
            )
            if hit:
                start, end, val = hit
                ok = run_validator(validator, val, args) if validator else False
                # ponytail: fixed 0.8 decay for a box pairing; scale by box
                # distance if confidence needs to rank layout hits.
                conf = min(0.95, spec_score * 0.8)
                if ok:
                    conf = min(1.0, conf + 0.05)
                out.append(
                    Candidate(
                        value=val, start=start, end=end, confidence=round(conf, 4),
                        method=MetadataMethod.ANCHOR, validated=ok,
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


def candidates_for(
    full: str,
    pages: list[str],
    fdef: dict,
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
    """
    opts = cfg.metadata_opts()
    limit = int(opts.get("max_candidates_per_field", 25))
    default_window = int(opts.get("default_anchor_window", 120))
    out: list[Candidate] = []
    for strategy in fdef.get("strategies", []):
        if strategy == "pattern":
            out.extend(_pattern_candidates(full, fdef, limit))
        elif strategy == "anchor":
            out.extend(_anchor_candidates(full, fdef, default_window, limit, words))
        elif strategy == "positional":
            out.extend(
                _positional_candidates(
                    full, fdef, cfg.templates, doc_class, words or [],
                    page_spans(pages), page_sizes or [],
                )
            )
    out.sort(key=_rank)
    return out


def scan_spans(full: str, fdef: dict) -> list[tuple[int, int, str]]:
    """Every place a value of this field's shape is printed, label or not.

    Runs the field's own `pattern.regex` and `anchor.value_regex` over the
    whole text, with no anchor label and no validator. These are not claims
    that a match is this field. Cascade uses them only to locate a value the
    model has already chosen, when the model wrote it in another format.
    """
    out: set[tuple[int, int, str]] = set()
    for spec, key in ((fdef.get("pattern") or {}, "regex"), (fdef.get("anchor") or {}, "value_regex")):
        rx = spec.get(key)
        if not rx:
            continue
        for m in _compile(rx, bool(spec.get("ignorecase", False))).finditer(full):
            s, e, val = _span_of(m)
            if 0 <= s < e:
                out.add((s, e, val))
    return sorted(out)


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
                cands = _anchor_candidates(full, fdef, default_window, limit, words)
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
