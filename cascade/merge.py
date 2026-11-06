"""Turn a grounded model answer into the record's class and fields.

The rule and checksum layer is a hard prior over the model:

- a value whose checksum passed is never removed or replaced, and a model
  value for a checksummed field is emitted only if it passes the same check;
- `validated` is true only when a checksum passed, never on a format check;
- a rule at short circuit specificity decides the class.

A rule value the model did not confirm is not emitted. Anchor rules are the
false positive problem this approach exists to fix, and every such drop is
recorded as `rule_only` so the choice is measured, not argued.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, NamedTuple, TypedDict

from baseline.config import Config
from baseline.metadata import CHECKSUM_VALIDATORS, NormCtx, normalize
from baseline.metadata.fields import Candidate, candidates_for
from baseline.schema import (
    Classification,
    ClassifyMethod,
    MetadataField,
    MetadataMethod,
    WordBox,
    page_for_offset,
)

from .config import ConfidenceTiers
from .ground import EXACT, LOOSE, NORMALISED, UNGROUNDED, Grounding, TextIndex, ground
from .prompt import VLMAnswer

# Shape checks for model-only identifiers (see _shape_ok).
ACCOUNT_NUMBER_MIN_DIGITS = 9
ACCOUNT_NUMBER_MAX_DIGITS = 18
INVOICE_NUMBER_MAX_CHARS = 30


class ClassProvenance(TypedDict):
    """Why the class was chosen."""

    source: str
    label: str
    vlm_label: str
    disagrees: bool


class ValueProvenance(TypedDict):
    """What happened to one value the model returned."""

    field: str
    vlm_value: str
    outcome: str
    n_matches: int
    rule_agrees: bool
    char_start: int | None
    char_end: int | None
    emitted: bool


class RuleOnlyProvenance(TypedDict):
    """A field the rules found but the model did not confirm, so nothing was emitted."""

    field: str
    outcome: str
    baseline_values: list[str]
    emitted: bool


FieldProvenance = ValueProvenance | RuleOnlyProvenance


@dataclass
class Decision:
    """The record's class and fields, with the provenance of each choice."""

    classification: Classification
    metadata: list[MetadataField]
    class_provenance: ClassProvenance
    field_provenance: list[FieldProvenance] = field(default_factory=list)


@dataclass(frozen=True)
class _DocContext:
    """Per document inputs every field decision reads."""

    full: str
    pages: list[str]
    cfg: Config
    tiers: ConfidenceTiers
    doc_class: str
    answer: VLMAnswer
    index: TextIndex
    norm_opts: dict[str, Any]
    words: list[WordBox] | None
    page_sizes: list[tuple[float, float]] | None


class _Trust(NamedTuple):
    method: MetadataMethod
    confidence: float
    validated: bool


def _unknown(cfg: Config) -> str:
    return str(cfg.classify_opts().get("unknown_label", "unknown"))


def decide_class(
    rules: dict[str, float], answer: VLMAnswer, cfg: Config, tiers: ConfidenceTiers
) -> tuple[Classification, ClassProvenance]:
    """Choose the class: a short circuit rule wins, otherwise the model's label.

    Args:
        rules: Rule score per class.
        answer: The model's checked answer.
        cfg: The loaded config.
        tiers: Confidence tiers from cascade.yaml.

    Returns:
        The classification and why it was chosen.
    """
    unknown = _unknown(cfg)
    short_circuit = float(cfg.classify_opts().get("rule_short_circuit", 0.9))
    all_scores = dict(sorted(rules.items()))
    if rules:
        # Ties break on name so the output is stable, as in the baseline.
        label, score = sorted(rules.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        if score >= short_circuit:
            prov = ClassProvenance(
                source="rule", label=label, vlm_label=answer.label, disagrees=answer.label not in (label, unknown)
            )
            return (
                Classification(
                    label=label, confidence=round(score, 4), all_scores=all_scores, method=ClassifyMethod.RULE
                ),
                prov,
            )
    confidence = 0.0 if answer.label == unknown else tiers.vlm_class
    prov = ClassProvenance(source="vlm", label=answer.label, vlm_label=answer.label, disagrees=False)
    return (
        Classification(label=answer.label, confidence=confidence, all_scores=all_scores, method=ClassifyMethod.VLM),
        prov,
    )


def _in_scope(fdef: dict[str, Any], doc_class: str, cfg: Config) -> bool:
    """The same scoping rule extract_fields applies."""
    scope = fdef.get("classes")
    if not scope or doc_class in scope:
        return True
    run_all = bool(cfg.metadata_opts().get("run_scoped_fields_when_unknown", True))
    return doc_class in ("", _unknown(cfg)) and run_all


_GROUND_RANK = {EXACT: 0, LOOSE: 1, NORMALISED: 2, UNGROUNDED: 3}
_OTHER_IDS = ("gstin", "pan", "ifsc", "phone")


def _alnum(s: str) -> str:
    return re.sub(r"[\W_]", "", s).upper()


def _shape_ok(name: str, raw: str, other_ids: set[str]) -> bool:
    """A model-only identifier must look like one; a rule-agreeing value skips this.

    ponytail: hard-coded for the two fields the audit saw confused; move to
    fields.yaml if a third field needs it.
    """
    if name == "account_number":
        digits = re.sub(r"\D", "", raw)
        return (
            not re.search(r"[A-Za-z]", raw)
            and ACCOUNT_NUMBER_MIN_DIGITS <= len(digits) <= ACCOUNT_NUMBER_MAX_DIGITS
            and _alnum(raw) not in other_ids
        )
    if name == "invoice_number":
        return bool(re.search(r"\d", raw)) and len(raw) <= INVOICE_NUMBER_MAX_CHARS
    return True


def _key(normalized: str | None, raw: str) -> str:
    return normalized if normalized is not None else raw


def _ranked(
    ctx: _DocContext, fdef: dict[str, Any], name: str, rule_cands: list[Candidate], multi: bool
) -> list[tuple[str, Grounding]]:
    """Ground every model value for the field; for a single value field, best first."""
    tried = [(v, ground(v, ctx.index, fdef, rule_cands, ctx.norm_opts)) for v in ctx.answer.fields.get(name, [])]
    if not multi:
        # One value is kept, so try the best first: rule agreement, then the
        # strongest grounding (exact, loose, normalised; one match over many),
        # then the model's order (the sort is stable).
        tried.sort(key=lambda vg: (not vg[1].rule_agrees, _GROUND_RANK[vg[1].outcome], vg[1].n_matches > 1))
    return tried


def _trust(
    ctx: _DocContext,
    name: str,
    g: Grounding,
    raw: str,
    rule_cands: list[Candidate],
    checksummed: bool,
    emitted: list[MetadataField],
) -> _Trust | str:
    """How far a grounded value is trusted, or the outcome that rejects it."""
    if checksummed:
        # A model value for a checksummed field is proven only if its span
        # matches a rule candidate with REGEX_CHECKSUM method (which is the
        # baseline's definition of proven).
        matches_proven = any(
            c.start == g.start and c.end == g.end and c.method is MetadataMethod.REGEX_CHECKSUM for c in rule_cands
        )
        if not matches_proven:
            return "checksum_failed"
        return _Trust(MetadataMethod.REGEX_CHECKSUM, 1.0, True)
    if g.rule_agrees:
        return _Trust(MetadataMethod.VLM_RULE_AGREE, ctx.tiers.agree, False)
    other_ids = {_alnum(m.value) for m in emitted if m.field in _OTHER_IDS}
    other_ids |= {_alnum(v) for f in _OTHER_IDS for v in ctx.answer.fields.get(f, [])}
    if not _shape_ok(name, raw, other_ids):
        return "shape_failed"
    return _Trust(MetadataMethod.VLM, ctx.tiers.ambiguous if g.n_matches > 1 else ctx.tiers.grounded, False)


def _build_field(
    ctx: _DocContext, fdef: dict[str, Any], name: str, start: int, end: int, trust: _Trust
) -> MetadataField:
    """The record field for an accepted span, normalised by the field's normalizer."""
    raw = ctx.full[start:end]
    normalized: str | None  # normalize() returns a bare tuple
    normalized, extras = normalize(
        fdef.get("normalizer"), raw, NormCtx(full=ctx.full, start=start, end=end, opts=ctx.norm_opts)
    )
    return MetadataField(
        field=name,
        value=raw,
        normalized_value=normalized,
        page=page_for_offset(ctx.index.spans, start),
        char_start=start,
        char_end=end,
        confidence=round(trust.confidence, 4),
        method=trust.method,
        validated=trust.validated,
        normalizer_version=ctx.cfg.normalizer_version,
        currency=extras.get("currency"),
    )


def _value_provenance(name: str, value: str, g: Grounding) -> ValueProvenance:
    return ValueProvenance(
        field=name,
        vlm_value=value,
        outcome=g.outcome,
        n_matches=g.n_matches,
        rule_agrees=g.rule_agrees,
        char_start=g.start,
        char_end=g.end,
        emitted=False,
    )


def _decide_field(
    ctx: _DocContext,
    fdef: dict[str, Any],
    name: str,
    baseline_values: list[MetadataField],
    emitted: list[MetadataField],
) -> tuple[list[MetadataField], list[FieldProvenance]]:
    """Fields to emit for one field definition, and their provenance."""
    checksummed = (fdef.get("pattern") or {}).get("validator") in CHECKSUM_VALIDATORS
    # Proven baseline values are kept unconditionally, before the scope check.
    kept = [m for m in baseline_values if m.method is MetadataMethod.REGEX_CHECKSUM] if checksummed else []
    seen = {_key(m.normalized_value, m.value) for m in kept}
    if not _in_scope(fdef, ctx.doc_class, ctx.cfg):
        # No provenance and no model values out of scope; proven values still go out.
        return kept, []

    rule_cands = candidates_for(ctx.full, ctx.pages, fdef, ctx.cfg, ctx.doc_class, ctx.words, ctx.page_sizes)
    multi = bool(fdef.get("multi"))
    prov: list[FieldProvenance] = []
    model_emitted = False
    for value, g in _ranked(ctx, fdef, name, rule_cands, multi):
        if kept and not multi:
            break
        item = _value_provenance(name, value, g)
        prov.append(item)
        if g.start is None or g.end is None:  # ungrounded
            continue
        trust = _trust(ctx, name, g, ctx.full[g.start : g.end], rule_cands, checksummed, emitted)
        if isinstance(trust, str):
            item["outcome"] = trust
            continue
        candidate = _build_field(ctx, fdef, name, g.start, g.end, trust)
        key = _key(candidate.normalized_value, candidate.value)
        if key in seen:
            continue
        seen.add(key)
        item["emitted"] = model_emitted = True
        kept.append(candidate)

    if not checksummed and not model_emitted and baseline_values:
        prov.append(
            RuleOnlyProvenance(
                field=name, outcome="rule_only", baseline_values=[m.value for m in baseline_values], emitted=False
            )
        )
    return kept, prov


def decide_fields(
    full: str,
    pages: list[str],
    cfg: Config,
    tiers: ConfidenceTiers,
    doc_class: str,
    answer: VLMAnswer,
    baseline_meta: list[MetadataField],
    words: list[WordBox] | None = None,
    page_sizes: list[tuple[float, float]] | None = None,
) -> tuple[list[MetadataField], list[FieldProvenance]]:
    """Choose the fields to emit from the grounded model values and the baseline.

    Args:
        full: The document's full text.
        pages: Text per page.
        cfg: The loaded config.
        tiers: Confidence tiers from cascade.yaml.
        doc_class: The class already decided.
        answer: The model's checked answer.
        baseline_meta: The baseline's fields for this document.
        words: Word boxes, for layout rules.
        page_sizes: Page sizes, for layout rules.

    Returns:
        The fields sorted by offset, and one provenance item per value tried.
    """
    norm_opts = cfg.fields.get("normalizer_options", {}) or {}
    index = TextIndex(full, pages)
    ctx = _DocContext(full, pages, cfg, tiers, doc_class, answer, index, norm_opts, words, page_sizes)
    from_baseline: dict[str, list[MetadataField]] = defaultdict(list)
    for m in baseline_meta:
        from_baseline[m.field].append(m)

    out: list[MetadataField] = []
    prov: list[FieldProvenance] = []
    for fdef in cfg.field_defs:
        name = fdef.get("name")
        if not name:
            continue
        kept, field_prov = _decide_field(ctx, fdef, name, from_baseline[name], out)
        out.extend(kept)
        prov.extend(field_prov)

    out.sort(key=lambda m: (m.char_start, m.field))
    return out, prov


def decide(
    full: str,
    pages: list[str],
    cfg: Config,
    tiers: ConfidenceTiers,
    rules: dict[str, float],
    answer: VLMAnswer,
    baseline_meta: list[MetadataField],
    words: list[WordBox] | None = None,
    page_sizes: list[tuple[float, float]] | None = None,
) -> Decision:
    """Decide the class, then the fields for that class.

    Args:
        full: The document's full text.
        pages: Text per page.
        cfg: The loaded config.
        tiers: Confidence tiers from cascade.yaml.
        rules: Rule score per class.
        answer: The model's checked answer.
        baseline_meta: The baseline's fields for this document.
        words: Word boxes, for layout rules.
        page_sizes: Page sizes, for layout rules.

    Returns:
        The decision with its provenance.
    """
    classification, class_prov = decide_class(rules, answer, cfg, tiers)
    metadata, field_prov = decide_fields(
        full, pages, cfg, tiers, classification.label, answer, baseline_meta, words, page_sizes
    )
    return Decision(classification, metadata, class_prov, field_prov)
