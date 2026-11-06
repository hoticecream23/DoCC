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

from baseline.config import Config
from baseline.metadata import CHECKSUM_VALIDATORS, NormCtx, normalize
from baseline.metadata.fields import candidates_for
from baseline.schema import (
    Classification,
    ClassifyMethod,
    MetadataField,
    MetadataMethod,
    WordBox,
    page_for_offset,
)

from .config import ConfidenceTiers
from .ground import EXACT, LOOSE, NORMALISED, UNGROUNDED, TextIndex, ground
from .prompt import VLMAnswer


@dataclass
class Decision:
    classification: Classification
    metadata: list[MetadataField]
    class_provenance: dict
    field_provenance: list[dict] = field(default_factory=list)


def _unknown(cfg: Config) -> str:
    return str(cfg.classify_opts().get("unknown_label", "unknown"))


def decide_class(
    rules: dict[str, float], answer: VLMAnswer, cfg: Config, tiers: ConfidenceTiers
) -> tuple[Classification, dict]:
    unknown = _unknown(cfg)
    short_circuit = float(cfg.classify_opts().get("rule_short_circuit", 0.9))
    all_scores = dict(sorted(rules.items()))
    if rules:
        # Ties break on name so the output is stable, as in the baseline.
        label, score = sorted(rules.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        if score >= short_circuit:
            prov = {
                "source": "rule",
                "label": label,
                "vlm_label": answer.label,
                "disagrees": answer.label not in (label, unknown),
            }
            return (
                Classification(
                    label=label, confidence=round(score, 4), all_scores=all_scores, method=ClassifyMethod.RULE
                ),
                prov,
            )
    confidence = 0.0 if answer.label == unknown else tiers.vlm_class
    prov = {"source": "vlm", "label": answer.label, "vlm_label": answer.label, "disagrees": False}
    return (
        Classification(label=answer.label, confidence=confidence, all_scores=all_scores, method=ClassifyMethod.VLM),
        prov,
    )


def _in_scope(fdef: dict, doc_class: str, cfg: Config) -> bool:
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
        return not re.search(r"[A-Za-z]", raw) and 9 <= len(digits) <= 18 and _alnum(raw) not in other_ids
    if name == "invoice_number":
        return bool(re.search(r"\d", raw)) and len(raw) <= 30
    return True


def _key(normalized: str | None, raw: str) -> str:
    return normalized if normalized is not None else raw


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
) -> tuple[list[MetadataField], list[dict]]:
    norm_opts = cfg.fields.get("normalizer_options", {}) or {}
    index = TextIndex(full, pages)
    from_baseline: dict[str, list[MetadataField]] = defaultdict(list)
    for m in baseline_meta:
        from_baseline[m.field].append(m)

    out: list[MetadataField] = []
    prov: list[dict] = []
    for fdef in cfg.field_defs:
        name = fdef.get("name")
        if not name:
            continue
        pattern = fdef.get("pattern") or {}
        checksummed = pattern.get("validator") in CHECKSUM_VALIDATORS
        multi = bool(fdef.get("multi"))

        kept: list[MetadataField] = []
        seen: set[str] = set()

        # For checksummed fields, keep proven baseline values unconditionally,
        # before the scope check.
        if checksummed:
            for m in from_baseline[name]:
                if m.method is MetadataMethod.REGEX_CHECKSUM:
                    kept.append(m)
                    seen.add(_key(m.normalized_value, m.value))

        # Model values are skipped for every out-of-scope field; only proven
        # baseline values (already in kept) survive.
        in_scope = _in_scope(fdef, doc_class, cfg)
        if not in_scope:
            # Out-of-scope fields produce no provenance record. Proven checksummed values
            # go to output regardless of scope.
            out.extend(kept)
            continue

        rule_cands = candidates_for(full, pages, fdef, cfg, doc_class, words, page_sizes)
        model_emitted = False
        tried = [(v, ground(v, index, fdef, rule_cands, norm_opts)) for v in answer.fields.get(name, [])]
        if not multi:
            # One value is kept, so try the best first: rule agreement, then the
            # strongest grounding (exact, loose, normalised; one match over many),
            # then the model's order (the sort is stable).
            tried.sort(key=lambda vg: (not vg[1].rule_agrees, _GROUND_RANK[vg[1].outcome], vg[1].n_matches > 1))
        for value, g in tried:
            if kept and not multi:
                break
            item = {
                "field": name,
                "vlm_value": value,
                "outcome": g.outcome,
                "n_matches": g.n_matches,
                "rule_agrees": g.rule_agrees,
                "char_start": g.start,
                "char_end": g.end,
                "emitted": False,
            }
            prov.append(item)
            if not g.grounded:
                continue
            raw = full[g.start : g.end]
            if checksummed:
                # A model value for a checksummed field is proven only if its span
                # matches a rule candidate with REGEX_CHECKSUM method (which is the
                # baseline's definition of proven).
                matches_proven = any(
                    c.start == g.start and c.end == g.end and c.method is MetadataMethod.REGEX_CHECKSUM
                    for c in rule_cands
                )
                if not matches_proven:
                    item["outcome"] = "checksum_failed"
                    continue
                method, confidence, validated = MetadataMethod.REGEX_CHECKSUM, 1.0, True
            elif g.rule_agrees:
                method, confidence, validated = MetadataMethod.VLM_RULE_AGREE, tiers.agree, False
            else:
                other_ids = {_alnum(m.value) for m in out if m.field in _OTHER_IDS}
                other_ids |= {_alnum(v) for f in _OTHER_IDS for v in answer.fields.get(f, [])}
                if not _shape_ok(name, raw, other_ids):
                    item["outcome"] = "shape_failed"
                    continue
                method, validated = MetadataMethod.VLM, False
                confidence = tiers.ambiguous if g.n_matches > 1 else tiers.grounded
            normalized, extras = normalize(
                fdef.get("normalizer"), raw, NormCtx(full=full, start=g.start, end=g.end, opts=norm_opts)
            )
            key = _key(normalized, raw)
            if key in seen:
                continue
            seen.add(key)
            item["emitted"] = True
            model_emitted = True
            kept.append(
                MetadataField(
                    field=name,
                    value=raw,
                    normalized_value=normalized,
                    page=page_for_offset(index.spans, g.start),
                    char_start=g.start,
                    char_end=g.end,
                    confidence=round(confidence, 4),
                    method=method,
                    validated=validated,
                    normalizer_version=cfg.normalizer_version,
                    currency=extras.get("currency"),
                )
            )

        if not checksummed and not model_emitted and from_baseline[name]:
            prov.append(
                {
                    "field": name,
                    "outcome": "rule_only",
                    "baseline_values": [m.value for m in from_baseline[name]],
                    "emitted": False,
                }
            )
        out.extend(kept)

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
    classification, class_prov = decide_class(rules, answer, cfg, tiers)
    metadata, field_prov = decide_fields(
        full, pages, cfg, tiers, classification.label, answer, baseline_meta, words, page_sizes
    )
    return Decision(classification, metadata, class_prov, field_prov)
