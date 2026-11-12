"""Run the pipeline against one cabinet of a client pack instead of the global vocabulary.

A pack cabinet has its own classes and its own index fields per class. This
module turns them into a derived Config, so every downstream stage (field
extraction, the cascade prompt, grounding) runs unchanged on the pack's names:

- classes are the cabinet's classes; "other" is the abstain label;
- each class's markers come from its name and aliases (the keyword prior);
- fields are the document-origin fields of each class, key first, scoped to
  that class and filed under the client's own field names. A field whose
  normalised name is in EXTRACTOR_BY_NAME reuses that extractor; any other
  date, number or text field reuses the dtype's extractor anchored on the
  client's own field label. erp, manual and fixed fields are never extracted.

The trained classifier does not know the pack's classes, so pack mode
classifies with the prior alone (``classify``).
"""

from __future__ import annotations

import copy
import re
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import Any

from .classify import rule_scores
from .clientpack import OTHER_CLASS, ClientPack, ClientPackError, FieldSpec, TaxonomyClass, load_pack
from .clientpack_rules import norm
from .config import Config
from .logging_setup import get_logger
from .rules import match_marker
from .schema import Classification, ClassifyMethod

log = get_logger(__name__)

PACK_KEY = "client_pack"  # pipeline.yaml section: {dir, cabinet}
HEADER_DEFAULT_CHARS = 600

# Keyword prior. A class name or alias found in the text scores by where it
# is found. All stay below the cascade's rule_short_circuit (0.9), so a model
# answer always overrides the prior.
PRIOR_HEADER_SCORE = 0.8
PRIOR_FIRST_PAGE_SCORE = 0.7
PRIOR_ANYWHERE_SCORE = 0.6
PRIOR_MIN_SCORE = 0.6  # below this the prior abstains to "other"
PRIOR_MIN_MARGIN = 0.1  # the best class must lead the next by this much
_PRIOR_SCOPES = (
    ("header", PRIOR_HEADER_SCORE),
    ("first_page", PRIOR_FIRST_PAGE_SCORE),
    ("anywhere", PRIOR_ANYWHERE_SCORE),
)

# Normalised client field name -> the existing extractor (fields.yaml) it means.
EXTRACTOR_BY_NAME = {
    "invoice no": "invoice_number",
    "invoice number": "invoice_number",
    "inv no": "invoice_number",
    "bill no": "invoice_number",
    "bill number": "invoice_number",
    "po no": "invoice_number",
    "po number": "invoice_number",
    "invoice date": "invoice_date",
    "bill date": "invoice_date",
    "date": "invoice_date",
    "document date": "invoice_date",
    "doc date": "invoice_date",
    "due date": "due_date",
    "total": "total_amount",
    "total amount": "total_amount",
    "grand total": "total_amount",
    "invoice amount": "total_amount",
    "net amount": "total_amount",
    "amount": "total_amount",
    "tax amount": "tax_amount",
    "vat amount": "tax_amount",
    "account no": "account_number",
    "account number": "account_number",
    "vendor name": "vendor_name",
    "supplier name": "vendor_name",
    "customer name": "buyer_name",
}
# Extractor whose value shape an unmapped field of each dtype borrows.
EXTRACTOR_BY_DTYPE = {"date": "invoice_date", "number": "total_amount", "text": "invoice_number"}
ROLE_ORDER = {"key": 0, "required": 1, "optional": 2}
LABEL_SPECIFICITY = 1.0


def _term_pattern(term: str) -> str | None:
    """Regex for a class name or alias: its words in order, any punctuation or spacing between."""
    words = re.findall(r"[0-9a-z]+", term.lower())
    if not words:
        return None
    return r"(?<![0-9a-z])" + r"[^0-9a-z]+".join(re.escape(w) for w in words) + r"(?![0-9a-z])"


def _class_entry(cls: TaxonomyClass) -> dict[str, Any]:
    terms = list(dict.fromkeys([cls.name, *cls.aliases]))
    markers = [
        {"kind": "regex", "pattern": pat, "scope": scope, "specificity": score, "term": term}
        for term in terms
        if (pat := _term_pattern(term))
        for scope, score in _PRIOR_SCOPES
    ]
    also = f" Also filed as: {', '.join(cls.aliases)}." if cls.aliases else ""
    return {"name": cls.name, "description": f"Documents of type {cls.name}.{also}", "markers": markers}


def _description(spec: FieldSpec) -> str:
    fmt = f", written as {spec.date_format}" if spec.date_format else ""
    key = " The document's own key number." if spec.role == "key" else ""
    return f"Client field {spec.name!r} ({spec.dtype}{fmt}), printed on the document.{key}"


def _labels(name: str) -> list[dict[str, Any]]:
    texts = dict.fromkeys(t for t in (name.lower().strip(" .:#"), norm(name)) if t)
    return [{"text": t, "specificity": LABEL_SPECIFICITY} for t in texts]


def _field_def(spec: FieldSpec, type_name: str, by_name: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """One fields.yaml entry for a client field, built on an existing extractor where one fits."""
    head = {"name": spec.name, "description": _description(spec), "classes": [type_name]}
    mapped = by_name.get(EXTRACTOR_BY_NAME.get(norm(spec.name), ""))
    if mapped is not None:
        fdef = {**copy.deepcopy(mapped), **head}
        if fdef.get("anchor"):
            fdef["anchor"]["labels"] = _labels(spec.name) + list(fdef["anchor"].get("labels") or [])
        return fdef
    shape = by_name.get(EXTRACTOR_BY_DTYPE.get(spec.dtype, ""))
    if shape is None or not shape.get("anchor"):
        return {**head, "type": "text", "strategies": []}  # the cascade asks the model; no rule finds it
    anchor = {**copy.deepcopy(shape["anchor"]), "labels": _labels(spec.name)}
    return {
        **head,
        "type": shape.get("type", "text"),
        "normalizer": shape.get("normalizer"),
        "strategies": ["anchor"],
        "anchor": anchor,
    }


def _field_defs(cfg: Config, pack: ClientPack) -> list[dict[str, Any]]:
    by_name = {f["name"]: f for f in cfg.field_defs if f.get("name")}
    out = []
    for cls in pack.classes:
        specs = [f for f in pack.fields.get(cls.name, []) if f.origin == "document"]
        for spec in sorted(specs, key=lambda f: ROLE_ORDER[f.role]):  # stable: key first, then file order
            out.append(_field_def(spec, cls.name, by_name))
    return out


def pack_config(cfg: Config, pack_dir: str | Path, cabinet: str) -> Config:
    """The Config for a pack-mode run: the cabinet's classes and fields in place of the global ones.

    Args:
        cfg: The loaded base config; not changed.
        pack_dir: The client pack directory.
        cabinet: Cabinet name or slug.

    Returns:
        A new Config. cabinet_id is the cabinet slug; "other" is the abstain label.
    """
    pack = load_pack(pack_dir, cabinet)
    pipeline = copy.deepcopy(cfg.pipeline)
    pipeline["cabinet_id"] = pack.cabinet.slug
    pipeline[PACK_KEY] = {"dir": str(Path(pack_dir).resolve()), "cabinet": pack.cabinet.name, "slug": pack.cabinet.slug}
    pipeline.setdefault("classification", {})["unknown_label"] = OTHER_CLASS
    pipeline.setdefault("metadata", {})["run_scoped_fields_when_unknown"] = False
    classes = [_class_entry(c) for c in pack.classes if c.name != OTHER_CLASS]
    fields = {**cfg.fields, "fields": _field_defs(cfg, pack), "templates": []}
    log.info("client pack loaded", extra={"cabinet": pack.cabinet.slug, "classes": len(classes)})
    return replace(cfg, pipeline=pipeline, classes={"classes": classes}, fields=fields)


def is_active(cfg: Config) -> bool:
    """True when cfg was built by pack_config.

    Args:
        cfg: A loaded config.

    Returns:
        Whether the run is in pack mode.
    """
    return bool((cfg.pipeline.get(PACK_KEY) or {}).get("slug"))


def resolve(cfg: Config, client_pack: tuple[str, str] | None = None) -> Config:
    """Apply a client pack from the arguments, else from pipeline.yaml client_pack, else none.

    Args:
        cfg: The loaded config.
        client_pack: (pack directory, cabinet) from the command line, or None.

    Returns:
        The pack-mode Config, or cfg unchanged when no pack is set.

    Raises:
        ClientPackError: A pack directory is set without a cabinet.
    """
    if is_active(cfg):
        return cfg
    spec = cfg.pipeline.get(PACK_KEY) or {}
    pack_dir, cabinet = client_pack or (spec.get("dir"), spec.get("cabinet"))
    if not pack_dir:
        return cfg
    if not cabinet:
        raise ClientPackError("a client pack needs a cabinet: pass --cabinet or set client_pack.cabinet")
    return pack_config(cfg, pack_dir, cabinet)


def inputs(cfg: Config, input_dir: str | Path) -> list[Path]:
    """The materialized copies of the cabinet's manifest under input_dir, in manifest order.

    Other files in the directory (results, reports) are never read as documents.

    Args:
        cfg: A pack-mode config.
        input_dir: The materialized cabinet directory.

    Returns:
        Each <doc_id><ext> that exists; missing ones are counted in the log.
    """
    spec = cfg.pipeline[PACK_KEY]
    pack = load_pack(spec["dir"], spec["slug"])
    root = Path(input_dir)
    wanted = [root / f"{row.doc_id}{PurePosixPath(row.rel_path).suffix.lower()}" for row in pack.manifest]
    found = [p for p in wanted if p.is_file()]
    if len(found) != len(wanted):
        log.warning("manifest documents not materialized", extra={"missing": len(wanted) - len(found)})
    return found


def evidence(full: str, pages: list[str], cfg: Config) -> dict[str, list[str]]:
    """Which class names and aliases the prior found, and where.

    Args:
        full: Full document text.
        pages: Per-page text.
        cfg: A pack-mode config.

    Returns:
        {class: ["<term> @ <scope>", ...]} for classes with a hit; terms are pack vocabulary, not document text.
    """
    header = int(cfg.classify_opts().get("header_chars", HEADER_DEFAULT_CHARS))
    out: dict[str, list[str]] = {}
    for cls in cfg.classes.get("classes", []):
        hits = [f"{m['term']} @ {m['scope']}" for m in cls.get("markers", []) if match_marker(m, full, pages, header)]
        if hits:
            out[cls["name"]] = hits
    return out


def classify(full: str, pages: list[str], cfg: Config) -> Classification:
    """Class from the keyword prior: best score, a margin over the runner-up, else "other".

    Args:
        full: Full document text.
        pages: Per-page text.
        cfg: A pack-mode config.

    Returns:
        A rule classification; method fallback and label "other" when it abstains.
        all_scores holds every class's prior score, the evidence for the choice.
    """
    scores = dict(sorted(rule_scores(full, pages, cfg).items()))
    # Ties break on name so the output is stable, as in the baseline.
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0])) + [(OTHER_CLASS, 0.0)] * 2
    (best, top), runner_up = ranked[0], ranked[1][1]
    if top < PRIOR_MIN_SCORE or top - runner_up < PRIOR_MIN_MARGIN:
        return Classification(
            label=OTHER_CLASS, confidence=round(top, 4), all_scores=scores, method=ClassifyMethod.FALLBACK
        )
    return Classification(label=best, confidence=round(top, 4), all_scores=scores, method=ClassifyMethod.RULE)
