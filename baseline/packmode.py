"""Run the pipeline against one cabinet of a client pack instead of the global vocabulary.

Over 400 lines: the prior, the field anchoring and the resume keys all read the same pack; see docs/parts/pack-mode.md.

A pack cabinet has its own classes and its own index fields per class. This
module turns them into a derived Config, so every downstream stage (field
extraction, the cascade prompt, grounding) runs unchanged on the pack's names:

- classes are the cabinet's classes; "other" (any case) is the abstain label;
- each class's terms are its name and aliases (the keyword prior);
- fields are the document-origin fields of each class, key first, scoped to
  that class and filed under the client's own field names. A field is anchored
  only on its own name and the spelling variants in SYNONYMS. It may borrow an
  existing extractor's value pattern and normaliser, never its labels.
  erp, manual and fixed fields are never extracted.

The trained classifier does not know the pack's classes, so pack mode
classifies with the prior alone (``classify``).
"""

from __future__ import annotations

import copy
import itertools
import json
import re
import sqlite3
import unicodedata
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import Any, NamedTuple

from . import db_registry
from .clientpack import OTHER_CLASS, ClientPack, ClientPackError, FieldSpec, TaxonomyClass, load_pack
from .config import Config
from .logging_setup import get_logger
from .schema import Classification, ClassifyMethod, compute_doc_id

log = get_logger(__name__)

PACK_KEY = "client_pack"  # pipeline.yaml section: {dir, cabinet}
HEADER_DEFAULT_CHARS = 600
# Pack-mode extraction defaults; an explicit extraction.max_pages or multi_frame_images wins.
PACK_MAX_PAGES = 500
PACK_MULTI_FRAME_IMAGES = True

# Keyword prior. A class name or alias found in the text scores by where it
# is found. All stay below the cascade's rule_short_circuit (0.9), so a model
# answer always overrides the prior.
PRIOR_HEADER_SCORE = 0.8
PRIOR_FIRST_PAGE_SCORE = 0.7
PRIOR_ANYWHERE_SCORE = 0.6
PRIOR_MIN_SCORE = 0.6  # below this the prior abstains to "other"
PRIOR_MIN_MARGIN = 0.1  # the best class must lead the next by this much
PRIOR_EPSILON = 1e-9  # 0.7 - 0.6 is 0.0999... in floats; the margin test allows for it
_PRIOR_SCOPES = (
    ("header", PRIOR_HEADER_SCORE),
    ("first_page", PRIOR_FIRST_PAGE_SCORE),
    ("anywhere", PRIOR_ANYWHERE_SCORE),
)

# Spelling variants of one word in a field name. Each group is one concept;
# no group crosses document concepts (PO, DO, GRN, Ref, Voucher ... stay apart).
SYNONYMS: tuple[tuple[str, ...], ...] = (
    ("no", "no.", "number", "num", "#"),
    ("inv", "invoice"),
    ("amt", "amount"),
    ("ref", "reference"),
    ("qty", "quantity"),
)
_SYNONYM_OF = {word: group for group in SYNONYMS for word in group}
# A field whose name holds one of these words is an identifier: text normaliser, never the amount one.
IDENTIFIER_TOKENS = frozenset({"no", "number", "num", "#", "id", "code", "ref", "reference"})
# Client field names whose value shape (not labels) an existing extractor gives.
SHAPE_BY_NAME = {"vendor name": "vendor_name", "supplier name": "vendor_name", "customer name": "buyer_name"}
# Extractor whose value shape a non-identifier field of each dtype borrows.
SHAPE_BY_DTYPE = {"date": "invoice_date", "number": "total_amount"}
SHAPE_KEYS = ("value_regex", "window", "direction", "ignorecase")
# Identifier value: right after the label, has a digit, keeps leading zeros and separators.
ID_VALUE_REGEX = r"^[\s:.#\[-]*((?=[^\s]*[0-9])[A-Za-z0-9](?:[A-Za-z0-9/_.,-]*[A-Za-z0-9])?)"
# Free text value: the rest of the label's line.
TEXT_VALUE_REGEX = r"^[ \t:#-]*(\S[^\n]{0,79}?)[ \t]*(?:\n|$)"
VALUE_WINDOW = 60
TEXT_NORMALIZER = "text_norm"
# A label must not be glued to a word; a one-word label must not follow another word on its line.
REJECT_GLUED = r"[^\W_]$"
REJECT_AFTER_WORD = r"[^\W_][ \t]?$"
ROLE_ORDER = {"key": 0, "required": 1, "optional": 2}
LABEL_SPECIFICITY = 1.0
RULE_DTYPES = ("date", "number", "text")


def _fold(text: str) -> str:
    """NFKC then casefold, so the prior matches any script and any case."""
    return unicodedata.normalize("NFKC", text).casefold()


def _words(text: str) -> list[str]:
    return re.findall(r"#|[^\W_]+", unicodedata.normalize("NFKC", text).lower())


def _term_pattern(term: str) -> re.Pattern[str] | None:
    """A class name or alias: its words in order, any punctuation or spacing between, Unicode aware."""
    words = re.findall(r"[^\W_]+", _fold(term))
    if not words:
        return None
    return re.compile(r"(?<![^\W_])" + r"[\W_]+".join(re.escape(w) for w in words) + r"(?![^\W_])")


def _is_reserved(name: str) -> bool:
    return name.strip().casefold() == OTHER_CLASS


def _class_entry(cls: TaxonomyClass) -> dict[str, Any]:
    terms = list(dict.fromkeys([cls.name, *cls.aliases]))
    # Markers keep generic rule scoring (rule_scores) working; the prior itself reads terms.
    markers = [
        {"kind": "regex", "pattern": pat.pattern, "scope": scope, "specificity": score, "term": term}
        for term in terms
        if (pat := _term_pattern(term))
        for scope, score in _PRIOR_SCOPES
    ]
    also = f" Also filed as: {', '.join(cls.aliases)}." if cls.aliases else ""
    description = f"Documents of type {cls.name}.{also}"
    return {"name": cls.name, "description": description, "terms": terms, "markers": markers}


def _description(spec: FieldSpec) -> str:
    fmt = f", written as {spec.date_format}" if spec.date_format else ""
    key = " The document's own key number." if spec.role == "key" else ""
    return f"Client field {spec.name!r} ({spec.dtype}{fmt}), printed on the document.{key}"


def _labels(name: str) -> list[dict[str, Any]]:
    """The client's field name and its SYNONYMS spellings; no other extractor's labels."""
    words = _words(name)
    texts = [" ".join(combo) for combo in itertools.product(*(_SYNONYM_OF.get(w, (w,)) for w in words))]
    texts += [t.replace(" #", "#") for t in texts if " #" in t]
    raw = unicodedata.normalize("NFKC", name).lower().strip(" :")
    return [{"text": t, "specificity": LABEL_SPECIFICITY} for t in dict.fromkeys([raw, *texts]) if t]


def is_identifier(spec: FieldSpec) -> bool:
    """A key, or a field named like a number or code: compared as text, never as an amount.

    Args:
        spec: The client field.

    Returns:
        Whether the field holds an identifier.
    """
    return spec.role == "key" or bool(IDENTIFIER_TOKENS & set(_words(spec.name)))


def _shape(spec: FieldSpec, by_name: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """Value pattern, normaliser and type for a field; None when no rule can read it."""
    if is_identifier(spec):
        anchor = {"value_regex": ID_VALUE_REGEX, "window": VALUE_WINDOW, "direction": "after"}
        return {"type": "text", "normalizer": TEXT_NORMALIZER, "anchor": anchor}
    source = SHAPE_BY_NAME.get(" ".join(_words(spec.name))) or SHAPE_BY_DTYPE.get(spec.dtype)
    if source is None:
        anchor = {"value_regex": TEXT_VALUE_REGEX, "window": VALUE_WINDOW, "direction": "after"}
        return {"type": "text", "normalizer": TEXT_NORMALIZER, "anchor": anchor}
    borrowed = by_name.get(source) or {}
    if not (borrowed.get("anchor") or {}).get("value_regex"):
        return None
    anchor = {k: copy.deepcopy(v) for k, v in borrowed["anchor"].items() if k in SHAPE_KEYS}
    return {"type": borrowed.get("type", "text"), "normalizer": borrowed.get("normalizer"), "anchor": anchor}


def _field_def(spec: FieldSpec, type_name: str, by_name: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """One fields.yaml entry for a client field, anchored on the client's own label only."""
    head = {"name": spec.name, "description": _description(spec), "classes": [type_name]}
    shape = _shape(spec, by_name) if spec.dtype in RULE_DTYPES else None
    labels = _labels(spec.name)
    if shape is None or not labels:
        return {**head, "type": "text", "strategies": []}  # the cascade asks the model; no rule finds it
    reject = REJECT_AFTER_WORD if len(_words(spec.name)) == 1 else REJECT_GLUED
    anchor = {**shape["anchor"], "labels": labels, "reject_before": reject}
    return {**head, **shape, "strategies": ["anchor"], "anchor": anchor}


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
        extraction.max_pages and multi_frame_images get pack defaults unless set.
    """
    pack = load_pack(pack_dir, cabinet)
    pipeline = copy.deepcopy(cfg.pipeline)
    pipeline["cabinet_id"] = pack.cabinet.slug
    pipeline[PACK_KEY] = {"dir": str(Path(pack_dir).resolve()), "cabinet": pack.cabinet.name, "slug": pack.cabinet.slug}
    pipeline.setdefault("classification", {})["unknown_label"] = OTHER_CLASS
    pipeline.setdefault("metadata", {})["run_scoped_fields_when_unknown"] = False
    extraction = pipeline.setdefault("extraction", {})
    extraction.setdefault("max_pages", PACK_MAX_PAGES)
    extraction.setdefault("multi_frame_images", PACK_MULTI_FRAME_IMAGES)
    classes = [_class_entry(c) for c in pack.classes if not _is_reserved(c.name)]
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


def resolve(cfg: Config, client_pack: tuple[str | None, str | None] | None = None) -> Config:
    """Apply a client pack from the arguments, else from pipeline.yaml client_pack, else none.

    A command-line cabinet with no command-line directory picks that cabinet
    of the configured pack. A relative configured dir resolves against the
    config directory; a command-line dir against the working directory.

    Args:
        cfg: The loaded config.
        client_pack: (pack directory or None, cabinet or None) from the command line, or None.

    Returns:
        The pack-mode Config, or cfg unchanged when no pack directory is set anywhere.

    Raises:
        ClientPackError: A pack directory is set without a cabinet.
    """
    if is_active(cfg):
        return cfg
    spec = cfg.pipeline.get(PACK_KEY) or {}
    cli_dir, cli_cabinet = client_pack or (None, None)
    pack_dir = Path(cli_dir) if cli_dir else (cfg.config_dir / spec["dir"] if spec.get("dir") else None)
    if pack_dir is None:
        return cfg
    cabinet = cli_cabinet or spec.get("cabinet")
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


def register(conn: sqlite3.Connection, cfg: Config) -> dict[str, str]:
    """Load a pack-mode run's cabinet into the store and map its documents to their cases.

    Args:
        conn: Open store.
        cfg: A config; nothing happens unless it is in pack mode.

    Returns:
        Materialized file stem (the manifest doc_id) to case_id, for rows that have one.
    """
    if not is_active(cfg):
        return {}
    spec = cfg.pipeline[PACK_KEY]
    pack = load_pack(spec["dir"], spec["slug"])
    db_registry.register_pack(conn, pack)
    return {row.doc_id: row.case_id for row in pack.manifest if row.case_id}


def resume_key(path: str | Path, cfg: Config) -> str:
    """What marks a document done on resume: the file stem (manifest doc_id) in pack mode, else the content hash.

    Two manifest documents with the same bytes have one content hash but two stems, so each gets a record.

    Args:
        path: The document.
        cfg: The run's config.

    Returns:
        The key compared against done_keys.
    """
    return Path(path).stem if is_active(cfg) else compute_doc_id(path)


def _records(output: Path) -> list[dict[str, Any]]:
    rows = []
    for line in output.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue  # a damaged line is re-run, as in existing_doc_ids
    return [r for r in rows if isinstance(r, dict)]


def done_keys(output: str | Path, cfg: Config) -> set[str]:
    """The resume keys already in a results file.

    Args:
        output: The results JSONL.
        cfg: The run's config.

    Returns:
        File stems in pack mode, doc_ids otherwise; empty when the file is missing.

    Raises:
        ClientPackError: In pack mode, the file holds records of another cabinet.
    """
    p = Path(output)
    if not p.exists():
        return set()
    rows = _records(p)
    if not is_active(cfg):
        return {str(r["doc_id"]) for r in rows if "doc_id" in r}
    others = sorted({str(r.get("cabinet_id")) for r in rows} - {cfg.cabinet_id})
    if others:
        raise ClientPackError(
            f"{p.name} holds records of cabinet {', '.join(others)}, not {cfg.cabinet_id}: "
            "write to a new output file, or pass --no-resume to replace it"
        )
    return {Path(str(r["filename"])).stem for r in rows if "filename" in r}


class _Hit(NamedTuple):
    start: int
    end: int
    cls: str
    term: str


def _hits(text: str, classes: list[dict[str, Any]]) -> list[_Hit]:
    """Every class term found in text; a term inside a longer term of another class at the same place is dropped."""
    folded = _fold(text)
    found = [
        _Hit(m.start(), m.end(), c["name"], term)
        for c in classes
        for term in c.get("terms", [])
        if (pat := _term_pattern(term))
        for m in pat.finditer(folded)
    ]
    return [
        h
        for h in found
        if not any(
            g.cls != h.cls and g.start <= h.start and h.end <= g.end and g.end - g.start > h.end - h.start
            for g in found
        )
    ]


def _scope_hits(full: str, pages: list[str], cfg: Config) -> dict[str, list[_Hit]]:
    header = int(cfg.classify_opts().get("header_chars", HEADER_DEFAULT_CHARS))
    first = pages[0] if pages else ""
    texts = {"header": first[:header], "first_page": first, "anywhere": full}
    classes = cfg.classes.get("classes", [])
    return {scope: _hits(texts[scope], classes) for scope, _ in _PRIOR_SCOPES}


def prior_scores(full: str, pages: list[str], cfg: Config) -> dict[str, float]:
    """Best prior score per class: the tier of the narrowest scope any of its terms is found in.

    Args:
        full: Full document text.
        pages: Per-page text.
        cfg: A pack-mode config.

    Returns:
        {class: score} for classes with a hit.
    """
    out: dict[str, float] = {}
    hits = _scope_hits(full, pages, cfg)
    for scope, score in _PRIOR_SCOPES:
        for h in hits[scope]:
            out[h.cls] = max(out.get(h.cls, 0.0), score)
    return dict(sorted(out.items()))


def evidence(full: str, pages: list[str], cfg: Config) -> dict[str, list[str]]:
    """Which class names and aliases the prior found, and where.

    Args:
        full: Full document text.
        pages: Per-page text.
        cfg: A pack-mode config.

    Returns:
        {class: ["<term> @ <scope>", ...]} for classes with a hit; terms are pack vocabulary, not document text.
    """
    hits = _scope_hits(full, pages, cfg)
    out: dict[str, list[str]] = {}
    for cls in cfg.classes.get("classes", []):
        found = [
            f"{term} @ {scope}"
            for term in cls.get("terms", [])
            for scope, _ in _PRIOR_SCOPES
            if any(h.cls == cls["name"] and h.term == term for h in hits[scope])
        ]
        if found:
            out[cls["name"]] = found
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
    scores = prior_scores(full, pages, cfg)
    # Ties break on name so the output is stable, as in the baseline.
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0])) + [(OTHER_CLASS, 0.0)] * 2
    (best, top), runner_up = ranked[0], ranked[1][1]
    if top < PRIOR_MIN_SCORE - PRIOR_EPSILON or top - runner_up < PRIOR_MIN_MARGIN - PRIOR_EPSILON:
        return Classification(
            label=OTHER_CLASS, confidence=round(top, 4), all_scores=scores, method=ClassifyMethod.FALLBACK
        )
    return Classification(label=best, confidence=round(top, 4), all_scores=scores, method=ClassifyMethod.RULE)
