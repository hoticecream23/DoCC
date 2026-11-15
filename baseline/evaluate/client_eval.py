"""Score a run over one client cabinet against the labels its client pack carries.

The pack's only gold is the folder a document sat in (its type) and, for most
single-document types, the key value in its file name. So this scores the
class and the key field and nothing else. Results are joined to the manifest
by file stem: ``clientpack.materialize`` names each copy ``<doc_id><ext>``.

Every labelled manifest document is in the denominator. A document with no
result line (the pipeline failed on it, or the run did not reach it) counts
as wrong. docs/parts/client-eval.md defines every number this reports.

Unlike the rest of this package it reads a client pack, so it imports
``baseline.clientpack``; that module is a data format, not pipeline code.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import BaseModel, Field

from ..clientpack import ClientPack, ManifestRow, load_pack
from ..clientpack_rules import norm
from .client_report import MIN_CONTAINED_CHARS
from .client_report import _pct as _pct  # re-exported for existing callers
from .client_report import render_client_report as render_client_report
from .io import load_jsonl
from .scoring import prf

UNKNOWN_LABEL = "unknown"  # the label the pipeline gives when it abstains
NO_RESULT = "(no result)"  # predicted class of a document with no result line
ABSTAINED = "(abstained)"  # predicted class of a document the pipeline did not class
DASHES = re.compile("[\u2010-\u2015\u2212\ufe58\ufe63\uff0d]")  # unicode dashes, mapped to "-"
FLOAT_ARTEFACT = re.compile(r"\d+\.0+")  # "4502424845.0": a number read as a float
SEPARATORS = re.compile(r"[\W_]+")  # anything not a letter or a digit


class ClassScore(BaseModel):
    """Precision and recall of one class (gold or predicted)."""

    precision: float
    recall: float
    support: int  # documents with this gold class
    predicted: int = 0  # documents predicted as this class


class CabinetScore(BaseModel):
    """Everything client-eval reports for one cabinet."""

    cabinet: str
    documents: int
    missing: int = 0  # in the manifest, no result line
    failed: int = 0  # has a result line, and the record carries diagnostics errors
    abstained: int = 0  # has a result line with no class, or the unknown label
    unlabeled: int = 0  # no gold label: reported, not scored
    duplicates: int = 0  # stems on more than one results line; the last line is used
    scored: int = 0  # labelled single documents
    correct: int = 0
    scored_missing: int = 0  # labelled single documents with no result line
    per_class: dict[str, ClassScore] = Field(default_factory=dict)
    key_scored: int = 0  # labelled documents with a key_gold
    key_exact: int = 0
    key_loose: int = 0
    key_contained: int = 0
    key_too_short: int = 0  # gold too short for the contained test
    key_class_ok: int = 0  # key documents whose class was right
    key_exact_class_ok: int = 0  # ... and whose key was exact
    cases: int = 0
    cases_majority: int = 0
    cases_any: int = 0
    mixed_cases: list[str] = Field(default_factory=list)
    no_result: list[str] = Field(default_factory=list)
    with_errors: list[str] = Field(default_factory=list)

    @property
    def accuracy(self) -> float:
        """Share of labelled single documents classed right; no result and abstentions count as wrong."""
        return self.correct / self.scored if self.scored else 0.0


def _stem(record: dict[str, Any]) -> str:
    return PurePosixPath(str(record.get("filename") or record.get("source_path") or "")).stem


def _label_key(label: str) -> str:
    return " ".join(label.split()).casefold()


def _class_names(pack: ClientPack) -> dict[str, str]:
    """Normalised class name or alias -> class name."""
    names: dict[str, str] = {}
    for cls in pack.classes:
        for name in (cls.name, *cls.aliases):
            names.setdefault(_label_key(name), cls.name)
    return names


def _resolve(names: dict[str, str], label: str) -> str:
    return names.get(_label_key(label), " ".join(label.split()))


def _predicted(names: dict[str, str], record: dict[str, Any] | None) -> str:
    if record is None:
        return NO_RESULT
    label = str((record.get("classification") or {}).get("label") or "")
    return ABSTAINED if _label_key(label) in ("", UNKNOWN_LABEL) else _resolve(names, label)


def canon_key(value: str) -> str:
    """Key form for the exact match: outer space trimmed, case folded, inner space collapsed, dashes to "-".

    Args:
        value: A key value.

    Returns:
        The canonical form.
    """
    return " ".join(DASHES.sub("-", value).split()).casefold()


def loose_key(value: str) -> str:
    """Key form for the loose match: canon_key with every space and punctuation mark removed.

    Args:
        value: A key value.

    Returns:
        Letters and digits only.
    """
    return SEPARATORS.sub("", canon_key(value))


def _defloat(value: str) -> str:
    """Map a float artefact such as 4502424845.0 to its integer part; anything else unchanged."""
    text = value.strip()
    return text.split(".")[0] if FLOAT_ARTEFACT.fullmatch(text) else value


def contained(gold: str, value: str) -> bool:
    """Gold appears in value as a whole token run, separators between its tokens ignored.

    Args:
        gold: The gold key, at least MIN_CONTAINED_CHARS letters and digits long.
        value: An extracted value.

    Returns:
        True when the gold's tokens appear in order with no letter or digit on either side.
    """
    tokens = [t for t in SEPARATORS.split(canon_key(gold)) if t]
    if not tokens:
        return False
    pattern = r"(?<![^\W_])" + r"[\W_]*".join(map(re.escape, tokens)) + r"(?![^\W_])"
    return re.search(pattern, canon_key(value)) is not None


def _values(field: dict[str, Any]) -> list[str]:
    return [_defloat(str(field[k])) for k in ("value", "normalized_value") if field.get(k) not in (None, "")]


def _key_hits(pack: ClientPack, row: ManifestRow, record: dict[str, Any] | None) -> tuple[bool, bool, bool | None]:
    """(exact, loose, contained) for one document; contained is None when the gold is too short to test."""
    fields = (record or {}).get("metadata") or []
    key = norm(pack.key_field(row.label))
    key_values = [v for f in fields if norm(str(f.get("field", ""))) == key for v in _values(f)]
    gold, gold_loose = canon_key(row.key_gold), loose_key(row.key_gold)
    exact = bool(gold) and any(canon_key(v) == gold for v in key_values)
    loose = bool(gold_loose) and any(loose_key(v) == gold_loose for v in key_values)
    if len(gold_loose) < MIN_CONTAINED_CHARS:
        return exact, loose, None
    return exact, loose, any(contained(row.key_gold, v) for f in fields for v in _values(f))


def _score_key(
    score: CabinetScore, pack: ClientPack, row: ManifestRow, record: dict[str, Any] | None, class_ok: bool
) -> None:
    exact, loose, inside = _key_hits(pack, row, record)
    score.key_scored += 1
    score.key_exact += exact
    score.key_loose += loose
    if inside is None:
        score.key_too_short += 1
    else:
        score.key_contained += inside
    if class_ok:
        score.key_class_ok += 1
        score.key_exact_class_ok += exact


def _per_class(pairs: list[tuple[str, str]]) -> dict[str, ClassScore]:
    """One row per gold or predicted class, no result and abstained included."""
    out = {}
    for name in sorted({g for g, _ in pairs} | {p for _, p in pairs}):
        tp = sum(g == name and p == name for g, p in pairs)
        fp = sum(g != name and p == name for g, p in pairs)
        fn = sum(g == name and p != name for g, p in pairs)
        res = prf(tp, fp, fn)
        out[name] = ClassScore(precision=res["precision"], recall=res["recall"], support=tp + fn, predicted=tp + fp)
    return out


def _score_cases(score: CabinetScore, cases: dict[str, list[tuple[str, str]]]) -> None:
    """Case gold is the majority label of its files. Majority: the most common class prediction equals it."""
    score.cases = len(cases)
    for case_id, pairs in cases.items():
        labels = Counter(g for g, _ in pairs)
        if len(labels) > 1:
            score.mixed_cases.append(case_id)
        gold = labels.most_common(1)[0][0]
        votes = Counter(p for _, p in pairs if p not in (NO_RESULT, ABSTAINED))
        score.cases_majority += bool(votes) and votes.most_common(1)[0][0] == gold
        score.cases_any += gold in votes


def _load_results(results_jsonl: str | Path) -> tuple[dict[str, dict[str, Any]], int]:
    """Records by stem, the last line winning, and the count of stems on more than one line."""
    records: dict[str, dict[str, Any]] = {}
    seen: Counter[str] = Counter()
    for record in load_jsonl(results_jsonl):
        stem = _stem(record)
        seen[stem] += 1
        records[stem] = record
    return records, sum(n > 1 for n in seen.values())


def _count_outcome(score: CabinetScore, row: ManifestRow, record: dict[str, Any] | None, predicted: str) -> None:
    if record is None:
        score.missing += 1
        score.no_result.append(row.rel_path)
        return
    if (record.get("diagnostics") or {}).get("errors"):
        score.failed += 1
        score.with_errors.append(row.rel_path)
    score.abstained += predicted == ABSTAINED


def score_cabinet(pack_dir: str | Path, cabinet: str, results_jsonl: str | Path) -> CabinetScore:
    """Score one cabinet's results against its pack.

    Args:
        pack_dir: The client pack.
        cabinet: Cabinet name or slug.
        results_jsonl: Results of a run over the materialized documents.

    Returns:
        The cabinet's scores.
    """
    pack = load_pack(pack_dir, cabinet)
    records, duplicates = _load_results(results_jsonl)
    names = _class_names(pack)
    score = CabinetScore(cabinet=pack.cabinet.name, documents=len(pack.manifest), duplicates=duplicates)
    pairs: list[tuple[str, str]] = []
    cases: dict[str, list[tuple[str, str]]] = {}
    for row in pack.manifest:
        record = records.get(row.doc_id)
        predicted = _predicted(names, record)
        _count_outcome(score, row, record, predicted)
        if not row.label:
            score.unlabeled += 1
            continue
        gold = _resolve(names, row.label)
        (cases.setdefault(row.case_id, []) if row.case_id else pairs).append((gold, predicted))
        if row.key_gold:
            _score_key(score, pack, row, record, predicted == gold)
    score.scored = len(pairs)
    score.correct = sum(g == p for g, p in pairs)
    score.scored_missing = sum(p == NO_RESULT for _, p in pairs)
    score.per_class = _per_class(pairs)
    _score_cases(score, cases)
    return score
