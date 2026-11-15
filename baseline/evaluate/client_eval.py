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
from .io import load_jsonl
from .scoring import prf

UNKNOWN_LABEL = "unknown"  # the label the pipeline gives when it abstains
NO_RESULT = "(no result)"  # predicted class of a document with no result line
ABSTAINED = "(abstained)"  # predicted class of a document the pipeline did not class
MIN_CONTAINED_CHARS = 6  # shorter gold keys are too short for the contained test
LIST_LIMIT = 50  # documents listed by name per report section
TENTHS_PER_PERCENT = 10
PERCENT = 100
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


def _pct(n: int, d: int) -> str:
    """Format n/d with a percentage floored to one decimal, so 100.0% means n == d; n/a when d is 0."""
    if not d:
        return "n/a"
    tenths = n * PERCENT * TENTHS_PER_PERCENT // d
    return f"{n}/{d} ({tenths // TENTHS_PER_PERCENT}.{tenths % TENTHS_PER_PERCENT}%)"


def _documents(n: int) -> str:
    return f"{n} document" if n == 1 else f"{n} documents"


HOW_TO_READ = [
    "## How to read this",
    "",
    "- Every labelled document in the manifest is counted. A document with no result counts as wrong.",
    "- Class: right when the predicted class equals the folder's type, or one of its aliases, "
    "ignoring case and spacing. Abstaining counts as wrong.",
    "- Key exact: the type's key field equals the file-name key after trimming, case folding, "
    "collapsing spaces and mapping dashes to '-'.",
    "- Key loose (ignoring separators): the same, with every space and punctuation mark removed.",
    "- Key contained: the file-name key appears in any extracted value as a whole run, not inside a longer "
    f"code. Keys under {MIN_CONTAINED_CHARS} letters and digits are too short to test.",
    "- The key field is read from the folder's (gold) type, even when the predicted class is wrong. "
    "A value written as a float ('123.0') is compared as '123'. The field's raw and normalized values "
    "are both compared; a match on either counts.",
    "- Only the class and the key field are scored: no other field is scored. Folder labels may be imperfect.",
    "",
]


def _overview(score: CabinetScore, header: list[str]) -> list[str]:
    return [
        f"# Client evaluation: {score.cabinet}",
        "",
        *[f"- {line}" for line in header],
        "",
        f"Documents in the manifest: {score.documents}. No result: {score.missing}. "
        f"Produced a result with errors: {score.failed}. Abstained: {score.abstained}. "
        f"Unlabelled (not scored): {score.unlabeled}. Duplicated results lines: {score.duplicates} "
        "(the last line of each is used).",
        "",
        *HOW_TO_READ,
    ]


def _classification(score: CabinetScore) -> list[str]:
    rows = [
        f"| {n} | {c.precision:.3f} | {c.recall:.3f} | {c.support} | {c.predicted} |"
        for n, c in score.per_class.items()
    ]
    return [
        "## Classification (single documents)",
        "",
        f"Accuracy {_pct(score.correct, score.scored)} \u2014 {_documents(score.scored_missing)} produced no result",
        "",
        "| class | precision | recall | gold | predicted |",
        "|---|---|---|---|---|",
        *rows,
        "",
    ]


def _key(score: CabinetScore) -> list[str]:
    testable = score.key_scored - score.key_too_short
    return [
        "## Key field",
        "",
        f"Documents with a file-name key: {score.key_scored}.",
        "",
        f"- Key exact: {_pct(score.key_exact, score.key_scored)}.",
        f"- Key loose (ignoring separators): {_pct(score.key_loose, score.key_scored)}.",
        f"- Key contained: {_pct(score.key_contained, testable)}; "
        f"{_documents(score.key_too_short)} too short to test.",
        f"- Class and key both right (exact): {_pct(score.key_exact_class_ok, score.key_scored)}. "
        f"Of the {_documents(score.key_class_ok)} whose class was right, the key was exact in "
        f"{_pct(score.key_exact_class_ok, score.key_class_ok)}.",
        "",
    ]


def _cases(score: CabinetScore) -> list[str]:
    mixed = ", ".join(score.mixed_cases) or "none"
    return [
        "## Case bundles",
        "",
        "The case's label is the majority label of its files. Majority: the most common class prediction "
        "across its files is the case label. Any-hit: at least one file is. A case with no classed file is wrong.",
        "",
        f"Cases: {score.cases}. Majority: {_pct(score.cases_majority, score.cases)}. "
        f"Any-hit: {_pct(score.cases_any, score.cases)}. Cases with mixed labels: {mixed}.",
        "",
    ]


def _listing(title: str, note: str, paths: list[str]) -> list[str]:
    more = [f"- and {len(paths) - LIST_LIMIT} more"] if len(paths) > LIST_LIMIT else []
    return [f"## {title}", "", note, "", *[f"- `{p}`" for p in paths[:LIST_LIMIT]], *more, ""]


def render_client_report(score: CabinetScore, results: str = "", header: list[str] | None = None) -> str:
    """Markdown report of one cabinet's scores.

    Args:
        score: From score_cabinet.
        results: The results file, shown in the header.
        header: More header lines, such as the run time and the stages that ran.

    Returns:
        The report text.
    """
    lines = [*([f"Results: `{results}`"] if results else []), *(header or [])]
    return "\n".join(
        [
            *_overview(score, lines),
            *_classification(score),
            *_key(score),
            *_cases(score),
            *_listing("No result", "No result line: counted as wrong everywhere.", score.no_result),
            *_listing(
                "Produced a result with errors",
                "The record carries errors. Its class and key are still scored as they are.",
                score.with_errors,
            ),
        ]
    )
