"""Score a run over one client cabinet against the labels its client pack carries.

The pack's only gold is the folder a document sat in (its type) and, for most
single-document types, the key value in its file name. So this scores the
class and the key field and nothing else. Results are joined to the manifest
by file stem: ``clientpack.materialize`` names each copy ``<doc_id><ext>``.

Unlike the rest of this package it reads a client pack, so it imports
``baseline.clientpack``; that module is a data format, not pipeline code.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import BaseModel, Field

from ..clientpack import ClientPack, ManifestRow, load_pack
from ..clientpack_rules import compact, norm
from .io import load_jsonl
from .scoring import prf

UNKNOWN_LABEL = "unknown"  # the label the pipeline gives when it abstains
PERCENT = 100


class ClassScore(BaseModel):
    """Precision and recall of one class."""

    precision: float
    recall: float
    support: int


class CabinetScore(BaseModel):
    """Everything client-eval reports for one cabinet."""

    cabinet: str
    documents: int
    missing: int  # in the manifest, not in the results
    failed: int  # the record carries diagnostics errors
    abstained: int  # no class, or the unknown label
    unlabeled: int  # no gold label: reported, not scored
    scored: int
    correct: int
    per_class: dict[str, ClassScore] = Field(default_factory=dict)
    key_scored: int = 0
    key_exact: int = 0
    key_contained: int = 0
    cases: int = 0
    cases_majority: int = 0
    cases_any: int = 0

    @property
    def accuracy(self) -> float:
        """Share of labeled single documents classed right; abstentions count as wrong."""
        return self.correct / self.scored if self.scored else 0.0


def _stem(record: dict[str, Any]) -> str:
    return PurePosixPath(str(record.get("filename") or record.get("source_path") or "")).stem


def _predicted(record: dict[str, Any]) -> str:
    label = str((record.get("classification") or {}).get("label") or "")
    return "" if label == UNKNOWN_LABEL else label


def _key_hits(pack: ClientPack, row: ManifestRow, record: dict[str, Any]) -> tuple[bool, bool]:
    """(exact match of the key field, gold contained in any extracted value)."""
    gold = compact(row.key_gold)
    key = norm(pack.key_field(row.label))
    values = [
        (norm(str(m.get("field", ""))), str(m.get("normalized_value") or m.get("value") or ""))
        for m in record.get("metadata", [])
    ]
    exact = any(name == key and compact(v) == gold for name, v in values)
    return exact, any(gold in compact(v) for _, v in values)


def _per_class(pairs: list[tuple[str, str]]) -> dict[str, ClassScore]:
    out = {}
    for name in sorted({g for g, _ in pairs}):
        tp = sum(g == name and p == name for g, p in pairs)
        fp = sum(g != name and p == name for g, p in pairs)
        fn = sum(g == name and p != name for g, p in pairs)
        res = prf(tp, fp, fn)
        out[name] = ClassScore(precision=res["precision"], recall=res["recall"], support=tp + fn)
    return out


def _score_cases(score: CabinetScore, cases: dict[str, list[tuple[str, str]]]) -> None:
    """Majority: the most common non-empty prediction equals the case label. Any-hit: one document does."""
    score.cases = len(cases)
    for pairs in cases.values():
        gold = pairs[0][0]
        votes = Counter(p for _, p in pairs if p)
        score.cases_majority += bool(votes) and votes.most_common(1)[0][0] == gold
        score.cases_any += gold in votes


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
    records = {_stem(r): r for r in load_jsonl(results_jsonl)}
    score = CabinetScore(
        cabinet=pack.cabinet.name,
        documents=len(pack.manifest),
        missing=0,
        failed=0,
        abstained=0,
        unlabeled=0,
        scored=0,
        correct=0,
    )
    pairs: list[tuple[str, str]] = []
    cases: dict[str, list[tuple[str, str]]] = {}
    for row in pack.manifest:
        record = records.get(row.doc_id)
        if record is None:
            score.missing += 1
            continue
        predicted = _predicted(record)
        score.failed += bool((record.get("diagnostics") or {}).get("errors"))
        score.abstained += not predicted
        if not row.label:
            score.unlabeled += 1
        elif row.case_id:
            cases.setdefault(row.case_id, []).append((row.label, predicted))
        else:
            pairs.append((row.label, predicted))
        if row.key_gold and row.label:
            exact, contained = _key_hits(pack, row, record)
            score.key_scored += 1
            score.key_exact += exact
            score.key_contained += contained
    score.scored = len(pairs)
    score.correct = sum(g == p for g, p in pairs)
    score.per_class = _per_class(pairs)
    _score_cases(score, cases)
    return score


def _pct(n: int, d: int) -> str:
    return f"{n}/{d} ({PERCENT * n / d:.1f}%)" if d else "0/0"


def render_client_report(score: CabinetScore, results: str = "") -> str:
    """Markdown report of one cabinet's scores.

    Args:
        score: From score_cabinet.
        results: The results file name, shown in the header.

    Returns:
        The report text.
    """
    lines = [
        f"# Client evaluation: {score.cabinet}",
        "",
        f"Results: `{results}`" if results else "",
        "",
        f"Documents in the manifest: {score.documents}. Missing from the results: {score.missing}. "
        f"Failed (record has errors): {score.failed}. Abstained: {score.abstained}. "
        f"Unlabeled (not scored): {score.unlabeled}.",
        "",
        "Only the class and the key field are scored. The pack has no gold for any other field, "
        "so no other field is scored.",
        "",
        "## Classification (single documents)",
        "",
        f"Accuracy: {_pct(score.correct, score.scored)}. Abstentions count as wrong.",
        "",
        "| class | precision | recall | support |",
        "|---|---|---|---|",
        *[f"| {n} | {c.precision:.3f} | {c.recall:.3f} | {c.support} |" for n, c in score.per_class.items()],
        "",
        "## Key field",
        "",
        "Gold is the file name stem. Exact: the type's key field equals it after case, space and punctuation "
        "are removed. Contained: it appears inside any extracted value.",
        "",
        f"Exact: {_pct(score.key_exact, score.key_scored)}. Contained: {_pct(score.key_contained, score.key_scored)}.",
        "",
        "## Case bundles",
        "",
        "The type labels the case, not each file. Majority: the most common prediction across the case's "
        "documents is the case type. Any-hit: at least one document is.",
        "",
        f"Cases: {score.cases}. Majority: {_pct(score.cases_majority, score.cases)}. "
        f"Any-hit: {_pct(score.cases_any, score.cases)}.",
        "",
    ]
    return "\n".join(lines)
