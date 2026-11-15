"""Markdown report of one cabinet's client-pack scores, from client_eval.score_cabinet.

docs/parts/client-eval.md defines every number this reports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .client_eval import CabinetScore

MIN_CONTAINED_CHARS = 6  # shorter gold keys are too short for the contained test
LIST_LIMIT = 50  # documents listed by name per report section
TENTHS_PER_PERCENT = 10
PERCENT = 100


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
        f"Attachments (files beside a document in its folder, not scored): {score.attachments}. "
        f"Labelled documents in a possible-duplicate group (same name and size in another folder): "
        f"{score.in_dup_group}.",
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
        "The case's label is the majority label of its files. All files right: every file is predicted as "
        "the case label; no result and abstained count as wrong. Any file right: at least one file is. "
        "Majority right: one class is predicted more often than any other (no result and abstained are not "
        "votes) and it is the case label; a tie is not right, and a case with no classed file is wrong.",
        "",
        f"Cases: {score.cases}. All files right: {_pct(score.cases_all, score.cases)}. "
        f"Any file right: {_pct(score.cases_any, score.cases)}. "
        f"Majority right: {_pct(score.cases_majority, score.cases)}. Cases with mixed labels: {mixed}.",
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
