"""The markdown summary of a workbook import: what the gold can and cannot claim."""

from __future__ import annotations

from collections import Counter

from .importer import WorkbookImport


def _headline(res: WorkbookImport, xlsx_path: str, root: str) -> list[str]:
    """Header and the bullet list of counts."""
    with_spans = sum(1 for r in res.rows if any("char_start" in e for e in r["metadata"]))
    lines = [
        "# Imported workbook",
        "",
        f"Source: `{xlsx_path}`",
        f"Root: `{root}`",
        "",
        f"- gold rows written: {len(res.rows)}",
        f"- documents carrying at least one gold field: {sum(1 for r in res.rows if r['metadata'])}",
        f"- documents carrying character offsets: {with_spans}",
        f"- gold fields: {sum(len(r['metadata']) for r in res.rows)}",
        f"- gold tags: {sum(len(r['tags']) for r in res.rows)}",
        f"- rows the reviewer marked as not belonging: {res.spurious_rows}",
        f"- rows a human added rather than corrected: {res.human_rows}",
        f"- confirmed abstentions, which carry no label: {len(res.confirmed_abstentions)}",
    ]
    if res.skipped_not_gold:
        lines.append(f"- rows skipped because IS_GOLD was not set: {res.skipped_not_gold}")

    verdicts = dict(sorted(res.counts.items()))
    if verdicts:
        lines += ["", "## Where the gold came from", "", "| row | verdict | count |", "| --- | --- | --- |"]
        for key, count in verdicts.items():
            kind, _, verdict = key.partition("_")
            lines.append(f"| {kind} | `{verdict}` | {count} |")
    return lines


def _limits(res: WorkbookImport) -> list[str]:
    """Why recall from this gold is an upper bound and classification is not."""
    return [
        "",
        "## What this gold cannot measure",
        "",
        "The workbook is an export of what the pipeline produced, so a row",
        "exists only where the pipeline emitted something or a human added one",
        "by hand. A field or tag the pipeline never found leaves no trace in",
        "the sheet and therefore none in this gold.",
        "",
        f"**Precision over these documents is honest. Recall is not.** Only "
        f"{res.human_rows} row(s) here were added by a human rather than "
        "corrected, so a false negative is almost unrepresented. Treat the "
        "metadata and tagging recall numbers from this file as an upper bound, "
        "and do not quote them as the recall of the system.",
        "",
        "Classification is the exception. Every gold document carries exactly",
        "one label and the reviewer either confirmed it or replaced it, so the",
        "classification score over this file is a real score.",
    ]


def _unresolved(res: WorkbookImport) -> list[str]:
    """Rows whose doc_id is not under the root."""
    if not res.unresolved_doc_ids:
        return []
    lines = [
        "",
        "## Rows pointing at content that is not here",
        "",
        "The doc_id is the SHA-256 of the file's bytes, so these rows were",
        "reviewed against a file that is not under `--root`. Same name is",
        "not same file. They are still in the gold, without a path, and will",
        "not join to any prediction until the content turns up.",
        "",
    ]
    return lines + [f"- `{d}`" for d in sorted(set(res.unresolved_doc_ids))]


def _abstentions(res: WorkbookImport) -> list[str]:
    """Abstentions the reviewer confirmed, which carry no label."""
    if not res.confirmed_abstentions:
        return []
    return [
        "",
        "## Abstentions confirmed as correct",
        "",
        "The reviewer agreed the pipeline was right not to answer. That is",
        "useful, but it is not a label: these documents still have a real",
        "class that nobody has written down, so they carry no classification",
        "gold. Put the true class in CLASS_CORRECTED to score them.",
        "",
        f"- {len(res.confirmed_abstentions)} document(s)",
    ]


def _missing_corrections(res: WorkbookImport) -> list[str]:
    """Rows marked wrong with no replacement value."""
    if not res.missing_correction:
        return []
    lines = [
        "",
        "## Rows marked wrong with nothing to replace them",
        "",
        "Nobody has said what the right answer is, so these produce no gold.",
        "",
    ]
    return lines + [f"- {d}" for d in sorted(set(res.missing_correction))]


def _orphans(res: WorkbookImport) -> list[str]:
    """Metadata or Tags rows with no gold document."""
    if not res.orphan_rows:
        return []
    counts = Counter(res.orphan_rows)
    lines = [
        "",
        "## Rows with no gold document",
        "",
        "A Metadata or Tags row whose DOC_ID is absent from the Documents",
        "tab, or whose document produced no gold row.",
        "",
    ]
    return lines + [f"- {k} ({v})" if v > 1 else f"- {k}" for k, v in sorted(counts.items())]


def _problems(res: WorkbookImport) -> list[str]:
    """Names the config does not declare."""
    if not res.problems:
        return []
    lines = [
        "",
        "## Names the config does not declare",
        "",
        "Nothing here was guessed at. Add the name to the relevant config",
        "file, or correct it in the sheet.",
        "",
    ]
    return lines + [f"- {p}" for p in res.problems]


def render_summary(res: WorkbookImport, xlsx_path: str, root: str) -> str:
    """What this gold file can and cannot be used to claim.

    Args:
        res: Result of `import_workbook`.
        xlsx_path: Workbook path, shown in the header.
        root: Document root, shown in the header.

    Returns:
        The summary as markdown, ending in a newline.
    """
    lines = _headline(res, xlsx_path, root)
    lines += _limits(res)
    lines += _unresolved(res)
    lines += _abstentions(res)
    lines += _missing_corrections(res)
    lines += _orphans(res)
    lines += _problems(res)
    return "\n".join(lines) + "\n"
