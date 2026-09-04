"""Turn a client supplied label CSV into a gold file the harness can score.

Classification.csv is the only real ground truth this project has, and it costs
nothing to use: it is File_Name,Document_Type over the labelled set. What it is
not is a gold file, for three reasons this module exists to handle.

1. Its labels are the client's, not ours. They go through taxonomy.yaml, so the
   mapping lives in exactly one place and a label that stops mapping to a
   declared class is a loud failure rather than a silent miss.
2. Eight of its fourteen labels are lossy: seven legal ones collapse to a
   single class because they describe what a document is used for rather than
   what it is. A score computed over them is a class level score and the report
   has to say so, so every row carries its original label alongside the mapped
   one and the summary counts how many rows were collapsed.
3. Its file names are not unique. Four names appear twice under two labels, and
   three of those four are different files that merely share a name. Rows are
   therefore resolved by <root>/<Document_Type>/<File_Name> first, which is
   exact, and keyed on a content addressed doc_id so two files sharing a name
   can never collide.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Config
from .logging_setup import get_logger
from .schema import compute_doc_id

log = get_logger(__name__)


@dataclass
class ImportResult:
    rows: list[dict[str, Any]] = field(default_factory=list)
    unmapped_labels: list[str] = field(default_factory=list)
    missing_files: list[str] = field(default_factory=list)
    lossy_rows: int = 0

    @property
    def collapsed(self) -> dict[str, list[str]]:
        """Mapped class to the client labels that collapsed into it."""
        out: dict[str, set[str]] = {}
        for r in self.rows:
            if r.get("label_is_lossy"):
                out.setdefault(r["label"], set()).add(r["client_label"])
        return {k: sorted(v) for k, v in sorted(out.items())}


def _resolve(root: Path, folder: str, name: str) -> Path | None:
    """Exact folder path first. Only then fall back to searching by name."""
    direct = root / folder / name
    if direct.is_file():
        return direct
    hits = sorted(p for p in root.rglob(name) if p.is_file())
    return hits[0] if len(hits) == 1 else None


def import_labels(
    csv_path: str | Path,
    root: str | Path,
    cfg: Config,
    name_column: str = "File_Name",
    label_column: str = "Document_Type",
) -> ImportResult:
    """Read the label CSV and emit gold rows, one per CSV row."""
    root = Path(root)
    mapping = cfg.client_label_map
    lossy = set(cfg.lossy_client_labels)
    if not mapping:
        raise ValueError("taxonomy.yaml declares no mappings; nothing to map through")

    res = ImportResult()
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            name = (row.get(name_column) or "").strip()
            client_label = (row.get(label_column) or "").strip()
            if not name or not client_label:
                continue
            if client_label not in mapping:
                res.unmapped_labels.append(client_label)
                continue
            target = mapping[client_label]
            if target is None:
                continue  # a recorded refusal to map, not an error
            path = _resolve(root, client_label, name)
            if path is None:
                res.missing_files.append(name)
                continue
            is_lossy = client_label in lossy
            res.lossy_rows += is_lossy
            res.rows.append({
                "doc_id": compute_doc_id(path),
                "path": str(path).replace("\\", "/"),
                "filename": path.name,
                "label": target,
                # Kept so a report can say what was collapsed and so the
                # mapping can be audited without re-reading the CSV.
                "client_label": client_label,
                "label_is_lossy": is_lossy,
            })

    if res.unmapped_labels:
        log.error("labels absent from taxonomy.yaml",
                  extra={"labels": sorted(set(res.unmapped_labels))})
    if res.missing_files:
        log.warning("csv rows with no file on disk",
                    extra={"count": len(res.missing_files)})
    return res


def render_summary(res: ImportResult, csv_path: str, root: str) -> str:
    """What the gold file can and cannot be used to claim."""
    lines = [
        "# Imported labels", "",
        f"Source: `{csv_path}`", f"Root: `{root}`", "",
        f"- gold rows written: {len(res.rows)}",
        f"- distinct doc_ids: {len({r['doc_id'] for r in res.rows})}",
        f"- rows carrying a lossy label: {res.lossy_rows}",
    ]
    if res.missing_files:
        lines.append(f"- csv rows with no file found: {len(res.missing_files)}")
    if res.unmapped_labels:
        lines.append(f"- labels not in taxonomy.yaml: {sorted(set(res.unmapped_labels))}")

    if res.collapsed:
        lines += ["", "## Collapsed labels", "",
                  "These client labels cannot be recovered from the document text.",
                  "Any score over them is a class level score, not a label level one.",
                  "", "| class | client labels collapsed into it |", "| --- | --- |"]
        for cls, labels in res.collapsed.items():
            lines.append(f"| `{cls}` | {', '.join(labels)} |")

    dupes = sorted({r["filename"] for r in res.rows
                    if sum(1 for x in res.rows if x["filename"] == r["filename"]) > 1})
    if dupes:
        lines += ["", "## File names appearing more than once", "",
                  "Keyed on content addressed doc_id, so these do not collide.", ""]
        lines += [f"- `{d}`" for d in dupes]
    return "\n".join(lines) + "\n"
