"""REPORT.md of a client pack: what the build left out, could not decide or changed, per cabinet.

``clientpack_build`` collects the findings on each in-progress cabinet and
calls ``render_report`` once at the end.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from .clientpack import EXCLUDED_FILE, FIELD_OVERRIDES_FILE, LABEL_MAP_FILE, PACK_FILE

if TYPE_CHECKING:
    from .clientpack_build import _Cabinet

OTHER_RENAMED = "Other (client)"  # a schema type called "other" would collide with the reserved class
OVERRIDES, AUTO = "overrides", "auto"  # the two sections of label_map.yaml per cabinet
LOOSE_NOTE = "covers too few of its words; add an override to use it"


def _label_sections(cab: _Cabinet) -> list[tuple[str, list[str]]]:
    return [
        (
            "Ambiguous labels",
            [f"- `{lb}`: candidates {', '.join(f'`{c}`' for c in cs)}" for lb, cs in cab.ambiguous.items()],
        ),
        (
            "Unmapped labels",
            [
                f"- `{lb}`: candidate `{c}` {LOOSE_NOTE}" if c else f"- `{lb}`: no schema type matched"
                for lb, c in cab.unmapped
            ],
        ),
        ("Overrides outside the schema", [f"- `{lb}` -> `{t}`" for lb, t in cab.foreign]),
        (
            "Aliases dropped (equal to another class name)",
            [f"- `{a}` of `{c}`: `{a}` is a class of its own" for a, c in cab.dropped_aliases],
        ),
        ("Wrapper folders skipped", [f"- `{w}`" for w in cab.wrappers]),
    ]


def _schema_sections(cab: _Cabinet) -> list[tuple[str, list[str]]]:
    notes = cab.field_notes
    conflicts = notes.conflicts
    return [
        ("Schema types merged", [f"- `{other}` merged into `{kept}`" for kept, other in cab.schema.merged]),
        ("Schema types renamed", [f"- `{raw}` renamed to `{OTHER_RENAMED}`" for raw in cab.schema.renamed]),
        (
            "Field definitions that disagree",
            [f"- `{t}` / `{f}`: {what} `{kept}` and `{other}`; used `{kept}`" for t, f, what, kept, other in conflicts],
        ),
        ("Unrecognised field types (read as optional)", [f"- `{t}` / `{f}`: `{v}`" for t, f, v in notes.bad_roles]),
        (f"Unknown {FIELD_OVERRIDES_FILE} entries (ignored)", [f"- {u}" for u in cab.unknown_overrides]),
    ]


def _file_sections(cab: _Cabinet) -> list[tuple[str, list[str]]]:
    return [
        ("Duplicates dropped", [f"- dropped {d} (kept {k})" for d, k in cab.duplicates]),
        (
            "Possible duplicates (same name and size, different folder, or a marked copy)",
            [f"- group `{g}`: " + ", ".join(f"`{p}`" for p in paths) for g, paths in cab.dup_groups],
        ),
        ("Oversize files", [f"- `{e.rel_path}`" for e in cab.excluded if e.reason == "oversize"]),
    ]


def _cabinet_report(cab: _Cabinet) -> list[str]:
    reasons = Counter(e.reason for e in cab.excluded)
    lines = [f"## {cab.name} (`{cab.slug}`)", ""]
    lines.append("Excluded: " + (", ".join(f"{r} {n}" for r, n in sorted(reasons.items())) or "none"))
    lines.append(f"Full list in `{cab.slug}/{EXCLUDED_FILE}`.")
    for title, items in [*_label_sections(cab), *_schema_sections(cab), *_file_sections(cab)]:
        if items:
            lines += ["", f"### {title}", "", *items]
    return [*lines, ""]


def render_report(cabinets: list[_Cabinet], skipped: list[str], unknown_cabinets: list[str]) -> str:
    """REPORT.md text.

    Args:
        cabinets: The cabinets as built.
        skipped: Workbooks under no cabinet.
        unknown_cabinets: field_overrides.yaml cabinets the inventory does not have.

    Returns:
        The report.
    """
    lines = [
        "# Client pack report",
        "",
        f"Set labels under `{OVERRIDES}:` in `{LABEL_MAP_FILE}` and re-run; `{AUTO}:` is rewritten each build.",
        f"Build time is in `{PACK_FILE}`.",
        "",
    ]
    if skipped:
        lines += ["## Workbooks under no cabinet (types not used)", "", *(f"- `{w}`" for w in skipped), ""]
    if unknown_cabinets:
        title = f"## Unknown cabinets in {FIELD_OVERRIDES_FILE} (ignored)"
        lines += [title, "", *(f"- `{c}`" for c in unknown_cabinets), ""]
    for cab in cabinets:
        lines += _cabinet_report(cab)
    return "\n".join(lines)
