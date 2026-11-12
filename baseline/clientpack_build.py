"""Build a client pack from the inventory and schema CSVs of one client share.

``build_pack`` screens out references, archives and oversize files, drops
duplicate copies across the whole share, maps folder labels to schema types,
finds case bundles and key values, and writes the files ``clientpack``
describes. label_map.yaml is read back first, so hand edits survive a re-run.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from .clientpack import (
    EXCLUDED_FILE,
    FIELDS_FILE,
    LABEL_MAP_FILE,
    MANIFEST_FILE,
    OTHER_CLASS,
    PACK_FILE,
    REPORT_FILE,
    TAXONOMY_FILE,
    CabinetEntry,
    ExcludedRow,
    FieldSpec,
    ManifestRow,
    PackInfo,
    PackSummary,
    TaxonomyClass,
    make_doc_id,
    read_rows,
    read_yaml,
    slugify,
    write_rows,
    write_yaml,
)
from .clientpack_rules import (
    IMAGE_EXT,
    SAMPLE_EXT,
    case_folders,
    dedup_rank,
    field_origin,
    is_reference_image,
    key_gold,
    match_label,
    norm,
    parse_dtype,
    parse_role,
    parse_source,
)
from .io import atomic_write_text
from .logging_setup import get_logger

log = get_logger(__name__)

DEFAULT_MAX_SIZE_MB = 50
KB_PER_MB = 1024
UNRECOGNISED_ROLE = "optional"

Row = dict[str, str]


@dataclass
class _Cabinet:
    """One cabinet while it is being built."""

    name: str
    rows: list[Row]
    types: dict[str, list[Row]]
    overrides: dict[str, str]
    docs: list[Row] = field(default_factory=list)
    excluded: list[ExcludedRow] = field(default_factory=list)
    duplicates: list[tuple[str, str]] = field(default_factory=list)
    mapping: dict[str, str] = field(default_factory=dict)
    unmapped: list[str] = field(default_factory=list)
    ambiguous: dict[str, tuple[str, ...]] = field(default_factory=dict)
    foreign: list[tuple[str, str]] = field(default_factory=list)
    bad_roles: list[tuple[str, str, str]] = field(default_factory=list)

    @property
    def slug(self) -> str:
        """Directory name of the cabinet in the pack."""
        return slugify(self.name)


def _rel_path(row: Row) -> str:
    return "/".join(p for p in (row["cabinet"], row["folder"], row["name"]) if p)


def _parts(row: Row, cabinet: str) -> tuple[str, ...]:
    """Folder parts below the cabinet, without a nested copy of the cabinet's own name."""
    parts = [p for p in row["folder"].split("/") if p]
    while parts and norm(parts[0]) == norm(cabinet):
        parts.pop(0)
    return tuple(parts)


def _label_of(row: Row, cabinet: str) -> tuple[str, str]:
    """The label a document carries and where it came from: its top folder, else its file stem."""
    parts = _parts(row, cabinet)
    return (parts[0], "folder") if parts else (PurePosixPath(row["name"]).stem, "filename")


def _exclude_reason(row: Row, has_samples: bool, type_labels: set[str], max_kb: int) -> str | None:
    kind = row["kind"]
    if kind in ("archive", "reference"):
        return kind
    ext = row["ext"].lower()
    if kind != "document" or ext not in SAMPLE_EXT | IMAGE_EXT:
        return "unsupported"
    if is_reference_image(row["name"], has_samples, type_labels):
        return "reference"
    return "oversize" if int(row["size_kb"] or 0) > max_kb else None


def _screen(cab: _Cabinet, max_kb: int) -> None:
    """Split the cabinet's rows into candidate documents and excluded files."""
    sample_folders = {(r["folder"], r["in_zip"]) for r in cab.rows if r["ext"].lower() in SAMPLE_EXT}
    type_labels = {norm(p) for r in cab.rows for p in _parts(r, cab.name)} | {norm(t) for t in cab.types}
    for row in cab.rows:
        has_samples = (row["folder"], row["in_zip"]) in sample_folders
        reason = _exclude_reason(row, has_samples, type_labels, max_kb)
        if reason:
            cab.excluded.append(ExcludedRow(rel_path=_rel_path(row), reason=reason))  # type: ignore[arg-type]
        else:
            cab.docs.append(row)


def _dedup(cabinets: list[_Cabinet]) -> None:
    """Keep one copy of each (name, size_kb) across the share; the rest are excluded as duplicates."""
    owner = {id(row): cab for cab in cabinets for row in cab.docs}
    groups: dict[tuple[str, str], list[Row]] = {}
    for cab in cabinets:
        for row in cab.docs:
            groups.setdefault((row["name"], row["size_kb"]), []).append(row)
    dropped: set[int] = set()
    for rows in groups.values():
        rows.sort(key=lambda r: dedup_rank(r["cabinet"], r["folder"], r["in_zip"]))
        for row in rows[1:]:
            cab = owner[id(row)]
            cab.excluded.append(ExcludedRow(rel_path=_rel_path(row), reason="duplicate"))
            cab.duplicates.append((_rel_path(row), _rel_path(rows[0])))
            dropped.add(id(row))
    for cab in cabinets:
        cab.docs = [r for r in cab.docs if id(r) not in dropped]


def _map_labels(cab: _Cabinet) -> None:
    """Resolve every label: label_map.yaml override first, then match_label."""
    for label in sorted({_label_of(d, cab.name)[0] for d in cab.docs}):
        if label in cab.overrides:
            target = cab.overrides[label]
            if target not in cab.types:
                cab.foreign.append((label, target))
        else:
            found = match_label(label, cab.types)
            target = found.target
            if found.ambiguous:
                cab.ambiguous[label] = found.candidates
            elif not target:
                cab.unmapped.append(label)
        cab.mapping[label] = target


def _field_specs(cab: _Cabinet, type_name: str) -> list[FieldSpec]:
    specs: dict[str, FieldSpec] = {}
    for row in cab.types.get(type_name, []):
        name = row["index_field"]
        role = parse_role(row["field_type"])
        if role is None:
            cab.bad_roles.append((type_name, name, row["field_type"]))
        role = role or UNRECOGNISED_ROLE
        dtype, date_format = parse_dtype(row["data_type"])
        origin = field_origin(name, row["field_type"], parse_source(row["source"]), role)
        spec = FieldSpec(name=name, dtype=dtype, date_format=date_format, role=role, origin=origin)  # type: ignore[arg-type]
        specs.setdefault(norm(name), spec)
    return list(specs.values())


def _manifest(cab: _Cabinet, fields: dict[str, list[FieldSpec]]) -> list[ManifestRow]:
    cases = case_folders(_parts(d, cab.name) for d in cab.docs)
    keyed = {t for t, specs in fields.items() if any(f.role == "key" for f in specs)}
    rows = []
    for d in cab.docs:
        parts = _parts(d, cab.name)
        label, source = _label_of(d, cab.name)
        target = cab.mapping.get(label, "")
        case_id = parts[1] if parts and parts[0] in cases else ""
        rel = _rel_path(d)
        row = ManifestRow(
            doc_id=make_doc_id(cab.name, rel),
            rel_path=rel,
            in_zip=d["in_zip"],
            case_id=case_id,
            label=target,
            label_source=source if target else "none",  # type: ignore[arg-type]
            key_gold=key_gold(d["name"]) if target in keyed and not case_id else "",
            pages=int(d["pages"]) if d["pages"].isdigit() else None,
            size_kb=int(d["size_kb"] or 0),
        )
        rows.append(row)
    return sorted(rows, key=lambda r: r.rel_path)


def _taxonomy(cab: _Cabinet, manifest: list[ManifestRow], all_types: bool) -> list[TaxonomyClass]:
    with_samples = {m.label for m in manifest if m.label}
    names = sorted((set(cab.types) | set(cab.mapping.values())) - {""}) if all_types else sorted(with_samples)
    classes = []
    for name in names:
        aliases = sorted(lb for lb, t in cab.mapping.items() if t == name and norm(lb) != norm(name))
        classes.append(TaxonomyClass(name=name, aliases=aliases, has_samples=name in with_samples))
    classes.append(TaxonomyClass(name=OTHER_CLASS, has_samples=any(not m.label for m in manifest)))
    return classes


def _write_cabinet(out: Path, cab: _Cabinet, all_types: bool) -> CabinetEntry:
    names = sorted(set(cab.types) | set(cab.mapping.values()) - {""})
    fields = {t: _field_specs(cab, t) for t in names}
    manifest = _manifest(cab, fields)
    classes = _taxonomy(cab, manifest, all_types)
    keep = {c.name for c in classes}
    folder = out / cab.slug
    folder.mkdir(parents=True, exist_ok=True)
    write_yaml(folder / TAXONOMY_FILE, {"classes": [c.model_dump() for c in classes]})
    write_yaml(folder / FIELDS_FILE, {t: [f.model_dump() for f in specs] for t, specs in fields.items() if t in keep})
    write_rows(folder / MANIFEST_FILE, list(manifest), list(ManifestRow.model_fields))
    excluded = sorted(cab.excluded, key=lambda e: e.rel_path)
    write_rows(folder / EXCLUDED_FILE, list(excluded), list(ExcludedRow.model_fields))
    with_samples = sum(c.has_samples for c in classes if c.name != OTHER_CLASS)
    return CabinetEntry(slug=cab.slug, name=cab.name, documents=len(manifest), types_with_samples=with_samples)


def _cabinets(inventory: list[Row], schema: list[Row], label_map: dict[str, dict[str, str]]) -> list[_Cabinet]:
    types: dict[str, dict[str, list[Row]]] = {}
    for row in schema:
        cabinet = row["workbook"].replace("\\", "/").split("/", 1)[0]
        types.setdefault(cabinet, {}).setdefault(row["document_type"], []).append(row)
    names = sorted({r["cabinet"] for r in inventory if r["cabinet"]})
    return [
        _Cabinet(
            name=name,
            rows=[r for r in inventory if r["cabinet"] == name],
            types={t: rows for t, rows in types.get(name, {}).items() if t},
            overrides={k: v for k, v in label_map.get(name, {}).items() if v},
        )
        for name in names
    ]


def _read_label_map(path: Path) -> dict[str, dict[str, str]]:
    data = read_yaml(path) if path.exists() else None
    return {str(c): {str(k): str(v or "") for k, v in (m or {}).items()} for c, m in (data or {}).items()}


def _report_sections(cab: _Cabinet) -> list[str]:
    reasons = Counter(e.reason for e in cab.excluded)
    lines = [f"## {cab.name} (`{cab.slug}`)", ""]
    lines.append("Excluded: " + (", ".join(f"{r} {n}" for r, n in sorted(reasons.items())) or "none"))
    lines.append(f"Full list in `{cab.slug}/{EXCLUDED_FILE}`.")
    sections = [
        (
            "Ambiguous labels",
            [f"- `{lb}`: candidates {', '.join(f'`{c}`' for c in cs)}" for lb, cs in cab.ambiguous.items()],
        ),
        ("Unmapped labels", [f"- `{lb}`: no schema type matched" for lb in cab.unmapped]),
        ("Overrides outside the schema", [f"- `{lb}` -> `{t}`" for lb, t in cab.foreign]),
        ("Duplicates dropped", [f"- `{d}` (kept `{k}`)" for d, k in cab.duplicates]),
        ("Oversize files", [f"- `{e.rel_path}`" for e in cab.excluded if e.reason == "oversize"]),
        ("Unrecognised field types (read as optional)", [f"- `{t}` / `{f}`: `{v}`" for t, f, v in cab.bad_roles]),
    ]
    for title, items in sections:
        if items:
            lines += ["", f"### {title}", "", *items]
    return [*lines, ""]


def _render_report(cabinets: list[_Cabinet]) -> str:
    lines = [
        "# Client pack report",
        "",
        f"Fill empty entries in `{LABEL_MAP_FILE}` and re-run; edits are kept. Build time is in `{PACK_FILE}`.",
        "",
    ]
    for cab in cabinets:
        lines += _report_sections(cab)
    return "\n".join(lines)


def build_pack(
    inventory_csv: str | Path,
    schema_csv: str | Path,
    out_dir: str | Path,
    *,
    all_types: bool = False,
    max_size_mb: float = DEFAULT_MAX_SIZE_MB,
    share_root: str = "",
) -> PackSummary:
    """Build or refresh a client pack.

    Args:
        inventory_csv: inventory.csv from scripts/inventory_share.py.
        schema_csv: schema.csv from the same script.
        out_dir: The pack directory; created if missing.
        all_types: Write every schema type to taxonomy.yaml, not only those with samples.
        max_size_mb: Larger documents are excluded as oversize.
        share_root: Recorded in pack.yaml for reference.

    Returns:
        Counts of what was written.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    label_map = _read_label_map(out / LABEL_MAP_FILE)
    cabinets = _cabinets(read_rows(Path(inventory_csv)), read_rows(Path(schema_csv)), label_map)
    for cab in cabinets:
        _screen(cab, int(max_size_mb * KB_PER_MB))
    _dedup(cabinets)
    for cab in cabinets:
        _map_labels(cab)
        label_map[cab.name] = {**label_map.get(cab.name, {}), **cab.mapping}
    entries = [_write_cabinet(out, cab, all_types) for cab in cabinets]
    info = PackInfo(
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), share_root=share_root, cabinets=entries
    )
    write_yaml(out / PACK_FILE, info.model_dump())
    write_yaml(out / LABEL_MAP_FILE, label_map)
    atomic_write_text(out / REPORT_FILE, _render_report(cabinets))
    summary = PackSummary(
        out_dir=str(out),
        cabinets=entries,
        documents=sum(e.documents for e in entries),
        excluded=sum(len(c.excluded) for c in cabinets),
        unmapped_labels=sum(len(c.unmapped) for c in cabinets),
        ambiguous_labels=sum(len(c.ambiguous) for c in cabinets),
    )
    log.info("client pack built", extra=summary.model_dump(exclude={"cabinets"}))
    return summary
