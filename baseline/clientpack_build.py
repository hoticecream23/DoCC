"""Build a client pack from the inventory and schema CSVs of one client share.

Over 400 lines on purpose: screening, layout, dedup and label_map all act on
the same in-progress ``_Cabinet``; splitting them would only move that state
between modules. Field specs live in ``clientpack_fields`` and REPORT.md in
``clientpack_report``. ``build_pack`` screens out references, archives and
oversize files, finds each document's label folder and case, drops duplicate
copies, maps labels to schema types and writes the files ``clientpack``
describes. label_map.yaml, field_overrides.yaml and the slugs in pack.yaml
are read first, so hand edits and folder names survive a re-run.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from .clientpack import (
    EXCLUDED_FILE,
    FIELD_OVERRIDES_FILE,
    FIELDS_FILE,
    LABEL_MAP_FILE,
    MANIFEST_FILE,
    OTHER_CLASS,
    PACK_FILE,
    REPORT_FILE,
    TAXONOMY_FILE,
    CabinetEntry,
    ClientPackError,
    ExcludedRow,
    FieldSpec,
    ManifestRow,
    PackInfo,
    PackSummary,
    TaxonomyClass,
    assign_slugs,
    make_doc_id,
    read_pack_info,
    read_rows,
    read_yaml,
    write_rows,
    write_yaml,
)
from .clientpack_fields import FieldNotes, RawOverrides, field_specs, read_field_overrides, unknown_overrides
from .clientpack_report import AUTO, OTHER_RENAMED, OVERRIDES, render_report
from .clientpack_rules import (
    IMAGE_EXT,
    SAMPLE_EXT,
    Folder,
    child_roles,
    clean_stem,
    dedup_rank,
    folder_tree,
    is_reference_image,
    key_gold,
    label_depth,
    match_label,
    norm,
    wrapper_folders,
)
from .io import atomic_write_text
from .logging_setup import get_logger

log = get_logger(__name__)

DEFAULT_MAX_SIZE_MB = 50
KB_PER_MB = 1024
HASH_COLUMN = "sha256"  # an inventory column with the file's content hash, when an inventory carries one
DUP_GROUP_LEN = 8  # hex characters of a possible-duplicate group id
MIN_DUP_PATHS = 2  # a possible-duplicate group spans at least two paths

Row = dict[str, str]
DedupKey = tuple[str, str, str]  # (cabinet, path below it, size_kb)


@dataclass(frozen=True)
class _Place:
    """Where a document sits: its label, where the label came from, its case, and document or attachment."""

    label: str
    source: str
    case_id: str
    role: str = "document"


@dataclass
class _Schema:
    """One cabinet's schema types, merged by normalised name."""

    types: dict[str, list[Row]] = field(default_factory=dict)
    display: dict[str, tuple[str, str]] = field(default_factory=dict)  # norm -> (display name, first spelling)
    merged: list[tuple[str, str]] = field(default_factory=list)
    renamed: list[str] = field(default_factory=list)


@dataclass
class _Cabinet:
    """One cabinet while it is being built."""

    name: str
    slug: str
    rows: list[Row]
    schema: _Schema
    overrides: dict[str, str]
    field_overrides: RawOverrides = field(default_factory=dict)
    docs: list[Row] = field(default_factory=list)
    place: dict[int, _Place] = field(default_factory=dict)
    excluded: list[ExcludedRow] = field(default_factory=list)
    duplicates: list[tuple[str, str]] = field(default_factory=list)  # (dropped, kept), each with loose or zip
    dup_group: dict[int, str] = field(default_factory=dict)  # id(doc) -> possible-duplicate group id
    dup_groups: list[tuple[str, list[str]]] = field(default_factory=list)
    wrappers: list[str] = field(default_factory=list)
    mapping: dict[str, str] = field(default_factory=dict)
    unmapped: list[tuple[str, str]] = field(default_factory=list)
    ambiguous: dict[str, tuple[str, ...]] = field(default_factory=dict)
    foreign: list[tuple[str, str]] = field(default_factory=list)
    dropped_aliases: list[tuple[str, str]] = field(default_factory=list)  # (alias, class it was mapped to)
    field_notes: FieldNotes = field(default_factory=FieldNotes)
    unknown_overrides: list[str] = field(default_factory=list)

    @property
    def types(self) -> dict[str, list[Row]]:
        """Schema type name -> its schema rows."""
        return self.schema.types


def _rel_path(row: Row) -> str:
    return "/".join(p for p in (row["cabinet"], row["folder"], row["name"]) if p)


def _parts(row: Row, cabinet: str) -> Folder:
    """Folder parts below the cabinet, without a nested copy of the cabinet's own name."""
    parts = [p for p in row["folder"].split("/") if p]
    while parts and norm(parts[0]) == norm(cabinet):
        parts.pop(0)
    return tuple(parts)


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


def _matcher(cab: _Cabinet) -> Callable[[str], bool]:
    """True for a folder name that an override or a schema type claims."""
    cache: dict[str, bool] = {}

    def matches(name: str) -> bool:
        if name not in cache:
            found = match_label(name, cab.types)
            cache[name] = bool(cab.overrides.get(name) or found.target or found.candidates)
        return cache[name]

    return matches


def _layout(cab: _Cabinet) -> None:
    """Label folder, case and role of every document, decided before dedup so copies do not change them.

    Each child folder of a label folder is judged on its own by ``child_roles``.
    """
    parts = {id(d): _parts(d, cab.name) for d in cab.docs}
    wrappers = wrapper_folders(parts.values(), _matcher(cab), bool(cab.types))
    _, holds = folder_tree(parts.values())
    depth = {k: label_depth(p, wrappers) for k, p in parts.items()}
    cab.wrappers = sorted({"/".join(parts[k][:n]) for k, i in depth.items() for n in range(1, i + 1)})
    children: dict[tuple[Folder, str], list[Row]] = defaultdict(list)
    for d in cab.docs:
        p, i = parts[id(d)], depth[id(d)]
        if len(p) > i + 1 and p[: i + 1] not in holds:
            children[(p[: i + 1], p[i + 1])].append(d)
    in_child: dict[int, tuple[str, str]] = {}  # id(doc) -> (case_id, role)
    for (_, child), docs in children.items():
        roles = child_roles(child, [clean_stem(d["name"]) for d in docs])
        for d, role in zip(docs, roles or ["document"] * len(docs), strict=True):
            in_child[id(d)] = ("" if roles else child, role)
    for d in cab.docs:
        p, i = parts[id(d)], depth[id(d)]
        if not p:
            cab.place[id(d)] = _Place(clean_stem(d["name"]), "filename", "")
            continue
        cab.place[id(d)] = _Place(p[i], "folder", *in_child.get(id(d), ("", "document")))


def _zip_stem_at(row: Row, parts: Folder, archives: set[tuple[str, str]]) -> int:
    """Index in parts of the folder that stands for a member's zip, or -1."""
    raw = [p for p in row["folder"].split("/") if p]
    stem = PurePosixPath(row["in_zip"]).stem
    hits = [i for i, p in enumerate(raw) if p == stem] if row["in_zip"] else []
    if not hits:
        return -1
    at = next((i for i in hits if ("/".join(raw[:i]), row["in_zip"]) in archives), hits[0])
    return at - (len(raw) - len(parts))


def _dedup_keys(
    cab: _Cabinet, row: Row, cabinets: dict[str, str], archives: set[tuple[str, str]]
) -> tuple[DedupKey, list[DedupKey], bool]:
    """A row's own key, the keys of the originals it may copy, and whether it sits under another cabinet's name.

    The own key is the path below the cabinet. A zip member also looks for the
    same path without its zip folder (a zip copied inside a sibling zip). A row
    under another cabinet's name looks for that path in that cabinet.
    """
    parts, size = _parts(row, cab.name), row["size_kb"]
    own = (cab.name, "/".join((*parts, row["name"])), size)
    lookups = []
    at = _zip_stem_at(row, parts, archives)
    if at >= 0:
        lookups.append((cab.name, "/".join((*parts[:at], *parts[at + 1 :], row["name"])), size))
    other = cabinets.get(norm(parts[0]), cab.name) if parts else cab.name
    if other != cab.name:
        rest = list(parts)
        while rest and norm(rest[0]) == norm(other):
            rest.pop(0)
        lookups.append((other, "/".join((*rest, row["name"])), size))
    return own, lookups, other != cab.name


def _duplicate_groups(
    rows: list[tuple[_Cabinet, Row]], keys: list[tuple[DedupKey, list[DedupKey], bool]]
) -> list[list[int]]:
    """Indexes of rows that are copies of each other, joined by own key or by a lookup key."""
    parent = list(range(len(rows)))

    def root(i: int) -> int:
        while parent[i] != i:
            i = parent[i]
        return i

    first: dict[DedupKey, int] = {}
    for i, (own, _, _) in enumerate(keys):
        j = first.setdefault(own, i)
        row, other = rows[i][1], rows[j][1]
        same_zip_entry = row["in_zip"] and (row["in_zip"], row["folder"]) == (other["in_zip"], other["folder"])
        if not same_zip_entry:  # two entries of one zip with one name are both kept
            parent[root(i)] = root(j)
    for i, (_, lookups, _) in enumerate(keys):
        for key in lookups:
            if key in first:
                parent[root(i)] = root(first[key])
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(rows)):
        groups[root(i)].append(i)
    return [g for g in groups.values() if len(g) > 1]


def _dedup(cabinets: list[_Cabinet]) -> None:
    """Keep one copy of each document; the rest are excluded as duplicates."""
    names = {norm(c.name): c.name for c in cabinets}
    archives = {c.name: {(r["folder"], r["name"]) for r in c.rows if r["kind"] == "archive"} for c in cabinets}
    rows = [(cab, row) for cab in cabinets for row in cab.docs]
    keys = [_dedup_keys(cab, row, names, archives[cab.name]) for cab, row in rows]
    dropped: set[int] = set()
    for group in _duplicate_groups(rows, keys):
        group.sort(
            key=lambda i: dedup_rank(keys[i][2], rows[i][1]["folder"], rows[i][1]["in_zip"], _rel_path(rows[i][1]))
        )
        kept = _describe(rows[group[0]][1])
        for i in group[1:]:
            cab, row = rows[i]
            detail = f"{_describe(row)}; kept {kept}"
            cab.excluded.append(ExcludedRow(rel_path=_rel_path(row), reason="duplicate", detail=detail))
            cab.duplicates.append((_describe(row), kept))
            dropped.add(id(row))
    for cab in cabinets:
        cab.docs = [r for r in cab.docs if id(r) not in dropped]


def _describe(row: Row) -> str:
    """A file's path, and whether it is loose or which zip it is a member of."""
    where = f"zip member `{row['name']}` of `{row['in_zip']}`" if row["in_zip"] else "loose file"
    return f"`{_rel_path(row)}` ({where})"


def _possible_duplicates(cab: _Cabinet) -> None:
    """Group kept documents with the same name and size in different folders; a content hash wins when present."""
    by_key: dict[str, list[Row]] = defaultdict(list)
    for d in cab.docs:
        by_key[d.get(HASH_COLUMN) or f"{d['name']}\n{d['size_kb']}"].append(d)
    for key, docs in sorted(by_key.items()):
        paths = sorted({_rel_path(d) for d in docs})
        if len(paths) < MIN_DUP_PATHS:
            continue
        group = hashlib.sha256(key.encode()).hexdigest()[:DUP_GROUP_LEN]
        cab.dup_groups.append((group, paths))
        for d in docs:
            cab.dup_group[id(d)] = group


def _map_labels(cab: _Cabinet) -> None:
    """Resolve every label: label_map.yaml override first, then match_label."""
    for label in sorted({cab.place[id(d)].label for d in cab.docs}):
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
                cab.unmapped.append((label, found.loose))
        cab.mapping[label] = target


def _doc_ids(cab: _Cabinet) -> dict[int, str]:
    """doc_id per document; a second zip entry with the same path gets its member index in the hash input."""
    seen: Counter[str] = Counter()
    ids = {}

    def order(r: Row) -> tuple[str, int]:
        member = r.get("member") or ""
        return _rel_path(r), int(member) if member.isdigit() else 0

    for d in sorted(cab.docs, key=order):
        rel = _rel_path(d)
        seen[rel] += 1
        ids[id(d)] = make_doc_id(cab.name, rel if seen[rel] == 1 else f"{rel}#{d.get('member') or seen[rel]}")
    return ids


def _manifest(cab: _Cabinet, fields: dict[str, list[FieldSpec]]) -> list[ManifestRow]:
    keyed = {t for t, specs in fields.items() if any(f.role == "key" for f in specs)}
    ids = _doc_ids(cab)
    rows = []
    for d in cab.docs:
        place = cab.place[id(d)]
        target = cab.mapping.get(place.label, "")
        with_key = target in keyed and place.source == "folder" and not place.case_id and place.role == "document"
        row = ManifestRow(
            doc_id=ids[id(d)],
            rel_path=_rel_path(d),
            in_zip=d["in_zip"],
            case_id=place.case_id,
            label=target,
            label_source=place.source if target else "none",  # type: ignore[arg-type]
            key_gold=key_gold(d["name"]) if with_key else "",
            pages=int(d["pages"]) if d["pages"].isdigit() else None,
            size_kb=int(d["size_kb"] or 0),
            member=d.get("member") or "",
            role=place.role,  # type: ignore[arg-type]
            dup_group=cab.dup_group.get(id(d), ""),
        )
        rows.append(row)
    return sorted(rows, key=lambda r: (r.rel_path, r.doc_id))


def _taxonomy(cab: _Cabinet, manifest: list[ManifestRow], all_types: bool) -> list[TaxonomyClass]:
    with_samples = {m.label for m in manifest if m.label}
    names = sorted((set(cab.types) | set(cab.mapping.values())) - {""}) if all_types else sorted(with_samples)
    classes = []
    taken = {norm(n) for n in names} | {OTHER_CLASS}
    for name in names:
        labels = sorted(lb for lb, t in cab.mapping.items() if t == name and norm(lb) != norm(name))
        cab.dropped_aliases += [(lb, name) for lb in labels if norm(lb) in taken]
        aliases = [lb for lb in labels if norm(lb) not in taken]
        classes.append(TaxonomyClass(name=name, aliases=aliases, has_samples=name in with_samples))
    classes.append(TaxonomyClass(name=OTHER_CLASS, has_samples=any(not m.label for m in manifest)))
    return classes


def _write_cabinet(out: Path, cab: _Cabinet, all_types: bool) -> CabinetEntry:
    names = sorted(set(cab.types) | set(cab.mapping.values()) - {""})
    fields = {t: field_specs(t, cab.types.get(t, []), cab.field_overrides, cab.field_notes) for t in names}
    _possible_duplicates(cab)
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


def _add_type(s: _Schema, row: Row) -> None:
    """File a schema row under its type, merging spellings that differ only in case and spacing."""
    raw = row["document_type"].strip()
    key = norm(raw)
    if not key:
        return
    if key not in s.display:
        s.display[key] = (OTHER_RENAMED if key == OTHER_CLASS else raw, raw)
        if key == OTHER_CLASS:
            s.renamed.append(raw)
    elif raw != s.display[key][1] and (s.display[key][0], raw) not in s.merged:
        s.merged.append((s.display[key][0], raw))
    s.types.setdefault(s.display[key][0], []).append(row)


def _read_schema(schema: list[Row], names: list[str]) -> tuple[dict[str, _Schema], list[str]]:
    """Schema types per cabinet, and the workbooks that sit under no cabinet.

    A workbook belongs to the cabinet named by its nearest folder that is a
    cabinet name, so a copied cabinet's workbook ("A/B/x.xlsx") goes to B.
    """
    by_norm = {norm(n): n for n in names}
    out: dict[str, _Schema] = {}
    skipped: list[str] = []
    for row in schema:
        folders = row["workbook"].replace("\\", "/").split("/")[:-1]
        cabinet = next((by_norm[norm(p)] for p in reversed(folders) if norm(p) in by_norm), "")
        if cabinet:
            _add_type(out.setdefault(cabinet, _Schema()), row)
        elif row["workbook"] not in skipped:
            skipped.append(row["workbook"])
    return out, skipped


def _cabinets(
    inventory: list[Row],
    schema: list[Row],
    overrides: dict[str, dict[str, str]],
    fields: dict[str, RawOverrides],
    recorded: dict[str, str],
) -> tuple[list[_Cabinet], list[str]]:
    names = sorted({r["cabinet"] for r in inventory if r["cabinet"]})
    schemas, skipped = _read_schema(schema, names)
    slugs = assign_slugs(names, recorded)
    cabinets = [
        _Cabinet(
            name=name,
            slug=slugs[name],
            rows=[r for r in inventory if r["cabinet"] == name],
            schema=schemas.get(name, _Schema()),
            overrides={k: v for k, v in overrides.get(name, {}).items() if v},
            field_overrides=fields.get(name, {}),
        )
        for name in names
    ]
    return cabinets, skipped


def _str_map(data: Any) -> dict[str, str]:
    return {str(k): str(v or "") for k, v in (data or {}).items()}


def _read_label_map(path: Path) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    """Hand-written overrides per cabinet, and the entries of an old flat label_map.yaml.

    Raises:
        ClientPackError: The file is not a mapping of mappings.
    """
    data = read_yaml(path) if path.exists() else None
    overrides: dict[str, dict[str, str]] = {}
    legacy: dict[str, dict[str, str]] = {}
    try:
        for cab, entry in (data or {}).items():
            sections = entry or {}
            if set(sections) <= {OVERRIDES, AUTO} and all(isinstance(v, (dict, type(None))) for v in sections.values()):
                overrides[str(cab)] = _str_map(sections.get(OVERRIDES))
            else:
                legacy[str(cab)] = _str_map(sections)
    except (AttributeError, TypeError) as e:
        raise ClientPackError(f"{LABEL_MAP_FILE} is not a mapping per cabinet") from e
    return overrides, legacy


def _migrate(overrides: dict[str, dict[str, str]], legacy: dict[str, dict[str, str]], schema: list[Row]) -> None:
    """Keep an old flat entry as an override only when it differs from what matching gives now."""
    schemas, _ = _read_schema(schema, list(legacy))
    for cab, entries in legacy.items():
        types = schemas.get(cab, _Schema()).types
        kept = {lb: t for lb, t in entries.items() if t and match_label(lb, types).target != t}
        overrides[cab] = {**kept, **overrides.get(cab, {})}


def _label_map(cabinets: list[_Cabinet], overrides: dict[str, dict[str, str]]) -> dict[str, dict[str, dict[str, str]]]:
    """label_map.yaml: overrides as read (also for cabinets not built now), auto regenerated."""
    data = {cab: {OVERRIDES: entries, AUTO: {}} for cab, entries in overrides.items()}
    for cab in cabinets:
        auto = {lb: t for lb, t in cab.mapping.items() if lb not in cab.overrides}
        data[cab.name] = {OVERRIDES: overrides.get(cab.name, {}), AUTO: auto}
    return dict(sorted(data.items()))


def _unknown_overrides(cabinets: list[_Cabinet], fields: dict[str, RawOverrides]) -> list[str]:
    """Note each cabinet's unknown field_overrides.yaml types and fields; return the unknown cabinets."""
    for cab in cabinets:
        cab.unknown_overrides = unknown_overrides(cab.field_overrides, cab.types)
    names = {cab.name for cab in cabinets}
    unknown = sorted(c for c in fields if c not in names)
    count = len(unknown) + sum(len(cab.unknown_overrides) for cab in cabinets)
    if count:
        log.warning(f"unknown {FIELD_OVERRIDES_FILE} entries are ignored; see {REPORT_FILE}", extra={"count": count})
    return unknown


def _pack_info(entries: list[CabinetEntry], recorded: dict[str, str], share_root: str) -> dict[str, Any]:
    """pack.yaml data; slugs of cabinets gone from the inventory stay reserved under ``retired``."""
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    current = {e.name for e in entries}
    retired = {n: s for n, s in sorted(recorded.items()) if n not in current}
    return PackInfo(created_at=created, share_root=share_root, cabinets=entries, retired=retired).model_dump()


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
    schema = read_rows(Path(schema_csv))
    overrides, legacy = _read_label_map(out / LABEL_MAP_FILE)
    _migrate(overrides, legacy, schema)
    fields = read_field_overrides(out / FIELD_OVERRIDES_FILE)
    recorded = info.slugs() if (info := read_pack_info(out)) else {}
    cabinets, skipped = _cabinets(read_rows(Path(inventory_csv)), schema, overrides, fields, recorded)
    for cab in cabinets:
        _screen(cab, int(max_size_mb * KB_PER_MB))
        _layout(cab)
    _dedup(cabinets)
    for cab in cabinets:
        _map_labels(cab)
    entries = [_write_cabinet(out, cab, all_types) for cab in cabinets]
    write_yaml(out / PACK_FILE, _pack_info(entries, recorded, share_root))
    write_yaml(out / LABEL_MAP_FILE, _label_map(cabinets, overrides))
    atomic_write_text(out / REPORT_FILE, render_report(cabinets, skipped, _unknown_overrides(cabinets, fields)))
    summary = _summary(out, entries, cabinets)
    log.info("client pack built", extra=summary.model_dump(exclude={"cabinets"}))
    return summary


def _summary(out: Path, entries: list[CabinetEntry], cabinets: list[_Cabinet]) -> PackSummary:
    return PackSummary(
        out_dir=str(out),
        cabinets=entries,
        documents=sum(e.documents for e in entries),
        excluded=sum(len(c.excluded) for c in cabinets),
        unmapped_labels=sum(len(c.unmapped) for c in cabinets),
        ambiguous_labels=sum(len(c.ambiguous) for c in cabinets),
    )
