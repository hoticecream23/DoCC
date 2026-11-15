"""Client packs: everything a run on one client share needs, built from its inventory.

A pack is a directory outside the repo, made on the client machine from the
two CSVs ``scripts/inventory_share.py`` writes. The Pydantic models below are
the format; docs/parts/client-pack.md describes it. Building lives in
``clientpack_build`` and the matching rules in ``clientpack_rules``. This
module holds the models, reads a pack back, and copies its documents out of
the share. Over 400 lines: the format models and the copy-out code share them.
"""

from __future__ import annotations

import csv
import hashlib
import shutil
import zipfile
import zlib
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .errors import DoccError
from .io import atomic_open
from .logging_setup import get_logger

log = get_logger(__name__)

PACK_VERSION = "1"
PACK_FILE = "pack.yaml"
LABEL_MAP_FILE = "label_map.yaml"
REPORT_FILE = "REPORT.md"
TAXONOMY_FILE = "taxonomy.yaml"
FIELDS_FILE = "fields.yaml"
MANIFEST_FILE = "manifest.csv"
EXCLUDED_FILE = "excluded.csv"
FIELD_OVERRIDES_FILE = "field_overrides.yaml"  # hand-written; build reads it and never writes it
OTHER_CLASS = "other"
DOC_ID_LEN = 16  # same length as a results doc_id

Dtype = Literal["text", "date", "number", "selection"]
Role = Literal["key", "required", "optional"]
Origin = Literal["document", "erp", "manual", "fixed"]
LabelSource = Literal["folder", "filename", "none"]
ExcludeReason = Literal["reference", "duplicate", "archive", "unsupported", "oversize"]


class ClientPackError(DoccError, ValueError):
    """A pack is missing, malformed, or would write inside the share."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CabinetEntry(_Model):
    """One cabinet as pack.yaml lists it."""

    slug: str
    name: str
    documents: int = Field(ge=0)
    types_with_samples: int = Field(ge=0)


class PackInfo(_Model):
    """pack.yaml."""

    version: str = PACK_VERSION
    created_at: str
    share_root: str = ""
    cabinets: list[CabinetEntry] = Field(default_factory=list)


class TaxonomyClass(_Model):
    """One class of a cabinet's taxonomy.yaml."""

    name: str
    aliases: list[str] = Field(default_factory=list)
    has_samples: bool


class FieldSpec(_Model):
    """One index field of a document type, in fields.yaml."""

    name: str
    dtype: Dtype
    date_format: str | None = None
    role: Role
    origin: Origin


class ManifestRow(_Model):
    """One document of a cabinet, in manifest.csv."""

    doc_id: str
    rel_path: str
    in_zip: str = ""
    case_id: str = ""
    label: str = ""
    label_source: LabelSource = "none"
    key_gold: str = ""
    pages: int | None = None
    size_kb: int = Field(ge=0)
    member: str = ""  # index of a zip member in the archive, so members with the same name stay apart

    @field_validator("pages", mode="before")
    @classmethod
    def _blank_pages(cls, v: Any) -> Any:
        return None if v == "" else v


class ExcludedRow(_Model):
    """One file left out of a cabinet, in excluded.csv."""

    rel_path: str
    reason: ExcludeReason


class ClientPack(_Model):
    """One cabinet of a pack, loaded for other code to use."""

    cabinet: CabinetEntry
    classes: list[TaxonomyClass]
    fields: dict[str, list[FieldSpec]]
    manifest: list[ManifestRow]

    def key_field(self, type_name: str) -> str:
        """Name of the type's key field, or "" when it has none.

        Args:
            type_name: Schema type name.

        Returns:
            The first field with role key.
        """
        return next((f.name for f in self.fields.get(type_name, []) if f.role == "key"), "")


class PackSummary(_Model):
    """What build_pack wrote."""

    out_dir: str
    cabinets: list[CabinetEntry]
    documents: int
    excluded: int
    unmapped_labels: int
    ambiguous_labels: int


def slugify(name: str) -> str:
    """Directory name for a cabinet.

    Args:
        name: Cabinet folder name.

    Returns:
        Lower-case letters, digits and underscores; "cabinet" if nothing is left.
    """
    slug = "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")
    return "_".join(p for p in slug.split("_") if p) or "cabinet"


def assign_slugs(names: list[str]) -> dict[str, str]:
    """A unique slug per cabinet name.

    Names are taken in sorted order. A slug already taken gets the first free
    suffix "_2", "_3" and so on, so "Acme-Ltd" after "Acme Ltd" is "acme_ltd_2".

    Args:
        names: Cabinet names.

    Returns:
        Name -> slug.
    """
    taken: set[str] = set()
    slugs = {}
    for name in sorted(set(names)):
        base = slug = slugify(name)
        n = 1
        while slug in taken:
            n += 1
            slug = f"{base}_{n}"
        taken.add(slug)
        slugs[name] = slug
    return slugs


def safe_member(name: str) -> str:
    """A zip member name as a safe relative path.

    Args:
        name: The name stored in the zip.

    Returns:
        Backslashes as slashes, with empty, "." and ".." parts and drive letters dropped.
    """
    parts = [p for p in name.replace("\\", "/").split("/") if p not in ("", ".", "..")]
    return "/".join(parts[1:] if parts and parts[0].endswith(":") else parts)


def make_doc_id(cabinet: str, rel_path: str) -> str:
    """Stable id of a document: hash of its cabinet and path, not its bytes.

    Args:
        cabinet: Cabinet name.
        rel_path: Path relative to the share root.

    Returns:
        16 hex characters.
    """
    return hashlib.sha256(f"{cabinet}\n{rel_path}".encode()).hexdigest()[:DOC_ID_LEN]


def write_yaml(path: Path, data: Any) -> None:
    """Write YAML all or nothing, keys in insertion order.

    Args:
        path: Target file.
        data: Plain data.
    """
    with atomic_open(path) as fh:
        yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True)


def read_yaml(path: Path) -> Any:
    """Read a YAML file.

    Args:
        path: File to read.

    Returns:
        The parsed data; None for an empty file.

    Raises:
        ClientPackError: The file is missing or not valid YAML.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    except (OSError, yaml.YAMLError) as e:
        raise ClientPackError(f"cannot read {path.name}: {type(e).__name__}") from e


def write_rows(path: Path, rows: list[BaseModel], header: list[str]) -> None:
    """Write models as CSV rows, None as an empty cell.

    Args:
        path: Target file.
        rows: Models whose fields are the header.
        header: Column names.
    """
    with atomic_open(path, newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "" if v is None else v for k, v in row.model_dump().items()})


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV as dicts; a BOM from Excel is dropped.

    Args:
        path: CSV file.

    Returns:
        One dict per row.

    Raises:
        ClientPackError: The file cannot be read.
    """
    try:
        with open(path, encoding="utf-8-sig", newline="") as fh:
            return list(csv.DictReader(fh))
    except OSError as e:
        raise ClientPackError(f"cannot read {path.name}: {type(e).__name__}") from e


def load_pack(pack_dir: str | Path, cabinet: str) -> ClientPack:
    """Read one cabinet of a pack.

    Args:
        pack_dir: The pack directory.
        cabinet: Cabinet name or slug.

    Returns:
        Its classes, fields by type and manifest rows.

    Raises:
        ClientPackError: The pack or the cabinet is missing or malformed.
    """
    root = Path(pack_dir)
    info = PackInfo.model_validate(read_yaml(root / PACK_FILE) or {})
    by_name = (c for c in info.cabinets if c.name == cabinet)
    entry = next(by_name, None) or next((c for c in info.cabinets if c.slug == cabinet), None)
    if entry is None:
        raise ClientPackError(f"no cabinet {cabinet!r} in {root / PACK_FILE}")
    cab = root / entry.slug
    taxonomy = read_yaml(cab / TAXONOMY_FILE) or {}
    fields = read_yaml(cab / FIELDS_FILE) or {}
    return ClientPack(
        cabinet=entry,
        classes=[TaxonomyClass.model_validate(c) for c in taxonomy.get("classes", [])],
        fields={t: [FieldSpec.model_validate(f) for f in specs or []] for t, specs in fields.items()},
        manifest=[ManifestRow.model_validate(r) for r in read_rows(cab / MANIFEST_FILE)],
    )


def _zip_location(share: Path, row: ManifestRow) -> tuple[Path, str]:
    """The zip on disk and the member name for a manifest row from a zip.

    The inventory writes a member's folder as "<zip folder>/<zip stem>/<inner
    dirs>". The zip stem can also be an ordinary folder name, so each place it
    occurs is tried until the zip exists on disk.
    """
    parts = PurePosixPath(row.rel_path).parts
    stem = PurePosixPath(row.in_zip).stem
    for i, part in enumerate(parts[:-1]):
        candidate = share.joinpath(*parts[:i], row.in_zip)
        if part == stem and candidate.is_file():
            return candidate, "/".join(parts[i + 1 :])
    raise ClientPackError(f"zip {row.in_zip!r} not found for {row.doc_id}")


def _member_info(archive: zipfile.ZipFile, row: ManifestRow, inner: str) -> zipfile.ZipInfo:
    """The archive entry of a manifest row: by its recorded index, else by its safe name."""
    infos = archive.infolist()
    if row.member.isdigit() and int(row.member) < len(infos):
        info = infos[int(row.member)]
        if safe_member(info.filename) == inner:
            return info
    found = next((i for i in infos if safe_member(i.filename) == inner), None)
    if found is None:
        raise KeyError(inner)
    return found


def _copy_one(share: Path, row: ManifestRow, dest: Path) -> None:
    if not row.in_zip:
        with open(share / row.rel_path, "rb") as src, atomic_open(dest, "wb") as fh:
            shutil.copyfileobj(src, fh)
        return
    archive_path, inner = _zip_location(share, row)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            data = archive.read(_member_info(archive, row, inner))
    except (zipfile.BadZipFile, KeyError, zlib.error, NotImplementedError, RuntimeError, EOFError) as e:
        raise ClientPackError(f"cannot extract {row.doc_id} from {row.in_zip}: {type(e).__name__}") from e
    with atomic_open(dest, "wb") as fh:
        fh.write(data)


class MaterializeResult(list[Path]):
    """The documents materialize made ready, and the doc_ids it could not copy.

    It is the list of ready paths, so callers that expect a list keep working.
    """

    def __init__(self, copied: list[Path], missing: list[str]) -> None:
        """Hold the result.

        Args:
            copied: Paths of the documents now in the work directory, in manifest order.
            missing: doc_ids whose source was missing or unreadable.
        """
        super().__init__(copied)
        self.missing = missing

    @property
    def copied(self) -> list[Path]:
        """Paths of the documents now in the work directory."""
        return list(self)


def materialize(pack_dir: str | Path, cabinet: str, share_root: str | Path, work_dir: str | Path) -> MaterializeResult:
    """Copy or extract each manifest document to work_dir/<slug>/<doc_id><ext>.

    Each copy is written atomically and a file already there is left alone,
    so a re-run only fetches what is missing. Nothing is written inside the
    share. A missing or unreadable source is logged and skipped.

    Args:
        pack_dir: The pack directory.
        cabinet: Cabinet name or slug.
        share_root: The share the inventory was taken of.
        work_dir: Where plain copies go.

    Returns:
        The ready paths in manifest order, with the doc_ids that failed in ``missing``.

    Raises:
        ClientPackError: work_dir is inside the share.
    """
    pack = load_pack(pack_dir, cabinet)
    share = Path(share_root).resolve()
    out = Path(work_dir).resolve() / pack.cabinet.slug
    if out == share or share in out.parents:
        raise ClientPackError("the work directory must be outside the share")
    out.mkdir(parents=True, exist_ok=True)
    paths, missing = [], []
    for row in pack.manifest:
        dest = out / f"{row.doc_id}{PurePosixPath(row.rel_path).suffix.lower()}"
        if not dest.exists():
            try:
                _copy_one(share, row, dest)
            except (OSError, ClientPackError) as e:
                log.warning("cannot copy", extra={"doc_id": row.doc_id, "error": type(e).__name__})
                missing.append(row.doc_id)
                continue
        paths.append(dest)
    extra = {"cabinet": pack.cabinet.name, "documents": len(paths), "missing": len(missing), "work": str(out)}
    log.info("materialized", extra=extra)
    return MaterializeResult(paths, missing)
