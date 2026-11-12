"""Inventory a client sample share: names, sizes and page counts only.

Walks a root whose first-level folders are clients (cabinets), looks inside
zip archives without extracting them, and writes one CSV row per file. No
document content is read into the output. With ``--headers-out`` it also
writes the first row of every sheet of each .xlsx file, which is where the
clients list their document types and index fields.

Standalone on purpose: standard library only, so this one file can be
copied to a machine without the repo. Page counts need PyMuPDF (PDF) and
Pillow (TIFF); without them the pages column is empty.

    python scripts/inventory_share.py "<share root>" --out inventory.csv
"""

from __future__ import annotations

import argparse
import csv
import functools
import io
import logging
import zipfile
from collections.abc import Iterator, Sequence
from dataclasses import astuple, dataclass, fields
from pathlib import Path, PurePosixPath

log = logging.getLogger("inventory_share")

DOCUMENT_EXT = {".pdf", ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp"}
REFERENCE_EXT = {".xlsx", ".xls", ".docx", ".doc", ".pptx", ".txt"}
ARCHIVE_EXT = {".zip"}
REFERENCE_PREFIXES = ("screenshot",)
LOCK_PREFIX = "~$"  # Office leaves these beside an open workbook; they are not files of interest
MAX_PAGE_PROBE_BYTES = 64 * 1024 * 1024  # do not load bigger files just to count pages
BYTES_PER_KB = 1024
HEADER_SCAN_ROWS = 10  # some workbooks put a title row above the real header
UNPARSED_PREVIEW_ROWS = 12  # rows shown for a sheet whose header was not recognised
SCHEMA_KEY = "document type"
# "index name"/"index field", never "indexing method": one workbook has both columns.
SCHEMA_COLUMNS = {
    "index name": "index_field",
    "index field": "index_field",
    "data type": "data_type",
    "field type": "field_type",
}
# Extra columns worth keeping: where a value comes from (scan, ERP view) and whether it is a key.
SCHEMA_EXTRAS = ("db_view", "db_field", "erp", "auto capture", "unique", "v/r", "indexing method", "db lookup")
SCHEMA_HEADER = ["workbook", "sheet", "document_type", "index_field", "data_type", "field_type", "source"]


@dataclass(frozen=True)
class Row:
    """One file on the share, or one member of a zip on the share."""

    cabinet: str
    folder: str
    name: str
    ext: str
    size_kb: int
    pages: str
    in_zip: str
    kind: str


def classify(name: str, ext: str) -> str:
    """Sort a file into document, reference, archive or other.

    Args:
        name: File name without directories.
        ext: Lower-case extension with the dot.

    Returns:
        "reference" for client notes and screenshots, "document" for scans,
        "archive" for zips, else "other".
    """
    if ext in REFERENCE_EXT or name.lower().startswith(REFERENCE_PREFIXES):
        return "reference"
    if ext in ARCHIVE_EXT:
        return "archive"
    return "document" if ext in DOCUMENT_EXT else "other"


def count_pages(data: bytes, ext: str) -> str:
    """Page count of a PDF or TIFF, or "" when it cannot be told.

    Args:
        data: File bytes.
        ext: Lower-case extension with the dot.

    Returns:
        The count as text; "1" for single-frame images; "" if the optional
        library is missing or the file does not open.
    """
    try:
        if ext == ".pdf":
            import fitz

            with fitz.open(stream=data, filetype="pdf") as doc:
                return str(doc.page_count)
        if ext in DOCUMENT_EXT:
            from PIL import Image

            with Image.open(io.BytesIO(data)) as img:
                return str(getattr(img, "n_frames", 1))
    except (ImportError, OSError, RuntimeError, ValueError) as exc:  # missing lib or unreadable file
        log.debug("no page count: %s", type(exc).__name__)
    return ""


def _row(cabinet: str, rel: PurePosixPath, size: int, data: bytes | None, in_zip: str) -> Row:
    ext = rel.suffix.lower()
    kind = classify(rel.name, ext)
    pages = count_pages(data, ext) if data is not None and kind == "document" else ""
    folder = "" if str(rel.parent) == "." else str(rel.parent)
    return Row(cabinet, folder, rel.name, ext, -(-size // BYTES_PER_KB), pages, in_zip, kind)


def _zip_rows(cabinet: str, rel: PurePosixPath, path: Path) -> Iterator[Row]:
    try:
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                member = rel.with_suffix("") / PurePosixPath(info.filename)
                small = info.file_size <= MAX_PAGE_PROBE_BYTES
                yield _row(cabinet, member, info.file_size, archive.read(info) if small else None, rel.name)
    except (zipfile.BadZipFile, OSError, RuntimeError) as exc:  # corrupt or password-protected
        log.warning("cannot read archive %s: %s", rel, type(exc).__name__)


def inventory(root: Path) -> Iterator[Row]:
    """Yield one row per file under ``root``, and per member of each zip.

    Args:
        root: Share root; its first-level folders are the cabinets.

    Yields:
        Rows in path order.
    """
    for path in sorted(p for p in root.rglob("*") if p.is_file() and not p.name.startswith(LOCK_PREFIX)):
        parts = path.relative_to(root).parts
        cabinet, rel = (parts[0], PurePosixPath(*parts[1:])) if len(parts) > 1 else ("", PurePosixPath(parts[0]))
        size = path.stat().st_size
        data = path.read_bytes() if size <= MAX_PAGE_PROBE_BYTES and rel.suffix.lower() in DOCUMENT_EXT else None
        yield _row(cabinet, rel, size, data, "")
        if rel.suffix.lower() in ARCHIVE_EXT:
            yield from _zip_rows(cabinet, rel, path)


def workbook_headers(root: Path) -> Iterator[tuple[str, str, str]]:
    """First row of every sheet of every .xlsx under ``root``.

    Args:
        root: Share root.

    Yields:
        (relative path, sheet name, header cells joined with " | ").
    """
    try:
        from openpyxl import load_workbook
    except ImportError:
        log.warning("openpyxl not installed; no workbook headers")
        return
    for path in sorted(p for p in root.rglob("*.xlsx") if not p.name.startswith(LOCK_PREFIX)):
        try:
            book = load_workbook(path, read_only=True, data_only=True)
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:  # locked, corrupt or not a real xlsx
            log.warning("cannot open %s: %s", path.name, type(exc).__name__)
            continue
        for sheet in book.worksheets:
            rows = list(sheet.iter_rows(min_row=1, max_row=UNPARSED_PREVIEW_ROWS, values_only=True))
            shown = rows[:1] if _find_header(rows) else rows  # unrecognised layout: show enough to read it
            for row in shown:
                yield str(path.relative_to(root)), sheet.title, " | ".join(_cell(c) for c in row if _cell(c))
        book.close()


def _cell(value: object) -> str:
    return "" if value is None else " ".join(str(value).split())


def _at(row: tuple[object, ...], index: int | None) -> str:
    return _cell(row[index]) if index is not None and index < len(row) else ""


def _find_header(rows: list[tuple[object, ...]]) -> tuple[int, dict[str, int]] | None:
    for at, row in enumerate(rows[:HEADER_SCAN_ROWS]):
        names = {_cell(c).lower(): i for i, c in enumerate(row) if _cell(c)}
        if SCHEMA_KEY in names:
            return at, names
    return None


def _sheet_schema(rows: list[tuple[object, ...]]) -> Iterator[tuple[str, str, str, str, str]]:
    found = _find_header(rows)
    if found is None:
        return
    at, names = found
    cols = {out: i for name, i in names.items() for key, out in SCHEMA_COLUMNS.items() if name.startswith(key)}
    extras = {name: i for name, i in names.items() if any(key in name for key in SCHEMA_EXTRAS)}
    doc_type = ""
    for row in rows[at + 1 :]:
        get = functools.partial(_at, row)
        doc_type = get(names[SCHEMA_KEY]) or doc_type  # merged cells: the type is only on its first row
        field = get(cols.get("index_field"))
        if not field:
            continue
        source = "; ".join(f"{name}={get(i)}" for name, i in extras.items() if get(i))
        yield doc_type, field, get(cols.get("data_type")), get(cols.get("field_type")), source


def workbook_schema(root: Path) -> Iterator[tuple[str, ...]]:
    """Document types and their index fields, as the clients wrote them.

    Reads only the schema columns (type, field name, data type, field type,
    and where the value comes from). Free-text remarks are left out.

    Args:
        root: Share root.

    Yields:
        Rows matching SCHEMA_HEADER.
    """
    try:
        from openpyxl import load_workbook
    except ImportError:
        log.warning("openpyxl not installed; no workbook schema")
        return
    for path in sorted(p for p in root.rglob("*.xlsx") if not p.name.startswith(LOCK_PREFIX)):
        try:
            book = load_workbook(path, read_only=True, data_only=True)
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:  # locked, corrupt or not a real xlsx
            log.warning("cannot open %s: %s", path.name, type(exc).__name__)
            continue
        for sheet in book.worksheets:
            rows = list(sheet.iter_rows(values_only=True))
            for found in _sheet_schema(rows):
                yield (str(path.relative_to(root)), sheet.title, *found)
        book.close()


def _write_csv(path: Path, header: list[str], rows: Sequence[Sequence[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as fh:  # BOM so Excel opens it as UTF-8
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    """Run the inventory.

    Args:
        argv: Arguments; defaults to the process arguments.

    Returns:
        Process exit code.
    """
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("root", type=Path, help="share root; first-level folders are the clients")
    ap.add_argument("--out", type=Path, default=Path("inventory.csv"))
    ap.add_argument("--headers-out", type=Path, help="also write the first row of each .xlsx sheet here")
    ap.add_argument("--schema-out", type=Path, help="also write document types and index fields from each .xlsx")
    ap.add_argument("--dry-run", action="store_true", help="count files, write nothing")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not args.root.is_dir():
        log.error("not a directory: %s", args.root)
        return 2
    rows = list(inventory(args.root))
    log.info("%d rows, %d documents", len(rows), sum(r.kind == "document" for r in rows))
    if args.dry_run:
        log.info("dry run: would write %s", args.out)
        return 0
    _write_csv(args.out, [f.name for f in fields(Row)], [astuple(r) for r in rows])
    if args.headers_out:
        _write_csv(args.headers_out, ["workbook", "sheet", "first_row"], list(workbook_headers(args.root)))
    if args.schema_out:
        _write_csv(args.schema_out, SCHEMA_HEADER, list(workbook_schema(args.root)))
    log.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
