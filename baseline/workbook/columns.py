"""The workbook's column spec, and the cell readers both directions share.

Export writes these columns and import reads them back, so they live in exactly
one place. A rename that touches only one side breaks the loop in silence.
"""

from __future__ import annotations

from typing import Any

# The four things a reviewer can say about a row, and the spellings seen in the
# sheet so far. Casing and spacing vary between reviewers and always will, so
# they are normalised here rather than policed in the sheet.
VERDICTS = {
    "ok": "ok",
    "correct": "ok",
    "wrong": "wrong",
    "incorrect": "wrong",
    "spurious": "spurious",
    "missing": "missing",
}
TRUTHY = {"true", "yes", "y", "1"}

# The column spec, in sheet order. Both directions read it, so a rename cannot
# desynchronise the export from the import.
#
# The trailing review columns of each tab are the reviewer's, and export leaves
# every one of them blank. Pre-filling IS_GOLD in particular would turn an
# unreviewed export into gold on the next import, which is the one mistake that
# would quietly invent a score out of nothing.
REVIEW_COLUMNS = ["VERDICT", "IS_GOLD", "REVIEWED_BY", "REVIEW_DATE", "NOTES"]

DOCUMENT_COLUMNS = [
    "DOC_ID", "RUN_ID", "EXTRACTED_AT", "SCHEMA_VERSION", "SOURCE_PATH", "FILE_NAME",
    "FILE_TYPE", "PAGE_COUNT", "EXTRACTION_METHOD", "PAGE_METHOD", "HAS_NATIVE_TEXT",
    "IS_SCANNED", "EXTRACTION_CONFIDENCE", "CHAR_COUNT", "TEXT_SHA256", "TEXT_PREVIEW",
    "DOC_CLASS", "CLASS_CONFIDENCE", "CLASS_METHOD", "CLASS_RUNNER_UP",
    "RUNNER_UP_CONFIDENCE", "ERROR_COUNT", "ERRORS", "LANGUAGE", "QUALITY",
    "CLASS_CORRECTED", "CLASS_VERDICT", "IS_GOLD", "REVIEWED_BY", "REVIEW_DATE", "NOTES",
]
METADATA_COLUMNS = [
    "FIELD_ROW_ID", "DOC_ID", "ROW_SOURCE", "FIELD", "VALUE", "NORMALIZED_VALUE",
    "CURRENCY", "PAGE", "CHAR_START", "CHAR_END", "CONFIDENCE", "METHOD", "VALIDATED",
    "NORMALIZER_VERSION", "CORRECTED_VALUE", "VERDICT", "IS_GOLD", "REVIEWED_BY",
    "REVIEW_DATE", "NOTES",
]
TAG_COLUMNS = [
    "DOC_ID", "TAG", "ROW_SOURCE", "CONFIDENCE", "METHOD",
    "VERDICT", "IS_GOLD", "REVIEWED_BY", "REVIEW_DATE",
]
TAXONOMY_COLUMNS = ["KIND", "NAME", "DESCRIPTION", "VALUE_TYPE", "FORMAT", "APPLIES_TO_CLASSES"]
LINE_ITEM_COLUMNS = [
    "DOC_ID", "TABLE_ID", "PAGE", "TABLE_METHOD", "N_ROWS", "N_COL",
    "ROW", "COL", "CELL_TEXT", "CHAR_START", "CHAR_END", "IS_HEADER",
]

# The full text never goes in a cell. PAGE_SEP carries a form feed and every
# offset is counted against the exact canonical string, so Excel stripping that
# character, rewriting the line endings or hitting its 32,767 character limit
# would silently invalidate every char_start in the Metadata tab. The sheet
# carries a hash and a length as a tripwire instead, and the JSONL stays the
# source of truth for text.
PREVIEW_CHARS = 200


def _cell(value: Any) -> str:
    """A cell as a stripped string. None, blanks and stray spacing all collapse."""
    if value is None:
        return ""
    return str(value).strip()


def _get(row: tuple, index: dict[str, int], column: str) -> str:
    """One column of one row as a stripped string, blank when absent."""
    i = index.get(column)
    return "" if i is None or i >= len(row) else _cell(row[i])


def _read_tab(wb, name: str) -> tuple[dict[str, int], list[tuple]]:
    """Header index plus the non empty rows of one tab.

    DOC_ID is filled only on the first row of each document, the way a person
    writes a grouped sheet. Carrying it down here is what stops half the rows
    orphaning.
    """
    if name not in wb.sheetnames:
        return {}, []
    ws = wb[name]
    rows = ws.iter_rows(values_only=True)
    try:
        header = [_cell(c).upper() for c in next(rows)]
    except StopIteration:
        return {}, []
    index = {h: i for i, h in enumerate(header) if h}

    out: list[tuple] = []
    carried = ""
    key = index.get("DOC_ID")
    for row in rows:
        if not any(c is not None and _cell(c) for c in row):
            continue
        row = list(row)
        if key is not None:
            here = _cell(row[key])
            if here:
                carried = here
            row[key] = carried
        out.append(tuple(row))
    return index, out
