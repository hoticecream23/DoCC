"""Walking a corpus: which files are documents, and why the rest are not.

Kept plus excluded always equals what is on disk. A file that is silently
skipped and a file that is silently processed are the same class of bug: the
reported total stops describing the input.
"""

from __future__ import annotations

from pathlib import Path

from .extract import SUPPORTED_EXT

# Files that a corpus contains but that are not documents. Each of these was a
# real finding on the client set, and each is recorded here rather than left
# implicit, because a run that quietly drops or quietly invents input reports a
# document count that does not mean what it appears to mean.
#
#   __MACOSX/ and ._*  A macOS archive writes an AppleDouble stub beside every
#                      real file, carrying the same name and extension. Eleven
#                      of them in Client_Documents are 367 to 639 byte stubs
#                      named .pdf and .docx, and they were being picked up and
#                      processed as documents. This is the one entry here that
#                      fixes an over-count rather than explaining an under-one.
#
#   .zip               Jindal.zip holds ten entries, every one of them byte
#                      identical to a loose file sitting next to it. Unpacking
#                      it would double count ten documents.
#
#   .xlsx              The nineteen spreadsheets in the client set are not
#                      documents at all. Each is a question answering
#                      evaluation set over one of the court judgments, with
#                      positive, negative, edge and adversarial sheets and the
#                      columns QS ID, Question, Answer, Context, Ground Truth.
#                      They are gold for a capability this pipeline does not
#                      have yet. Extracting them as documents would classify a
#                      test set as a legal document and pollute every corpus
#                      level number with it.
IGNORED_DIR_NAMES = {"__MACOSX"}
APPLEDOUBLE_PREFIX = "._"
EXCLUDED_EXT = {
    ".zip": "archive",
    ".xlsx": "evaluation_set",
    ".xls": "evaluation_set",
}


def _exclusion_reason(path: Path) -> str | None:
    """Why this file is not a document, or None if it is one.

    Order matters. A macOS stub is checked before the extension, because the
    whole problem with those files is that they carry a supported one.
    """
    if path.name.startswith(APPLEDOUBLE_PREFIX):
        return "macos_resource_fork"
    if IGNORED_DIR_NAMES.intersection(part for part in path.parts):
        return "macos_resource_fork"
    suffix = path.suffix.lower()
    if suffix in EXCLUDED_EXT:
        return EXCLUDED_EXT[suffix]
    if suffix not in SUPPORTED_EXT:
        return "unsupported_extension"
    return None


def scan_documents(root) -> tuple[list, dict[str, list[str]]]:
    """Every document under root, plus everything left out and why.

    Returns the kept paths and a mapping of reason to the paths excluded for
    it. Callers report the second half. A file that is silently skipped and a
    file that is silently processed are the same class of bug: the total stops
    describing the input.
    """
    r = Path(root)
    if r.is_file():
        return [r], {}
    if not r.is_dir():
        # A mistyped path would otherwise walk nothing and report a clean run.
        raise FileNotFoundError(f"no such file or directory: {r}")

    kept, excluded = [], {}
    for p in sorted(r.rglob("*"), key=lambda p: str(p).lower()):
        if not p.is_file():
            continue
        reason = _exclusion_reason(p)
        if reason is None:
            kept.append(p)
        else:
            excluded.setdefault(reason, []).append(str(p))
    return kept, excluded


def iter_documents(root) -> list:
    """Every supported file under root, sorted so runs are reproducible."""
    return scan_documents(root)[0]
