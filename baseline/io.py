"""Crash-safe file writes and the shared --dry-run report.

A whole-file output is written to a temp file beside the target and moved
into place with os.replace, so a crash leaves the old file or the new one,
never half of either.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO

from .logging_setup import get_logger

log = get_logger(__name__)


@contextmanager
def atomic_open(
    path: str | Path, mode: str = "w", encoding: str | None = "utf-8", newline: str | None = None
) -> Iterator[IO]:
    r"""Open a temp file beside path and move it over path on a clean exit.

    Args:
        path: Final file.
        mode: Write mode, "w" or "wb".
        encoding: Text encoding, ignored for binary modes.
        newline: As for open(); pass "\n" for JSONL.

    Yields:
        The open temp file. On an exception it is deleted and path is untouched.
    """
    target = Path(path)
    temp = target.with_suffix(target.suffix + ".tmp")
    binary = "b" in mode
    try:
        with open(temp, mode, encoding=None if binary else encoding, newline=None if binary else newline) as fh:
            yield fh
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)


def atomic_write_text(path: str | Path, text: str, encoding: str = "utf-8") -> None:
    """Path.write_text, but all or nothing.

    Args:
        path: Final file.
        text: Whole file content.
        encoding: Text encoding.
    """
    with atomic_open(path, encoding=encoding) as fh:
        fh.write(text)


def report_dry_run(outputs: Iterable[str | Path | None], documents: int | None = None) -> int:
    """Log what a command would write.

    Args:
        outputs: Paths the command would write; falsy entries are skipped.
        documents: How many documents the command would read, if known.

    Returns:
        Exit code 0.
    """
    paths = [str(p) for p in outputs if p]
    log.info("dry run, nothing written", extra={"would_write": paths, "documents": documents})
    return 0


def cut_partial_tail(path: str | Path) -> bool:
    """Drop a last line that a crash left without its newline.

    Append-only JSONL outputs cannot be written atomically, so a crash can
    leave half a record at the end. Resuming after it would glue the next
    record onto that fragment.

    Args:
        path: The JSONL file. A missing file is left alone.

    Returns:
        True when a partial line was cut.
    """
    target = Path(path)
    if not target.exists():
        return False
    # ponytail: reads the whole file once per resume; seek from the end if outputs reach GBs.
    content = target.read_bytes()
    cut = content.rfind(b"\n") + 1
    if cut >= len(content):
        return False
    log.warning("dropping a partial last line", extra={"path": str(target)})
    with open(target, "r+b") as fh:
        fh.truncate(cut)
    return True
