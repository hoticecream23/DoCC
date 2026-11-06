"""File reading for the scorer. Scoring modules stay free of IO."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file, skipping blank lines.

    Args:
        path: File to read.

    Returns:
        One dict per non blank line.

    Raises:
        ValueError: A line is not valid JSON. The message names file and line.
    """
    rows = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{i} is not valid JSON: {exc}") from exc
    return rows
