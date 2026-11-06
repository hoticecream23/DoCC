"""Scoring. Reads any JSONL that matches the output schema.

This is deliberately implementation blind. It does not import the pipeline
and it does not care how a record was produced, so the baseline and whatever
replaces it are scored by exactly the same code. Nothing in this package may
import from the rest of `baseline` except `logging_setup`.

- `io.py`            reading JSONL inputs
- `scoring.py`       joining gold to predictions, and per component scores
- `metadata_scores.py` metadata value, span and validated scores
- `calibration.py`   confidence calibration and gold-covered field scores
- `table_scores.py`  content and adjacency scores for tables
- `report.py`        whole files scored end to end and written up
- `compare.py`       two runs diffed against the same gold
"""

from .compare import compare, render_comparison
from .io import load_jsonl
from .metadata_scores import eval_metadata
from .report import evaluate, render_report
from .scoring import (
    KEYS,
    coverage_at_error,
    eval_classification,
    eval_extraction,
    eval_tagging,
    join,
    levenshtein,
    prf,
)
from .table_scores import eval_tables

__all__ = [
    "KEYS",
    "compare",
    "coverage_at_error",
    "eval_classification",
    "eval_extraction",
    "eval_metadata",
    "eval_tables",
    "eval_tagging",
    "evaluate",
    "join",
    "levenshtein",
    "load_jsonl",
    "prf",
    "render_comparison",
    "render_report",
]
