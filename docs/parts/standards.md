# Coding standards pass

Date: 2026-11-06. Branch: `small-fixes`, commits 0f867f9..3f2b1d0 (10 commits,
106 files, +7788 / -3489). Plan and status: `PLAN-standards.md`. Standards
source: the four .docx files in the "coding standards final draft" folder
(v19 is authoritative; csd_5 and crs supplement it).

## What changed

- Tooling: ruff, black, isort, mypy, pytest-cov configured in `pyproject.toml`
  (line length 120, Google docstrings, `disallow_untyped_defs`, coverage gate
  85). Pre-commit hooks and a GitHub Actions workflow (lint, format, types,
  bandit, tests with tesseract) added. `pip install -e ".[dev]"`.
- Errors: one hierarchy in `baseline/errors.py` (`DoccError`); every re-raise
  uses `from e`; every remaining `except Exception` has a `noqa: BLE001`
  reason; logs carry a truncated exception summary, never raw document text.
- Writes: every output file goes through `baseline/io.py` (temp file beside
  the target, then `os.replace`); cascade and baseline resume trim a partial
  trailing JSONL line. All CLIs take `--dry-run`; `--resume/--no-resume`
  replaces `--no-resume`.
- Demo: no default VLM model. The server starts without one; cascade mode
  returns 503 with the `ConfigError` message.
- Structure: `baseline/cli.py` only parses and dispatches; train and
  tune-thresholds logic moved to `baseline/training.py`; graph rendering to
  `baseline/graph_report.py`; scoring split into `evaluate/{io,calibration,
  metadata_scores,compare}.py`; workbook summary to `workbook/summary.py`.
  Every function is 50 lines or fewer. Modules over 400 lines carry a
  header note saying why.
- Tests: `tests/unit` (438, ~15 s, `pytest -m "not integration" --no-cov`)
  and `tests/integration` (153). No assertions on private attributes.

## Before and after

| Measure | Before | After |
|--|--|--|
| ruff findings (standards config) | 2111 | 0 |
| black / isort files to change | 60 / 8 | 0 / 0 |
| mypy errors (`disallow_untyped_defs`) | 109 | 0 |
| functions > 50 lines | 30 | 0 |
| `except Exception` without reason | 25 | 0 |
| non-atomic output writers | 10 | 0 |
| custom exception modules | 5 scattered | 1 shared |
| tests / coverage | 560 / 84.6 % | 591 / 87.3 % |
| CI, pre-commit, coverage gate | none | all three |

## Not applied (not applicable to this repo)

S47 MkDocs/OpenAPI sync, S56/S57 `backend/src` layout, S61 trace_id
middleware, S63 async LLM IO, S66-S68 experiment/prompt/embedding
registries, S72-S76 PII redaction/RBAC/TLS, S80 PR size (repo-wide pass),
S82-S90 team review process.
