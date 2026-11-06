# PLAN-standards: bring DoCC into line with the coding standards

Date: 2026-11-06. Branch: `small-fixes`. Orchestrator plan from one standards
extraction and four audits (mechanical, errors/logging/security,
naming/structure, tests/CI).

Standards source: four .docx files in
`C:\Users\advay\Downloads\coding standards final draft-20260908T064751Z-1-001`.
Consolidated list (S1-S90): scratchpad `standards/STANDARDS.md`.
Authoritative: `coding_standards_final_v19`. Supplements: csd_5 (handbook),
crs (review process). Measured baseline: scratchpad `audit/BASELINE.md`.

## Cause

The repo has no lint, format, type, or coverage configuration and no CI
(S35-S37, S53). So every mechanical rule is unenforced and has drifted.
The review-type rules show the usual symptoms of that: no shared exception
module, catch-all excepts, non-atomic output writes, oversize functions.

## Evidence (before)

| Measure | Value |
|--|--|
| ruff violations (120 cols, google docstrings, ANN/D/UP/B/N/E/W/F/I/BLE/TD/PLR2004) | 2111 (tests 1657, baseline 335, cascade 61, scripts 38, demo 20) |
| top codes | ANN001 713, ANN201 448, D103 406, PLR2004 120, ANN202 74, D101 45, D102 41, N806 38 |
| black --check (120) would change | 60 of 85 files |
| isort --check (profile black) | 8 files |
| mypy strict / lenient errors | 109 / 62 |
| functions > 50 lines | 30 (29 in source) |
| modules > 400 lines (source) | 5: graph.py 549, cli.py 518, extract.py 489, scoring.py 453, importer.py 427 |
| `except Exception` sites | 25 (6 are documented boundaries) |
| custom exception classes, scattered | 6 in 5 modules, no shared module |
| outputs truncated in place (no tmp+rename) | cascade/runner.py:166, baseline/cli.py:219,276, tables/records.py:242 |
| silent default model | demo/server.py:46 |
| print() in library code | 8 |
| tests | 560 pass in 108 s, coverage 84 % (gate wants 85) |
| CI / pre-commit / tool config | none |
| secrets, TODOs, raw-text logging | 0 / 0 / 0 (clean) |

## Decisions

- Line length 120 (S11 over black default, conflict C1).
- tests/ exempt from ANN, D, PLR2004: the standard targets public APIs.
- Shared exceptions live in `baseline/errors.py` (repo has no `src/utils`;
  S29 adapted). cascade and demo import from it if they already depend on
  baseline; otherwise each package gets its own `errors.py`.
- Modules > 400 lines: split only where a function is also > 50 lines
  (graph.py, cli.py, importer.py). Otherwise add the header comment (S18).
- S80 (PR <= 300 LOC) cannot hold for a repo-wide compliance pass; commit
  per phase instead.
- Not applicable, skipped: S47 (MkDocs/OpenAPI), S56/S57 (backend/src
  layout), S61 (trace_id middleware), S63, S66-S68 (prompt/embedding
  registries), S72-S76 (PII/RBAC/TLS), S82-S90 (team process).

## Fix tasks

### Phase 0 (serial, one worker, touches every file)
T0 Tooling and autofix. Files: pyproject.toml, .pre-commit-config.yaml,
   .github/workflows/ci.yml, .gitignore, requirements-dev.txt (or
   optional-dependencies), every .py (black/isort/ruff --fix).
   Done: `ruff check .`, `black --check .`, `isort --check .` clean except
   ANN/D/PLR2004/PLR0915 residue; tests pass; one commit.

### Phase 1 (parallel, disjoint files)
T1 Errors and logging. Files: new baseline/errors.py; baseline/{extract,
   ocr,classify,tagging,pipeline,logging_setup}.py, baseline/metadata/
   {fields,normalizers,validators}.py, baseline/tables/records.py (except
   sites only), cascade/{vlm,prompt}.py, demo/server.py.
   Done: no `except Exception` outside documented boundaries; every
   re-raise uses `from e`; all custom errors subclass one base in
   errors.py; demo has no silent default model; logs truncate `str(exc)`;
   tests pass.
T2 Atomic writes, CLI hygiene. Files: cascade/runner.py, cascade/cli.py,
   baseline/cli.py, baseline/tables/records.py (writer only), scripts/*.py,
   demo/__main__.py, new baseline/io.py (atomic_write helper).
   Done: every results/provenance writer goes tmp+rename; `--dry-run` on
   every CLI; `--no-resume` -> `--resume/--no-resume` BooleanOptionalAction;
   no print() in library code; tests pass.

### Phase 2 (parallel by file set, after phase 1 commits)
T3 cascade/, demo/, scripts/: ANN, D (google), mypy strict clean, N806,
   PLR2004 -> named constants, functions <= 50 lines.
T4 baseline/{evaluate,tables,workbook}/: same, plus split importer.py.
T5 baseline/*.py and baseline/metadata/: same, plus split graph.py and
   cli.py (one module per command group), header comments where > 400.
   Done (each): `ruff check <paths>` 0; `mypy --strict-ish <paths>` 0;
   no function > 50 lines in <paths>; tests pass.

### Phase 3 (one worker)
T6 Tests. Files: tests/**. Move to tests/unit and tests/integration
   (demo server, vlm stub, corpus runs); replace private-attr assertions
   (test_graph.py:232, test_offsets.py:185); add tests for rules, schema,
   logging_setup, config, cli, errors, io; coverage gate 85 in pyproject.
   Done: `pytest --cov` >= 85 and passes.

### Phase 4
T7 Verify (Sonnet): re-run every BASELINE.md command; fill "after" table;
   check each Done line. T8 Docs: docs/parts/standards.md, README tooling
   section.

## Status log
- done: standards extraction, 4 audits, this plan, T0 (commits 0f867f9,
  d01bcfc; ruff 2111 -> 393 under new config, black/isort clean, 560 pass)
- done: T2 (ad4ef21: baseline/io.py atomic writes, --dry-run, --resume
  flag, crash-safe cascade resume), T1 (b4244b6: baseline/errors.py, 14
  justified except-Exception sites left, no default model in demo).
  Ruff after phase 1: 369. 582 tests pass.
- done: T3 (2397371), T4 (5edaa68), T5 (f3d0571). After phase 2: ruff 4,
  mypy 0 (60 files), black/isort clean, 0 functions > 50 lines, 584 pass,
  coverage 86.7 %. New modules: baseline/{training,graph_report}.py,
  evaluate/{io,calibration,metadata_scores,compare}.py, workbook/summary.py.
- done: T6 (caa17e5: tests/unit 438 + tests/integration 153, gate 85,
  coverage 87.3 %, ruff 0). T7 verify: 4 defect groups found (7 long
  modules without header note, 6 non-atomic writers, CI lacks tesseract,
  demo probe too narrow).
- done: T8 (3f2b1d0: 6 more atomic writers, header notes on 8 large
  modules, tesseract in CI, demo probe widened). Final re-measure by the
  orchestrator: ruff 0, black/isort 0, mypy 0 (60 files), functions > 50
  lines 0, except-without-reason 0, 592 tests, coverage 87.3 %.
  Result written to docs/parts/standards.md.
- in progress: none
- blocked: none. CI has not yet run on GitHub (branch not pushed).
