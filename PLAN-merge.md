# PLAN-merge: first steps toward merging DoCC with the DCS MES shell

Date: 2026-11-09. Branch: `small-fixes`. Decisions come from
docs/parts/assimilation.md (SQLite first, FastAPI as a thin shell, DoCC
schema is the data authority).

## Tasks

M1 SQLite store and cabinet_id. Files: baseline/db.py (new), schema.py,
   config.py, pipeline.py, cli.py, config/pipeline.yaml,
   docs/parts/schemas-data.md, tests/unit/test_db.py,
   tests/integration/test_db_roundtrip.py.
   Done: `db-import` twice gives equal counts; `db-export` round-trips;
   pipeline writes a row after the JSONL commit; DB failure never blocks
   JSONL; five checks green.
M2 FastAPI shell. Files: api/** (new), pyproject.toml (api extra, httpx in
   dev), .github/workflows/ci.yml, README.md, tests/*/test_api*.py.
   Done: /health, /documents/process, /documents/process_async,
   /jobs/{id}; BadInput->400, ConfigError/Unavailable->503; X-Request-ID
   echoed; in-memory JobStore behind a protocol; five checks green.
M3 Wire API to DB (after M1 and M2): DB-backed JobStore, GET /documents
   list from the DB, process routes upsert on completion. Done: restart
   keeps job state; list endpoint matches db rows; five checks green.

M2b API hardening (from the automated security review of a627ed7):
   Content-Length pre-check -> 413; bounded job queue -> 503 Retry-After;
   finished-job eviction; upload dir cleanup; move require_model and
   vlm_settings out of demo/server.py into cascade/.

M1b cabinet_id through the cascade runner; cascade upserts to the store;
   workbook import writes review_events (idempotent by hash).

## Status log
- done: M2 (a627ed7), M1 (3c7e6b7: baseline/db.py 8 tables, db-import /
  db-export, pipeline hook), M2b (a464386: 413 body limit, bounded queue,
  helpers moved to cascade/config.py), M1b (25c74bb). Coverage 89.1 %.
- done: M3 (20a658a: SqliteJobStore with stale-job failover, GET
  /documents list/detail/review_events, API upserts, cascade --cabinet,
  manifest path from config root), cleanup (c9623a6). Orchestrator
  verification on 20a658a: 656 tests pass, coverage 87.3 %; on c9623a6
  ruff/black/mypy clean, 0 functions > 50 lines, 0 large modules without
  a note, 0 broad excepts without reason.
- in progress: none
- pushed 2026-11-10 at 90e2a4e; CI green on py3.10 and 3.12 (bandit B608
  false positives in db.py list queries resolved with static SQL).
- blocked: none. Open from workers: baseline/config.py has
  no dedicated api db key (API reuses db.path in root); no job TTL;
  review_events written only when a store already exists.
- next (not started): real client re-baseline, per-client schemas,
  Arabic OCR check, HITL UI on top of the new routes.

## Client test preparation (2026-11-12)
- done, local only, not pushed: e36c83f (scripts/inventory_share.py: files,
  page counts, index-field schema), e894e46 (client packs: per-cabinet
  taxonomy, field schema, labeled manifest, client-eval), 1c073c5 (pack
  mode for baseline and cascade, keyword class prior, client field names,
  vlm.input image|text, client-run command, page limit). Verified by the
  orchestrator: 732 tests, coverage 90.2 %, all six checks clean.
- scoring available without annotation: class from folder name, key field
  from file name. No other field is scored.
- waiting on the user: exact model tag and server address; headers.csv for
  two workbooks the schema export could not parse; confirmation of one
  cabinet's two types; commit and Python on the client machine.
- next: push (needs the user's OK), git pull on the client machine, build
  the pack there, client-run per cabinet, paste back the report.
