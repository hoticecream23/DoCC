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

## Store additions from the Phase 2 database spec (2026-11-15)
- done, local only: de1891b (tenants, cabinets, doc_types, field_defs,
  audit_log, schema_migrations; documents.routing_status and case_id;
  review_events.verdict and is_gold; runs.trace_id; effective_fields read
  helper; register_pack; four API read routes), f1ffb47 (function size).
- rejected and not built: PostgreSQL-only, outbox, fact/dimension tables,
  generated document ids, in-row corrected values, bbox integer columns.
- open risk from the worker: routing_status may mark most documents
  "review", because any unvalidated field without a checksum or format
  rule is high priority. Check on real data before relying on it.
- asked of the colleague, unanswered: the actual .sql files; which engine
  statement is current; where per-client field role and origin live; how
  a multi-file case is keyed.

## Code review of the client-test and store work (2026-11-15)
Paused here: usage limit reached. A fresh session can resume from this log.
- pass 1 (4 reviewers): 9 high, ~15 medium bugs, all reproduced. Fixed in
  0dcc49c (scoring), 3bcc003 (pack builder, inventory), d8a2590 + 92f89ba
  (store, API), acc1453 + b825bec (pack mode, text mode), 54d00db (loose
  ends). Verified by the orchestrator at that point: 918 then 925 tests,
  coverage 91 %, all static checks clean.
- pass 2 (2 reviewers): every pass-1 bug confirmed fixed except server
  path exposure (partly). No new high. New medium/low items dispatched:
  - ad14069 landed: pack-mode label rule, layouts, Arabic-Indic digits,
    integer normaliser, diagnostics notes (verified only in the worker's
    scratch worktree, 975 tests).
  - IN FLIGHT when paused, each commits its own paths when done:
    (a) pack builder + scoring round 2 (per-folder cases, stable slugs,
        merge conflicts, tie rules) in baseline/clientpack*.py,
        baseline/evaluate/, baseline/clientrun.py;
    (b) store + API round 2 (paths out of records, refused-import
        sidecars, review-event idempotency, connect locking, one API
        process per store) in baseline/db*.py, api/;
    (c) notes wiring in baseline/pipeline.py and cascade/runner.py
        (page-limit messages become notes, new outcome not_sent).
- TO DO on resume: check `git status` and `git log` for what (a), (b),
  (c) committed; finish or redo anything left uncommitted; run all six
  checks and the full suite on the combined tree; one short confirmation
  review of round-2 fixes; then ask the user before pushing, and build a
  new zip. Local branch is ahead of origin (last push 3fcc86d).
- The zip on the client machine (3fcc86d) predates the review. Its scores
  are unreliable: partial runs can show 100 %, real documents are dropped
  as duplicates, and a wrong key can be returned.

## Review closed (2026-11-15)
- Three review passes by independent agents, every finding reproduced by
  script before it was reported, every fix carrying a regression test.
  Pass 1: 9 high, ~15 medium. Pass 2: 0 high, ~10 medium. Pass 3
  (confirmation): 1 high (case id = folder name only), 2 medium; all fixed
  in b24cb47, 112fbdd, 490f3ac.
- Final verification by the orchestrator on 490f3ac: 1127 tests pass
  (765 before the review), coverage 91.8 %, ruff/black/isort/mypy/bandit
  clean, no function over 50 lines, no client names in tree or history.
- Known and accepted, not fixed: the API has no authentication or tenant
  isolation (documented; bind to loopback); byte-identical twins in one
  cabinet collapse to one store row; the built-in amount pattern drops a
  leading minus sign (pre-existing); a document id is a content hash, so
  the same file cannot belong to two cabinets in the store (refused with
  an audit row; a composite key is the planned fix).
- Not reviewed a fourth time: the three pass-3 fixes were verified by
  their own regression tests and the full suite only.
