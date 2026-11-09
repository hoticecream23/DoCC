# Assimilation plan: DoCC and the DCS MES specs

Date: 2026-11-09. Input: the 21-document "DCS MES" spec set (15 core docs,
6 client docs, visuals) and the DoCC codebase on `small-fixes`. Three
independent analyses (persistence, architecture, requirements) feed this.
Client names, people, hosts and share paths are deliberately left out.

## What the spec set is

A target design for a banking document platform, LLM-assisted, dated
2026-10-30 to 11-05. Same domain as DoCC, not a description of it. DoCC is
the working code; the specs are the proposed shell around it. Treat the
specs' numbers (latency, precision, uptime, ROI, "production", "approved")
as unverified. Do not repeat them to the client.

## Decision 1: database

**Choose SQLite now, with a schema that ports to PostgreSQL later.**

Why:
- The client box is a single on-prem GPU machine, offline. Stated volume is
  about 100 files across 13 client/document-type pairs. PostgreSQL adds
  install, backup and ops work with no benefit at that size.
- The spec's "outbox" (Postgres with SQLite fallback and replay) solves a
  Postgres-outage problem a one-box SQLite deployment does not have.
- DoCC's JSONL design is already crash-safe (results line as commit marker,
  partial-tail trim on resume). Keep JSONL as the append log and import
  source; make SQLite the queryable, authoritative store for review and
  cabinet state.

Schema authority: DoCC's Pydantic models in `baseline/schema.py`. The spec's
three tables (documents, hitl_feedback, outbox_events) cannot hold what DoCC
produces. Gaps in the spec's model: field offsets, page, method and
normalizer version; point-coordinate bboxes (the spec uses 0-1000 ints and
has no table for them); cascade provenance and abstention reasons; tables
and line items; model manifest; run and schema versions; per-field review
verdicts and human-added rows from the workbook; any cabinet/tenant column.

Tables to create: documents, fields, tags, tables, provenance,
review_events, runs, models; every row keyed by `cabinet_id` plus DoCC's
hash `doc_id` (keep the hash, not the spec's UUID). Bbox blobs stay on disk;
store path and hash.

Steps, in order:
1. Add `cabinet_id` to DoCC records and config.
2. `baseline/db.py`: stdlib `sqlite3`, WAL mode, portable SQL only.
3. Idempotent `import-jsonl` command; delete-and-reimport test.
4. Pipeline writes the DB row after the results line is committed.
5. Workbook import writes `review_events`.
6. Demo and API read lists from the DB; add `export-jsonl`.
7. Generate PostgreSQL DDL from the same schema when a second box or a
   second team needs it.

## Decision 2: architecture

| Proposal | Call | Reason |
|--|--|--|
| FastAPI + Uvicorn | Adopt, as a thin shell | DoCC already has Pydantic models; the demo router is hand-rolled |
| trace_id middleware, JSON logs | Adopt | cheap; DoCC's JSON logging exists |
| PII masking in logs and exports | Adopt | real requirement; no code exists yet on either side |
| 5-Stack classifier | Adapt | maps to DoCC baseline + cascade; expose two modes, not five stacks. The spec's fusion trusts VLM self-reported confidence; DoCC grounds values in text and should keep doing so |
| Celery/Redis, MinIO | Reject for now | extra services on a closed single box; use a thread pool and a SQLite job table |
| Postgres + outbox | Reject for now | see Decision 1 |
| JWT/OIDC/RBAC | Defer | the spec's own doc 14 marks it future; LAN firewall plus a static API key first |
| Hard-coded vLLM host | Reject | goes in `vlm.allowed_hosts` config |

Integration seams, DoCC code wrapped as is:
1. `baseline.pipeline.process_document` behind a sync `/process` route
   (upload to a temp file first).
2. `cascade.runner.process_document` as a background job (12-40 s per
   document measured; never synchronous).
3. `cascade.vlm.VLMClient` and the `Backend` protocol pointed at the
   colleague's vLLM endpoint via `vlm.allowed_hosts`.

Response schema: pick DoCC `DocumentRecord`; map the spec's
`DocumentProcessResponse` onto it, not the reverse.

## Conflicts to settle in the next discussion

- VLM latency: spec says 0.4-0.7 s; DoCC measured 11-40 s per document.
- Class count: 22 in one spec doc, 15 in another; the client docs imply
  per-client class sets.
- Model: spec names 2B/5B models and an AI judge; DoCC measured qwen3.5:9b
  on Ollama. Settle one model and pin it (standard S64).
- Domain: spec is banking (Aadhaar, dossiers); client documents are
  ERP-linked invoices, GRNs, vouchers and insurance forms.
- Auth: "production" in two docs, "future" in another.
- Volumes: the client docs disagree with each other on file counts.
- Not stated anywhere and must be asked: Arabic/RTL need, SLAs, internet
  policy on the box, ERP access method.

## Requirements coverage

27 must-haves extracted (shall/must wording). DoCC: 4 done, 14 partial,
9 missing. Missing: PII masking, cabinet router, ERP tier-2 hydration,
per-client tenancy, database, stateless scale, sub-1.5 s latency, TLS,
search. Of these, scale, latency and TLS are the spec author's targets,
not the client's.

## Backlog for the merged system

| # | Item | Effort | Head start |
|--|--|--|--|
| 1 | Re-baseline on real client documents; label 30-50 per cabinet | M | DoCC |
| 2 | Per-client class sets and field schemas; per-field VLM prompts | M | DoCC |
| 3 | Arabic/bilingual OCR check (DoCC is `eng` only) | M | neither |
| 4 | Cabinet/tenant model and router | M | spec |
| 5 | One SQLite database for records, corrections, audit (Decision 1) | M | spec schema + DoCC records |
| 6 | HITL review UI with correction capture and bbox highlight | L | spec UI + DoCC spans/workbook |
| 7 | Tier-2 ERP hydration, start with a CSV lookup | L | neither |
| 8 | PII redaction in logs and APIs; disk encryption; access control | S-M | neither |
| 9 | Service layer: FastAPI routes, share watcher, job table, VLM timeout with fallback | M | spec API + DoCC pipeline |
| 10 | Multi-page table and line-item extraction | L | DoCC (tables v0) |

Deferred: Celery, MinIO, Vault, Kubernetes, Prometheus, the 5-stack matrix,
OIDC, LayoutLMv3.

## What DoCC may claim today

Grounded spans, local-only inference, reproducible runs, macro F1 0.971
with cascade on 20 documents (state the sample size). Nothing else from the
spec's metrics.
