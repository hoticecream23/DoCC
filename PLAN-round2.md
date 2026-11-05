# PLAN-round2: improvement round 2

Date: 2026-11-05. Branch: `small-fixes`. Inputs: docs/parts/round2-cascade.md, round2-tables-graph.md, round2-classification.md (Sonnet audits), plus what round 1 deferred (PLAN-round1.md).

## What the audits found

- **Classification:** after 67f5e59, SROIE still has 85 invoice-predicted and 39 unknown receipts. Evidence that is receipt-only on the client set (0 hits on any client class): a GST ID / GST Reg No line (114 of the 124 misses; 103 with the due-date / bill-to / remit gate), a GST summary or rounding line (76), a standalone CASH line plus a CHANGE line (57). Together, gated: 111 of 124. Two docs look like real invoices and no text rule should reach them. tax_form: only "tax statement" is a clean marker (+1 doc). id_document: OCR text is noise on 3 of 4; nothing proposed.
- **The SVM:** trained on 42 synthetic samples, 7 classes, no tax_form or court_document, never saw client text. Every one of the 16 client abstentions has model top-probability .16-.24. Gold for classification exists (112 labelled client docs), so the frozen rule "do not retrain before gold exists" no longer blocks a retrain; leakage does, so any retrain is scored by cross-validation only.
- **Cascade:** the 5 "omitted" values were junk rule values the model correctly dropped. Real FPs on covered fields: 3 wrong-field account_number, 1 rule override (SIDBI invoice_number). Proposals: per-field prompt definitions with negatives (about -4 FP), a shape gate on model-only account_number / invoice_number (about -3 FP), `maxItems: 1` and empty-list-never-empty-string (0 to +1 TP). Fuzzy grounding gains nothing on these docs.
- **Tables / graph / workbook:** tables are scored only on 5 synthetic gold tables; line items never become records and nothing consumes tables; the graph is a spike with no consumer; the workbook shows confidence as a bare number and its import only produces gold rows.
- **Eval:** buyer_name misses on case only ("John doe").

## Tasks (ranked)

| # | Task | Files | Done check | Risk |
| --- | --- | --- | --- | --- |
| R2-1 | Receipt markers: GST ID/Reg No, GST summary/rounding, CASH+CHANGE, each gated on no due date / bill-to / remit, at a specificity that beats the invoice markers; `tax statement` marker for tax_form | config/classes.yaml, tests/test_taxonomy.py | SROIE accuracy .656 -> ≥ .90, client set: no class loses precision, macro F1 not down; suite green | low |
| R2-2 | Cascade prompt and gates: per-field definitions with negatives, shape gate on model-only account_number / invoice_number, `maxItems: 1`, empty list never empty string | cascade/prompt.py, cascade/merge.py or ground.py, tests | 20-doc re-run: covered-field FP down ≥ 3, TP not down; classification unchanged | medium: model run needed |
| R2-3 | Retrain the TF-IDF SVM on the 112 labelled client docs (labels from the folder name in source_path), 5-fold stratified CV, calibrated; keep model_min .45 and every rule threshold; write the new models/ and the manifest; report CV coverage / precision / macro F1 vs the current model; final model trained on all 112 but quoted numbers are CV only | baseline/classify.py (train), models/, models/MANIFEST.sha256, tests | CV: coverage up, precision 1.0 kept on every class, macro F1 up; corpus/ run still green; suite green; independent review before accept | high: changes the model artefact |
| R2-4 | Layout fallback on the image route: `anchor.direction: right_of_label / below_label` using the word-box sidecar, after the text rule misses, offsets preserved | baseline/metadata/fields.py, baseline/tables/geometry.py (reuse), config/fields.yaml, tests | SROIE image route value F1 .314 -> up, text route unchanged, corpus/ byte-identical | medium |
| R2-5 | Eval: case-insensitive match for name fields (buyer_name, vendor_name) via their normalizer, not the harness | baseline/metadata/normalizers.py, config/fields.yaml | gold_reviewed buyer_name +1 TP; nothing else moves | low |
| R2-6 | Workbook: a REVIEW_PRIORITY column from the calibration bucket of each field, written only to new exports | baseline/workbook/exporter.py, tests/test_workbook.py | export shows the column; review.xlsx untouched; suite green | low |
| R1-6 | (carried) config loaded once in cli main; cascade retry statuses and merge 0.9 to cascade.yaml; one yaml loader | baseline/cli.py, cascade/vlm.py, cascade/merge.py, cascade/config.py | suite green | low |

Deferred: line-item records with an arithmetic check (needs table gold first); real table gold (human work); retraining with new classes (needs a taxonomy decision).

## Status log

- 2026-11-05: audits done, plan written. R2-1 dispatched into the free slot; K3+K4 and R1-4 still running.
- 2026-11-05: R2-2 dispatched after K3+K4 landed. Running: R1-4 (OSD), R2-1 (markers), R2-2 (cascade prompt). Queued: R2-3 (retrain, CV), R2-4 (layout fallback), R2-5, R2-6, R1-6.
- 2026-11-05: R2-1 5679e67: SROIE accuracy .656 -> .958, coverage .892 -> .997, invoice predictions 85 -> 14, unknown 39 -> 1; client tax_form R .20 -> .40, macro F1 .765 -> .795, coverage .911 -> .920, no precision lost. Risk noted: 'GST No.' Indian invoices with no due date / bill-to. R2-3 dispatched.
- 2026-11-05: R2-4 dispatched. Running: R2-2, R2-3, R2-4.
- 2026-11-05: R2-1 regex fix f957a17 (no atomic group, linear whitespace, 561 tests). R2-3 cfe4818 adopted: model retrained on 112 client docs (old: 42 synthetic, 7 classes); 5-fold CV full pipeline coverage .920 -> .955, macro F1 .795 -> .816, min class precision .80 unchanged; old model kept in models/previous/. Independent leakage review dispatched. NOTE: models/*.joblib are gitignored, so the manifest in git now describes model files that must be shipped alongside it. R2-6 dispatched.
- 2026-11-05: R2-6 0315524 (REVIEW_PRIORITY on new exports; importer reads by header name). Retrain review: ACCEPT; in-fold vectoriser and calibration confirmed, labels from folders, CV numbers reproduced exactly against models/previous; two script fixes requested (dedupe doc_id e075127fa5dba90e across folds; --baseline-models default models/previous). Pipeline CV numbers are optimistic because the rules were tuned on the same 112 docs.
- 2026-11-05: R2-4: box-based label/value pairing built and measured on the SROIE image route: micro F1 .440 -> .439, total_amount unchanged, no gain (PROJECT_STATE's .314 is stale; HEAD is .440). Commit 79c5c7e (mechanism off) reverted in ebacd44: no dead code. Finding kept here. R2-5 dispatched.
- 2026-11-05: CV script fixes 815f9fa (111 docs: coverage .919 -> .955, macro F1 .795 -> .816). R2-2 664a458 done (see PLAN-cascade.md). R1-6 dispatched. Running: R2-5, R1-6.
- 2026-11-05: R2-5 e777799 (upper-case name normaliser) changed no number: the scorer compares our normalised value against the gold's raw text, so a case mismatch survives any normaliser. Reverted in the commit after it. Follow-up for a future round: casefold or normalise the gold side in the scorer's covered view (frozen-eval question).
- 2026-11-05: R1-6 0565a1f (config once in cli main, vlm.retry_statuses in cascade.yaml, one read_yaml). **Round 2 complete.** Final suite at HEAD: 560 passed in 91 s. Results: docs/parts/rounds.md.
