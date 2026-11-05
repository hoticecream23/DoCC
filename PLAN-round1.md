# PLAN-round1: improvement round 1

Date: 2026-11-05. Branch: `small-fixes`. Inputs: docs/parts/round1-ocr.md, round1-layout.md, round1-calibration-debt.md (Sonnet audits), docs/parts/cascade-pass.md (K1).

## What the audits found

- **OCR (critical path):** current path is a LANCZOS upscale to a 1000 px short side, `--psm 3`, no deskew/binarisation/OSD. On 8 images no global preprocessing variant beat it (mean word confidence .627 vs .56-.61). Per-file gains have opposite signs. OpenCV is not installed. The "ID route" proposal is refused by HANDOFF.md:81.
- **Layout:** SROIE text input carries no boxes, so a box-based fallback cannot help there; the image route has boxes and 24 of 134 label docs miss a number. Most remaining text-route misses are text-only: `TOTAL INCLUDES GST` (6 docs), `RECEIPT NO.` regex (1), `TL` alias (1); 7 of 10 sampled invoice_number misses have a blank label with no value anywhere.
- **Calibration:** confidences are ordinal only; validated fields score .983 precision against a nominal 1.0; the harness cannot bucket by confidence. SROIE classification answers 322 of 361 at rule confidence with .037 accuracy, flagged earlier as a label mismatch and never verified.
- **Debt:** cli.py 9 config loads, 50% covered; evaluate/report.py 16% covered; 7 of the 10 slowest tests are in test_pipeline.py; suite 127 s; 8 `ponytail:` comments.

## Tasks (ranked)

| # | Task | Files | Done check | Risk |
| --- | --- | --- | --- | --- |
| R1-1 | Text rule fixes from the layout audit: `total includes gst`, `TL` alias, `RECEIPT NO.` regex, two-line vendor names (second all-caps line joined) | config/fields.yaml, baseline/metadata/fields.py (vendor only) | SROIE text: total_amount and invoice_number recall up, no precision down; gold_reviewed unchanged or up; suite green | low |
| R1-2 | SROIE classification: verify whether .037 is a gold-label string mismatch in scripts/convert_sroie.py or a real misclassification (receipts titled "TAX INVOICE"); fix the conversion if that is it, else report | scripts/convert_sroie.py, datasets/sroie manifest | SROIE classification accuracy explained with numbers; converter fix measured | low |
| R1-3 | Eval: confidence buckets (0.5-0.7, 0.7-0.9, 0.9-1.0) with precision per bucket per field, plus ECE, in the report | baseline/evaluate/scoring.py, report.py, tests/test_evaluate.py | report shows buckets on existing predictions; tests; suite green | low |
| R1-4 | OCR OSD rotation: when a scanned page's mean word confidence < 0.6, run `image_to_osd`, rotate if it says 90/180/270 with confidence, re-OCR once | baseline/ocr.py or extract.py | measured on the 40 labelled image files via the classification harness: accuracy .525 -> must not drop, and page confidence up on rotated pages; suite green; byte-identical on corpus/ | medium: time per bad page |
| R1-5 | Test speed: one session-scoped pipeline fixture for test_pipeline.py's slow tests | tests/test_pipeline.py, tests/conftest.py | suite wall time down ≥ 20%, same test count | low |
| R1-6 | M1+M3: config loaded once in cli main; cascade retry statuses and merge 0.9 default into cascade.yaml; one yaml loader | baseline/cli.py, cascade/vlm.py, cascade/merge.py, cascade/config.py | suite green, cmd_* imports kept | low |

Deferred to round 2: layout fallback on the image route (round1-layout.md proposal 2, measure on SROIE image-route F1 .314); per-rule precision calibration (needs R1-3); workbook flags on weak fields.

## Status log

- 2026-11-05: audits done, plan drafted. Waiting on K1 (cascade diagnosis) before dispatch; cascade fixes run first.
- 2026-11-05: R1-1 (fields.yaml text fixes), R1-2 (SROIE label check), R1-3 (eval calibration buckets) dispatched to Opus in parallel while K1 runs on the GPU. R1-4 (OSD rotation), R1-5 (test fixture), R1-6 (M1+M3) queued.
- 2026-11-05: R1-2 verdict: not a label bug. SROIE predicts invoice 310 / unknown 39 / receipt 12 because the invoice marker fires on the 'TAX INVOICE' title (classes.yaml:23). Converted into R1-7 (receipt vs tax-invoice disambiguation: settlement line and no due date / bill-to / remit), measured on SROIE and the 112-doc client set; dispatched to Opus.
- 2026-11-05: R1-3 fc0a977 (calibration section + ECE in the eval; SROIE field ECE .41, classification ECE .90). Harness flaw found: fields the gold set never labels (invoice_number, email, buyer_name on SROIE) count as FP; round-2 item: score only fields the gold set covers. R1-5 dispatched.
- 2026-11-05: R1-7 67f5e59: receipt marker (tax invoice + settlement line, no due date/bill-to/remit). SROIE accuracy .033 -> .656, receipt R .033 -> .656, invoice predictions 310 -> 85; client set unchanged (macro F1 .765). ~40 SROIE misses print only 'Invoice No', out of reach of this rule. K2 dispatched into the freed slot.
- 2026-11-05: R1-5 81c2199: suite 123 s -> 96 s (22%), two session fixtures, no weaker assertions. K3+K4 dispatched.
- 2026-11-05: R1-1 efca31c: SROIE total_amount F1 .717 -> .729, vendor_name .590 -> .633, invoice_number reads 233 -> 234; gold_reviewed unchanged. R1-4 dispatched.
- 2026-11-05: R1-4 a5c5464: OSD on pages with confidence < .6 (8 of 40 images triggered, 1 rotated, .27 -> .68), 40-image classification accuracy .700 -> .700 (note: .525 -> .700 since the audit pass began), +20-30% time on those files, corpus byte-identical. Kept as a correctness feature; small measured gain. Round 1 implementation complete except R1-6 (carried into round 2).
- 2026-11-05: **Round 1 complete** (R1-6 landed in round 2 as 0565a1f). Results: docs/parts/rounds.md.
