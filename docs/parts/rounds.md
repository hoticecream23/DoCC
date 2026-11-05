# Cascade pass and improvement rounds 1 and 2, 2026-11-05

Owner document. Plans and status logs: PLAN-cascade.md, PLAN-round1.md, PLAN-round2.md. Audit reports: docs/parts/cascade-pass.md, round1-*.md, round2-*.md. All on branch `small-fixes`.

## Headline numbers

| Metric | Before (start of 2026-11-05) | After | Where |
| --- | --- | --- | --- |
| Cascade vs baseline, 20 reviewed docs, classification macro F1 | .811 vs .953 (old notes: .758 vs .953) | .811 vs .953, coverage .75 vs 1.00 | PLAN-cascade.md |
| Cascade vs baseline, 20 docs, metadata on gold-covered fields | .560 vs .552 (looked like a drop) | .560 vs .595 | K2, R2-2 |
| Cascade fallbacks on 20 docs | 2 | 0 | K4, R2-2 |
| SROIE classification accuracy (361 public receipts) | .033 | .958, coverage .997 | R1-7, R2-1 |
| Client set (112 docs) macro F1 / coverage | .765 / .911 | .795 / .920 (rules), .816 / .955 (CV with retrained model) | R2-1, R2-3 |
| 40 labelled image files, classification accuracy | .525 (PROJECT_STATE) | .700 | measured in R1-4 |
| SROIE text total_amount F1 | .717 | .729 | R1-1 |
| SROIE text vendor_name F1 | .590 | .633 | R1-1 |
| SROIE covered-field F1 (honest view) | not available | .635 | K2 |
| Test suite wall time | 123 s | 96 s | R1-5 |
| Tests | 527 | 560 passed, 91 s | |

## What changed

- **Eval harness:** calibration buckets and ECE (fc0a977); a gold-covered-fields view that stops punishing a system for reading fields the gold never labels (ace372a). Existing numbers and JSON keys unchanged.
- **Cascade:** merge ranks every model value and prefers rule agreement (35bde23); 400 retries once with half the pages, 500 retried (5e73df6); per-field prompt definitions with negatives, exact-string wording, maxItems 1, shape gate on model-only identifiers (664a458); invoice_number normaliser (f54fe16).
- **Classification:** receipt titled "tax invoice" with a settlement line and no due date is a receipt (67f5e59); gated GST-line, rounding and cash/change markers plus a tax_form marker (5679e67, f957a17); classifier retrained on 111 client docs under 5-fold CV, old model kept in models/previous (cfe4818, 815f9fa).
- **Fields:** totals that include GST, wrapped vendor names, dotted receipt labels (efca31c).
- **OCR:** OSD rotation on pages below .6 confidence (a5c5464).
- **Workbook:** REVIEW_PRIORITY column on new exports (0315524).
- **Debt:** shared pipeline fixtures (81c2199); config loaded once, cascade constants in cascade.yaml (0565a1f).

## Rejected after measurement

- Box-based label/value pairing on the image route: micro F1 .440 -> .439. Built, measured, reverted (ebacd44).
- Upper-case name normaliser: no number moved because the scorer compares against raw gold text. Reverted (a59d266). The fix belongs in the scorer.
- Global OCR preprocessing variants (Otsu, psm 4/6, 3300 px upscale): none beat the current path on 8 images (round1-ocr.md). Not implemented.

## Things to know

- `models/*.joblib` are gitignored. The committed manifest now describes the retrained model; ship `models/` with it, or `baseline run` refuses to load.
- The 112-doc client numbers are optimistic: rules were tuned on those docs this session. The CV protocol isolates the model, not the rules.
- Cascade run-to-run variance is about one document in 20 on classification.
- Two SROIE docs (OJC MARKETING) look like real invoices and stay predicted invoice on purpose.
- Open: scorer casefolding on the covered view; line-item records need table gold first; id_document OCR text is noise on 3 of 4 docs.
