# Round 2: classification gaps after 67f5e59

Read-only audit. No repo code changed. Gold files not opened. SROIE gold = receipt for all 361 (task statement). Client gold class = folder name in source_path. Inputs: scratchpad r1_7/sroie_after.jsonl and client_after.jsonl (text.full). Script: scratchpad/r2cls.py.
SROIE state in the file: 237 receipt, 85 invoice, 39 unknown (the brief said 236 correct; the file has 237 = .656).

## 1. Evidence table (regex presence on canonical text)

SROIE columns: 85 invoice-predicted | 39 unknown | 237 correct receipts. Client columns: hits on all 112 docs; non-receipt hits named.

| evidence | 85 | 39 | 237 | client hits (112) |
|---|---|---|---|---|
| "GST ID / GST Reg No / GST No" (`\bgst\s*(?:reg\w*\.?\s*)?(?:id\|no)\b`) | 81 | 33 | 175 | 0 |
| "GST summary" | 45 | 22 | 150 | 0 |
| "rounding" / "round adj" | 31 | 18 | 125 | 0 |
| ROC / reg-no in brackets `(123456-X)` | 50 | 31 | 176 | 0 |
| qty@price lines (>=2 lines `1@4.80`) | 5 | 2 | 26 | 0 |
| standalone CASH line + CHANGE line | 39 | 18 | 156 | 0 |
| total + rounding | 31 | 14 | 124 | 0 |
| qty x price anywhere | 7 | 4 | 43 | 0 |
| tel/ROC/reg no label | 12 | 15 | 37 | 0 |
| bill no | 0 | 6 | 5 | 1 tax doc |
| served by | 6 | 1 | 1 | 14 (legal docs) TRAP |
| cashier | 15 | 20 | 118 | 1 Invoice TRAP |
| cashier + thank you | 7 | 17 | 67 | 1 Invoice TRAP |
| thank you | 33 | 33 | 161 | 15 (10 Invoice) TRAP |
| HH:MM | 65 | 37 | 207 | 10 TRAP |
| cash/change/tendered/rounding word | 84 | 36 | 233 | 28 (5 Invoice) TRAP |
| "invoice no" without "tax invoice" | 40 | 0 | 2 | 21 (20 Invoice, 1 PO) TRAP |
| "receipt" word | 6 | 7 | 75 | 21 TRAP |

Findings: "invoice no" alone is unusable (fires on 20 of 25 client invoices). The 40 "invoice no" docs are not reachable by that string. They are reachable by the GST ID marker. Bare "cash/change" words, cashier, thank you and HH:MM all fire on client invoices or legal docs.
Both of 85 and 39 print Malaysian GST identity lines. The client set has none.

## 2. Proposed markers (all 0 hits on all 112 client docs)

Counts are over the 124 misses (85 + 39). "gated" = also no due date / payment terms / bill-to / remit line, as in 67f5e59.

1. GST registration line: `\bgst\s*(?:reg\w*\.?\s*)?(?:id|no)\b`. Ungated: 81 + 33 = 114. Gated: 70 + 33 = 103.
2. "GST summary" or "rounding" / "round adj" line. Ungated: 48 + 28 = 76 alone. Gated: 37 + 28 = 65. Adds few beyond marker 1 (use as second evidence, or skip).
3. Standalone `CASH` line and `CHANGE` line (multiline, line starts with the word). Ungated: 39 + 18 = 57. Gated: 28 + 18 = 46. It catches docs where the amount sits on the next line, which the 67f5e59 settlement regex misses.
4. Two or more `1@4.80` / `2X 5.90` item lines. 5 + 2 = 7. Weak, only for stragglers.

Union of 1+2+3 (ungated): 83 invoice + 39 unknown = 122 of 124. Gated: 72 + 39 = 111 of 124.
Trade-off: the gate drops 11 docs that print bill-to / due date. Those are real receipts by gold. Keep the gate unless the team accepts the trap risk. Marker 1 is Malaysia-specific. A client Indian GST invoice would print GSTIN and not match, but "GST No." on one would match. Re-check on new client data. Score must beat the invoice markers (0.97, as in 67f5e59).

Unreachable by markers 1+2+3 (2 docs, both invoice-predicted): X00016469670.txt and X00016469671.txt (OJC MARKETING SDN BHD). They print TAX INVOICE, INVOICE NO, BILL TO, cashier, items; no GST ID, rounding or CASH/CHANGE lines. They look like real invoices; no text rule can separate them from a client invoice. Treat as label noise. The ROC-bracket pattern would reach neither, so it is not proposed.
Docs only reachable by marker 4 (qty@price lines): none found beyond 1 to 3, so marker 4 can be dropped.

## 3. tax_form and id_document (client)

"tax document" folder = 5 docs. Tax_form recall .20 = 1/5 (the ITR acknowledgement, 0.75).
- 2 docs are tax invoices (Dubai NG LLC, Gujarat Freight). Predicted invoice. config/taxonomy.yaml:29 says the folder is a subject area, mapping lossy. No tax_form marker should capture these, and they are probably right as invoice. The real ceiling for tax_form recall is lower than the folder count suggests (about 3/5).
- CA Final Direct Tax Summary Notes (unknown, model .21): study notes, not a form.
- Real property TAX STATEMENT (unknown, .16): a government statement. This is the only miss a marker can reach.

Evidence (tax doc 5 | other 107): form 16 0 | 1; assessment year 1 | 4 (legal docs); tds 0 | 2; pan 1 | 9 (6 salary slips); financial year 0 | 2; income tax 2 | 13; tax (any) 5 | 41; tax statement/summary 2 | 0; tax invoice 2 | 0; 1040/W-2/1099 0 | 1.
Only "tax statement / tax summary" is clean (0 other hits). "Form 16", "assessment year", "tds", "pan", "financial year" fire on legal docs and salary slips, so they are unusable alone.

Proposed tax_form markers: (a) `\b(?:property\s+)?tax\s+statement\b`, header scope, spec ~0.8: 1 of 4 misses, 0 other. (b) `\btax\s+summary\b` only if the team counts study notes as tax_form: not recommended. Do not add "tax invoice" (it would break invoice P). Net gain: +1 doc (R .20 -> .40), plus macro-F1 effect small.

id_document (4 docs, R .25 = 1/4, only the PAN card, 0.93): Aadhaar, passport, voter id are OCR garbage (100-930 chars, mostly noise, "Your Aadhaar" fragment). "income tax" fires on the PAN card only. No ID route may be built (HANDOFF.md:81). Nothing proposed.

## 4. Baseline model

- models/classifier/meta.json: n_samples 42, 7 classes x 6 samples, 5-fold CV. Labels: bank_statement, contract, id_document, invoice, purchase_order, receipt, salary_slip. Training set is the synthetic corpus only. PROJECT_STATE.md:896 and :996 say the same ("trained on synthetic data only", "never been run on the real corpus").
- It has no tax_form, court_document, legal or notes class. With 7 classes the top probability is near uniform on client-like text (.16-.24 on the 16 abstentions).
- It never saw client-like text, so its probability cannot reach 0.45.
- Retrain proposal (do not run): needs 112 labelled client docs, of which classes with < 2 examples must be merged or dropped (calibration needs >= 2 per class; build_estimator uses the smallest class count for CV folds). Class sizes: Writ 16, Salary 15, Invoice 25, Receipt 7, PO 6, tax 5 (only 3 real tax_form), ID 4, plus legal groups of 5 each. Needs a held-out split or cross-validation by doc (112 docs is thin: 5-fold gives ~20 per fold, 4-5 docs per rare class). Needs the fold report plus a client-set evaluation on docs unseen in training, else the macro F1 gain is training-set leakage. Models come with `baseline train --annotations FILE --output-model DIR`, then `record_hash` updates models/MANIFEST.sha256. Risk: gold and training would overlap, so a fair score needs nested CV. Expected effect: low for rule-covered classes; it would help abstentions only if it learns the layout of the 9 no-rule docs, which is 1-2 docs per class.

## 5. Retrain (R2-3)

Training set: the 112 labelled client docs. Text is the stored canonical `text.full` in results_classification.jsonl. The label is the folder name in `source_path`, mapped by config/taxonomy.yaml (no folder dropped): court_document 46, invoice 25, salary_slip 15, receipt 7, purchase_order 6, tax_form 5, contract 4, id_document 4. The old model (42 synthetic docs, 7 classes x 6) is kept outside git under models/previous/.

Protocol: `python scripts/cv_classifier.py results_classification.jsonl OUT [--baseline-models models/previous]`. One doc_id is filed twice (Stroyproekt, under Litigation Strategy Notes and Written Submissions, same bytes). CV keeps one copy, so the CV runs on 111 docs (court_document 45) and no copy is tested against its own twin. 5-fold StratifiedKFold, shuffle, seed 0. Each fold fits `build_estimator` (the same pipeline as `train`) on the training docs only. Held-out docs are scored by the model alone (top probability >= model_min .45) and by the full `classify()` (rules + model). The "current" columns are the previous model from models/previous/, scored on the same held-out docs. Thresholds unchanged. All numbers below are CV; none come from the all-112 fit.

Per fold (coverage / min class precision / macro F1):

| fold | current model | new model | current full | new full |
|---|---|---|---|---|
| 0 | .000 / - / .000 | .826 / 1.00 / .583 | 1.00 / .500 / .697 | 1.00 / .500 / .697 |
| 1 | .000 / - / .000 | .909 / 1.00 / .750 | .864 / 1.00 / .736 | .955 / 1.00 / .875 |
| 2 | .000 / - / .000 | .773 / 1.00 / .375 | .864 / .800 / .700 | .955 / .833 / .739 |
| 3 | .000 / - / .000 | .773 / 1.00 / .429 | 1.00 / .833 / .939 | 1.00 / .833 / .939 |
| 4 | .000 / - / .000 | .773 / 1.00 / .429 | .864 / 1.00 / .667 | .864 / 1.00 / .667 |
| mean | .000 / - / .000 | .811 / 1.00 / .513 | .918 / .827 / .748 | .955 / .833 / .783 |
| pooled | .000 / - / .000 | .811 / 1.00 / .550 | .919 / .800 / .795 | .955 / .800 / .816 |

Pooled precision among answered, full pipeline, current -> new: court_document 45/45 -> 45/45, invoice 23/26 -> 25/28, salary_slip 14/14 -> 15/15, receipt 5/5 -> 6/6, purchase_order 4/4 -> 4/4, tax_form 2/2 -> 2/2, contract 4/5 -> 4/5, id_document 1/1 -> 1/1. Model alone, new: every class it answers is 1.0 (contract 2/2, purchase_order 1/1); the current model answers nothing at .45.

The full decision changes on 4 docs, all from unknown to the right class (2 invoice, 1 receipt, 1 salary_slip). The other errors (2 tax_form -> invoice, 1 PO -> invoice, 1 PO -> contract) are rule answers and the same under both models.

Decision: adopted. Full-pipeline coverage rises (.919 -> .955 pooled), no class loses precision, macro F1 rises (.795 -> .816 pooled). contract and id_document have 4 docs each: reported, not claimed. 111 docs is a thin set. The rules were also tuned on these docs, so the full-pipeline numbers are optimistic for both models alike. Final model: `baseline train --kind classifier` on all 112 docs, manifest updated.
