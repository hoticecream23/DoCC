# Cascade pass K1: baseline vs cascade v2 on the 20 reviewed documents

Date: 2026-11-05. Model: Ollama `qwen3.5:9b-16k`. Predictions and reports are in the scratchpad `k1\` folder (baseline.jsonl, cascade.jsonl, cascade.jsonl.provenance, baseline_report.md, cascade_report.md, diff.txt).
21 files were run. Gold has 21 rows, and the harness matches 20. The voter ID file (`Sample_voter_id.jpg`) is "unmatched gold" in both runs, so it is not scored.
Other agents edited `config/classes.yaml` and `config/fields.yaml` at 13:40 and 13:41, after both runs had loaded their config (baseline 13:31, cascade started 13:32). The numbers below used the config as committed.

## Eval summaries

| Metric | Baseline | Cascade v2 |
| --- | --- | --- |
| classification macro F1 | .811 | .971 |
| classification coverage | .750 (15/20) | .950 (19/20) |
| accuracy when answered | 1.000 | 1.000 |
| metadata value micro F1, all fields | .512 (P .412, R .677) | .449 (P .316, R .774) |
| metadata TP / FP / FN | 21 / 30 / 10 | 24 / 52 / 7 |
| metadata micro F1, fields with gold support only* | .560 (P .477, R .677) | .552 (P .429, R .774) |
| metadata span micro F1 | .189 | .073 |
| validated precision | 1.000 | 1.000 |
| tagging micro F1 | .533 | .526 |

*Gold support means the field has at least one TP or FN in the report. `due_date`, `phone` and `vendor_name` have none, so every value emitted there counts as a false positive. This is inferred from the report tables. Gold was not read.

Per-class F1 (baseline -> cascade): id_document .500 -> 1.000; invoice 1.000 -> 1.000; purchase_order 1.000 -> 1.000; receipt .667 -> .857; salary_slip .889 -> 1.000.

Per-field metadata (baseline -> cascade, with TP/FP/FN):

| field | baseline P/R/F1 (TP,FP,FN) | cascade P/R/F1 (TP,FP,FN) |
| --- | --- | --- |
| aadhaar | 0/0/0 (0,0,1) | 0/0/0 (0,0,1) |
| account_number | 0/0/0 (0,3,1) | .143/1/.250 (1,6,0) |
| buyer_name | 0/0/0 (0,1,1) | 0/0/0 (0,6,1) |
| due_date | 0/0/0 (0,3,0) | 0/0/0 (0,4,0) |
| email | .714/1/.833 (5,2,0) | .714/1/.833 (5,2,0) |
| invoice_date | .455/1/.625 (5,6,0) | .556/1/.714 (5,4,0) |
| invoice_number | 0/0/0 (0,5,1) | 0/0/0 (0,8,1) |
| pan | 1/1/1 (2,0,0) | 1/1/1 (2,0,0) |
| phone | not emitted | 0/0/0 (0,8,0) |
| tax_amount | 1/1/1 (3,0,0) | .600/1/.750 (3,2,0) |
| total_amount | .500/.500/.500 (6,6,6) | .667/.667/.667 (8,4,4) |
| vendor_name | 0/0/0 (0,4,0) | 0/0/0 (0,8,0) |

## Main finding

The 2026-10-22 metadata drop is not a loss of correct values. Against the baseline the cascade lost zero true positives. TP rose 21 -> 24, FN fell 10 -> 7, recall rose .677 -> .774.
Micro F1 fell only because precision fell: 22 more false positives. Of these, 13 are on fields gold does not label (`phone` +8, `vendor_name` +4, `due_date` +1). The rest are `buyer_name` +5, `account_number` +3, `invoice_number` +3 and `tax_amount` +2, minus 4 fewer on `total_amount` and `invoice_date`.
On gold-supported fields F1 is flat (.560 vs .552).
Whether the extra values are real errors needs gold for those fields. This pass does not read gold.

So "every metadata loss" means two lists: baseline values the cascade dropped (table A, none was a TP) and the 7 cascade FNs (table B).

## Table A: baseline value dropped or replaced by the cascade

| doc (file) | field | baseline | cascade | cause | decision rule |
| --- | --- | --- | --- | --- | --- |
| c66b Aadhar_card.webp | invoice_date | 1983-08-10 (a birth date) | none | (c) model omitted, rule_only drop. Gain. | cascade/merge.py:182 |
| b8aa passport | invoice_date | 2011-10-11 | none | (c) same. Gain. | merge.py:182 |
| 8330 Sample_voter_id.jpg | invoice_date | 0002-20-21 | none | (c) same. Gain (doc unscored). | merge.py:182 |
| 0829 equipment PO.webp | total_amount | 1.00 | none | (c) model omitted | merge.py:182 |
| 4d58 Equipment_po.jpg | total_amount | 30.00 | none | (c) model said `$0.00`, not grounded (merge.py:139), rule value then dropped | merge.py:139, 182 |
| 4d58 Equipment_po.jpg | invoice_number, invoice_date | PO-001234, 00-00-20 | none | (c) model omitted | merge.py:182 |
| 4d58 Equipment_po.jpg | vendor_name | "ABC Supplies Inc. Buyer Name" | "ABC Supplies Inc." | model value replaced an over-long rule value | merge.py:154-157, ground.py:185 |
| 67be Online_transactionReceipt.png | vendor_name | YOUR | none | model value `[YOUR COMPANY NAME]` not grounded (template placeholder), rule value dropped. Gain. | merge.py:139, 182 |
| 01ee SalarySlip.jpg | vendor_name | "SALARY FOR THE MONTH OF JUL 2013" | none | (c) model omitted. Gain. | merge.py:182 |
| 7c39 SIDBI PO.pdf | invoice_number | 2018MAR16/L00197407 | DIV/24102025/OUT/59885 | model value replaced rule value (neither scored TP) | merge.py:154-157 |
| 7c39 SIDBI PO.pdf | total_amount | 5.00 (wrong) | 4532321.00 | model grounded `45,32,321/-` and it agrees with a rule candidate. Gain, now TP. | merge.py:154, ground.py:196-199 |
| 0b93 SalarySlip2.jpg | account_number | 100000043634 (wrong) | 235362724352 | model grounded and agrees with rule. Gain, now TP. | merge.py:154 |

The rule_only drop fired on 8 doc-fields. None removed a TP.

## Table B: remaining cascade misses (the 7 cascade FNs)

finish_reason was `stop` for every doc that got an answer. No doc has more than 2 pages, so the max_pages 4 cap never cut a page.

| doc | field | gold value | cascade | cause | decision rule |
| --- | --- | --- | --- | --- | --- |
| 0b93 SalarySlip2.jpg | total_amount | 518692.00 | 1531194.00 | (g) the model listed two values, the first (wrong) grounded and was emitted, the second (correct) was never tried | merge.py:132 `if kept and not multi: break` |
| 2cb1 image-537.png | total_amount | 378.00 | none | (h) model read `$378.00` from the image. The value is not in the OCR text, so grounding refuses it | merge.py:139, ground.py:202 |
| 94af payment-receipt-template.webp | total_amount | 4044.00 | none | (h) model read `$4,044.60` (also a misread). Not in OCR text | merge.py:139, ground.py:202 |
| c66b Aadhar_card.webp | aadhaar | 3208 (last four) | none | (i) model returned `XXXX XXXX 3208`. It grounds exactly, but a masked number cannot pass the checksum gate | merge.py:150-151 |
| 99035 NG LLC invoice.jpg | invoice_number | 02-140305 | `02 - 140305` | (f) value grounds exactly, but the `upper_strip` normalizer keeps spaces around the hyphen, so the harness sees a different string | merge.py:154-157; invoice_number `normalizer: upper_strip` in config/fields.yaml |
| e97326 pay_slip.jpg | total_amount | 41682.00 | 45879.00 (baseline anchor) | (j) VLM call failed (HTTP 500 after 3 attempts), baseline fallback | cascade/runner.py exception path |
| 2cb1 image-537.png | buyer_name | "John doe" | "John Doe" | (f) case only, same as baseline. Gold casing vs extracted casing | merge.py:154 |

PurchaseReceipt.jpg also fell back (HTTP 400 after 1.1 s) and stays `unknown`. It is the one abstained document in cascade classification.

## Cause counts (docs per cause)

| cause | docs |
| --- | --- |
| (a) normalisation stops grounding | 0 |
| (b) model value overrode a correct rule value | 0 |
| (c) model omitted the field, rule_only drop (junk or wrong rule values only) | 5 |
| (d) truncation or off-schema | 0 |
| (e) page cap | 0 |
| (f) harness normalisation or casing mismatch | 2 |
| (g) first model value wins over a later correct one | 1 |
| (h) value is in the image but not in the OCR text | 2 |
| (i) masked checksummed value | 1 |
| (j) VLM call failed, baseline fallback | 2 |

## Timings

Baseline: 39.5 s for 21 docs including model load, about 1.9 s per doc (`--workers 1`).
Cascade: 516 s for 21 docs (13:32:02 to 13:40:38), 24.6 s per doc. VLM time per successful doc ran 11 to 40 s (SIDBI PO, 2 pages, 39.5 s). Failed calls: pay_slip 96.5 s over 3 attempts (HTTP 500), PurchaseReceipt 1.2 s (HTTP 400).

## Fix proposals, ranked by expected metadata gain

1. Try every model value for a single-valued field. File `cascade/merge.py:132`. Replace the `break` with a pass that keeps the grounded value that also has `rule_agrees`, else the first grounded one. Expected gain: +1 TP (0b93 total_amount), total_amount F1 .667 -> .750, micro F1 about +.02. Grounding stays as strict as now.
2. Fix the VLM failures. Files `cascade/vlm.py` retry path and the render settings in `config/cascade.yaml`. Retry an HTTP 400 once at lower `max_long_edge` and log the body, and treat HTTP 500 as retryable with a shorter timeout. Expected gain: both fallback docs get a model answer, up to +1 TP (pay_slip total_amount) and classification coverage .95 -> up to 1.00.
3. Collapse spaces around `-` and `/` in the invoice number normalizer. Files `baseline/metadata/normalizers.py` (`_n_upper_strip`, line 108) and the `invoice_number` entry in `config/fields.yaml`. Expected gain: +1 TP (99035). It changes both baseline and cascade, so re-measure both.
4. Fix the measurement, not the model. `phone`, `vendor_name` and `due_date` have no gold, which adds 13 false positives. Add gold for them in the next review round, or score only fields with gold. Expected effect: micro F1 about .45 -> .55 with no value changes.
5. Masked Aadhaar (`XXXX XXXX 3208`). Add a masked last-four candidate to the aadhaar rules (`baseline/metadata/fields.py`) so a masked model value can agree with a rule candidate at `cascade/merge.py:150`. Expected gain: +1 TP. It loosens the checksum gate for masked numbers, so decide that deliberately first.

Not proposed: values the OCR text lacks (cause h) stay refused by design. The fix is better OCR, not softer grounding.
