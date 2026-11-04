# A4 audit: classification abstentions

Source: results_classification.jsonl (112 docs, 16 abstained), re-run through baseline.classify with config/ and models/classifier. Gold class below is the folder name in `source_path` (the eval report only gives aggregate mismatch counts, not per doc). Gold files were not opened. Scripts are in the scratchpad (ab.py, ab2.py, ab3.py); no repo code changed.

## Abstention checks in `classify()` (baseline/classify.py)

| # | name | line | tests | default |
|---|------|------|-------|---------|
| 1 | rule short-circuit | ~212 | best rule marker score >= bar: answer at once | 0.9 |
| 2 | model_min_confidence | ~224 | calibrated model top prob >= bar: answer; else fall through | 0.45 |
| 3 | rule_min_confidence | ~245 | best rule score >= bar after model declined: answer | 0.7 |
| 4 | final fallback | ~255 | none passed: label `unknown` | - |

baseline/rules.py has no abstention logic (marker matching only). Records carry no reason field (`diagnostics` is not used for it), so the firing check was inferred by re-running.

## Count per refusing check (16 abstentions)

- Check 2 (model) refuses all 16. Top model prob is only 0.16-0.24 on every abstained doc (8 classes, near uniform). The model never reaches 0.45 on any of them. It is not a useful gate here.
- Check 3 (rule_min) is the deciding refusal. 7 docs: one weak marker fired (0.55 or 0.6), below 0.7. 9 docs: no rule fired at all.

## Tables

Verdict key: CR = correct refusal under the frozen bars; RW = check reasoning wrong; OCR = text too poor. "gap" = no marker covers this wording (rule coverage, not a wrong check).

### receipt (recall .14, 6 fn: 5 unknown, 1 invoice)
| doc id | gold | refusing check | score | verdict |
|---|---|---|---|---|
| 67becdd9e048aae5 | receipt | rule_min | rule 0.55, model 0.197 | CR (single header "receipt" keyword); OCR conf 0.57 |
| 267b2c031a078207 | receipt | rule_min | rule 0.55, model 0.18 | CR; OCR conf 0.55 |
| a1514daaf4eeb5df | receipt | rule_min | rule 0.55, model 0.177 | CR |
| 2cb1bb6a83da685a | receipt | rule_min (no rule) | no rule, model 0.174 | gap: no receipt word at all (no "receipt" keyword, no "total") |
| 7167bb348212cb74 | receipt | rule_min (no rule) | no rule, model 0.181 | gap; only "total" present |

(The 6th fn, receipt -> invoice, is a wrong answer, not an abstention. Out of scope.)

### tax_form (recall .20, 4 fn: 2 unknown, 2 invoice)
| doc id | gold | refusing check | score | verdict |
|---|---|---|---|---|
| 7cd32a637cc6ee8f | tax_form | rule_min (no rule) | no rule, model 0.21 | RW candidate: text contains "income tax"/"assessment year"/"previous year" wording and "gross"/"deduction", yet no marker fired. Needs a look at which tax_form marker was expected (OCR conf 0.94, so text is good) |
| e87b0d3787721da2 | tax_form | rule_min (no rule) | no rule, model 0.163 | OCR (conf 0.42; none of the tax terms present) |

### id_document (recall .25, 3 fn, all unknown)
| doc id | gold | refusing check | score | verdict |
|---|---|---|---|---|
| c66b1e523ff5e844 | id_document | rule_min | rule 0.6, model top is contract 0.216 | CR (lone weak keyword; aadhaar / "date of birth" class); OCR 0.61 |
| b8aa62b47b7049a7 | id_document | rule_min | rule 0.6, model id_document 0.226 | CR; OCR 0.40 |
| 83304e91bf6833a4 | id_document | rule_min (no rule) | no rule, model 0.16 | OCR (100 chars, conf 0.27) |

Note: HANDOFF forbids an ID route. The two 0.6 docs are lowered by design (classes.yaml:95).

### purchase_order (recall .50, 3 fn: 1 unknown, 1 contract, 1 invoice)
| doc id | gold | refusing check | score | verdict |
|---|---|---|---|---|
| e07a7318117427cb | purchase_order | rule_min (no rule) | no rule, model 0.206 (top is contract) | RW candidate: "purchase order" appears at char 59 (the title) but is followed by "No/#/." so the negative lookahead at classes.yaml:80 and :81 blocks both strong markers. Vendor/ship-to/delivery-date is absent within 4000 chars. Also contains the word "invoice" |

### other weak classes (not asked, same cause)
| doc id | gold | refusing check | score | verdict |
|---|---|---|---|---|
| 7ab54a38dc45f7b9, a694ff0fe30431fa, 01ee710d660e7f8d | salary_slip | rule_min (no rule) | model 0.18-0.24 | RW candidate / gap: all contain "net salary" and "gross"/"deduction"; none contain "net pay" or "salary slip". classes.yaml:69 needs literal "net pay" |
| c8ba653f89fa5fe9 | invoice | rule_min | rule 0.6, model 0.167 | CR |
| 3630e310da96e958 | invoice | rule_min (no rule) | no rule, model 0.186 | gap: 219 chars, OCR 0.98, no "invoice" word |

## Totals
- 16 abstentions: 7 weak-rule (rule 0.55-0.6 < 0.7), 9 no rule fired.
- Correct refusals under frozen bars: 7 weak-rule docs (strictly), plus 2 OCR (e87b.., 8330..).
- Judged wrong or likely wrong reasoning: 5 (3 salary slips, 1 purchase order, 1 tax doc). Others are marker coverage gaps, which is a config question, not a bar change.
- Caveat: verdicts for the "no rule" docs come from regex presence checks on the stored OCR text, not from reading the text. Confirm each before editing.

## Fix proposals (reasoning, not bars)
1. salary_slip marker (classes.yaml:69): accept "net salary" as an alternative to "net pay" in the `all` slot, keeping the `any: gross/deductions/...` requirement. Gain: salary_slip +3 tp, recall .80 -> 1.00 (12/15 to 15/15). Risk: low, still two terms required.
2. purchase_order lookahead (classes.yaml:80-81): the "No/#/." exclusion exists to ignore "purchase order number" references on invoices. Allow it when the phrase is within the first ~100 chars (the title line) and there is no "invoice"-class strong marker. Gain: +1 tp, PO recall .50 -> .67. Risk: medium; check it does not turn invoices into POs on the 112 set (precision is now 1.0 for PO).
3. tax_form (7cd32a637cc6ee8f): find why "assessment year" / "previous year" (marker at classes.yaml:109, 0.75) did not fire, since both phrases look present. Likely whitespace or OCR spelling not matched by the keyword rule; or the marker needs `any` plus `all`. Gain: +1 tp, recall .20 -> .40. Verify first.
4. Receipt corroboration (cannot be done without a bar change if it just lifts 0.55): instead add a real receipt marker for amount-paid/cash/change/total-with-tax-lines, so the doc earns its own >= 0.7. Gain: up to +3 (the 3 docs with a 0.55 header hit), recall .14 -> .57. Needs a sample of what those receipts say; unverified.
5. Model gate: the model is near uniform (max 0.24), so the 0.45 bar cannot matter. Retraining is forbidden before gold exists, so no action. Record this as the reason the rule layer decides everything.

Combined best case if 1-4 all hold: +8 tp of 16 abstentions; overall accuracy .81 -> about .88. Do not lower rule_min (0.7) or model_min (0.45).
