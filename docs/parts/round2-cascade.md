# Round 2: cascade v2 accuracy levers (read-only audit)

Date: 2026-11-05. Sources: k1 run (cascade.jsonl, provenance); k34 diff (no eval report there, so k1 reports used). Gold was not opened, so "FP" below means "emitted value not in baseline"; true FP status is unverified.

## 1. Decision points

- **Prompt** (cascade/prompt.py:15-26, 73-82). One generic system prompt: copy values character for character, prefer the page-text form, empty list when absent. Fields are listed as `name (applies to: classes): description` with no examples and no negative cases. Descriptions come from config/fields.yaml (total_amount "The final amount payable, including tax"; account_number "A bank account number").
- **Schema** (prompt.py:55-70). Class is an enum; every field is a required `array<string>`. No per-field pattern, no maxItems, no quote/evidence key. `parse_answer` (110-145) is strict: an extra, missing or undeclared field rejects the whole answer.
- **Grounding** (cascade/ground.py:178-202). Three steps, first match wins: exact (125), loose = whitespace collapse + casefold (135), normalised = field normalizer equal on value vs `scan_spans`/rule candidates (152). Boundary checks at 100-114 (no start/end inside a longer word or joined number; one page only). No fuzzy tolerance and no confidence floor. Rule agreement is checked at 196-199.
- **Merge** (cascade/merge.py:85-196). Checksummed fields: proven baseline values kept, a model value emitted only if its span equals a REGEX_CHECKSUM candidate (150-165). Single-value fields: model values sorted by rule_agrees, grounding rank, n_matches (139, K3). Confidence: agree (162), else grounded/ambiguous by n_matches (165-166). A rule value the model did not return is dropped as `rule_only` (190-192).
- **Calibration** (config/cascade.yaml:41-45). agree .9, grounded .75, ambiguous .5, vlm_class .8. Unmeasured. Nothing here thresholds on them, so changing them does not change emitted values.

## 2. "Model omitted 5": correction

In PLAN-cascade and cascade-pass.md the 5 "(c) model omitted" are not FNs. They are baseline values dropped by `rule_only`, and none was a TP. The real FN causes (7) are in Table B of cascade-pass.md. Raw-output audit of the 5 omissions (fields and values only):

| doc | field | model raw | result | class |
| --- | --- | --- | --- | --- |
| c66b Aadhar | invoice_date | [] | rule 1983-08-10 (a birth date) dropped | (a) never mentioned; correct, a gain |
| b8aa passport | invoice_date | [] | dropped | (a), gain |
| 8330 voter ID | invoice_date | [] | dropped (doc unscored) | (a), gain |
| 0829 equipment PO | total_amount | [""] (empty string) | rule 1.00 dropped | (a) empty value |
| 4d58 Equipment_po | total_amount, invoice_number, invoice_date | total `$0.00`; other two [] | `$0.00` ungrounded; rule 30.00, PO-001234, 00-00-20 dropped | total: (c) refused, not in OCR text (OCR may have read `$0.00` as `30.00`, so the model may be right); other two (a) |

No case of (b) wrong field name (the schema forbids it) or (d) truncation (finish_reason stop everywhere).

Real model-side FNs among the 7:
- 0b93 total_amount: model returned ['15,31,194.00', '5,18,692.00']; the correct second value was never tried in k1 (fixed by K3).
- 2cb1 total_amount: `$378.00`; "378" is not in the OCR text (subtotal $350.00 is). Grounding refused.
- 94af total_amount: `$4,044.60`; "4,044" is not in the OCR text. Refused.
- c66b aadhaar: `XXXX XXXX 3208` grounded exact, then `checksum_failed` (merge.py:158).
- 2cb1 buyer_name: model `John Doe`, grounded exact; gold casing differs. Harness item.

## 3. Extra FPs on covered fields (cascade-new values vs baseline, k1)

Every one is grounded `exact` with `rule_agrees=False`. Zero are ungrounded hallucinations: the model returns values that are printed on the page.

| doc | field | cascade value | baseline | class |
| --- | --- | --- | --- | --- |
| 99035 NG | account_number | 02080402 (printed `02-08-0402`, date-like) | none | wrong field |
| 54ca, c4be Everon | account_number | 40044448 | none | wrong field (likely customer/reference id) |
| 01ee SalarySlip | account_number | 20610714105832 | none | plausible; gold may omit |
| c4be, f4db | buyer_name | 1199 SEIU ... | none | plausible; role or gold sparsity |
| 7c39 SIDBI | buyer_name | SMALL INDUSTRIES DEVELOPMENT BANK OF INDIA | none | plausible |
| 4d58, 94af | buyer_name | XYZ Company, Sarah Johnson | none | template placeholder, right as printed |
| f4db | invoice_number | 32101 | none | plausible |
| 2cb1 | invoice_number | 2025-1124-01 | none | plausible |
| 94af | invoice_number | 1234567 | none | template placeholder |
| 7c39 SIDBI | invoice_number | DIV/24102025/OUT/59885 (k1); DIV/27092022/OUT/16743, rule-agreeing (k34) | 2018MAR16/L00197407 | model replaced rule value; the PO prints 5 numbers |
| 99035 NG | tax_amount | 19.75 | none | plausible (5% of 395) |
| 7c39 SIDBI | tax_amount | 194398.00 | none | likely wrong (line item, not total tax) |
| f4db, 4ca3 payslip | total_amount | 199.00, 96436.00 | none | plausible; payslip may be gross not net |
| 2cb1 | invoice_date | 2025-11-24 | none | plausible |

Counts: hallucinated 0; wrong field 3 (all account_number); right value wrong normalisation 0 in k1 (NG invoice_number `02 - 140305` fixed by K5, now `02-140305` in k34); rule value overridden by model 1 (7c39 invoice_number); the other ~9 are model-only reads of fields the baseline rules missed, plausibly correct but unlabelled or labelled differently (unverifiable without gold).
Main point: the FP rise is mostly real extra reads, not grounding failures. A blunt precision gate would cut TPs too.

## 4. Proposals, ranked (expected TP gain minus FP risk)

1. **Per-field prompt definitions with examples and negatives** (prompt.py:73-82, config/fields.yaml descriptions). account_number: "bank account, 9-18 digits, labelled Account or A/c; not a date, phone, customer or reference id". total_amount: "net payable incl. tax; for payslips the net pay, not gross". invoice_number: "the one number that labels this document; ignore numbers of other documents". buyer_name: "billed-to party, not the sender". Expected: about 3 account_number FPs and 1-2 total/tax FPs removed, 0 TP lost. Measure: re-run the 20 docs (~12 min), compare covered-view F1 per field. Risk: low; could suppress the plausible 01ee account number.
2. **Shape gate on model-only values** (merge.py:164-166). When `rule_agrees` is False for account_number and invoice_number, require the field's baseline pattern (digit length, not a date shape); otherwise drop or keep at `ambiguous`. Deterministic, model independent. Expected: the same 3 account FPs. Risk: medium for formats outside the pattern. Measure: 20 docs plus the SROIE image route (219 unscored invoice_number reads give a regression check).
3. **maxItems 1 in the schema for single-valued fields, and "return an empty list, never an empty string"** (prompt.py:63). Removes the 0829 `[""]` case and list-of-two ambiguity. Expected 0-1 TP, FP neutral. Risk: loses the 0b93 second-value rescue if K3 is not in place; with K3 it is a no-op there. Measure: 20 docs.
4. **Fuzzy grounding with a strict offset check** (ground.py, new step after normalised). No help for the measured misses: the 2 "OCR lacks value" cases have no such digits in the text ("378" and "4,044" absent). Expected 0 TP on the 20 docs. Risk: new FP route, and PLAN-cascade says do not soften grounding. Defer. If tried: rapidfuzz ratio >= .9, same page, length within 1, digits identical.
5. **Never let an ungrounded model value beat a validated rule value.** Already true (merge.py:117-121 keeps checksummed values; an ungrounded value never replaces anything). No change.
6. **Calibration numbers** (cascade.yaml:41-45). No effect on emitted values until a downstream threshold exists. Skip.
7. **Masked Aadhaar** (merge.py:150-151). +1 TP but it loosens the checksum gate. Leave as a deliberate decision.

## 5. Research (search snippets only; pages not read)

Constrained decoding (vLLM `json_schema` with xgrammar, NIM structured generation) guarantees output shape, while `json_object` only guarantees valid JSON. Cascade already sends a strict schema, so the lever left is content, not shape ([ExtractBench](https://arxiv.org/pdf/2607.29677), [NVIDIA NIM structured generation](https://docs.nvidia.com/nim/vision-language-models/1.7.0/structured-generation.html)). VLMs localise poorly (best IoU 14.4% on ExStrucTiny), so tools keep coordinates away from the model ([ExStrucTiny review](https://www.themoonlight.io/review/exstructiny-a-benchmark-for-schema-variable-structured-information-extraction-from-document-images)). The closest open pipeline, [paperpin](https://pypi.org/project/paperpin/), matches exact, then normalised, then fuzzy, with per-field-type rules, and returns `not_found` rather than inventing a box. That is the ground.py ladder minus the fuzzy step. [extracture](https://pypi.org/project/extracture/) uses a rapidfuzz sliding window and checks a source quote. [llmground](https://pypi.org/project/llmground/) marks fuzzy matches as weak, not grounded. All treat fuzzy hits as a review flag.
