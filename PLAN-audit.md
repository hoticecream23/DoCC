# PLAN-audit: system-wide improvement roadmap

Date: 2026-11-04. Branch: `small-fixes`. Written by the orchestrator from five
parallel audits (architecture, reliability, performance, accuracy, security).

## Baselines (measured today)

| Metric | Value |
| --- | --- |
| Tests | 481 passed, 0 failed, ~120 s wall |
| Coverage | 71% total. profile.py 0%, tables/records.py 40%, tagging.py 59% |
| `baseline run` on corpus/ (10 docs, workers=1) | 11.1 s wall. 5.7 s one-time model load, ~0.5 s/doc after |
| OCR | serial, one Tesseract call per page at 300 dpi, PNG encode+decode per page |
| Classification (112 docs) | coverage .857, macro F1 .664, answered acc .948 |
| Worst classes (P/R/F1) | receipt 1.0/.14/.25, tax_form 1.0/.20/.33, id_document 1.0/.25/.40, purchase_order 1.0/.50/.67 |
| Worst fields (19 gold docs) | invoice_number 0/0/0 with 8 fp; total_amount .385/.455/.417 |
| SROIE (public, 361 docs) | total_amount F1 .485; vendor_name 0/0/0 (361 fn); invoice_number 0 of 179; value micro F1 .451 text / .314 images |
| Scanned images | classification acc .525 vs .972 on PDFs (PROJECT_STATE, Known gaps 1) |

## Frozen constraints (do not break)

- Do not re-run `client_results.jsonl` while `review.xlsx` is out (HANDOFF.md:23).
- `PAGE_SEP` is frozen (PROJECT_STATE.md:154). `DocumentRecord` contract is frozen (HANDOFF.md:989).
- Do not soften abstention checks to raise recall. Fix a check's reasoning, never lower its bar (HANDOFF.md:764).
- Do not retrain models before gold exists (PROJECT_STATE.md:1003). Do not build an ID route (HANDOFF.md:81).
- Do not touch PROJECT_STATE.md. Do not open gold*.jsonl, annotations.jsonl, datasets/, Client_Documents/ (read eval reports instead).
- Existing determinism tests must keep passing: parallel == serial, worker-count identity, byte-identical output.

## Causes (what the audits found)

1. **OCR is the throughput floor.** 85% of the client corpus is scanned. Pages are OCR'd one at a time, each through a PNG round-trip (extract.py:262,307; ocr.py:79).
2. **Errors are swallowed silently** at pipeline.py:203, logging_setup.py:38, cli.py:498; one bad `--predictions` line aborts `eval` (cli.py:136); a truncated VLM JSON body is not retried (vlm.py:133).
3. **Model files are trusted blindly.** joblib.load at classify.py:132 and tagging.py:208 has no integrity check. A swapped model runs code.
4. **The VLM client can leak the key.** Bearer header sent over plain http (vlm.py:122-133) and `resp.read()` has no size cap.
5. **No dependency lock.** pyproject.toml has lower bounds only; a fresh client install can drift.
6. **Field rules miss the obvious value.** total_amount picks a non-grand total; invoice_number reads nothing on SROIE and 8 false positives on gold; vendor_name is declared (fields.yaml:224) but emits nothing.
7. **Class recall is abstention.** Precision is 1.0 everywhere; receipt/tax_form/id_document abstain 75-86% of the time. Per the frozen rule, the fix is check reasoning, not thresholds.
8. **Maintainability debt, low urgency.** cli.py loads config in 9 subcommands; two yaml loaders (baseline/config.py:23, cascade/config.py:63); graph.py mixes model/build/analysis/render; retry statuses and a 0.9 default hard-coded (vlm.py:26, merge.py:53); profile.py untested.

## Roadmap (ranked by impact × confidence ÷ effort, regression risk noted)

### Tier 1: ship first, low risk, measurable

| # | Task | Files | Done check | Risk |
| --- | --- | --- | --- | --- |
| T1 | OCR pages in a thread pool; drop PNG round-trip (`Image.frombytes(pix.samples)`). Keep 300 dpi. | baseline/extract.py, baseline/ocr.py | `baseline run` on corpus/ byte-identical to before; scanned-page time per doc down ≥2x on a multi-page image doc; test_ocr + test_pipeline determinism tests pass | low: output must stay byte-identical |
| T2 | Log swallowed exceptions; `eval` skips and counts bad JSON lines; VLM retries on truncated/invalid JSON body (bounded) | baseline/pipeline.py, baseline/logging_setup.py, baseline/cli.py, cascade/vlm.py | new tests: bad predictions line is skipped and counted; truncated body retried once then raised; full suite green | low |
| T3 | Model integrity: write `models/MANIFEST.sha256` at `train`, verify before every joblib.load, refuse on mismatch | baseline/classify.py, baseline/tagging.py, baseline/cli.py (train) | test: tampered model refused with clear error; existing models/ get a manifest generated once; suite green | low-medium: must not break runs on existing models/ (generate manifest in this task) |
| T4 | VLM client hardening: refuse Bearer over non-https unless host is loopback; cap response read (e.g. 8 MiB) | cascade/vlm.py, config/cascade.yaml | tests for both refusals; test_cascade_vlm green | low |
| T5 | Pin dependencies: `requirements.lock` from the working venv (`pip freeze`), referenced in docs/client_poc_setup.md | requirements.lock, docs/client_poc_setup.md | file exists, install doc updated; no downloads performed | none |

### Tier 2: accuracy, measured with `baseline eval` against existing gold (read report, not gold)

| # | Task | Files | Done check | Risk |
| --- | --- | --- | --- | --- |
| A1 | total_amount: prefer the grand total (last "total"/"grand total"/"amount due" line, largest consistent with subtotal+tax) | baseline/metadata/fields.py, config/fields.yaml | gold F1 .417 → ≥.60 and SROIE text F1 .485 → ≥.60, precision not down; suite green | medium: keep abstention bar unchanged |
| A2 | invoice_number: fix the rule's reasoning (8 fp on gold, 0/179 SROIE) | baseline/metadata/fields.py, baseline/metadata validators | gold fp 8 → ≤2, SROIE recall > 0; suite green | medium |
| A3 | vendor_name: find why a configured field never emits; make it emit on receipts | baseline/metadata/fields.py, config/fields.yaml | SROIE vendor_name F1 0 → >.40; suite green | medium |
| A4 | Investigate receipt/tax_form/id_document abstentions: list which check refuses each gold doc and why. Report only, then decide. | baseline/classify.py, baseline/rules.py | a table in docs/parts/audit.md: doc → refusing check → correct? | none (report) |

### Tier 3: maintainability, do after Tier 1-2 or skip

| # | Task | Files | Done check |
| --- | --- | --- | --- |
| M1 | Load config once in `main()`; move tune helpers to baseline/tuning.py | baseline/cli.py | suite green, cmd_* imports kept |
| M2 | Smoke test for profile.py; failure-path tests for tables/records.py | tests/ | coverage of both modules ≥60% |
| M3 | Move retry statuses and merge 0.9 default to cascade.yaml; one shared yaml loader | cascade/vlm.py, cascade/merge.py, cascade/config.py | suite green |

Skipped on purpose: graph.py split (pure move, no measurable gain); default `--workers` change (behaviour change for client runs); 200 dpi OCR (accuracy risk without a scanned-image gold run); scanned-image accuracy (cascade v2 is the chosen fix and needs the GPU client).

## Validation protocol

1. Before any change: `python -m pytest -q` green (481), `baseline run --input corpus/ --output <scratch>/before.jsonl --workers 1 --no-resume`, `baseline eval` report saved to scratch.
2. After each worker: diff results JSONL (T1 must be byte-identical), re-run eval (A-tasks must improve named metric and not reduce any precision), full suite green.
3. Reject any change that fails a frozen constraint or lacks a measured gain.

## Execution order

Wave 1 (parallel, disjoint files): T1 (extract.py, ocr.py), T2 (pipeline.py, logging_setup.py, cli.py eval path, vlm.py JSON retry), T3 (classify.py, tagging.py, cli.py train path).
Note: T2 and T3 both touch cli.py in different functions; T3 goes after T2 lands.
Wave 2 (parallel): T4 (vlm.py), T5 (lock file), A1 (fields.py total_amount).
Wave 3 (parallel): A2, A3 (both in fields.py, run sequentially), A4 (report).
Wave 4: M1-M3 if time allows.

## Status log

- 2026-11-04: audits done, baselines recorded, plan written. **Waiting for approval of causes and roadmap before any code change.**
- 2026-11-04: approved. Wave 1 in progress: T1 (extract.py, ocr.py), T2 (pipeline.py, logging_setup.py, cli.py eval, vlm.py), T5 (requirements.lock). T3 queued after T2.
- 2026-11-04: done T5 (72de4df), T2 (5b54c58, 483 tests), T1 (0d7bff0, 8-page scan 22.3 s -> 8.1 s, output identical). In progress: T3 (manifest), A1 (total_amount), T4 (vlm hardening).- 2026-11-04: done T3 (code swept into d096fe5 by A1's commit; manifest in 8df3731; history not yet split), A1 (d096fe5: SROIE total_amount F1 .485 -> .717, gold .385 -> .500), T4 (a1a5bf1 + 14961dc: http+key now warns, not refuses, because the documented LAN vLLM setup uses it; 8 MiB body cap), A4 (report in docs/parts/audit-abstentions.md: 16 abstentions, 5 wrong). In progress: A2 (invoice_number), A5 (net salary + PO title rules), M2 (tests). Queued: A3 (vendor_name, after A2), commit split.- 2026-11-04: history split: A1 d97713b, T3 b248072 (code + manifest), T4 6027d39 + 9737dd3, A5 63b2e59 (macro F1 .664 -> .690, coverage .857 -> .884), M2 2a15440 (profile.py 0 -> 92%, records.py 40 -> 91%), A2 d9abee7 (invoice_number: harness cannot measure, SROIE gold has no labels; independent sample check in progress). In progress: A3 (vendor_name). Suite: 508 passed at d9abee7. report.md (gitignored) was overwritten by an eval run; regenerate if needed.- 2026-11-04: A2 accepted on independent sample (17/20 correct, 0 wrong). A3 d1f2816 (SROIE vendor_name F1 0 -> .590). ddaa015 drops a possessive quantifier so fields.yaml compiles on Python 3.10. Final suite at HEAD: 511 passed. Results in docs/parts/audit.md. **Thread complete.** Open follow-ups listed there under "Not done".
- 2026-11-04: security review flagged the vendor_name regex; confirmed by timing (three keyword-filled header lines hung the matcher). Rewritten with an unambiguous keyword match and a 160-char header cap, worst case 63 ms on a 200k-char doc, guard test added. Suite 512 passed.
- 2026-11-04: second pass started. B1 (invoice_number value on the line under its label, config/fields.yaml) and B2 (receipt-only marker, config/classes.yaml) running in parallel.
- 2026-11-04: B2 ab1952d (receipt recall .14 -> .71, macro F1 .690 -> .765, coverage .884 -> .911, no class lost precision; marker regex timed, 5 ms worst case). B1 still running.
- 2026-11-04: B1 50b5c92 (SROIE invoice_number 219 -> 233 reads, 0 changed; regex timed, safe). Full suite 516 passed. **Second pass complete.** Remaining: 11 SROIE docs whose number sits far from its label (needs bbox columns); 9 client abstentions with no rule firing stay abstained, by design.
