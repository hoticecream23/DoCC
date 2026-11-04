# Audit thread: system-wide improvement pass, 2026-11-04

Owner document for the `audit` thread. Plan and status log: `PLAN-audit.md` at the repo root. Abstention analysis: `docs/parts/audit-abstentions.md`.

## Result

Twelve commits on `small-fixes`, 5b54c58 through ddaa015. Nothing pushed. Suite 481 -> 511 tests, all green, ~100 s.

| Metric | Before | After | Commit |
| --- | --- | --- | --- |
| OCR, 8-page scanned PDF, workers=1 | 22.3 s | 8.1 s (2.8x), output identical | 0d7bff0 |
| Classification macro F1 (112 docs) | .664 | .690 | 63b2e59 |
| Classification coverage | .857 | .884 (precision 1.0 kept) | 63b2e59 |
| salary_slip recall | .80 | .93 | 63b2e59 |
| purchase_order recall | .50 | .67 | 63b2e59 |
| total_amount F1, SROIE text (361) | .485 | .717 | d97713b |
| total_amount F1, gold_reviewed (20) | .385 | .500 | d97713b |
| vendor_name F1, SROIE text | 0 (never emitted) | .590 (P .60 / R .58) | d1f2816 |
| invoice_number, SROIE | 179 junk reads (DATE, QTY, CASHIER read as numbers) | 219 reads; independent sample of 20: 17 correct, 3 junk removed, 0 wrong. Not harness-measurable: SROIE gold has no invoice_number | d9abee7 |
| Coverage: baseline/profile.py | 0% | 92% | 2a15440 |
| Coverage: baseline/tables/records.py | 40% | 91% | 2a15440 |
| Swallowed exceptions | 3 silent sites, 1 abort-on-bad-line | logged; tune-thresholds skips and counts bad lines; VLM retries a non-JSON body | 5b54c58 |
| Model loading | joblib.load unverified | sha256 manifest checked before every load; `baseline train` writes it | b248072 |
| VLM client | key over http silently; unbounded body | warns on key over http to non-loopback; 8 MiB body cap (`vlm.max_response_bytes`) | 6027d39, 9737dd3 |
| Dependencies | lower bounds only | `requirements.lock`, 26 pins, referenced in the client guide | 72de4df |

## Things to know

- `models/MANIFEST.sha256` is committed; `models/*.joblib` are gitignored. A client who copies or retrains models needs the matching manifest, or `baseline run` refuses to load them. `baseline train` writes it.
- The documented LAN vLLM setup (http + `DOCC_VLM_KEY`) now logs one warning per run instead of failing. Use https off the LAN.
- `requirements.lock` was produced on Python 3.14. Python 3.10 may need `tomli` and `exceptiongroup`, which are not in it.
- The vendor_name rule emits with .60 precision on receipts. On the client corpus it runs on invoice / receipt / purchase_order only; names that wrap to two lines come out truncated. The first version of its regex backtracked catastrophically (three skipped header lines of repeated "invoice" hung the matcher); it was rewritten so the skipped-line keyword match is unambiguous and the captured header is capped at 160 characters. Worst measured case is now 63 ms on a 200k-character document, with a guard test.
- `report.md` (gitignored) was overwritten by an eval run during A5. Regenerate with `baseline eval` if it is needed.
- `client_results.jsonl` and `review.xlsx` were not touched. All measurement runs wrote to the scratchpad.

## Not done (deliberately)

- Receipt / tax_form / id_document abstentions: 9 of 16 have no rule firing at all; the unverified receipt marker (audit-abstentions.md, fix 3) needs a check against the trap cases first.
- 8 SROIE receipts abstain on invoice_number where "INV NO:" is followed by "DATE :". Recall only.
- M1 (load config once in cli.py) and M3 (move cascade constants to yaml): no measurable gain, skipped.
- graph.py split, default `--workers`, 200 dpi OCR, scanned-image accuracy: see PLAN-audit.md "Skipped on purpose".
