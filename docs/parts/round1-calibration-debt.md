# Round 1: confidence calibration and maintainability debt

Date 2026-11-05. Branch small-fixes. No repo code changed. Scratch output: scratchpad\r1cal\.

## 1. Confidence calibration

### How confidences are assigned
- Metadata fields (baseline/metadata/fields.py:57-76): hand-set per field in config/fields.yaml as
  `confidence_validated` (1.0 x5, 0.95 x2, 0.9, 0.7, 0.6) and `confidence_unvalidated`
  (0.3 to 0.9). Anchor templates default to 0.85 (fields.py:228). Label-based hits are scored from
  the label `specificity` (fields.py:114-161).
- Classification (baseline/classify.py): rule score = best marker `specificity` in
  config/classes.yaml. 87 markers; values 0.6 to 0.96, mostly 0.85 to 0.96. These are author guesses.
  Model fallback uses its probability (classify.py:230,259).
- Cascade (cascade/merge.py:153-175, config/cascade.yaml:41): fixed tiers (regex_checksum 1.0,
  agree, grounded, ambiguous, vlm_class). Constants per provenance, not per rule.
- Tags (baseline/tagging.py:112): per-rule constant, default 0.9.

### Is anything downstream using them?
Barely.
- Decision use: classify.py:204-205 (`rule_min_confidence` 0.7, `model_min_confidence` 0.45) gates
  rule vs model vs unknown. Cascade `rule_short_circuit` 0.9 gates VLM skip (merge.py:53).
- Tag condition `min_confidence`: supported (tagging.py:79-82) but used by exactly one rule
  (config/tags.yaml:100, floor 0.7).
- Field sort: fields.py:238 uses confidence to break ties after `validated`.
- Display only: workbook exporter (exporter.py:165-198) writes the numbers to columns. The graph
  does not read confidence.
- Nothing abstains, routes to review, or thresholds on field confidence in the exporter or graph.

### Measured calibration
The harness cannot bucket by predicted confidence for fields. Eval report and JSON give only:
- precision of fields marked `validated`;
- classification `coverage_at_{1,5,10}pct_error` (scoring.py:121-165), the only confidence-aware
  metric.
There is no per-doc match output (the report prints a truncated "Misses" list only), and gold files
may not be opened. So per-field per-bucket precision was NOT measured. Stopped at this step as
instructed. (Note: `eval --json` wrote no file on the SROIE run; not investigated.)

Data points that exist (from `python -m baseline eval`, sroie_after.jsonl vs gold_text):

| Source | Predicted | Observed |
| --- | --- | --- |
| SROIE test, 361 docs, fields marked validated | 1.0 for checksum fields | 0.983 precision (2 wrong). A checksum "1.0" is not exact. |
| gold_reviewed, 20 docs (older run a2/rev_after.md) | rules 0.85+ | accuracy when answered 1.00, coverage 0.70, validated precision 1.000. Only 20 docs. |
| SROIE classification (receipts, no matching class in taxonomy) | 322 of 361 answered with rule confidence | accuracy when answered 0.037; coverage_at_5pct_error 0.000 at threshold 1.000. Out of domain, but confidence did not flag it. |

Reading: confidence is ordinal at best. Values like 0.93 vs 0.94 carry no information. The SROIE
classification row shows high confidence with wrong answers when the taxonomy lacks the true class.
(That run also shows invoice_number 219 fp and vendor_name 361 fn; these are field or mapping gaps,
not calibration.)

### Research (one paragraph)
Standard post-hoc options are Platt scaling (a 1-D logistic fit of correctness on the score) and
isotonic regression (monotone piecewise-constant), both fit on a held-out set. Studies report mixed
winners: isotonic gives the best top-class calibration with enough data, Platt wins on log loss and
is more stable on small sets; isotonic can overfit and pile high confidence onto wrong answers when
the calibration set is small, and about 50 examples already gives noticeable gains. For a
rule-based extractor the simplest fit is the empirical precision per rule (or per field and
provenance tier) with Beta or Wilson shrinkage toward a prior, and Platt over that for rules with
few matches. One search ran; it found no source that treats per-rule precision directly, so this is
an inference. Sources seen: arXiv 2411.16742 (Text-to-SQL calibration, rescale), arXiv 2509.23665
(Calibration Meets Reality), arXiv 2102.10809 (Local calibration).

## 2. Maintainability debt

### PLAN-audit Tier 3 items, current state
- M1: baseline/cli.py (512 lines) calls `load_config_cached(args.config)` in 9 places (lines 77,
  112, 204, 228, 253, 279, 322, 342, 363). cascade/cli.py:27 does the same. Tune helper
  cmd_tune_thresholds is at line 110 and calls tagging.tune_thresholds. cli.py coverage is 50%.
- M3: cascade/vlm.py:26 `_RETRY_STATUS = {429,500,502,503,504}` is hard-coded. cascade/merge.py:53
  `rule_short_circuit` default 0.9 is hard-coded in code, not in config/pipeline.yaml (which owns
  rule_min_confidence). Two YAML loaders: baseline/config.py:23 and cascade/config.py:65.

### Ten slowest tests (suite: 527 passed, 127 s)
1. 14.04s test_pipeline.py::test_parallel_output_is_byte_identical_to_serial
2. 11.30s test_cascade_cli.py::test_run_against_a_local_server_writes_records_and_provenance
3. 9.73s test_ocr.py::test_batch_with_ocr_is_byte_identical_across_worker_counts
4. 9.71s test_pipeline.py::test_default_config_is_deterministic_apart_from_timings
5. 6.34s test_pipeline.py::test_same_input_gives_byte_identical_output
6. 5.48s test_pipeline.py::test_no_resume_reprocesses_everything
7. 3.27s test_cascade_runner.py::test_batch_writes_one_record_and_one_provenance_line_per_document
8. 3.20s test_pipeline.py::test_batch_completes_despite_a_broken_file
9. 3.14s test_pipeline.py::test_resume_skips_documents_already_written
10. 3.07s test_pipeline.py::test_timings_are_recorded_by_default

Together about 69 s of 127 s. Seven are in test_pipeline.py and run the real pipeline.

### Coverage (total 82%); modules under 60%
- baseline/evaluate/report.py 16% (renders the eval report)
- baseline/labels.py 27% (import-labels)
- baseline/cli.py 50% (several subcommands untested)
- baseline/__main__.py, cascade/__main__.py, demo/__main__.py 0% (3-4 line stubs, ignore)
Next lowest (70-76%): tagging.py 70%, classify.py 72% (model fallback 93-170), metadata/fields.py 75%, graph.py 76%.

### TODO / FIXME
None in .py or .yaml.

### Ponytail ledger (8 entries)
| Location | Defers |
| --- | --- |
| baseline/extract.py:170 | Scan-quality thresholds fit one batch (Yaumi 36-45); retune per batch. |
| baseline/extract.py:207 | Fixed thread pool per document; with --workers N it is N x threads. |
| config/fields.yaml:109 | Positional pairing on flattened text; use bbox columns if layouts vary. |
| config/fields.yaml:284 | One header line only; a name that wraps to line 2 is missed. |
| config/fields.yaml:320 | Title patterns are indiankanoon only. |
| config/fields.yaml:380 | Supreme Court and High Courts only; tribunals and lower courts not covered. |
| demo/server.py:80 | First frame only for multi-page TIFF. |
| demo/server.py:202 | VLMError branch rarely fires since the runner falls back. |

## 3. Ranked proposals

1. Add confidence buckets to the eval: per-field per-bucket precision plus ECE, and an optional
   per-doc details output. Benefit: first real calibration measurement; unblocks the rest.
   Effort: small (~60 lines + test). Risk: low. Files: baseline/evaluate/scoring.py, report.py,
   tests/test_evaluate.py. Also lifts report.py coverage from 16%.
2. Replace hand-set confidences with empirical per-rule precision from gold (a `calibrate` command
   that writes confidence_* into fields.yaml, Wilson-shrunk; Platt only for rules with few
   matches). Benefit: numbers become meaningful, thresholds become tunable. Effort: medium.
   Risk: medium (overfit to SROIE; tiny reviewed set; needs a held-out split). Depends on 1.
   Files: baseline/metadata/fields.py, config/fields.yaml, config/classes.yaml, baseline/cli.py.
3. Make confidence do something: flag fields under a floor in the workbook for review, and add
   `min_confidence` to more tag rules. Benefit: reviewer time goes to weak fields. Effort: small.
   Risk: low, but only worth doing after 2. Files: baseline/workbook/exporter.py, config/tags.yaml.
4. M1/M3 cleanup: one `_cfg(args)` helper in cli.py, retry statuses and `rule_short_circuit` into
   config, one shared YAML loader. Benefit: removes hard-codes, shrinks cli.py. Effort: small.
   Risk: low (merge.py and vlm.py have 98-99% coverage). Files: baseline/cli.py, cascade/vlm.py,
   cascade/merge.py, cascade/config.py, baseline/config.py.
5. Cut test time: run the pipeline once in a module-scoped fixture and reuse the output across the
   pipeline tests. Benefit: about 30-40 s per run. Effort: small. Risk: low if determinism tests
   still run each config separately. Files: tests/test_pipeline.py, tests/test_ocr.py.
