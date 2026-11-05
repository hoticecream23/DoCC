# Where every schema lives

Index only. Field tables, examples and loaders are in `docs/parts/schemas-data.md` (what the system writes and reads) and `docs/parts/schemas-config.md` (what you can configure, and the schema sent to the model). Updated 2026-11-05.

## The one contract

`DocumentRecord`, `baseline/schema.py:142`, schema version 1.2.0. Pydantic, strict. Both `baseline run` and `cascade run` write one of these per document as JSONL. The eval harness scores it without importing the pipeline. Every metadata value carries `page`, `char_start`, `char_end` into `text.full`, and `verify_offsets()` proves it. `PAGE_SEP` and the record are frozen.

## Data schemas

| What | Authoritative definition | Version | Written by | Read by |
| --- | --- | --- | --- | --- |
| DocumentRecord (text, classification, metadata, tags, diagnostics) | baseline/schema.py:142 | 1.2.0 | pipeline.py, cascade/runner.py | eval, workbook, graph, tables, demo |
| Word boxes sidecar `.bboxes` | schema.py:179 (WordBox), :197 (BBoxSidecar); writer pipeline.py:211 | with record | pipeline | tables, cascade render |
| Cascade provenance sidecar | cascade/runner.py:87 (plain dict, incl. `raw`) | none | cascade runner | demo (drops `raw`), audits |
| Gold rows (fields, classification, tables) | baseline/evaluate/scoring.py:15 loader | none | humans, scripts/convert_sroie.py, workbook import | eval |
| Eval scores JSON and report | baseline/evaluate/report.py:23; keys `calibration`, `covered`, `calibration_covered` added 2026-11-05 | additive | `baseline eval` | people, docs |
| TableRecord (`tables.jsonl`) | schema.py:308 | 1.1.0 | `baseline tables` | eval |
| Graph (`graph.json`) | baseline/graph.py:29-43 | none | `baseline graph` | nothing downstream |
| Review workbook sheets and columns, incl. REVIEW_PRIORITY | baseline/workbook/columns.py, exporter.py:101 | none | `export-workbook` | reviewers, `import-workbook` (produces gold rows) |
| Demo API (`/api/status`, `/api/run`) | demo/server.py:95 | none | demo server | demo/index.html |
| Model manifest `models/MANIFEST.sha256` | baseline/classify.py:27-55 (sha256sum format) | none | `baseline train` | every model load |
| Training rows | baseline/cli.py:46; scripts/cv_classifier.py | none | annotations / folder labels | `baseline train` |

## Config schemas

| File | Loader and validation | Notes |
| --- | --- | --- |
| config/pipeline.yaml | baseline/config.py:116 `load_config`, plain dict | section is `classification`, thresholds model_min .45, rule_min .7, rule_short_circuit .9 |
| config/taxonomy.yaml | same, optional | class ids and folder mapping |
| config/classes.yaml | same, dict | rule markers: kind, pattern, scope, specificity, gates. Regexes must compile on Python 3.10 and run linear |
| config/fields.yaml | same, dict | per-field strategies: anchor (labels, window, direction, value_regex, reject_before/after/value), pattern; normalizers and validators by name |
| config/tags.yaml, tables.yaml, graph.yaml | same | |
| config/cascade.yaml | cascade/config.py:63, pydantic strict, extra keys refused | vlm (incl. retry_statuses, max_response_bytes, extra_body), render, confidence block |
| JSON schema sent to the VLM | cascade/prompt.py:73 `response_schema`, generated from fields.yaml and taxonomy | 18 classes, 22 fields, maxItems 1, per-field negatives |
| demo/Modelfile, .claude/launch.json | Ollama and the in-app browser | |

## Known gaps (from the maps)

- Every baseline config file is a plain dict: a typo in a key is silently ignored or falls back to a default. Only cascade.yaml refuses unknown keys. An unknown marker `kind` never fires; an unknown `scope` acts as `anywhere`; an unknown normalizer or validator only logs a warning.
- `runtime.workers` and `version` in pipeline.yaml are never read.
- `load_jsonl` (scoring.py:15) and `read_jsonl` (schema.py:345) are duplicates.
- Repo-root `results.jsonl` and `tables.jsonl` are older versions (1.0.0) and `scores.json` predates the calibration keys; they are artefacts, not the contract.
