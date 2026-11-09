# Data schemas

Where each data shape is defined, who writes it, who reads it, and a trimmed real example.
Line numbers are on branch `small-fixes`. Examples come from files at the repo root; some of
those files are older than the code (noted per section).

## 1. DocumentRecord (one line of results.jsonl)

- Definition: `baseline/schema.py:142` (`DocumentRecord`), parts at lines 70-140.
- Version: `SCHEMA_VERSION = "1.3.0"` (`schema.py:18`). The field `schema_version` carries it. 1.3.0 added `cabinet_id`.
- Base class `Strict` (`schema.py:28`): `extra="forbid"`, `validate_assignment=True`.
- Writer: `baseline/pipeline.py:101` `process_document`, serialised by `dumps` (`pipeline.py:185`), batch writer `run_batch`. Cascade writes the same shape (`cascade/runner.py`).
- Readers: `baseline/evaluate` (as plain dicts), `baseline/graph.py`, `baseline/tables/records.py`, `baseline/workbook/exporter.py`, `scripts/cv_classifier.py`, the demo (`record.model_dump`).
- Constants: `PAGE_SEP = "\n\f\n"` (line 21, joins pages, all offsets count against it); `DOC_ID_LEN = 16` (line 23); `Confidence` = float in [0,1].
- `verify_offsets()` (line 160): for each metadata item, `text.full[char_start:char_end]` must equal `value`. Returns a list of problems; empty means clean.
- `read_jsonl` (line 345) reads one record per line. It iterates the handle, not `splitlines()`, because PAGE_SEP holds a form feed.

DocumentRecord

| field | type | constraint |
| --- | --- | --- |
| doc_id | str | length exactly 16, hex. First 16 hex of SHA-256 of file bytes (`compute_doc_id`) |
| source_path | str | path as scanned |
| filename | str | kept apart from source_path |
| schema_version | str | default `SCHEMA_VERSION` |
| cabinet_id | str | default `"default"` (`DEFAULT_CABINET`); set from `cabinet_id` in `pipeline.yaml` or `run --cabinet`. Lines written before 1.3.0 have no key and load as `"default"` |
| text | TextRecord | |
| classification | Classification | |
| metadata | list[MetadataField] | default empty |
| tags | list[Tag] | default empty |
| diagnostics | Diagnostics | |

TextRecord (`schema.py:70`)

| field | type | constraint |
| --- | --- | --- |
| full | str | must equal `PAGE_SEP.join(pages)` (`check_consistent`) |
| pages | list[str] | no page may contain PAGE_SEP |
| method | ExtractionMethod | enum below |
| char_count | int | >= 0, must equal `len(full)` |
| extraction_confidence | float | 0..1 |
| page_methods | list[ExtractionMethod] | default empty, one per page |

Classification (`schema.py:97`)

| field | type | constraint |
| --- | --- | --- |
| label | str | class name, or the unknown label |
| confidence | float | 0..1 |
| all_scores | dict[str, float] | each 0..1, default empty |
| method | ClassifyMethod | enum below |

MetadataField (`schema.py:104`)

| field | type | constraint |
| --- | --- | --- |
| field | str | field name from `config/fields.yaml` |
| value | str | raw text, must equal the slice at the offsets |
| normalized_value | str or null | required key, may be null |
| page | int | >= 0 |
| char_start | int | >= 0 |
| char_end | int | >= 0, not before char_start |
| confidence | float | 0..1 |
| method | MetadataMethod | enum below |
| validated | bool | true only when a checksum or format check passed |
| normalizer_version | str | default "1.0.0" |
| currency | str or null | amount fields only |

Tag (`schema.py:128`): `tag: str`, `confidence: float 0..1`, `method: TagMethod`.

Diagnostics (`schema.py:134`)

| field | type | constraint |
| --- | --- | --- |
| page_count | int | >= 0 |
| is_scanned | bool | |
| has_native_text | bool | |
| timings_ms | dict[str, float] | default empty; cascade adds `cascade_render`, `cascade_vlm`, `cascade_decide` |
| errors | list[str] | default empty |

Enums

| enum | line | values |
| --- | --- | --- |
| ExtractionMethod | 32 | native, ocr, hybrid, none, ocr_rescued (1.1.0) |
| ClassifyMethod | 44 | tfidf_svm, rule, fallback, vlm (1.2.0) |
| MetadataMethod | 52 | regex_checksum, regex_format, anchor, positional, vlm, vlm_rule_agree (last two 1.2.0) |
| TagMethod | 64 | keyword, tfidf_svm, derived |

Example (`results.jsonl`, first record, text cut; this file carries `schema_version` 1.0.0, older than the code)

```json
{"doc_id": "dadc997a9cee4ceb", "source_path": "corpus/bank_statement.pdf", "filename": "bank_statement.pdf",
 "schema_version": "1.0.0",
 "text": {"full": "STATEMENT OF ACCOUNT\nBank: HDFC Bank...", "pages": ["STATEMENT OF ACCOUNT..."], "method": "native",
          "char_count": 453, "extraction_confidence": 0.98, "page_methods": ["native"]},
 "classification": {"label": "bank_statement", "confidence": 0.94, "all_scores": {"bank_statement": 0.94}, "method": "rule"},
 "metadata": [{"field": "account_number", "value": "502010123456789", "normalized_value": "502010123456789",
               "page": 0, "char_start": 53, "char_end": 68, "confidence": 0.95, "method": "anchor",
               "validated": false, "normalizer_version": "1.0.0", "currency": null}],
 "tags": [{"tag": "bank_details_present", "confidence": 0.97, "method": "derived"}],
 "diagnostics": {"page_count": 1, "is_scanned": false, "has_native_text": true,
                 "timings_ms": {"extract": 168.204, "classify": 5.724}, "errors": []}}
```

## 2. Word-box sidecar (`<output>.bboxes/<doc_id>.json.gz`)

- Definition: `baseline/schema.py:179` `WordBox`, `schema.py:197` `BBoxSidecar` (both `Strict`). No separate version field.
- Writer: `baseline/pipeline.py:211` `write_sidecar`. One gzip JSON file per document, named `<doc_id>.json.gz`, mtime pinned to 0 so bytes are reproducible. Nothing is written when `words` is empty. Directory defaults to `<output>.bboxes` (`pipeline.py:262`; cascade `runner.py:160`).
- Readers: `baseline/tables/records.py:217` (geometry tables), `baseline/metadata/fields.py` (positional fields), cascade merge (`decide` receives `sidecar.words` and `page_sizes`).

BBoxSidecar

| field | type | note |
| --- | --- | --- |
| doc_id | str | joins to DocumentRecord |
| page_sizes | list[(float, float)] | per page, width and height in points |
| words | list[WordBox] | |

WordBox

| field | type | note |
| --- | --- | --- |
| page | int | >= 0 |
| text | str | |
| x0, y0, x1, y1 | float | page coordinates |
| conf | float or null | OCR confidence; null for native text |
| char_start, char_end | int or null | offset in canonical text when mappable |

Example (`results.jsonl.bboxes/023336296a5f0e8a.json.gz`, 32 words, first one shown)

```json
{"doc_id": "023336296a5f0e8a", "page_sizes": [[595.0, 842.0]],
 "words": [{"page": 0, "text": "PAYMENT", "x0": 142.0, "y0": 124.0, "x1": 235.0, "y1": 140.0,
            "conf": 0.96, "char_start": 0, "char_end": 7}]}
```

## 3. Cascade provenance sidecar (`<output>.provenance`)

- Definition: no pydantic class. It is a plain dict built in `cascade/runner.py:87` (`prov`), carried in `Result.provenance` (`runner.py:34`). Class decision part: `cascade/merge.py:50-67` (`decide_class`). Field decision part: `cascade/merge.py:123-221` (`decide_fields`). No version.
- Writer: `cascade/runner.py` `run_batch` (about lines 150-190). One JSON object per line in `<output>.provenance`, written in step with `results.jsonl` (same open mode, so resume appends to both).
- Readers: the demo (`demo/server.py:121` copies a subset). `cascade/cli.py` documents the file. No eval code reads it.
- `raw` is the unparsed model reply text (`reply.content`). It is `null` until a VLM reply arrives, and stays `null` when the call failed.
- No provenance file exists in the repo, so the example below is built from the code, not copied from a run.

Top-level keys

| key | type | meaning |
| --- | --- | --- |
| doc_id | str | same as the record |
| filename | str | same as the record |
| model | str or null | model name the backend reported |
| finish_reason | str or null | VLM finish reason |
| pages_sent | int | text pages sent (after retry reduction) |
| images_sent | int | page images sent |
| reduced_pages | int | present only after a context-overflow retry; the page budget used |
| fallback | bool | true when cascade failed and the baseline answer was kept |
| errors | list[str] | fallback message plus dropped-offset problems |
| raw | str or null | raw model reply |
| class | dict or null | class decision, below; null on fallback |
| fields | list[dict] | per-field decisions, below; empty on fallback |
| timings_ms | dict | `cascade_render`, `cascade_vlm`, `cascade_decide` (0.0 when deterministic timings are on) |

`class` keys: `source` ("rule" or "vlm"), `label`, `vlm_label`, `disagrees` (bool; true only when a rule answer was kept and the model gave a different non-unknown label).

`fields` items, one per model value tried: `field`, `vlm_value`, `outcome`, `n_matches`, `rule_agrees`, `char_start`, `char_end`, `emitted`.
`outcome` is the grounding result (`exact`, `loose`, `normalised`, `ungrounded`; `cascade/ground.py:28-31`) or is overwritten by `checksum_failed` or `shape_failed`.
A second item kind has `outcome: "rule_only"`, `baseline_values` (list of str) and `emitted: false`. It means the baseline had values and the model gave none that were kept. Out-of-scope fields produce no item.

Example (constructed from the code)

```json
{"doc_id": "dadc997a9cee4ceb", "filename": "bank_statement.pdf", "model": "<model>", "finish_reason": "stop",
 "pages_sent": 1, "images_sent": 1, "fallback": false, "errors": [], "raw": "{\"label\": \"bank_statement\", ...}",
 "class": {"source": "vlm", "label": "bank_statement", "vlm_label": "bank_statement", "disagrees": false},
 "fields": [{"field": "account_number", "vlm_value": "502010123456789", "outcome": "exact", "n_matches": 1,
             "rule_agrees": true, "char_start": 53, "char_end": 68, "emitted": true}],
 "timings_ms": {"cascade_render": 40.1, "cascade_vlm": 5200.3, "cascade_decide": 3.2}}
```

## 4. Gold formats the harness reads

- Loader: `baseline/evaluate/scoring.py:15` `load_jsonl` (one JSON object per line; blank lines skipped; bad JSON raises `ValueError` with file and line). Join: `scoring.py:39` `join`.
- Join keys: `KEYS = ("doc_id", "source_path", "path")` (`scoring.py:9`). Each key value is tried in full and as its basename. Match order is doc_id, then path, then filename. Unmatched gold rows go to `unmatched_gold`; unmatched predictions to `unscored_predictions`.
- Writers: `baseline workbook import` (importer, section 7), `baseline/labels.py` (client labels), hand-made files.
- The shape below is taken from what the scorers read, not from any gold file.

| gold key | used by | shape |
| --- | --- | --- |
| doc_id / source_path / path | `join` | any one; doc_id preferred |
| text | `eval_extraction` (`scoring.py:111`) | string, or dict with `full`. Rows without it are skipped |
| label | `eval_classification` (`:168`), `eval_calibration` (`:375`) | class name string, compared after strip |
| metadata | `eval_metadata` (`:225`), calibration, `covered_fields` (`:419`) | list of items: `field` (required), `normalized_value` or `value` (normalized preferred; `_norm_of`), optional `char_start` and `char_end` (both needed for span scoring) |
| tags | `eval_tagging` (`:316`) | list of tag-name strings (a set is built from it). Predictions use dicts with `tag` |
| tables | `eval_tables` (`table_scores.py:73`) | list of tables; each table has `cells`, a list of rows, each row a list of cell strings. `None` or missing skips the doc. Matched against the tables sidecar |

Example (shape only, built from the loader code)

```json
{"doc_id": "<16 hex>", "label": "invoice",
 "metadata": [{"field": "total_amount", "normalized_value": "11800.00"}],
 "tags": ["contains_pii"],
 "tables": [{"cells": [["Item", "Qty"], ["Widget", "2"]]}]}
```

## 5. Eval outputs

- Definition: no pydantic class. Built by `baseline/evaluate/report.py:23` `evaluate` and returned as a dict; the CLI writes it with `--json` (`baseline/cli.py:298`, `indent=2, sort_keys=True`). The markdown is `render_report` (`report.py:56`). Comparison: `compare` (`report.py:287`), `render_comparison` (`report.py:352`). No version.
- Readers: humans; `report.py` renders sections from the dict.

Top-level keys of the scores JSON

| key | source | content |
| --- | --- | --- |
| gold_count, pred_count, matched | `evaluate` | row counts |
| unmatched_gold, unscored_predictions | `evaluate` | first 20 keys each |
| extraction | `eval_extraction` | `scored`, `cer`, `wer`; only `{"scored": 0}` when no gold text |
| classification | `eval_classification` | `n`, `answered`, `abstained`, `coverage`, `accuracy_overall`, `accuracy_when_answered`, `macro_f1`, `per_label` (precision, recall, f1, tp, fp, fn), `confusion` (top 10, key "truth -> got"), `coverage_at_{1,5,10}pct_error`, `threshold_at_{1,5,10}pct_error` |
| metadata | `eval_metadata` | `value` and `span`, each with `per_field`, `micro`, `macro_f1`; `span_scored_docs`; `validated_precision`; `validated_wrong`; `errors` (list of strings) |
| tagging | `eval_tagging` | `per_tag`, `micro`, `macro_f1` |
| tables | `eval_tables` | `scored`, `gold_tables`, `pred_tables`, `detected_tables`, `detection_recall`, `cells` and `adjacency` (precision, recall, f1, tp, fp, fn) |
| calibration | `eval_calibration` (`scoring.py:375`) | `fields.overall`, `fields.per_field.<name>`, `classification`; each is a `bucket_stats` dict |
| covered | `eval_covered` (`scoring.py:424`) | `fields` (sorted names the gold labels), `micro`, `macro_f1`, `unscored_predictions` (int), `unscored_per_field` |
| calibration_covered | `eval_covered` | same shape as `calibration.fields` (`overall`, `per_field`), gold-covered fields only |

`bucket_stats` (`scoring.py:352`): `buckets` keyed `0.5-0.7`, `0.7-0.9`, `0.9-1.0` (`BUCKETS`, line 349), each `{n, precision (accuracy for classification), mean_confidence}`; `n` (items in a bucket); `below_0.5` (count under 0.5); `ece` (null when `n` is 0).

Markdown report sections, in order (`render_report`): `# Evaluation report`, `## Extraction`, `## Classification`, `## Metadata` (with `### Misses`), `## Gold-covered fields`, `## Calibration` (`### Field values`, `### Field values, gold-covered fields only`, `### Classification (answered documents)`), `## Tagging`, `## Tables`. Covered, Calibration and Tables print only when their data is present. Comparison report: `# Comparison`, `## Per field`, `## Per tag`, `## Regressions`.

Example (`scores.json` at the root, trimmed). This file predates the new keys: it has no `calibration`, `covered` or `calibration_covered`. Their shape comes from `scoring.py`.

```json
{"gold_count": 10, "pred_count": 10, "matched": 10, "unmatched_gold": [], "unscored_predictions": [],
 "extraction": {"scored": 1, "cer": 0.0079, "wer": 0.0625},
 "classification": {"n": 10, "answered": 9, "abstained": 1, "coverage": 0.9, "accuracy_overall": 0.9,
                    "accuracy_when_answered": 1.0, "macro_f1": 1.0, "coverage_at_5pct_error": 0.9},
 "metadata": {"span": {"note": "gold has no offsets", "per_field": {}}, "span_scored_docs": 0, "validated_precision": 1.0},
 "tables": {"scored": 10, "gold_tables": 5, "pred_tables": 4, "detected_tables": 4, "detection_recall": 0.8,
            "cells": {"precision": 1.0, "recall": 0.8033, "f1": 0.8909}}}
```

Shape of the new keys (from code)

```json
{"calibration": {"fields": {"overall": {"buckets": {"0.9-1.0": {"n": 0, "precision": 0.0, "mean_confidence": 0.0}},
                                         "n": 0, "below_0.5": 0, "ece": null},
                            "per_field": {}},
                 "classification": {"buckets": {}, "n": 0, "below_0.5": 0, "ece": null}},
 "covered": {"fields": [], "micro": {}, "macro_f1": 0.0, "unscored_predictions": 0, "unscored_per_field": {}},
 "calibration_covered": {"overall": {}, "per_field": {}}}
```

## 6. Tables output and graph output

### 6a. Tables sidecar (`tables.jsonl`)

- Definition: `baseline/schema.py:308` `TableRecord`, with `Table` (278), `TableCell` (254), `TableDiagnostics` (299), `TableMethod` (245: geometry, text_grid, tab_delimited).
- Version: `TABLES_SCHEMA_VERSION = "1.1.0"` (`schema.py:242`). It is separate from the record version.
- Writer: `baseline/tables/records.py:207` `build` (calls `build_record`, line 51); one `model_dump` per line.
- Readers: `evaluate` (`tables_path`), workbook exporter (Line items tab), the tables report (`tables.md`).
- Joins to DocumentRecord on `doc_id`. `TableRecord.verify_offsets(full)` (line 320) checks placed cells slice back to their text.

TableRecord

| field | type | constraint |
| --- | --- | --- |
| doc_id | str | length 16 |
| source_path, filename | str | copied from the record |
| schema_version | str | default "1.1.0" |
| tables | list[Table] | |
| diagnostics | TableDiagnostics | `abstained` (list of reasons), `strategies_tried`, `timings_ms`, `errors` |

Table: `table_id` (`<doc_id>-p<page>-t<n>`), `page` >= 0, `n_rows`, `n_cols` >= 0, `method`, `confidence` 0..1 (ordering signal only), `header` (list[str], empty if none found), `cells`, `char_start`/`char_end` (null unless every cell placed). Property `fill` = cells / (rows x cols).

TableCell: `row`, `col` >= 0; `row_span`, `col_span` >= 1 (always 1 today); `text`; `page`; `char_start`/`char_end` (int or null); `x0 y0 x1 y1` (float or null); `is_header` (bool).

Example (`tables.jsonl`, first record, one cell; the file carries `schema_version` 1.0.0, older than the code)

```json
{"doc_id": "dadc997a9cee4ceb", "source_path": "corpus/bank_statement.pdf", "filename": "bank_statement.pdf",
 "schema_version": "1.0.0",
 "tables": [{"table_id": "dadc997a9cee4ceb-p0-t0", "page": 0, "n_rows": 5, "n_cols": 5, "method": "text_grid",
             "confidence": 0.76, "header": ["Date", "Description", "Debit", "Credit", "Balance"],
             "cells": [{"row": 0, "col": 0, "row_span": 1, "col_span": 1, "text": "Date", "page": 0,
                        "char_start": 120, "char_end": 124, "x0": 56.0, "y0": 157.43, "x1": 77.12, "y1": 171.17,
                        "is_header": true}]}]}
```

### 6b. Graph (`graph.json`)

- Definition: plain dataclasses `Node` (`baseline/graph.py:29`) and `Edge` (`:36`), container `Graph` (`:43`), serialised by `Graph.to_dict`. No pydantic class and no version field.
- Writer: `baseline/graph.py:524` `build`: `{"nodes", "edges", "stats"}`, `indent=2, sort_keys=True`. Also optional `graph.dot` and `graph.md`.
- Reader: reports only (`to_dot`, `render_report`).
- Node: `id`, `kind`, plus kind-specific attributes flattened into the object. Edge: `src`, `dst`, `type`, plus flattened attributes. Node ids are prefixed: `doc:<doc_id>`, `org:<key>`, `acct:<ifsc>:<number>`. When `add_node` repeats an id, lists union and scalars keep the first value.
- Kinds seen in `graph.json`: `document` (`filename`, `label`, `method`, `page_count`), `organization` (`pan`, others), `account` (`number`, `ifsc`).
- Edge types seen: `mentions_org`, `issued_by`, `pays_to`, `held_by`, `references`, `duplicate_of`.
- `stats`: `documents`, `organizations`, `accounts`, `edges`, `edges_by_type`, `duplicate_groups` (list of `{documents, key}`), `unverified_ids_ignored`.

Example (`graph.json`, trimmed)

```json
{"nodes": [{"id": "acct:HDFC0001234:502010123456789", "kind": "account", "ifsc": "HDFC0001234", "number": "502010123456789"},
           {"id": "doc:019f595f4e9783a4", "kind": "document", "filename": "id_card.txt", "label": "id_document",
            "method": "native", "page_count": 1}],
 "edges": [{"src": "acct:HDFC0001234:502010123456789", "dst": "org:AAPFU0939F", "type": "held_by"}],
 "stats": {"documents": 10, "organizations": 2, "accounts": 2, "edges": 21,
           "edges_by_type": {"held_by": 2, "pays_to": 6}, "unverified_ids_ignored": 0}}
```

## 7. Review workbook (`review.xlsx`)

- Definition: column lists in `baseline/workbook/columns.py` (`DOCUMENT_COLUMNS`, `METADATA_COLUMNS`, `TAG_COLUMNS`, `TAXONOMY_COLUMNS`, `LINE_ITEM_COLUMNS`, `REVIEW_COLUMNS`, `VERDICTS`, `TRUTHY`). No version field in the file.
- Writer: `baseline/workbook/exporter.py:101` `export_workbook`. Refuses to overwrite an existing file unless `overwrite`. `merge_from` carries an earlier review (`merge.py`). Strings are forced to string type so `=...` OCR text is not a formula.
- Readers: `baseline/workbook/importer.py:140` `import_workbook`, and `merge.py`. Both read by header name (upper-cased), not by position. `DOC_ID` is carried down from the first row of each group.
- Reviewer columns are left blank on export. `IS_GOLD` is never pre-filled.
- Sheets, in order: `Documents`, `Metadata`, `Tags`, `Taxonomy`, `Line items`.
- Verdict spellings (`VERDICTS`): ok/correct become `ok`; wrong/incorrect become `wrong`; `spurious`; `missing`. Truthy `IS_GOLD`: true, yes, y, 1.

Columns

| sheet | columns |
| --- | --- |
| Documents | DOC_ID, RUN_ID, EXTRACTED_AT, SCHEMA_VERSION, SOURCE_PATH, FILE_NAME, FILE_TYPE, PAGE_COUNT, EXTRACTION_METHOD, PAGE_METHOD, HAS_NATIVE_TEXT, IS_SCANNED, EXTRACTION_CONFIDENCE, CHAR_COUNT, TEXT_SHA256, TEXT_PREVIEW (200 chars), DOC_CLASS, CLASS_CONFIDENCE, CLASS_METHOD, CLASS_RUNNER_UP, RUNNER_UP_CONFIDENCE, ERROR_COUNT, ERRORS, LANGUAGE, QUALITY, then reviewer: CLASS_CORRECTED, CLASS_VERDICT, IS_GOLD, REVIEWED_BY, REVIEW_DATE, NOTES |
| Metadata | FIELD_ROW_ID, DOC_ID, ROW_SOURCE, FIELD, VALUE, NORMALIZED_VALUE, CURRENCY, PAGE, CHAR_START, CHAR_END, CONFIDENCE, METHOD, VALIDATED, NORMALIZER_VERSION, then reviewer: CORRECTED_VALUE, VERDICT, IS_GOLD, REVIEWED_BY, REVIEW_DATE, NOTES, then export-only REVIEW_PRIORITY |
| Tags | DOC_ID, TAG, ROW_SOURCE, CONFIDENCE, METHOD, then reviewer: VERDICT, IS_GOLD, REVIEWED_BY, REVIEW_DATE |
| Taxonomy | KIND (class, field, tag), NAME, DESCRIPTION, VALUE_TYPE, FORMAT, APPLIES_TO_CLASSES. Generated from config, never typed |
| Line items | DOC_ID, TABLE_ID, PAGE, TABLE_METHOD, N_ROWS, N_COL, ROW, COL, CELL_TEXT, CHAR_START, CHAR_END, IS_HEADER (one row per table cell) |

`ROW_SOURCE` is `pipeline` for exported rows and `human` for rows a reviewer added (`merge.py:229`).

REVIEW_PRIORITY (`exporter.py:62` column name, `:66` `review_priority`): value `HIGH`, `MEDIUM` or `LOW`. HIGH when confidence < 0.7, or not validated and the method is not `regex_checksum` or `regex_format`. LOW when confidence >= 0.9 and validated. Otherwise MEDIUM. The importer and merge ignore it.

What import produces: a gold JSONL (section 4 shape), one row per document the reviewer gave a class. Row keys: `doc_id`, `label`, `metadata` (list of `{field, normalized_value}` plus `page`/`char_start`/`char_end` for `ok` rows only), `tags` (list of strings), plus `path` and `filename` when resolved. Rules: `ok` keeps the pipeline value; `wrong` and `missing` take the corrected value (run through the field normalizer); `spurious` produces nothing. For tags, `wrong` and `spurious` drop the tag; `ok` and `missing` add it. A row needs a truthy `IS_GOLD` when that column exists. Unknown class, field or tag names are reported in `WorkbookImport.problems` (`importer.py:60`), not guessed. Import gives precision-honest gold but no recall for items the pipeline never emitted.

Example (shape of one Metadata row; no real row is quoted from `review.xlsx`)

```text
FIELD_ROW_ID=<doc_id>_1, DOC_ID=<doc_id>, ROW_SOURCE=pipeline, FIELD=total_amount, VALUE=11,800.00,
NORMALIZED_VALUE=11800.00, CONFIDENCE=0.95, VALIDATED=FALSE, ..., VERDICT=<blank>, IS_GOLD=<blank>, REVIEW_PRIORITY=HIGH
```

## 8. Demo API (`demo/server.py`)

- Definition: no schema class; dicts built in `make_server` (`demo/server.py:95`). No version. The server binds 127.0.0.1 and checks `Host` and (for POST) `Origin`. Errors return `{"error": "<message>"}` with 400, 403, 404, 413 (file over 25 MB), 502, 503 or 500.
- Reader: `demo/index.html`.

Endpoints

| method and path | response |
| --- | --- |
| GET `/` | `demo/index.html` |
| GET `/api/status` | `{"cascade": {"base_url": str, "model": str, "reachable": bool}}` |
| GET `/api/corpus` | sorted list of file names in `corpus/` |
| POST `/api/run` | see below |

POST `/api/run` request (query string; the body is the file bytes for an upload)

| param | meaning |
| --- | --- |
| mode | `baseline` (default) or `cascade`; anything else is 400 |
| corpus | name of a file in `corpus/` (no body) |
| filename | upload name; the body holds the file (`Content-Length` required, max 25 MB) |

One of `corpus` or `filename` is required.

Response (200)

| key | type | note |
| --- | --- | --- |
| mode | str | echoes the mode |
| record | object | a DocumentRecord (section 1), `model_dump(mode="json")` |
| pages | list | page previews from `render_pages` (`:62`): `{index, png_base64, width, height}`; PDFs up to 20 pages at 110 dpi, images one page, other types an empty list |
| total_ms | float | wall time of the run |
| provenance | object | cascade mode only: `model`, `finish_reason`, `pages_sent`, `images_sent`, `fallback`, `errors`, `timings_ms`. The `raw`, `class` and `fields` keys are not sent |

Example (cascade mode, trimmed; shape from the code)

```json
{"mode": "cascade", "record": {"doc_id": "...", "classification": {"method": "vlm"}, "...": "..."},
 "pages": [{"index": 0, "png_base64": "...", "width": 909, "height": 1286}],
 "total_ms": 5300.0,
 "provenance": {"model": "<model>", "finish_reason": "stop", "pages_sent": 1, "images_sent": 1,
                "fallback": false, "errors": [], "timings_ms": {"cascade_vlm": 5200.3}}}
```

## 9. Training manifest and training-set record

### 9a. `models/MANIFEST.sha256`

- Format: `sha256sum` text. One line per model file: 64 hex digest, two spaces, path relative to `models/` with forward slashes. A leading `*` on the name is accepted when reading.
- Writer: `baseline/classify.py:34` `record_hash` (called after `joblib.dump`; entries sorted by name). Reader: `classify.py:27` `_read_manifest`, `classify.py:55` `load_verified`. A missing entry or a digest mismatch raises `ModelIntegrityError`; the model is not loaded. Test: `tests/test_model_manifest.py`.
- `models/<kind>/meta.json` sits beside each model. Classifier keys (`classify.py:161`): `kind`, `labels`, `class_counts`, `n_samples`, `cv_folds`.

```text
524303345792f7be7ab84cc4d5f3c1ff90b5962c41b62c7473f1ca3d9d636a7e  classifier/model.joblib
aaa7a5620436288a0b296674a6b143e463333ee73e82ff4c7c2adcf06f53fe8c  tagger/model.joblib
```

### 9b. Training-set record (`baseline train --annotations`)

- Reader: `baseline/cli.py:46` `_read_annotations` (JSONL, sorted by `doc_id`, `path` or `text` for reproducibility); `cmd_train` at `cli.py:81`; text from `_text_for` (`cli.py:53`).
- Row shape: `{path | text, label?, tags?}`. `doc_id` is also used for sorting and for `tune-thresholds`.

| key | type | use |
| --- | --- | --- |
| text | str | inline canonical text; used first when present |
| path | str | file to extract when `text` is absent; a row with neither raises `ValueError` |
| doc_id | str | optional; sort and join key |
| label | str | classifier target; rows without it are skipped for the classifier |
| tags | list[str] | tagger targets; rows without it are skipped for the tagger |

```json
{"path": "corpus/invoice_native.pdf", "label": "invoice", "tags": ["contains_pii"]}
```

### 9c. `scripts/cv_classifier.py` input

- Reader: `scripts/cv_classifier.py:32` `load_docs`. Input is a results JSONL (section 1 shape), not an annotation file. It uses `source_path`, `doc_id`, `text.full` and `text.pages` (strings, or dicts with `text`).
- The label is the parent folder name of `source_path` (second to last path part), mapped through `config/taxonomy.yaml`. Folders with no mapping are dropped and counted. Duplicate `doc_id`s are skipped.
- Output goes to the `OUT_DIR` argument.

## Which is authoritative

1. DocumentRecord and all its parts, enums, `PAGE_SEP`, `DOC_ID_LEN`, `SCHEMA_VERSION`: `baseline/schema.py` (the pydantic class wins over any doc or older results file).
2. Word boxes: `WordBox` and `BBoxSidecar` in `baseline/schema.py`; file layout from `write_sidecar` in `baseline/pipeline.py:211`.
3. Tables sidecar: `TableRecord` in `baseline/schema.py`; `TABLES_SCHEMA_VERSION` there.
4. Cascade provenance: the code in `cascade/runner.py:87` and `cascade/merge.py` (no class, no version).
5. Gold and eval input: the loader code in `baseline/evaluate/scoring.py` and `table_scores.py` (no class).
6. Scores JSON and report: `evaluate` and `render_report` in `baseline/evaluate/report.py`.
7. Graph: `Graph.to_dict` in `baseline/graph.py` (no class, no version).
8. Workbook: `baseline/workbook/columns.py` for columns; `exporter.py` and `importer.py` for behaviour.
9. Demo API: `demo/server.py`.
10. Manifest and training rows: `baseline/classify.py` and `baseline/cli.py`.

Known differences between definitions and files (facts only)

- Root `results.jsonl` and `tables.jsonl` carry `schema_version` 1.0.0; the code writes 1.2.0 and 1.1.0.
- Root `scores.json` has no `calibration`, `covered` or `calibration_covered`.
- `load_jsonl` (`scoring.py:15`) and `read_jsonl` (`schema.py:345`) are two copies of the same reader.
- Tags are dicts in a record (`Tag`) but plain strings in gold and training rows.
- Tables in gold are lists of rows of strings; in the sidecar they are `Table` objects with `TableCell` items.
- Line items `IS_HEADER` (`exporter.py`) is computed as "table has a header and row is 0"; `TableCell.is_header` is stored on the cell.

## SQLite store (`docc.sqlite`)

- Definition: `baseline/db.py` (`SCHEMA_DDL`). Stdlib `sqlite3`, WAL mode, foreign keys on. Portable SQL only: TEXT, INTEGER and REAL columns, JSON kept as TEXT, composite keys instead of autoincrement, upserts as `INSERT ... ON CONFLICT ... DO UPDATE`. The DDL should port to PostgreSQL unchanged.
- Location: `db.path` in `pipeline.yaml`, default `docc.sqlite`, resolved beside the results file. `null` turns the store off.
- Rule: the JSONL is the log and the source of truth. The pipeline writes and flushes each results line first, then upserts the record (`run_batch` in `baseline/pipeline.py`). A store error is logged and the batch goes on. `db-import` rebuilds or repairs the store from the JSONL at any time, and is idempotent: importing the same file twice gives the same row counts.

| table | key | main columns |
| --- | --- | --- |
| documents | doc_id | cabinet_id, source_path, filename, doc_type (classification label), confidence, classify_method, schema_version, page_count, run_id, bbox_path, bbox_sha256, record_json (the full line), created_at, updated_at |
| fields | (doc_id, ord) | one row per MetadataField: name, value, normalized_value, page, char_start, char_end, method, validated (0/1), normalizer_version, confidence, currency |
| tags | (doc_id, ord) | tag, confidence, method |
| tables | (doc_id, table_id) | page, n_rows, n_cols, method, confidence, table_json, sidecar_path (the tables JSONL) |
| provenance | doc_id | model, finish_reason, fallback (0/1), abstention_reason (joined `errors` when fallback), prov_json (the whole cascade line, raw reply included) |
| review_events | event_id | doc_id, field, old_value, new_value, source, created_at. Created empty; the workbook import will write it |
| runs | run_id | started_at, config_hash (sha256 of every loaded YAML), schema_version, cabinet_id |
| models | name | version (null: the manifest has none), sha256 from `models/MANIFEST.sha256`, recorded_at |

- Child tables join to `documents` on doc_id; `documents.cabinet_id` scopes them. `doc_id` is already the content hash, so there is no separate hash column.
- `upsert_document` replaces a document's `fields` and `tags` rows in one transaction. It does not touch `tables`, `provenance` or `review_events`.
- Word boxes stay on disk in `<results>.bboxes/`; the store holds their path and sha256.
- Export rebuilds the JSONL from `record_json`, ordered by doc_id.
- Public API: `connect`, `upsert_document`, `import_jsonl` (returns `ImportStats`), `export_jsonl`, `list_documents`, `start_run`, `record_models`.

Commands (both take `--dry-run`)

```text
python -m baseline db-import results.jsonl [--db PATH] [--tables tables.jsonl] [--provenance results.jsonl.provenance]
python -m baseline db-export out.jsonl [--db PATH] [--cabinet ID]
```
