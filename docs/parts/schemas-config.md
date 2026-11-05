# Config schemas and the VLM response schema

All YAML lives in `config/`. `baseline/config.py` loads it. Only `cascade.yaml` is validated by a schema (pydantic). Every other file is a plain dict and nothing checks its keys at load time.

Loader entry points (all in `C:\Users\advay\Desktop\dev\DoCC\baseline\config.py`):
- `read_yaml` (line 19): one `yaml.safe_load`; an empty file gives `{}`.
- `_load` (line 26): raises `FileNotFoundError` if absent, `ValueError` if the top level is not a mapping.
- `Config` (line 36): frozen dataclass of dicts. `load_config` (line 116) builds it. Required: pipeline, classes, fields, tags. Optional (`_load_optional`, line 111; missing gives `{}`): graph, tables, taxonomy.
- `load_config_cached` (line 132): per-process cache for workers.

Rule stated in the module docstring: no class, field, tag or pattern name appears in code. It all comes from YAML.

## config/pipeline.yaml

Loader: `Config.pipeline`, `baseline/config.py:116` (dict, unvalidated). Read through `extract_opts` (line 84, section `extraction`), `classify_opts` (line 87, section `classification`), `tagging_opts` (line 90), `metadata_opts` (line 93). The section is named `classification`, not `classify`. `runtime` is read directly in `baseline/pipeline.py:107` and `:259`, and `baseline/cli.py:376`.

| Key | Type / default (in file) | Meaning | Read at |
|---|---|---|---|
| `version` | 1 | Not read by code. | none |
| `extraction.native_char_threshold` | int, 100 | A page with fewer native chars is treated as scanned. | `baseline/extract.py:266` |
| `extraction.strip_before_count` | bool, true | Ignore whitespace when judging that threshold. | `extract.py:274` |
| `extraction.ocr.engine` | str, `tesseract` | Key in the OCR engine registry. | `extract.py:112` |
| `extraction.ocr.binary` | str or null | Absolute path to tesseract; null means PATH. | OCR engine |
| `extraction.ocr.lang` | str, `eng` | Tesseract language. | OCR engine |
| `extraction.ocr.dpi` | int, 300 | Render DPI for scanned PDF pages. | `extract.py:272` |
| `extraction.ocr.psm` | int, 3 | Tesseract page segmentation mode. | OCR engine |
| `extraction.ocr.timeout_s` | int, 120 | Per-call timeout. | OCR engine |
| `extraction.ocr.threads` | int, NOT in the file | OCR threads per document. Code default `min(4, cpu_count)` (`OCR_THREADS`, `extract.py:241`). Add the key to override. | `extract.py:273` |
| `extraction.ocr.min_short_side` | int, 1000 | Upscale image files so the short side reaches this. 0 disables. | `extract.py:168` |
| `extraction.ocr.pink_paper.{min_red_over_green,max_green_blue_gap,psm}` | 34, 10, 6 | Red-channel OCR rule for pink paper images. Remove the block to turn off. | `extract.py:193` |
| `extraction.quality_gate.{enabled,threshold,min_tokens}` | true, 0.02, 40 | Re-OCR pages whose native text layer looks like bad substituted OCR. `enabled` defaults to false if absent. | `extract.py:358-363` |
| `extraction.native_confidence` | float, 0.98 | Confidence given to native text. | `extract.py:269` |
| `extraction.capture_bboxes` | bool, true | Keep word boxes (needed by tables and positional fields). | `extract.py:268` |
| `classification.rule_short_circuit` | float, 0.9 | A rule at or above this wins and skips the model. | `baseline/classify.py:203` |
| `classification.model_min_confidence` | float, 0.45 | Model top score below this is discarded (falls to weak rule). | `classify.py:205` |
| `classification.rule_min_confidence` | float, 0.7 | Weak rule bar, used after the model declines or is absent. | `classify.py:204` |
| `classification.header_chars` | int, 600 | Size of the `header` scope, chars from start of page 0. Also used by tagging. | `classify.py:73` |
| `classification.model_dir` | str, `models/classifier` | Trained classifier location. | `classify.py` |
| `classification.unknown_label` | str, `unknown` | Label for abstention. Also used by the VLM schema. | `classify.py:202` |
| `metadata.default_anchor_window` | int, 120 | Chars around an anchor label to hunt for a value. A field's own `anchor.window` wins. | `metadata/fields.py:308` |
| `metadata.drop_on_offset_mismatch` | bool, true | Drop a field whose offsets do not slice back to its value. | `fields.py:312` |
| `metadata.strict_offsets` | bool, false | Raise instead of dropping (tests). | `fields.py:309` |
| `metadata.max_candidates_per_field` | int, 25 | Candidate cap per field. | `fields.py:307` |
| `metadata.run_scoped_fields_when_unknown` | bool, true | Class-scoped fields still run when class is unknown. | `fields.py:311` |
| `tagging.model_dir` | str, `models/tagger` | Trained tagger location. | tagging |
| `tagging.default_threshold` | float, 0.5 | Used when a tag has no `threshold`. | `baseline/tagging.py:29` |
| `runtime.deterministic_timings` | bool, false | Zero the timings so output is byte identical. | `pipeline.py:107` |
| `runtime.log_level` | str, INFO | Log level. | `pipeline.py:259` |
| `runtime.workers` | int, 1 | NOT read by any code. Worker count comes from CLI `--workers` (`cli.py:411`). | none |

Excerpt:
```yaml
classification:
  rule_short_circuit: 0.9
  model_min_confidence: 0.45
  rule_min_confidence: 0.7
  header_chars: 600
  unknown_label: unknown
extraction:
  ocr: {engine: tesseract, dpi: 300, psm: 3, min_short_side: 1000}
runtime: {workers: 1, deterministic_timings: false}
```

## config/taxonomy.yaml

Loader: `Config.taxonomy`, optional (`config.py:116`), dict, unvalidated. Accessors: `client_label_map` (line 55), `lossy_client_labels` (line 60). Consumers: `baseline/labels.py`, `baseline/workbook/exporter.py:233`.

Class ids are NOT defined here. They are defined in `classes.yaml` (`Config.class_names`, line 51). This file maps the client's folder labels (14 labels) onto them.

| Key | Meaning |
|---|---|
| `version` | 1, unused. |
| `mappings[]` | `label` (client folder name / `Document_Type`), `class` (a name in classes.yaml, or null to refuse mapping), `lossy` (true if the class cannot recover the label; score collapsed), `n` (count, informational). |
| `unscored_classes` | Classes the client labelled set never uses (`bank_statement`, `court_document`). |
| `conflicts[]` | Files appearing under two labels: `file`, `labels`, `same_bytes`. Documentation for humans. |

```yaml
mappings:
  - {label: "Invoice", class: invoice, lossy: false, n: 25}
  - {label: "tax document", class: tax_form, lossy: true, n: 5}
  - {label: "Writ Petitions & Appeals", class: court_document, lossy: true, n: 16}
```
Seven legal labels all map to `court_document`. Class ids in classes.yaml: invoice, receipt, bank_statement, salary_slip, purchase_order, id_document, contract, tax_form, court_document, address_proof, kyc_form, customer_application, loan_application, sanction_letter, repayment_schedule, account_opening_form, account_closure_form.

## config/classes.yaml

Loader: `Config.classes`, required, dict, unvalidated (`config.py:116`). Rule engine: `baseline/rules.py` (`match_marker` line 37, `best_marker_score` line 64); per-class scoring `rule_scores` at `baseline/classify.py:72`; cascade at `classify.py:195-266`. Tags reuse the same marker machinery.

Class entry keys: `name` (id), `description` (also sent to the VLM; `check_descriptions` fails if blank), `markers` (list).

Marker keys:

| Key | Values | Meaning |
|---|---|---|
| `kind` | `regex` or `keywords` | Any other kind never fires (silent). |
| `pattern` | regex string | For `regex`. Case-insensitive unless `case_sensitive: true`. Empty pattern never fires. |
| `any` / `all` | list of terms | For `keywords`. Whole-word match (`(?<!\w)term(?!\w)`, case-insensitive). Every `all` term must be present AND at least one `any` term, if `any` is given. Both empty never fires. |
| `scope` | `first_page`, `header`, `anywhere` | Text slice: page 0; first `header_chars` chars of page 0; whole document. Default `anywhere`. An unknown scope silently acts as `anywhere` (`rules.py:28`). |
| `specificity` | float, default 0.5 | How much this marker alone proves the class. |
| `case_sensitive` | bool | Regex only. |

There is no `gate` key. "Gates" are written inside the regex as negative lookaheads, for example `\A(?![\s\S]*\b(?:due\s+date|payment\s+terms?|bill(?:ed)?\s+to|remit)\b)...` on the receipt markers (no due date, terms, bill-to or remit line).

Combination: a class score is the highest specificity among its firing markers (no sum, no vote); classes are independent. Then in `classify`: best rule at or above `rule_short_circuit` (0.9) wins and skips the model; else the model if its top score is at or above `model_min_confidence` (0.45); else best rule if at or above `rule_min_confidence` (0.7); else `unknown`. Ties break on class name. So a marker at 0.6 or 0.55 can never classify alone; it only supports.

Regex constraints (from comments in classes.yaml, and `tests/test_taxonomy.py:406`, `tests/test_vendor_name.py:24`): Python `re` on Python 3.10, so no atomic groups or possessive quantifiers (3.11+). The file emulates an atomic group with lookahead plus a named backreference `(?=(?P<c>...))(?P=c)`. Patterns must stay linear on long repeated input; each run of spaces is owned by one quantifier. Tests check this.

Invoice marker example:
```yaml
- {kind: keywords, all: ["invoice number", "invoice date"], scope: anywhere, specificity: 0.96}
- {kind: regex, pattern: '\bproforma\s+invoice\b', scope: first_page, specificity: 0.93}
```
Receipt marker example:
```yaml
- {kind: regex, pattern: '\b(?:payment|cash|money)\s+receipt\b', scope: first_page, specificity: 0.94}
- {kind: keywords, any: [receipt, "thank you for your payment"], scope: header, specificity: 0.55}
```

## config/fields.yaml

Loader: `Config.fields`, required, dict, unvalidated. Accessors: `field_defs` (line 65, `fields:`), `templates` (line 69), `normalizer_version` (line 81, default "1.0.0"). Extraction: `baseline/metadata/fields.py` (`_pattern_candidates` line 49, `_anchor_candidates` line 91, `_positional_candidates` line 172, strategy loop `candidates_for` line 241, `extract_fields` line 297). Top-level keys: `version`, `normalizer_version` ("1.1.0"), `fields` (22 entries), `normalizer_options`, `templates` (empty list).

Field entry keys:

| Key | Meaning |
|---|---|
| `name` | Field id. Also a key of the VLM schema. |
| `description` | Required for the VLM prompt (checked by `cascade.prompt.check_descriptions`). |
| `classes` | Optional list. Field runs only for these classes (and for unknown class if `run_scoped_fields_when_unknown`). Absent means all classes. |
| `type` | `id`, `text`, `date`, `amount` etc. Descriptive. |
| `normalizer` | Name in `NORMALIZERS` (`baseline/metadata/normalizers.py:147`): `id_upper`, `digits_only`, `lower_strip`, `upper_strip`, `ref_upper`, `text_norm`, `date_iso`, `currency`. Unknown name: warning, raw value kept. |
| `multi` | bool. Keep all candidates instead of the first. Also removes `maxItems: 1` in the VLM schema. |
| `strategies` | Ordered list from `pattern`, `anchor`, `positional`. Candidates from all are sorted by rank. An unknown name is silently skipped. |
| `pattern.regex` | Regex; group 1 is the value. |
| `pattern.validator` | Name in `VALIDATORS` (`validators.py:175`): `pan`, `aadhaar`, `gstin`, `ifsc`, `luhn`, `email`, `phone_in`, `date_sane`, `amount_sane`, `invoice_number`. Unknown name: warning, treated as not validated. |
| `pattern.validator_args` | Optional dict passed to the validator. |
| `pattern.confidence_validated` / `confidence_unvalidated` | Floats; defaults 1.0 / 0.5. |
| `pattern.ignorecase` | bool, default false. |
| `anchor.window` | Chars after/before the label to search. Falls back to `metadata.default_anchor_window`. |
| `anchor.direction` | `after` (default), `before`, `both`. |
| `anchor.labels[]` | `{text, specificity}`. Label text is matched lowercase; highest specificity tried first. |
| `anchor.value_regex` | Regex whose group 1 is the value, applied in the window. |
| `anchor.reject_before` | Regex on text just before the label; a hit drops the label (e.g. `[a-z]$` rejects a label glued to a word). |
| `anchor.reject_after` | Regex on text just after the value; a hit drops the value (e.g. "excl", "qty"). |
| `anchor.reject_value` | Regex; a value that matches is dropped (invoice number must not be a date). |
| `anchor.validator`, `validator_args`, `ignorecase` | Same as pattern. |

`normalizer_options`: `date_iso.dayfirst` (true), `currency.{default, lookbehind, symbols}`. `templates`: positional regions (`class`, `match.regex`, `fields.<name>.{page,region}`, `confidence`); empty today.

YAML anchors: `reject_value: &date_value '(...)'` on `invoice_number`, reused as `value_regex: *date_value` on `invoice_date` and `due_date`. Plain YAML aliasing, resolved by `safe_load`.

Excerpt:
```yaml
- name: invoice_date
  description: "The date the invoice, receipt or order was issued."
  classes: [invoice, receipt, purchase_order, bank_statement]
  type: date
  normalizer: date_iso
  strategies: [anchor, pattern]
  anchor:
    window: 60
    direction: after
    value_regex: *date_value
    labels:
      - {text: "invoice date", specificity: 1.0}
      - {text: "date", specificity: 0.55}
  pattern:
    regex: '\b([0-9]{4}[/.-]...)\b'
    validator: date_sane
    confidence_validated: 0.6
    confidence_unvalidated: 0.3
```
Field names: pan, aadhaar, gstin, ifsc, card_number, email, phone, invoice_number, invoice_date, due_date, total_amount, tax_amount, account_number, vendor_name, buyer_name, petitioner, respondent, judgment_date, bench, court, citation, case_number.

## config/tags.yaml

Loader: `Config.tags`, required, dict, unvalidated. Accessors `tag_defs` (line 73), `derived_tag_rules` (line 77). Consumer: `baseline/tagging.py` (thresholds line 29, keyword scoring ~line 34, derived rules `derived_scores` line 92).

| Key | Meaning |
|---|---|
| `tags[].name` | Tag id. |
| `tags[].description` | Human text. |
| `tags[].threshold` | Emit threshold; default `tagging.default_threshold`. Tuned by `baseline tune-thresholds`, which rewrites the file (`save_yaml`). |
| `tags[].patterns[]` | Same marker shape as classes.yaml (`kind`, `pattern`, `any`, `all`, `scope`, `specificity`). Empty list means model or derived only. |
| `derived[].tag` | Tag this rule emits. |
| `derived[].confidence` | Float, default 0.9. |
| `derived[].all_of` / `any_of` | Conditions. Metadata condition: `field`, `present`, `validated`, `min_confidence`, `value_matches` (regex). Document condition: `doc` (`page_count`, `extraction_confidence`, `is_scanned`, `char_count`, `method`) with `gt`, `lt`, `gte`, `lte`, `eq`. |

```yaml
- name: gst_applicable
  threshold: 0.5
  patterns:
    - {kind: regex, pattern: '\b(?:c?gst|sgst|igst|gstin)\b', specificity: 0.9}
derived:
  - tag: multi_page
    confidence: 1.0
    all_of: [{doc: page_count, gt: 1}]
```

## config/tables.yaml

Loader: `Config.tables`, optional, dict, unvalidated (`config.py:116`). Passed whole to `baseline/tables/*` (`cli.py:369`). All reads use `.get(key, default)`.

| Key | Default | Meaning |
|---|---|---|
| `version` | 1 | Unused. |
| `strategies` | `[tab_delimited, geometry, text_grid]` | Run in order, first hit wins per document (`tables/records.py:64`). Unknown name: skipped. |
| `min_rows`, `min_cols`, `min_cells` | 2, 2, 6 | A band must clear all three. |
| `geometry.row_overlap_ratio` | 0.5 | Vertical overlap to share a line. |
| `geometry.cell_gap_spaces` | 1.6 | Gap in space widths that ends a cell. |
| `geometry.space_width_percentile` | 25 | Percentile of in-line gaps taken as one space. |
| `geometry.min_space_width` | 1.0 | Fallback space width. |
| `geometry.column_overlap_ratio` | 0.4 | Horizontal overlap to share a column (complete linkage). |
| `geometry.max_row_gap_heights` | 2.0 | Row break, in median line heights, that ends a band. |
| `geometry.min_column_support` | 2 | Fewer cells than this: abstain. |
| `text_grid.min_separator_width` | 2 | Blank columns that make a separator. |
| `text_grid.min_multi_cell_rows` | 0.75 | Fraction of lines that must straddle a separator. |
| `text_grid.max_band_lines` | 400 | Longer bands are judged as they stand (search is quadratic). |
| `header.require_full_row`, `header.max_digit_ratio` | true, 0.1 | First row is a header only if full and nearly digit-free (`tables/assemble.py:17`). |

```yaml
strategies: [tab_delimited, geometry, text_grid]
min_cells: 6
geometry: {cell_gap_spaces: 1.6, column_overlap_ratio: 0.4}
```

## config/graph.yaml

Loader: `Config.graph`, optional, dict, unvalidated (`config.py:116`). Consumer: `baseline/graph.py` (`gcfg.get(...)`, lines 192-217, 293-341). `cli.py:348` refuses to run when it is empty.

| Key | Meaning |
|---|---|
| `version` | Unused. |
| `identity.organization.keys` | Verified fields that create or merge an organisation (`gstin`, `pan`). |
| `identity.organization.gstin_embeds_pan` | Chars 3-12 of GSTIN equal the PAN; joins GSTIN-only and PAN-only docs. |
| `identity.account.keys`, `.qualifier` | Account identity field (`account_number`) and the field that qualifies it (`ifsc`). |
| `roles.issued_by`, `roles.billed_to` | Role to field name (`vendor_name`, `buyer_name`). |
| `role_max_distance` | 400. Chars; give up beyond. |
| `role_min_margin` | 60. Winner must be this much closer than runner-up. |
| `references.field`, `.owner_classes`, `.edge` | Field that links docs (`invoice_number`), classes that own a number, edge name. |
| `duplicates.match_on`, `.same_class` | Fields that must match for a duplicate (`invoice_number`, `total_amount`); require same class. |

```yaml
identity:
  organization: {keys: [gstin, pan], gstin_embeds_pan: true}
roles: {issued_by: vendor_name, billed_to: buyer_name}
role_max_distance: 400
duplicates: {match_on: [invoice_number, total_amount], same_class: true}
```

## config/cascade.yaml

Loader: `load_cascade_config` at `cascade/config.py:63` (not part of `Config`). It calls `CascadeConfig.model_validate(read_yaml(...))`. Models derive from `Strict` (`baseline/schema.py:28`, `extra="forbid"`), so a misspelt key fails before the first document. Models: `VLMSettings` (line 18), `RenderSettings` (line 36), `ConfidenceTiers` (line 43), `CascadeConfig` (line 56).

| Key | Type / default | Meaning |
|---|---|---|
| `version` | int, required | Schema version (1). |
| `vlm.base_url` | str, required | OpenAI-compatible endpoint. Loopback only unless listed in `allowed_hosts`. |
| `vlm.model` | str or null, default null | Null refuses to run; set here or `--model`. |
| `vlm.api_key_env` | str or null | Name of an env var holding the key. |
| `vlm.timeout_s` | float > 0, required (180) | Request timeout. |
| `vlm.retries` | int 0-10, required (2) | Retry count. |
| `vlm.backoff_s` | float >= 0, required (2.0) | Backoff between retries. |
| `vlm.max_tokens` | int > 0, required (2048) | Response token budget. |
| `vlm.max_response_bytes` | int > 0, default 8 MiB | Bigger responses are refused. |
| `vlm.allowed_hosts` | list[str], default [] | Non-loopback hosts allowed. |
| `vlm.retry_statuses` | frozenset[int], default {429,500,502,503,504} | HTTP codes worth retrying. 400 is not. |
| `vlm.extra_body` | dict, default {} | Extra request fields (thinking off: `reasoning_effort: none`, `chat_template_kwargs.enable_thinking: false`). Never overrides fields `VLMClient` sets. |
| `render.dpi` | int, 0 < x <= 600 (150) | Page render DPI. |
| `render.max_long_edge` | int > 0 (1600) | Pixel cap per image. |
| `render.max_pages` | int > 0 (4) | Pages sent. |
| `render.max_text_chars` | int >= 0 (12000) | Total page text sent beside images. |
| `confidence.agree/grounded/ambiguous/vlm_class` | `Confidence` (0-1) (0.9/0.75/0.5/0.8) | Confidence tiers. Validator: `agree >= grounded >= ambiguous`. |

There is no `calibration` block. The `confidence` block holds unmeasured starting values, to be replaced by fitted calibration once reviewed gold exists.

Demo overrides (`demo/server.py:41-46`, `vlm_settings`): env `DOCC_VLM_BASE_URL` sets `vlm.base_url`, `DOCC_VLM_MODEL` sets `vlm.model` (applied with `model_copy(update=...)`, so not re-validated). If `vlm.model` is null and neither env var is set, the demo uses `http://127.0.0.1:11434/v1` with `qwen3.5:9b-16k`.

```yaml
vlm:
  base_url: http://127.0.0.1:8000/v1
  model: null
  timeout_s: 180
  retries: 2
  max_tokens: 2048
  extra_body: {reasoning_effort: none, chat_template_kwargs: {enable_thinking: false}}
render: {dpi: 150, max_long_edge: 1600, max_pages: 4, max_text_chars: 12000}
confidence: {agree: 0.9, grounded: 0.75, ambiguous: 0.5, vlm_class: 0.8}
```

## JSON schema sent to the VLM

Builder: `response_schema(cfg)` at `C:\Users\advay\Desktop\dev\DoCC\cascade\prompt.py:73`. Prompt text: `_system_text` (line 95), `build_messages` (line 107). Answer check: `parse_answer` (line 132), which raises `VLMOutputError` for non-JSON, extra or missing keys, class not in the enum, undeclared or missing fields, or non-string lists. `check_descriptions` (line 65) fails early if any class or field has no `description`.

How it is built from config:
- `class`: `enum` is `cfg.class_names` plus `classification.unknown_label` (18 values today).
- `fields.properties`: one entry per `fields.yaml` field with a name, each `{"type":"array","items":{"type":"string"}}`. `"maxItems": 1` is added unless the field has `multi: true` or is in `_OFTEN_SEVERAL = ("email","phone")`. Today `pan` and `gstin` (multi) plus `email` and `phone` have no cap; the other 18 are capped at 1.
- `required` lists every field name, so the model must return every key (empty list when absent). `additionalProperties: false` at both levels.
- The system prompt lists each class as `- name: description`, and each field as `- name (applies to: classes|any class): description` plus a negative from `_NOT` (line 41) for `account_number`, `invoice_number`, `total_amount`, `tax_amount`, `vendor_name`, `buyer_name`. Descriptions are therefore config; the negatives are in code.

Generated schema, trimmed (produced by calling `response_schema(load_config())`, no model call):
```json
{
  "type": "object",
  "properties": {
    "class": {"type": "string", "enum": ["invoice", "receipt", "...", "account_closure_form", "unknown"]},
    "fields": {
      "type": "object",
      "properties": {
        "pan":     {"type": "array", "items": {"type": "string"}},
        "aadhaar": {"type": "array", "items": {"type": "string"}, "maxItems": 1},
        "email":   {"type": "array", "items": {"type": "string"}}
      },
      "required": ["pan", "aadhaar", "gstin", "...22 names"],
      "additionalProperties": false
    }
  },
  "required": ["class", "fields"],
  "additionalProperties": false
}
```

## demo/Modelfile and .claude/launch.json

- `C:\Users\advay\Desktop\dev\DoCC\demo\Modelfile`: Ollama model `qwen3.5:9b` with `num_ctx 16384`; build with `ollama create qwen3.5:9b-16k -f demo/Modelfile`.
- `C:\Users\advay\Desktop\dev\DoCC\.claude\launch.json`: one configuration, `demo`, runs `python -m demo` on port 8765.

## Validation gaps

Only `cascade.yaml` has a schema. The rest are dicts read with `.get(key, default)`, so a typo drops the setting without any message.

- `pipeline.yaml`: no schema. A misspelt key (for example `rule_short_circut`) is silently ignored and the code default applies. A misspelt section name (`classify:` instead of `classification:`) drops the whole section. Wrong types raise only when `int()` or `float()` runs. `runtime.workers`, `version` and `extraction.ocr.threads` (absent) are never read from the file.
- `classes.yaml`: no schema. A misspelt `kind` or an unknown `scope` does not error: an unknown kind never fires, an unknown scope acts as `anywhere` (`rules.py:28,42`). Missing `specificity` defaults to 0.5, so a typo makes the marker too weak to classify. A bad regex raises `re.error` at first use, on the first document. A class without `name` raises `KeyError`. A missing `description` is caught only by cascade `check_descriptions`.
- `fields.yaml`: no schema. Misspelt anchor or pattern keys (`reject_befor`, `confidence_validatd`) are ignored. Unknown `normalizer` or `validator` names log a warning and the value is kept raw or treated as unvalidated (`normalizers.py:164`, `validators.py:194`). An unknown `strategies` entry is skipped without a message (`fields.py:261`). An invalid `direction` matches neither `after` nor `before`, so the label yields nothing. Bad regexes raise on first use.
- `tags.yaml`: no schema. Same marker leniency as classes. A derived rule with a typo in `field` or a condition key simply never fires; a rule without `tag` is skipped.
- `tables.yaml`: no schema; every key has a code default, so typos are ignored. An unknown strategy name is skipped. If the file is missing or empty, `baseline tables` refuses to run (`cli.py:369`).
- `graph.yaml`: no schema; typos fall back to defaults (for example `role_min_margin` to 0). Missing or empty file makes `baseline graph` refuse (`cli.py:348`).
- `taxonomy.yaml`: optional, no schema. Missing `mappings` raises in `labels.py:74`. A mapping to a class name not in classes.yaml is not checked here; labels absent from the file are reported by `labels.py:107`.
- `cascade.yaml`: validated. Extra or misspelt keys, out of range values and unordered confidence tiers fail at load. Gap: the demo's env override uses `model_copy(update=...)`, which skips validation.
