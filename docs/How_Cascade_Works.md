# How cascade v2 works

Cascade v2 is the vision-model pipeline in `cascade/`. This page explains how
one document goes through it, why it is built this way, what it costs, and
what the measured numbers say. The design is
`docs/superpowers/specs/2026-10-05-cascade-v2-vlm-grounding-design.md`. Where
the code and the design disagree, the code wins, and this page says so.

## 1. The idea in one paragraph

The baseline runs first, and its rules and checksums are the hard prior. A
local vision-language model (VLM) then reads the page images and the canonical
text that the baseline extracted. The model returns a class and a list of
values per field, in a fixed JSON schema. Every model value must be grounded:
it must be found at an exact span of the canonical text, so that
`text.full[char_start:char_end] == value`. A value that cannot be found is
refused. Rules and checksums win ties: a checksum-proven value is never
removed, and a rule at short-circuit strength decides the class. The model
server is local (loopback or an allowlisted host), so no document content
leaves the machine or the client network.

## 2. One document through cascade

The entry point is `cascade/runner.py:process_document`. The stages run in
this order. Any exception after the baseline run sends the document to the
fallback path (section 2.9).

### 2.1 Baseline run

- **Input:** the file path.
- **Output:** the baseline `DocumentRecord` and the word-box sidecar.
- **Code:** `baseline.pipeline.process_document`, called unchanged.
- **What it gives cascade:** the canonical text (`text.full`, the pages
  joined by `PAGE_SEP = "\n\f\n"`, `baseline/schema.py`), the rule metadata,
  the word boxes, and the page sizes.
- **Failure:** if the baseline cannot read the file, `process_document`
  raises. This is the only case with no record. `run_batch` counts it as
  `failed` and continues.

All offsets are into the baseline's text. Cascade and the baseline are scored
against the same string for the same file.

### 2.2 Render

- **Input:** the file path and the `render` block of `config/cascade.yaml`.
- **Output:** a list of PNG bytes, one per page.
- **Code:** `cascade/render.py:render_pages`.
- **Config:** `dpi: 150`, `max_long_edge: 1600`, `max_pages: 4`.

PDF pages are rendered with PyMuPDF at `dpi`. Image files give frame 0 only,
because that is the only frame the baseline extractor reads. `.docx` and
`.txt` give no images, so the model gets text only. Each image is scaled down
with Lanczos until its long edge is at most `max_long_edge` pixels. Only the
first `max_pages` pages are rendered, and only the same pages of text are
sent.

**Failure:** a document with no image and no text raises `nothing to send`
and falls back.

### 2.3 Prompt

- **Input:** the config, the page texts, the images, `max_text_chars`.
- **Output:** a two-message chat (system and user) and a JSON schema.
- **Code:** `cascade/prompt.py:build_messages`, `_system_text`,
  `response_schema`.
- **Config:** `max_text_chars: 12000`, the total page text sent across all
  pages. Class and field names and descriptions come from
  `config/classes.yaml` and `config/fields.yaml`.

The **system message** tells the model to:

- copy each value exactly as printed, and not reformat dates or amounts;
- copy the value as it appears in the page text, when it is there;
- return an empty list for a missing field, never an empty string, and never
  guess;
- return the one value that fits the field, not a nearby field;
- use `unknown` when the document is none of the classes.

Then it lists every class with its description, and every field as
`name (applies to: classes): description`. Six fields also carry a negative
definition from `_NOT` in `prompt.py`. For example, `vendor_name` is "Who
issues the document and is paid, not the customer", and `account_number` is
"Not a GSTIN, PAN, IFSC, phone, customer, reference or invoice number, nor a
date". These were added in round 2 for the confusions the audit saw
(`docs/parts/round2-cascade.md`).

The **user message** holds the page text, labelled `--- page N ---`, then each
image as a base64 PNG data URL.

The **schema** has `class` as an enum of the class names plus `unknown`, and
`fields` as an object with every field required, each an array of strings.
Single-value fields get `maxItems: 1`. Multi fields and `email` and `phone`
do not, because a page often prints several of them (`_OFTEN_SEVERAL`).
`additionalProperties` is false at both levels.

`run_batch` calls `check_descriptions` first, so a class or field with no
description fails the run before the first document.

### 2.4 VLM call

- **Input:** the messages and the schema.
- **Output:** a `VLMReply` with the content string, the model name the server
  reports, and `finish_reason`.
- **Code:** `cascade/vlm.py:VLMClient.complete`.
- **Config:** the `vlm` block of `config/cascade.yaml`.

| Key | Value | Meaning |
| --- | --- | --- |
| `base_url` | `http://127.0.0.1:8000/v1` | The OpenAI-compatible server. vLLM on a GPU box, Ollama on a laptop. |
| `model` | `null` | Null refuses to run. Set it, or pass `--model`. |
| `api_key_env` | `null` | Name of an environment variable that holds the server's key. |
| `timeout_s` | `180` | Timeout per HTTP attempt. |
| `retries` | `2` | Extra attempts after the first, so 3 attempts in total. |
| `backoff_s` | `2.0` | Sleep before retry n is `backoff_s * 2^(n-1)`: 2 s, then 4 s. |
| `max_tokens` | `2048` | The answer budget sent to the server. |
| `max_response_bytes` | `8388608` | A bigger response body is refused (8 MiB). |
| `allowed_hosts` | `[]` | Non-loopback hosts that are allowed. |
| `retry_statuses` | `[429, 500, 502, 503, 504]` | HTTP codes worth another attempt. |
| `extra_body` | see below | Server-specific request fields. |

The request is a POST to `<base_url>/chat/completions` with `temperature: 0`
and `response_format: {type: json_schema, strict: true}`. The fields in
`extra_body` go first, so they never override the fields the client sets.
`extra_body` turns thinking off: `reasoning_effort: none` for Ollama, and
`chat_template_kwargs.enable_thinking: false` for vLLM. A thinking model spends
the whole `max_tokens` budget on thinking and returns no answer
(`docs/client_poc_setup.md`, section 0).

**Retries.** A status in `retry_statuses`, a connection error, a timeout, or a
body that is not JSON uses one attempt and tries again. Any other HTTP error
raises at once. After the last attempt the client raises
`gave up after N attempts`.

**HTTP 400.** Ollama sends 400 when the prompt overflows the context. The
client raises `VLMBadRequest`, and the runner retries once with
`max(1, max_pages // 2)` pages and images. It records `reduced_pages` in the
provenance. A second failure falls back. This retry is not in the design. It
was added in K4 (`PLAN-cascade.md`).

**The 16k context.** Ollama's default context is 4,096 tokens. The invoice
request in the demo was 4,846 tokens, so it failed with 400 (`PLAN-demo.md`).
One page needs about 5,000 tokens (`docs/client_poc_setup.md`). The fix for
the laptop is `demo/Modelfile`: it builds `qwen3.5:9b-16k` from `qwen3.5:9b`
with `num_ctx 16384`. Build it once with
`ollama create qwen3.5:9b-16k -f demo/Modelfile`. On a server, start Ollama
with `OLLAMA_CONTEXT_LENGTH=16384`, or vLLM with `--max-model-len 16384`.

### 2.5 Parse

- **Input:** the reply content string.
- **Output:** a `VLMAnswer` with a label and a dict of field to values.
- **Code:** `cascade/prompt.py:parse_answer`.

The parser is strict. It raises `VLMOutputError`, and the document falls back,
when the content is not JSON, is not an object, has keys other than `class`
and `fields`, has a class that is not declared, has an undeclared field, has a
missing field, or has a field that is not a list of strings. Each value is
stripped. Empty strings are dropped, and duplicates are removed in order. A
field with no values left is left out of the answer.

### 2.6 Ground

- **Input:** one model value, a per-document `TextIndex`, the field
  definition, the ranked rule candidates for the field.
- **Output:** a `Grounding` with an outcome, a span, the number of matches,
  and `rule_agrees`.
- **Code:** `cascade/ground.py:ground`.

Three levels, and the first that matches wins:

1. **Exact.** The value occurs verbatim in `text.full`.
2. **Loose.** The value occurs after whitespace runs are collapsed and case is
   folded on both sides. The folded copy keeps a map back to the original
   characters, so the span is the document's own slice. A match that starts or
   ends inside a casefold expansion (for example `ß` to `ss`) is refused.
3. **Normalised.** The field's own normalizer gives the same value for the
   model value and for a place in the text. The places are the spans
   `scan_spans` finds for the field's shape, plus the rule candidates. This
   grounds a model `2021-04-08` against a printed `08/04/2021`.

Every match must also be usable:

- it lies on one page and never crosses `PAGE_SEP`;
- it does not start or end inside a longer word or number. A `-`, `/` or `,`
  between two alphanumerics joins them, and a `.` joins two digits. So `140`
  does not ground inside `02-140305`.

When there are several matches, the first match that a rule candidate agrees
with wins. Rule candidates arrive ranked best first, so a value printed twice
lands where the strongest rule put it. "Agrees" means the same span, or an
overlapping span with the same normalised value. Otherwise the first match is
taken and `rule_agrees` is false.

**Refused:** an empty value, and a value with no usable match. The outcome is
`ungrounded`, and the value is never emitted. The emitted value is always
`full[s:e]`, never the model's string.

### 2.7 Merge and decide

- **Code:** `cascade/merge.py:decide`, `decide_class`, `decide_fields`.
- **Config:** the `confidence` block of `config/cascade.yaml`, and
  `rule_short_circuit: 0.9` and `unknown_label: unknown` in
  `config/pipeline.yaml`.

**Class.** If the top rule score is at or above `rule_short_circuit`, the rule
decides. The method is `rule` and the confidence is the rule score. The
provenance logs `disagrees` when the model named a different real class.
Otherwise the model's label is used, with method `vlm` and confidence
`vlm_class` (0.8). A model `unknown` gets confidence 0.0. That is an
abstention.

**Fields.** For each field in `fields.yaml`:

1. **Checksummed fields.** Every baseline value with method `regex_checksum`
   is kept first, even when the field is out of scope.
2. **Scope.** A field scoped to other classes gets no model value. The
   scoping rule is the baseline's.
3. **Ranking.** For a single-value field, the model values are sorted: rule
   agreement first, then exact over loose over normalised, then one match over
   many, then the model's order. The first value that survives is kept. A
   multi field keeps every distinct value. This was K3 (`PLAN-cascade.md`).
4. **Per value:**
   - not grounded: skipped;
   - checksummed field: emitted only if the span equals a rule candidate with
     method `regex_checksum`. Then the method is `regex_checksum`, confidence
     1.0, `validated: true`. Otherwise the outcome is `checksum_failed`;
   - rule agrees: method `vlm_rule_agree`, confidence `agree` (0.9);
   - model only: the **shape gate** runs. A model-only `account_number` must
     have no letters, 9 to 18 digits, and must not equal a GSTIN, PAN, IFSC or
     phone value already seen. A model-only `invoice_number` must hold a digit
     and be at most 30 characters. A failure is `shape_failed`. A pass gets
     method `vlm`, confidence `grounded` (0.75) for one match or `ambiguous`
     (0.5) for several.
5. **Rule-only drop.** For a non-checksummed field, if the model emitted
   nothing, the baseline's values are not emitted. The provenance records
   `rule_only` with the baseline values. The anchor rules' false positives are
   what this approach exists to fix.

`validated` is true only when a checksum passed. That keeps validated
precision at 1.0 (`docs/parts/cascade-pass.md`).

**Confidence tiers.** `agree: 0.9`, `grounded: 0.75`, `ambiguous: 0.5`,
`vlm_class: 0.8`. The config marks them as starting points, not measured.
`cascade/config.py` refuses a config where `agree >= grounded >= ambiguous`
does not hold. No threshold reads them yet, so they do not change which
values are emitted (`docs/parts/round2-cascade.md`).

### 2.8 Record and provenance

After merge, `_drop_bad_offsets` checks every field again against
`full[char_start:char_end] == value`. A bad one is dropped and written to
`diagnostics.errors`. Tags are applied to the merged metadata with the
baseline's tagger. The record has the baseline's `doc_id`, path, filename and
text, cascade's class, metadata and tags, and diagnostics with the baseline's
timings plus `cascade_render`, `cascade_vlm` and `cascade_decide`.

The record is a normal `DocumentRecord`, schema version 1.2.0
(`baseline/schema.py`). That version adds `ClassifyMethod.VLM`,
`MetadataMethod.VLM` and `MetadataMethod.VLM_RULE_AGREE`.

The **provenance** is one JSON line per document in
`<output>.provenance`. It holds `doc_id`, `filename`, `model` (as the server
reports it), `finish_reason`, `pages_sent`, `images_sent`, `reduced_pages`
when the 400 retry ran, `fallback`, `errors`, `raw` (the model's full answer
string), `class` (source, label, model label, `disagrees`), `fields` (one item
per value tried: model value, outcome, number of matches, `rule_agrees`, span,
`emitted`), and `timings_ms`. Word boxes go to `<output>.bboxes`, as in the
baseline.

The demo's cascade mode returns only `model`, `finish_reason`, `pages_sent`,
`images_sent`, `fallback`, `errors` and `timings_ms`. It drops `raw` and the
per-field items (`demo/server.py`).

### 2.9 Fallback

Any exception from render to tags is caught. The runner logs it, sets
`fallback: true` in the provenance, and writes the baseline's class, metadata
and tags with the error in `diagnostics.errors`. A model outage degrades to
the baseline, never to an empty record.

## 3. Why grounding

The design is precision first. The harness scores span F1, which needs
offsets, and validated precision, which punishes false proof
(design spec, "Where the points are"). A model that returns correct values
without locations scores zero on span F1.

A hallucination is a value the document does not contain. It cannot be
grounded, so it is refused. Grounding is both the offset step and the guard
against invention. In the K1 run, the model returned `[YOUR COMPANY NAME]` on
a receipt template. It did not ground, so it was refused
(`docs/parts/cascade-pass.md`, table A). The round-2 audit found zero
ungrounded hallucinations among the extra values: every new value was printed
on the page (`docs/parts/round2-cascade.md`, section 3).

The cost is recall on bad OCR. On `image-537.png` the model read `$378.00`
from the image, but `378` is not in the OCR text, so the value was refused. On
the payment receipt template it read `$4,044.60`, which is also a misread
(`docs/parts/round2-cascade.md`). That costs recall, never precision. The
plan says not to soften grounding (`PLAN-cascade.md`, constraints).

**Worked example from the demo** (`PLAN-demo.md`):

- `invoice_native.pdf`: a native PDF. Cascade sent 2 pages and took 32.9 s.
  All 13 fields came out as `vlm_rule_agree`. So the model and a rule found
  each value at the same place, and each has confidence 0.9 (`agree`). No
  fallback.
- `receipt_scanned.pdf`: a scan. Cascade sent 1 page and took 13 s.
  `vendor_name` is the payer, at 75%, method `vlm`. No rule agreed, the value
  grounded at one span, so it got the `grounded` tier, and it is not
  validated. It is printed on the page, so grounding accepts it, but it is the
  wrong party. Grounding proves a value is on the page. It does not prove that
  the value is in the right field.

## 4. How it is built

### Package layout

`cascade/` sits beside `baseline/`. It imports from `baseline` and never
changes it.

| File | Job |
| --- | --- |
| `render.py` | page images |
| `prompt.py` | system prompt, schema, strict parser |
| `vlm.py` | the HTTP client and the local-only guard |
| `ground.py` | value to span |
| `merge.py` | class decision, field ranking, gates, methods |
| `runner.py` | one document, fallback, batch run |
| `config.py` | validated `cascade.yaml` |
| `cli.py` | `cascade run` |

Cascade writes the same `DocumentRecord` JSONL, so `baseline eval` and
`baseline compare` score it with no change. The batch runner reuses the
baseline's resumable JSONL writing. No test needs a GPU: the runner tests use
a `FakeBackend` (`tests/test_cascade_runner.py`), and the client tests use a
local stub HTTP server (`tests/vlm_stub.py`, used by
`tests/test_cascade_vlm.py`).

### Design decisions and trade-offs

- **Baseline first, on CPU.** The model never defines the text, and every
  document has a fallback. The cost is a full baseline run before each model
  call.
- **Images capped.** 150 dpi, long edge 1600, 4 pages, 12,000 text characters.
  One big scan cannot exhaust GPU memory. A longer document loses its later
  pages, but no K1 document had more than 2 pages
  (`docs/parts/cascade-pass.md`).
- **No fine-tuning.** A layout model needs hundreds of labelled documents per
  type. The team holds 20 reviewed documents and does not know the client's
  types (design spec, "Approach"). A new type is a YAML edit.
- **OpenAI-compatible API.** vLLM and Ollama both serve it, so the model and
  the server are configuration. The client uses only `urllib`.
- **Thinking off.** Without it, the answer is empty and every document falls
  back (`docs/client_poc_setup.md`).
- **Temperature 0.** Runs repeat on the same server and model, but not
  exactly: one class flipped between two runs (`PLAN-cascade.md`, K3+K4).

### Safety rules (`cascade/vlm.py`)

- The host must be loopback (`localhost` or a loopback IP) or listed in
  `vlm.allowed_hosts`. The check runs on the base URL and on the final URL.
- The URL must be `http` or `https`, with no whitespace, control characters,
  user info, `%`, `\`, `@`, query or fragment.
- Redirects are never followed. Proxy environment variables are ignored.
- An API key sent over plain `http` to a non-loopback host logs a warning.
- A response bigger than `max_response_bytes` is refused.
- The VLM call is cascade's only network call.

## 5. What it costs

All timings are on the developer laptop: RTX 3070, 8 GB, Ollama
`qwen3.5:9b-16k`.

| Measure | Value | Source |
| --- | --- | --- |
| Baseline, per document | about 1.9 s (39.5 s for 21 docs) | `docs/parts/cascade-pass.md` |
| Cascade, per document | 24.6 s (516 s for 21 docs) | `docs/parts/cascade-pass.md` |
| VLM time per successful document | 11 to 40 s | `docs/parts/cascade-pass.md` |
| `invoice_native.pdf`, 2 pages | 32.9 s | `PLAN-demo.md` |
| `receipt_scanned.pdf`, 1 page | 13 s | `PLAN-demo.md` |
| Earlier laptop test | about 20 s per document | `docs/client_poc_setup.md` |
| A failed call, 3 attempts on HTTP 500 | 96.5 s | `docs/parts/cascade-pass.md` |

Pages sent: at most 4 (`max_pages`). Tokens: the 2-page invoice request was
4,846 tokens (`PLAN-demo.md`), and one page needs about 5,000
(`docs/client_poc_setup.md`). The answer budget is 2,048 tokens
(`max_tokens`). The GPU client's cost is unknown until its hardware is known.

## 6. What the numbers say

All runs are on the 20 human-reviewed documents. Gold has 21 rows; the voter
ID file is unmatched in both runs and is not scored
(`docs/parts/cascade-pass.md`). "Covered" F1 counts only fields that the gold
labels at least once (K2, `PLAN-cascade.md`).

| Run | Class macro F1 | Coverage | Metadata TP/FP/FN | Micro F1 | Covered F1 | Fallbacks |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline, K1 | .811 | .75 | 21 / 30 / 10 | .512 | .560 | n/a |
| Cascade, K1 | .971 | .95 | 24 / 52 / 7 | .449 | .552 | 2 |
| Cascade, after K3+K4 | .953 | 1.00 | 25 / 53 / 6 | .459 | .568 | 1 |
| Cascade, after round 2 | .953 | 1.00 | 25 / 49 / 6 | .476 | .595 | 0 |

Sources: `docs/parts/cascade-pass.md` (K1 rows) and `PLAN-cascade.md` (the
two later rows). K5 left the baseline unchanged on the 20 documents
(`PLAN-cascade.md`). Validated precision is 1.000 for both pipelines in K1.
Accuracy when answered is 1.000 for both (`docs/parts/cascade-pass.md`).

**The October "drop".** The 2026-10-22 run showed metadata micro F1 falling
from .488 to .440 (`PLAN-cascade.md`, session notes). K1 found that cascade
lost no correct value. TP rose 21 to 24, and recall rose .677 to .774. F1 fell
because of 22 more false positives, and 13 of these are on `phone`,
`vendor_name` and `due_date`, which the gold never labels. The harness counts
every unlabelled value as a false positive (`docs/parts/cascade-pass.md`).

**Remaining misses and why** (`docs/parts/cascade-pass.md`, table B;
`docs/parts/round2-cascade.md`):

- **OCR lacks the value** (2 documents). The model read the amount from the
  image, but the OCR text does not hold it, so grounding refuses it. The fix is
  better OCR, not softer grounding.
- **Masked Aadhaar** (1). The model returned `XXXX XXXX 3208`. It grounds
  exactly, but a masked number cannot pass the checksum gate.
- **Casing** (1). `John Doe` against gold `John doe`. A harness item.
- **Ranking** (1, `SalarySlip2.jpg` total). K3 lets the merge try the second
  value, but the plan reports that neither target miss was fixed by K3+K4.
- **Run-to-run variance.** One document's class flipped between runs, so
  classification moved from .971 to .953 (`PLAN-cascade.md`).

Span F1 fell from .189 to .073 in K1 (`docs/parts/cascade-pass.md`). The
sources do not explain this drop.

## 7. Running it

```
python -m cascade [--config DIR] [--root DIR] [--log-level LEVEL] run \
    --input DIR --output results.jsonl \
    [--model NAME] [--base-url URL] [--bbox-dir DIR] [--no-resume]
```

`--model` and `--base-url` override `cascade.yaml`. The run writes
`results.jsonl`, `results.jsonl.provenance` and `results.jsonl.bboxes/`, and
resumes by default. Score it with `baseline eval` and compare with
`baseline compare`, as for any run.

**Demo.** Run `python -m demo` and open `http://127.0.0.1:8765`. Pick cascade
mode. With `vlm.model` null in `cascade.yaml`, the demo uses Ollama at
`http://127.0.0.1:11434/v1` with `qwen3.5:9b-16k`. Override with
`DOCC_VLM_BASE_URL` and `DOCC_VLM_MODEL` (`demo/server.py`).

**Client GPU server.** Run vLLM or Ollama on the GPU server with a 16k context,
listening on the local network. On the VM, set `base_url`, `model` and
`allowed_hosts` in `config/cascade.yaml`. The full steps are in
`docs/client_poc_setup.md`.

## 8. Open questions and next levers

- **Better OCR on the worst pages.** Two misses are values the model sees but
  the OCR text lacks. A VLM reader for those pages is the design's named next
  step.
- **Gold for unlabelled fields.** `phone`, `vendor_name` and `due_date` need
  gold before precision on them can be judged.
- **Fitted calibration.** The tiers are unmeasured. They need gold to fit.
- **Masked Aadhaar.** A masked last-four rule candidate would add one TP, but
  it loosens the checksum gate. It is a deliberate decision, not a fix.
- **The GPU client's hardware and documents.** They set the model size, the
  classes, the fields, and whether the VLM skip path is needed.

## Where the code and the design disagree

The code wins in each case.

- **Checksummed model values.** The design says to run the model value
  through the field's validator. The code emits it only when its span equals a
  rule candidate with method `regex_checksum` (`merge.py:decide_fields`).
- **Ambiguous tier.** The design gives `ambiguous` to any value with several
  matches. The code gives `agree` to a rule-agreeing value even when it has
  several matches, and `ambiguous` only to a model-only value.
- **Not in the design.** The shape gate (`shape_failed`), the per-field
  negatives, `maxItems: 1`, value ranking, and the HTTP 400 half-pages retry
  were added in the accuracy pass (`PLAN-cascade.md`).
- **Normalised grounding places.** The design names the baseline's pattern and
  anchor scanners. The code uses `scan_spans` plus the rule candidates.
