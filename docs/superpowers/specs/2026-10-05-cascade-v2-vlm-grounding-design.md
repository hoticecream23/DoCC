# Cascade v2: VLM extraction with grounding

Date: 2026-10-05
Status: approved in discussion 2026-10-05
Supersedes: `2026-09-17-cascade-architecture-design.md` for stages 4 and 5
and for the CPU first constraint. That spec's output contract, evaluation
protocol, knowledge graph write rules and provenance sidecar still stand.

## Why the overhaul

Two drivers.

1. **The GPU client.** The team has been pointed at a client that has GPUs.
   Its hardware and documents are not known yet. The old spec was CPU first,
   chose LiLT partly for CPU cost, and tuned several stages to the Yaumi
   corpus. None of that should be assumed for a client we know nothing about.
2. **Competition.** Teammates are building rival pipelines. All of them are
   scored on our harness: `DocumentRecord` JSONL through `baseline eval` and
   `baseline compare`, on shared gold.

Constraints that did not change:

- **Local models only.** No document content leaves the company. Models are
  self hosted on the client's GPUs.
- **Permissive licences only.** Apache-2.0, MIT or equivalent, checked before
  a model is adopted.
- **The output contract holds.** Every metadata value carries character
  offsets that slice back to the canonical text.

The timeline is open ended, so this optimises for the best final system, and
the first step is the one that generalises to unknown documents.

## Where the points are

The harness does not only score whether a value is right.

- **Span F1** needs character offsets into the canonical text. A pipeline that
  returns values without locations scores zero on it.
- **`validated_precision`** rewards values that are proven, and punishes a
  pipeline that claims proof it does not have.
- **Coverage at 1, 5 and 10 percent error** rewards knowing when to abstain.

A rival that returns correct values but no offsets, or that never abstains,
loses on all three. Cascade v2 is designed to win those three, on top of
value accuracy.

## Approach

Chosen over two alternatives:

- **A fine tuned layout model** (LiLT and similar; LayoutLMv3 is excluded as
  CC-BY-NC). Fast and precise, with native offsets, but it needs hundreds of
  labelled documents per document type and does little zero shot. We hold 20
  reviewed documents and do not know the client's types. Worth revisiting
  later as a distilled fast path.
- **A GPU tiered cascade** (rules, then a small encoder, then a VLM on the
  residue). Cheapest per page, but it has the most thresholds to set, and each
  needs gold. Its cheap tiers were tuned on Yaumi.

What was chosen: **an open weight vision language model reads every page,
and every value it returns is grounded back into the canonical text before it
may be emitted.** The rule and checksum layer stays as a hard prior.

The one idea kept from the tiered cascade is a skip path: a document whose
class is decided by a rule at short circuit specificity and whose fields are
all checksum proven may skip the VLM. It is not built in v1. It is added only
once throughput is measured and turns out to matter.

## Scope of this sub-project

**In:** cascade v1. A runner that produces `DocumentRecord` JSONL using a VLM
plus grounding, scored by the unchanged harness.

**Out, each its own later sub-project:**

- The document grouper. The rule it uses (`CASH #` on Yaumi) may not exist at
  the new client.
- The knowledge graph in the run.
- LoRA fine tuning on reviewed gold.
- Fitted calibration. Needs gold.
- The VLM skip path.
- A rule set for the GPU client's documents.

**Untouched:** `baseline/` entirely, the tables pass, and tagging, which
cascade calls rather than reimplements.

## Pipeline

```
file
  -> baseline.extract          native text, quality gate, OCR, word boxes   reused
  -> render page images                                                    new
  -> baseline rules            rule class scores, checksummed IDs          reused
  -> VLM                       page images + page text + JSON schema
                               -> {class, fields}                          new
  -> ground                    each VLM value -> span in canonical text    new
  -> merge and decide          checksum prior, class decision, confidence  new
  -> baseline tagging          tags from the merged metadata               reused
  -> DocumentRecord JSONL + provenance sidecar
```

### Text and geometry

`baseline.extract` is called unchanged. It provides the canonical text, the
pages, the per page extraction methods and the word boxes. The VLM never
defines the text: offsets are always into the text the baseline extractor
produced, so cascade and the baseline are scored against the same string for
the same file.

### Page images

- PDF pages are rendered with PyMuPDF at a configured DPI.
- Image files are read directly. Multi frame TIFFs yield one image per frame,
  in the same order the extractor reads them.
- Formats with no page image (`.docx`, `.txt`) send text only.
- Long edges are capped at a configured pixel size, so one oversized scan
  cannot exhaust GPU memory.
- A document longer than a configured page limit sends its first pages only,
  and the provenance records which pages were sent.

### VLM backend

- One client for an **OpenAI compatible chat completions endpoint**. vLLM
  serves this on the client's GPUs and Ollama on a development laptop, so the
  model and the server are configuration, not code.
- The endpoint must be local. The client refuses a URL whose host is not a
  loopback address or a host named in config, so a mistyped URL cannot send a
  document off site.
- Written on the standard library (`urllib.request`), so no new HTTP
  dependency.
- Every call has a timeout and a bounded number of retries with backoff.
- A call that fails, times out, or returns output that does not parse against
  the schema is recorded in `diagnostics.errors`. The document then falls back
  to the baseline's own classification and fields, so a VLM outage degrades
  to the baseline rather than to empty records.
- Requests use temperature 0 so a run is repeatable on the same server and
  model.

The default model is chosen by experiment on the client's hardware. It is a
config value, and its licence is checked before it is adopted.

### Prompt and schema

- The response is forced to a JSON schema built from config. `class` is an
  enum of `classes.yaml` names plus the unknown label. `fields` maps each
  `fields.yaml` name to a list of strings.
- Each class and field carries a one line description in YAML, which goes in
  the prompt. A new client document type is a YAML edit.
- The prompt instructs the model to copy values exactly as printed. Grounding
  then enforces that rather than trusting it.
- Field names, class names and descriptions come from config. None appear in
  code.

### Grounding

This is the correctness core. **A VLM value that cannot be located in the
canonical text is not emitted**, because the contract requires
`full[char_start:char_end] == value`. This is also the guard against invented
values: a value the document does not contain cannot be grounded.

For each value, in order, the first step that matches wins:

1. **Exact.** The value occurs verbatim in the canonical text.
2. **Loose.** The value occurs once whitespace runs are collapsed and case is
   folded on both sides. Matching is done on a normalised copy of the text
   that keeps an index back to the original characters, and the span reported
   is the original slice, so `full[s:e]` is the document's own spelling.
3. **Normalised.** Candidate spans come from the baseline's own pattern and
   anchor scanners for that field. A candidate matches when its normalised
   value equals the VLM value's normalised value, using the field's own
   normalizer. This is what grounds a VLM `2021-04-08` against a printed
   `08/04/2021`, and `4,044.00` against `4044`.

Rules for every step:

- The span never crosses `PAGE_SEP`. A value spread across two pages is not a
  single field value.
- The emitted `value` is always `full[s:e]`, never the VLM's string.
- **More than one match:** a span that a rule candidate also produced is
  preferred. Otherwise the first occurrence is taken, its confidence is set to
  the ambiguous tier, and the provenance records how many matches there were.
- **No match:** the value is dropped and recorded in the provenance as
  `ungrounded`, with the VLM's string.

### Merge and decide

**Checksummed fields** (`CHECKSUM_VALIDATORS`):

- A rule proven value is always kept, and no VLM output can remove or replace
  it.
- A VLM value for a checksummed field is run through the same validator after
  grounding. If it fails, it is dropped. If it passes, it is emitted as
  `regex_checksum`, since the checksum is what proves it.

**`validated`** is set true only when a checksum passed. Cascade does not
mark a value validated on a format check such as `date_sane`. This settles,
for cascade, the open handoff question about the `validated` contract, and it
is what keeps `validated_precision` at 1.0. The knowledge graph keys entities
on validated values, which is a second reason not to weaken it.

**Other fields:**

- The VLM value is emitted when grounded.
- If the VLM returns nothing for a field that a rule candidate covers, the
  rule candidate is not emitted. The VLM read the page and found no value, and
  the rule layer's false positive rate on anchor fields is the problem this
  approach exists to fix. Recorded in the provenance as `rule_only`, so the
  decision can be measured rather than argued.
- A multi field emits each distinct grounded value once.

**Class:**

- A rule score at or above `rule_short_circuit` decides the class. The rules
  are never wrong when they answer on the labelled set, and that is the
  strongest evidence in the system.
- Otherwise the VLM's class is used. `unknown` from the VLM is an abstention.
- A disagreement between a rule above the short circuit and the VLM is logged
  to the provenance.
- A field scoped to classes in `fields.yaml` follows the same scoping as the
  baseline: emitted when its scope includes the decided class, or when the
  document is unknown.

**Confidence:** fixed tiers from config until there is gold to fit
calibration on. Values below are the starting points, not measured.

| case | tier |
|---|---|
| checksum passed | 1.0 |
| VLM grounded at a span a rule candidate also produced | `agree` |
| VLM grounded at a unique span | `grounded` |
| VLM grounded at one of several matching spans | `ambiguous` |
| class decided by rule | the rule score |
| class decided by VLM | `vlm_class` |

Every report produced from these says the confidences are uncalibrated.

## Output contract

- `DocumentRecord` JSONL, unchanged in shape.
- Schema version to **1.2.0**, an additive change: `ClassifyMethod.VLM`,
  `MetadataMethod.VLM` and `MetadataMethod.VLM_RULE_AGREE`.
  `baseline/evaluate/` does not read method values, so scoring needs no
  change. The workbook exporter writes method values as strings, so it needs
  none either.
- Every record passes `verify_offsets()` before it is written. A record that
  fails has the failing fields removed and the problem written to
  `diagnostics.errors`; it is never written with a bad offset.
- **Provenance sidecar**, `<output>.provenance`, one JSON line per document:
  the model the server reports, pages and images sent, raw VLM output, per
  value grounding outcome (`exact`, `loose`, `normalised`, `ungrounded`,
  `checksum_failed`, `rule_only`) with the number of matching spans, class
  decision and source, and timings per step. The run log names the server.
- Word boxes go to the same `.bboxes` sidecar the baseline writes.

## Package layout

```
cascade/
  __init__.py
  __main__.py
  render.py      page images from a document
  vlm.py         OpenAI compatible client, timeouts, retries, local only guard
  prompt.py      JSON schema and prompt from config
  ground.py      value -> span in the canonical text
  merge.py       checksum prior, class decision, confidence tiers
  runner.py      one document in, record and provenance out; batch running
  cli.py         cascade run
config/cascade.yaml
```

`cascade` imports from `baseline` and never modifies it. If a baseline part
needs a change for cascade, that change is made in `baseline` and must leave
the baseline's own outputs identical, checked by `compare` on `gold.jsonl`.

The batch runner reuses the baseline's resumable, order stable JSONL writing.
It runs documents in one process by default, since the GPU server does the
heavy work and batches across requests itself.

## Testing

- Test first throughout.
- **No GPU in the test suite.** A fake backend returns canned JSON per
  document, so every test is deterministic and runs on any machine.
- Grounding carries the most tests: exact, loose on whitespace, loose on case,
  normalised date, normalised amount, ambiguous with and without a rule
  candidate, ungrounded, a hallucinated value, a value that would cross
  `PAGE_SEP`, and a value that occurs only inside a longer token.
- Merge: a proven ID survives a VLM that omits it; a VLM ID that fails its
  checksum is dropped; `validated` is never true without a checksum; the class
  short circuit wins over the VLM.
- Backend: timeout, retry, malformed JSON, schema violation and a non local
  URL, against a local stub HTTP server.
- End to end: the synthetic corpus through the runner with the fake backend.
  Every record passes `verify_offsets()`, and with a backend that always
  fails the output matches the baseline's classification and fields.
- Once a real model runs: `baseline compare --fail-on-regression` against the
  baseline on `gold.jsonl` and on the reviewed client gold.

## Risks

- **GPU cost per page.** Unknown until the hardware is. The skip path is the
  answer if it matters, and is measured before it is built.
- **Grounding recall on bad OCR.** A value the VLM reads correctly from the
  image, but which OCR mangled in the text, cannot be grounded and is
  dropped. That costs recall, never precision. The provenance counts it, so it
  can be measured, and it is the evidence that would justify a VLM reader for
  the worst pages later.
- **Ambiguous spans cost span F1.** A value printed twice is located at the
  first occurrence unless a rule candidate says otherwise. Word box grounding
  from the VLM's own coordinates is a possible later improvement.
- **Model drift.** A server upgrade can change outputs. The provenance records
  the model for every document and the run log names the server, so two runs
  that differ can be explained.

## Open questions

1. The GPU client's hardware, which sets the model size.
2. The GPU client's document types, which set the classes and fields.
3. A pages per day target, which decides whether the skip path is needed.
