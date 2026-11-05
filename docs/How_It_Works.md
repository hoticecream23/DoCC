# How it works

A walkthrough for the live demo. Each number names its source.

## 1. What it is

The system writes one JSON record per document: text, type, fields and tags.
The current system is the baseline. It is deterministic and runs on a CPU only. It
uses regex, checksums, TF-IDF, linear SVMs and Tesseract OCR. It makes no LLM
calls (README.md). It sets the floor that a smarter approach must beat, and
freezes the output contract so one harness can score any implementation. The successor is cascade v2.
In cascade v2, a local vision model reads each page. Cascade v2 writes the same
record, and the same harness scores it (PROJECT_STATE.md).

## 2. One document, end to end

`baseline/pipeline.py`, function `process_document`, runs these stages in
order: extract, classify, metadata, tag. Each stage is timed. A failed stage
records an error and returns a safe default, so the document still gets a
record.

The rule for every stage: **refuse rather than guess.** A wrong answer costs
more than a missing one. On weak evidence the system says `unknown` or emits
nothing. This is how classification keeps precision at 1.0 while coverage grows
(docs/parts/audit.md).

### 2.1 Text extraction

- **In:** a file. PDF, DOCX, TXT or an image.
- **Out:** the canonical text by page, each page's route, and a confidence.
- **How:** a PDF page with a text layer is read directly (*native*). A page
  with fewer than 100 characters of native text goes to OCR
  (config/pipeline.yaml). A mix is *hybrid*; `page_methods` shows each page. DOCX and TXT are read directly. Images always go
  to OCR (README.md).
- **Quality gate:** a text layer can be somebody else's bad OCR. A page that
  scores above 0.02 is re-read through OCR as `ocr_rescued`
  (config/pipeline.yaml).
- **How it fails:** a poor scan gives poor text, and every later stage inherits
  it. A corrupt file gives an error in `diagnostics.errors`.
- **In the demo:** the text route (native, ocr or hybrid), the per-page
  methods, the confidence, and the `extract` timing.

### 2.2 Classification

- **In:** the canonical text.
- **Out:** a label, a confidence, and the method (rule or model).
- **How:** first the rules in `config/classes.yaml`. Each class has keyword or
  regex markers, each with a scope and a specificity. Two examples for
  `invoice`:
  - the words "invoice number" and "invoice date" together, specificity 0.96;
  - the regex `\bproforma\s+invoice\b` on the first page, specificity 0.93.

  A rule at 0.9 or above decides the class outright and skips the model
  (config/pipeline.yaml, `rule_short_circuit`). Otherwise a calibrated TF-IDF
  SVM decides. A model answer below 0.45 is discarded (`model_min_confidence`).
  A weaker rule at 0.7 or above counts only after the model declines
  (`rule_min_confidence`). If nothing passes, the label is `unknown`.
- **How it fails:** a document with none of the expected words gets `unknown`,
  by design. The model is trained on synthetic data only and
  cannot predict 9 of the 17 classes. The rule layer carries every class
  (PROJECT_STATE.md, "Known gaps" 3).
- **In the demo:** the label, the confidence, the method, and the `classify`
  timing.

### 2.3 Metadata fields

- **In:** the canonical text, the class, and the word boxes.
- **Out:** fields, each with value, normalized value, page, `char_start`,
  `char_end`, confidence, method and `validated` (baseline/schema.py).
- **How:** `config/fields.yaml` defines each field with one or more strategies:
  - **regex** for values with a fixed shape;
  - **anchor**: find a label word, then look for the value near it;
  - **positional**: read a fixed spot on a known template. No templates are
    registered yet (PROJECT_STATE.md, "Known gaps" 6).

  Example: `total_amount` runs only on invoice, receipt, purchase_order and
  salary_slip. It looks up to 80 characters after a label such as
  "grand total" (specificity 1.0) or "net pay" (0.9). It refuses a label whose
  own line names another quantity, such as "sub total" or "tax total"
  (config/fields.yaml).

  Validators check IDs with real checksums: PAN, Aadhaar, GSTIN, IFSC and card
  numbers. A value that fails its checksum is not emitted at all. On real
  scans, a failed checksum was always misread text (README.md).
- **Offsets:** slice the canonical text with a field's offsets and you get the
  value back. This is checked at build and again at write. A field whose
  offsets do not slice back is dropped (config/pipeline.yaml,
  `drop_on_offset_mismatch`).
- **How it fails:** an anchor field fails when the document prints no label.
  On a receipt the vendor is just the first line. A receipt also prints
  subtotal, rounding, total, cash and change, and the anchor can pick the wrong
  one (PROJECT_STATE.md, "Measured on public data").
- **In the demo:** the field table, with offsets and page for each value. Note
  which values show `validated: true`.

### 2.4 Tagging

- **In:** the text, the fields, and facts such as page count and whether the
  document is scanned.
- **Out:** a list of tags, each with a confidence and a source method.
- **How:** a document has one type but many tags. Tags come from keywords, a model, or the metadata itself. A
  validated GSTIN is stronger evidence for `gst_applicable` than the word "GST",
  and the tag records which source it came from (README.md).
- **In the demo:** the tag list, with the method next to each tag.

### 2.5 Post passes: tables and graph

Both run later over a results file and never change the record. Tables try
tab-delimited rows, word-box geometry, then whitespace columns, and refuse a
band whose columns do not line up. The graph joins organisations only on
checksummed IDs such as GSTIN or PAN, never on names (README.md). The demo may
not show these passes.

## 3. The contract

`baseline/schema.py` defines `DocumentRecord`. It holds:

- `doc_id`: a hash of the file contents, so the same file always gets the same
  id (README.md);
- `text`: the full text, the pages, the extraction method, the per-page
  methods and the confidence;
- `classification`: label, confidence, all scores, method;
- `metadata`: the list of fields described above;
- `tags`;
- `diagnostics`: page count, scanned flag, per-stage timings in ms, errors.

**Why offsets and a page on every value.** A reviewer can jump to the value.
The harness can score whether the system
pointed at the right place (span scoring), not only whether the value is right.
A pipeline that returns values without locations scores zero on span F1
(cascade v2 design spec). Offsets also block invented values: if a value is
not in the text, there is no span to point at.

**Why the eval harness never imports the pipeline.** `baseline/evaluate/`
imports nothing from the rest of `baseline` except logging, and keeps its own
JSONL reader (PROJECT_STATE.md, "Code layout"). It reads any JSONL that matches
the schema. So every pipeline is scored by identical code, and none can pass by
sharing a bug with its scorer.

## 4. How it is measured

**The harness** (baseline/evaluate/scoring.py) reports these numbers for
classification: coverage (the share of documents answered), accuracy when
answered, macro F1, and coverage at a fixed error rate. A system that may say
`unknown` is judged on how much it answers while it stays under an error budget
(README.md). For fields it reports value F1 and span F1. It also reports
`validated_precision`. Every field marked validated claims proof, so this
number must be exactly 1.0 (README.md).

**The gold sets.** There is a labelled set from the client corpus for
classification, a small reviewed gold for fields, and a public set. The public
set is SROIE: 361 scanned receipts from ICDAR 2019, under CC-BY-4.0. Nobody
here tuned on it, so it gives the honest number (PROJECT_STATE.md, "Measured on
public data").

**Key numbers, latest** (all from docs/parts/audit.md, 2026-11-04):

| Metric | Value |
| --- | --- |
| Classification macro F1, 112 docs | .765 |
| Classification coverage | .911, precision 1.0 kept |
| `total_amount` F1, SROIE text, 361 docs | .717 |
| `total_amount` F1, reviewed gold, 20 docs | .500 |
| `vendor_name` F1, SROIE text | .590 (P .60 / R .58) |
| OCR, 8-page scanned PDF, 1 worker | 22.3 s to 8.1 s, output identical |

The macro F1 and coverage come from the second audit pass. Before the audit
they were .664 and .857 (docs/parts/audit.md).

**What the numbers do not cover.**

- The SROIE field numbers above use the organisers' own transcribed text. That
  takes OCR out of the measurement. Our own OCR costs about a third of the
  field scores (PROJECT_STATE.md, "Measured on public data").
- On the 40 labelled image files, classification accuracy is 0.525, against
  0.972 on the PDFs (PROJECT_STATE.md, "Known gaps" 1). **Scanned-image
  accuracy is the critical path.** Nothing else moves the overall number as
  much.
- Metadata, tagging, tables and the graph have little or no gold from the real
  corpus yet (PROJECT_STATE.md, "Known gaps" 2).
- Span scoring rests on 11 documents, at micro F1 0.2105. That is too few to
  trust the second decimal (PROJECT_STATE.md, "Known gaps" 11).

## 5. Cascade v2

Cascade v2 is the successor. The code is in `cascade/`, beside `baseline/`.

1. The baseline runs first. Its record is the fallback.
2. Cascade renders page images: 150 DPI, long edge 1600 px, first 4 pages
   (PROJECT_STATE.md, "Cascade v2").
3. A local vision model reads the images. It answers `{class, fields}` under a
   JSON schema built from `classes.yaml` and `fields.yaml`.
4. **Grounding.** Every value must be found in the canonical text: exactly,
   then ignoring whitespace and case, then through the field's normalizer. The
   emitted value is always the document's own text, with its offsets. A value
   that cannot be grounded is dropped. This is the guard against invented
   values.
5. **Rules and checksums are a hard prior.** A rule at the short-circuit level
   decides the class. A checksum-proven value is never removed. A rule value
   that the model did not confirm is dropped and recorded as `rule_only`.
6. Any failure after the baseline record falls back to the baseline answer. No
   document is lost.
7. Cascade writes the record plus a provenance sidecar. The sidecar records the
   model's raw answer and what happened to every value (README.md).

**What it needs to run.** A GPU server that serves an OpenAI-compatible model
endpoint (vLLM or Ollama). `docs/client_poc_setup.md` gives the setup and a
model choice by GPU memory.

**Status.** Cascade v2 has run on a real model once, on 2026-10-22, on the
developer laptop: an RTX 3070 with 8 GB, Ollama serving `qwen3.5:9b`
(Apache-2.0). Thinking was turned off through `vlm.extra_body` (commit
e4d6384), and Ollama ran with `OLLAMA_CONTEXT_LENGTH=16384`. On 20
human-reviewed documents, baseline to cascade (session notes, 2026-10-22; not
yet in a checked-in report):

| Metric | Baseline | Cascade v2 |
| --- | --- | --- |
| Classification macro F1 | 0.758 | 0.953 |
| Classification coverage | 0.70 | 1.00 |
| Metadata micro F1 | 0.488 | 0.440 |

The metadata drop is not yet investigated. Cascade v2 has not yet run on the
client's GPU server.

## 6. Safety and ops

- **Model manifest:** a sha256 check before every model load
  (docs/parts/audit.md).
- **Network:** the baseline makes none. The cascade client talks only to
  loopback or `vlm.allowed_hosts`, follows no redirects, ignores proxies, and
  caps a response at 8 MiB (PROJECT_STATE.md; docs/parts/audit.md).
- **Locked dependencies:** `requirements.lock`, 26 pins (docs/parts/audit.md).
- **Documents stay local:** cascade sends pages only to the configured model
  server on the client's hardware (cascade v2 design spec).
- **Deterministic:** same input, same bytes, with any worker count; timings
  excepted. A broken file never stops a batch (README.md).

## 7. Running the demo

```bash
python -m demo
```

Open http://127.0.0.1:8765. Pick a file from `corpus/` and run it. The page
shows the class, the fields, the tags, and the stage timings.

`corpus/` holds synthetic documents made by `tests/make_corpus.py`. It holds no
client data. Three suggested files:

| File | What it shows |
| --- | --- |
| `invoice_native.pdf` | The native route. The text layer is read directly, so extraction is fast. Look for the class method and the offsets on each field. |
| `receipt_scanned.pdf` | The OCR route. The page has no text layer, so Tesseract reads it. Compare the `extract` timing and the extraction confidence with the native file. |
| `salary_slip.docx` | The DOCX route. Text is read directly. A DOCX has no page breaks, so it is one canonical page (PROJECT_STATE.md, "Known gaps" 7). |

Optional extras: `invoice_hybrid.pdf` shows a native page and an OCR page in
the same document (README.md). `broken.pdf` is a damaged file. Use it to show
how the system reports a failure. Try it before the demo.

The Baseline / Cascade toggle picks the pipeline. Cascade needs Ollama running.
Run `ollama pull qwen3.5:9b` once. Then run
`ollama create qwen3.5:9b-16k -f demo/Modelfile` once. The default 4096-token
context rejects a two-page invoice, so this model sets 16384 tokens. As an
alternative, run `set OLLAMA_CONTEXT_LENGTH=16384` and then `ollama serve`.
The status pill shows whether the server answers. Set `DOCC_VLM_BASE_URL` and
`DOCC_VLM_MODEL` to use another server or model. The default is
`http://127.0.0.1:11434/v1` and `qwen3.5:9b-16k`.

## 8. Questions you may get

- **Why abstain instead of guess?** A wrong label flows downstream as truth. An
  `unknown` goes to a person. We keep precision at 1.0 and grow coverage.
- **Why not an LLM for everything?** The baseline sets the floor, so we know
  what a model adds. Cascade v2 does use a vision model, but it grounds every
  value in the text and keeps checksums as a hard prior.
- **What happens on a bad scan?** OCR confidence drops, IDs fail their
  checksums and are not emitted, and the class may be `unknown`. The record
  still exists and says what happened. Scan quality is the main limit today.
- **How do you add a class?** Edit `config/classes.yaml`: a name and its
  markers. Fields and tags are YAML edits too. No Python changes (README.md).
- **How do you know a change helped?** `baseline compare` scores two runs on
  the same gold and lists regressions. It can fail a CI run on a regression
  (README.md).
