# How it works

A walkthrough for the live demo. Each number names its source.

## 1. What it is

The system writes one JSON record per document: text, type, fields and tags.
The baseline is deterministic and runs on a CPU only: regex, checksums,
TF-IDF, linear SVMs and Tesseract OCR, no LLM calls (README.md). It sets the
floor and freezes the output contract. In cascade v2, a local vision model
reads each page. It writes the same record, and the same harness scores it
(PROJECT_STATE.md).

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
- **How it fails:** a poor scan gives poor text, and every later stage
  inherits it. Pages below .6 OCR confidence get an orientation check and
  are rotated if needed (commit a5c5464).
- **In the demo:** the text route, the per-page methods, the confidence.

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
  These abstention thresholds did not change in rounds 1 and 2
  (docs/parts/round2-classification.md, "Retrain").
- **New rules** (rounds 1 and 2):
  - a receipt titled "tax invoice" is a receipt when it prints a settlement
    line and no due date, bill-to or remit line (commit 67f5e59);
  - a GST registration line, a rounding line, or standalone CASH and CHANGE
    lines also mark a receipt. Each is *gated*: it counts only when the
    document has no due date, payment terms, bill-to or remit line. A
    tax_form marker ("tax statement") was added too (5679e67, f957a17;
    docs/parts/rounds.md).
- **The model** is a TF-IDF SVM retrained on 111 labelled client documents
  under 5-fold cross-validation. The old model had 42 synthetic samples and
  7 classes (docs/parts/round2-classification.md, "Retrain"). The old model
  is kept in `models/previous/`.
- **How it fails:** a document with none of the expected words gets `unknown`,
  by design. The model knows only the 8 classes in the client set. Rules
  carry the rest of the 17 classes (PROJECT_STATE.md). An Indian invoice that
  prints "GST No." and no due date or bill-to can look like a receipt
  (PLAN-round2.md, R2-1 risk).
- **In the demo:** the label, the confidence and the method.

### 2.3 Metadata fields

- **In:** the canonical text, the class, and the word boxes.
- **Out:** fields, each with value, normalized value, page, `char_start`,
  `char_end`, confidence, method and `validated` (baseline/schema.py).
- **How:** `config/fields.yaml` defines each field with one or more strategies:
  - **regex** for values with a fixed shape;
  - **anchor**: find a label word, then look for the value near it;
  - **positional**: a fixed spot on a known template. None registered yet.

  Example: `total_amount` looks up to 80 characters after a label such as
  "grand total". It refuses a label whose line names another quantity, such
  as "sub total" (config/fields.yaml).

  Validators check IDs with real checksums: PAN, Aadhaar, GSTIN, IFSC and card
  numbers. A value that fails its checksum is not emitted (README.md).
- **Offsets:** slice the canonical text with a field's offsets and you get the
  value back. A field whose offsets do not slice back is dropped
  (config/pipeline.yaml).
- **How it fails:** an anchor field fails when the document prints no label.
  A receipt prints subtotal, rounding, total, cash and change, and the anchor
  can pick the wrong one (PROJECT_STATE.md, "Measured on public data").
- **In the demo:** the field table, with offsets and page for each value. Note
  which values show `validated: true`.

### 2.4 Tagging

- **In:** the text, the fields, and facts such as page count and whether the
  document is scanned.
- **Out:** a list of tags, each with a confidence and a source method.
- **How:** a document has one type but many tags. Tags come from keywords, a
  model, or the metadata. Each tag records its source (README.md).
- **In the demo:** the tag list, with the method next to each tag.

### 2.5 Post passes: tables and graph

Both run later over a results file and never change the record. The graph
joins organisations only on checksummed IDs, never on names (README.md). The
demo does not show these passes.

## 3. The contract

`baseline/schema.py` defines `DocumentRecord`. It holds:

`doc_id` (a hash of the file contents), `text`, `classification`,
`metadata`, `tags`, and `diagnostics` (page count, scanned flag, timings,
errors) (README.md).

**Why offsets and a page on every value.** A reviewer can jump to the value.
The harness can score whether the system pointed at the right place (span
scoring). Offsets also block invented values: if a value is not in the text,
there is no span to point at.

**Why the eval harness never imports the pipeline.** `baseline/evaluate/`
keeps its own JSONL reader and imports nothing else from `baseline` except
logging (PROJECT_STATE.md, "Code layout"). Every pipeline is scored by
identical code, and none can pass by sharing a bug with its scorer.

## 4. How it is measured

**The harness** (baseline/evaluate/scoring.py) reports coverage (the share
of documents answered), accuracy when answered, and macro F1 for
classification. For fields it reports value F1, span F1 and
`validated_precision`, which must be exactly 1.0 (README.md).

**The gold sets.** There is a labelled set from the client corpus for
classification, a small reviewed gold for fields, and a public set. The public
set is SROIE: 361 scanned receipts from ICDAR 2019, under CC-BY-4.0
(PROJECT_STATE.md). The round 1 and 2 receipt rules were built from SROIE
misses, so SROIE classification is no longer an untuned number
(PLAN-round1.md, R1-7; docs/parts/round2-classification.md).

**Key numbers, latest** (2026-11-05, after rounds 1 and 2):

| Metric | Value | Source |
| --- | --- | --- |
| SROIE classification accuracy, 361 receipts | .958, coverage .997 (was .033) | docs/parts/rounds.md |
| Client set macro F1 / coverage, 112 docs, rules | .795 / .920 | docs/parts/rounds.md |
| Client set macro F1 / coverage, 111 docs, 5-fold CV | .816 / .955 | docs/parts/round2-classification.md |
| Classification accuracy, 40 scanned images | .700 (was .525) | docs/parts/rounds.md |
| `total_amount` F1, SROIE text | .729 | docs/parts/rounds.md |
| `vendor_name` F1, SROIE text | .633 | docs/parts/rounds.md |
| Covered-field F1, SROIE text | .635 | docs/parts/rounds.md |
| `total_amount` F1, reviewed gold, 20 docs | .500 | docs/parts/audit.md |
| OCR, 8-page scanned PDF, 1 worker | 22.3 s to 8.1 s, output identical | docs/parts/audit.md |

**The gold-covered fields view.** The SROIE gold labels only a few fields.
The old score counted every other field we read, such as `invoice_number`, as
a false positive, even when the value was right. The covered view scores only
the fields the gold labels, so a field the gold never labels is not scored at
all (docs/parts/rounds.md; PLAN-round1.md, R1-3).

**What the numbers do not cover.**

- The client-set numbers are optimistic. The rules were tuned on the same
  documents. Cross-validation isolates the model, not the rules
  (docs/parts/rounds.md, "Things to know").
- The SROIE field numbers use the organisers' own transcribed text. That
  takes OCR out of the measurement. Our own OCR costs about a third of the
  field scores (PROJECT_STATE.md, "Measured on public data").
- **Scanned-image accuracy is still the critical path.** It rose from .525 to
  .700 on the 40 images, against .972 on the PDFs (docs/parts/rounds.md;
  PROJECT_STATE.md).
- Metadata, tagging, tables and the graph have little gold from the real
  corpus. Span scoring rests on 11 documents, at micro F1 0.2105
  (PROJECT_STATE.md, "Known gaps" 2 and 11).

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
7. Cascade writes the record plus a provenance sidecar: the model's raw
   answer and what happened to every value (README.md).

**What it needs to run.** A GPU server with an OpenAI-compatible endpoint
(vLLM or Ollama); see `docs/client_poc_setup.md`.

**Status.** Cascade v2 runs on the developer laptop: an RTX 3070 with 8 GB,
Ollama serving `qwen3.5:9b` (Apache-2.0), thinking off, 16384-token context.
On 20 human-reviewed documents (PLAN-cascade.md; docs/parts/rounds.md):

| Metric | Baseline | Cascade v2 |
| --- | --- | --- |
| Classification macro F1 | .811 | .953 |
| Classification coverage | .75 | 1.00 |
| Metadata F1, gold-covered fields | .560 | .595 |
| Fallbacks to the baseline | | 0 (was 2) |

Cascade now beats the baseline on both classification and metadata. The
earlier metadata "drop" was the old scoring: it counted fields the gold never
labels as false positives (PLAN-cascade.md). Run-to-run variance is about one
document in 20 on classification (docs/parts/rounds.md).

Cascade v2 has not yet run on the client's GPU server.

For how cascade is built, read docs/How_Cascade_Works.md.

## 6. Safety and ops

- **Model manifest:** a sha256 check before every model load
  (docs/parts/audit.md). `models/*.joblib` are not in git. Ship `models/`
  with the committed manifest, or `baseline run` refuses to load
  (docs/parts/rounds.md).
- **Review order:** new workbook exports have a REVIEW_PRIORITY column
  (commit 0315524).
- **Network:** the baseline makes none. The cascade client talks only to
  loopback or `vlm.allowed_hosts` and caps a response at 8 MiB
  (docs/parts/audit.md).
- **Locked dependencies:** `requirements.lock`, 26 pins (docs/parts/audit.md).
- **Documents stay local:** cascade sends pages only to the configured model
  server on the client's hardware (cascade v2 design spec).
- **Deterministic:** same input, same bytes, with any worker count; timings
  excepted. A broken file never stops a batch (README.md).

## 7. Running the demo

```bash
python -m demo
```

Open http://127.0.0.1:8765. Pick a file from `corpus/` and run it.

`corpus/` holds synthetic documents made by `tests/make_corpus.py`. It holds no
client data. Three suggested files:

| File | What it shows |
| --- | --- |
| `invoice_native.pdf` | The native route: fast extraction. Look at the class method and the field offsets. |
| `receipt_scanned.pdf` | The OCR route. Compare the `extract` timing and confidence with the native file. |
| `salary_slip.docx` | The DOCX route. No page breaks, so one canonical page (PROJECT_STATE.md). |

Extras: `invoice_hybrid.pdf` mixes a native and an OCR page. `broken.pdf`
shows how a failure is reported.

**Cascade mode.** The Baseline / Cascade switch in the top bar picks the
pipeline. The VLM pill beside it shows `VLM · <model>` when the server
answers, and `VLM offline` when it does not. Run a file in one mode, then in
the other. The comparison strip then shows the label, the confidence and the
field count for both. In cascade mode the Model card shows the model, the
pages sent, and whether the run fell back to the baseline.

Cascade needs Ollama running. Run these once:

```bash
ollama pull qwen3.5:9b
ollama create qwen3.5:9b-16k -f demo/Modelfile
```

The default 4096-token context rejects a two-page invoice. `demo/Modelfile`
sets 16384 tokens (PLAN-demo.md). Set `DOCC_VLM_BASE_URL` and
`DOCC_VLM_MODEL` to use another server or model. The default is
`http://127.0.0.1:11434/v1` and `qwen3.5:9b-16k`. Expect about 33 s for
`invoice_native.pdf` and 13 s for `receipt_scanned.pdf` (PLAN-demo.md).

**On `receipt_scanned.pdf`** the model names the payer as `vendor_name`, at
75%, method `vlm` (PLAN-demo.md). Say so plainly. The value is grounded: it
is in the text. But it is the wrong party. Grounding stops invented values,
not misread roles. That is why the model's confidence and method are shown,
and why a person reviews the record.

## 8. Questions you may get

- **Why abstain instead of guess?** A wrong label flows downstream as truth. An
  `unknown` goes to a person. We keep precision at 1.0 and grow coverage.
- **Why not an LLM for everything?** The baseline sets the floor, so we know
  what a model adds. Cascade v2 does use a vision model, but it grounds every
  value in the text and keeps checksums as a hard prior.
- **What happens on a bad scan?** IDs fail their checksums and are not
  emitted, and the class may be `unknown`. The record still says what
  happened.
- **How do you add a class?** Edit `config/classes.yaml`. No Python changes
  (README.md).
- **How do you know a change helped?** `baseline compare` scores two runs on
  the same gold and lists regressions (README.md).
- **Why did SROIE accuracy jump from .03 to .96?** SROIE receipts are titled
  "tax invoice", so the invoice marker fired on almost all of them. A new rule
  calls such a document a receipt when it prints a settlement line and no due
  date, bill-to or remit line. Gated GST, rounding and cash markers did the
  rest (PLAN-round2.md, R2-1).
- **Is the model trained on our documents now?** Yes. It is trained on the
  labelled client documents and measured under 5-fold cross-validation, so no
  document is scored by a model that saw it. But the rules were tuned on the
  same documents, so the pipeline number is optimistic
  (docs/parts/round2-classification.md, "Retrain").
