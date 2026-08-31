# Project state

Last updated: 2026-09-01 (eval harness added)

## What this is

A baseline document processing pipeline. Cheap, deterministic, classical ML
only. Its job is to be the floor a neural approach must beat, and to lock down
an output schema so an evaluation harness can score any implementation
interchangeably.

## Status

All four components are built, wired, and tested. Evaluation harness
is in. 187 tests pass.

| Component | State | Notes |
| --- | --- | --- |
| Text extraction | Done | All three routes verified against a real engine: native, ocr, hybrid. |
| Classification | Done | Rule layer plus calibrated TF-IDF SVM, both verified. |
| Metadata | Done | All three strategies implemented. Positional has no templates registered yet. |
| Tagging | Done | Keyword, model, and derived rules all firing. Threshold tuning verified. |
| CLI | Done | run, train, tune-thresholds, profile, eval, compare all exercised. |
| Eval harness | Done | Schema driven, does not import the pipeline. |

## Verified behaviour

- 10 document synthetic corpus processes end to end, 46 metadata fields, 0 offset mismatches
- Tesseract 5.4.0 installed and driving the OCR route for real
- Byte identical output across reruns and across 1 vs 3 vs 4 workers
- Resume skips already processed doc_ids and picks up only new files
- A corrupt PDF records an error and the batch still completes
- Missing Tesseract degrades to a recorded error, not a crash
- A brand new document class defined only in YAML classifies with no code change
- OCR output is reproducible, so byte determinism holds with OCR in the loop

## The canonical text contract

This is the single most load bearing decision. Everything downstream depends
on it being stable.

- `PAGE_SEP = "\n\f\n"` defined once in `schema.py`
- `text.full` is exactly `PAGE_SEP.join(text.pages)`, asserted at build time
- Page texts are sanitised so they can never themselves contain `PAGE_SEP`
- Every `char_start` / `char_end` in `metadata` is measured against `text.full`
- Word bounding boxes in the sidecar also use canonical offsets, not page local
- `verify_offsets()` runs at write time and any mismatch is logged and recorded

Changing `PAGE_SEP` invalidates every stored offset. Treat it as frozen.

## OCR verification

Tesseract 5.4.0.20240606 (eng), installed at `C:\Program Files\Tesseract-OCR`.
The Windows installer does not add it to PATH, so the binary path is set in
`pipeline.yaml -> extraction.ocr.binary`. Null there means "look on PATH".
That path is machine specific and is the one thing to change when this repo
moves to another box.

Measured against `receipt.txt` and `receipt_scanned.pdf`, which hold identical
content down one native route and one OCR route:

| metric | value |
| --- | --- |
| character similarity | 0.9941 |
| positional word accuracy | 0.938 |
| errors | 2, both punctuation (a dropped colon, a colon read as a period) |
| extracted fields agreeing across routes | 4 of 4, exactly |
| extraction_confidence | 0.980 native vs 0.884 OCR |

The anchor machinery absorbed the punctuation loss: `Account Number.` still
matched the `account number` anchor, and the OCR'd IFSC still passed its
checksum. That is the behaviour the char ngram features exist to protect.

`invoice_hybrid.pdf` exercises the hybrid route, page 0 native and page 1
scanned. Offsets stay correct across the seam where the extraction method
itself changes, fields come out of both pages, and word boxes from both
routes coexist in one canonical offset space. This was the last untested
path in the extractor.

The profile command now computes the OCR quality proxy properly: mean page
confidence per route, and for documents carrying both routes a like for like
confidence drop. Currently 0.083 on the synthetic corpus.

## Evaluation harness

`baseline eval` and `baseline compare`, in `evaluate.py`. It imports nothing
from the pipeline. It reads JSONL and the schema, so the baseline and any
replacement are scored by identical code. That was the whole point of fixing
the schema first.

What it measures:

- Extraction: CER and WER against reference text, whitespace normalised
  because layout is not a recognition error
- Classification: macro F1, plus coverage at 1, 5 and 10 percent error, which
  is the metric that matters for a system allowed to abstain
- Metadata: per field precision and recall, scored separately by value (was
  the answer right) and by span (did it point at the right place)
- Tagging: per tag precision and recall, micro and macro

`validated_precision` is the alarm. Any field marked validated claims its
format proved it, so that number must be exactly 1.0. It currently is.

`compare` diffs two runs against the same gold and lists regressions.
`--fail-on-regression` exits non zero so it can gate CI.

Gold for the synthetic corpus lives in `tests/make_gold.py`, hand written
from the templates. It records what a careful annotator would mark, not what
the baseline produces. Gold carries values but no offsets, so span scoring
stays dark until there is real annotation.

### Bugs the harness found immediately

Worth recording, because these are exactly what an unmeasured pipeline hides.

1. **Soft validators were claiming `regex_checksum`.** A passing length check
   reported the same method as a verified GSTIN, so nothing downstream could
   tell a proof from a guess. Now only `pan`, `aadhaar`, `gstin`, `ifsc` and
   `luhn` report `regex_checksum`. Everything else reports `regex_format`
   even when it passes. This moved `validated_precision` from 0.895 to 1.0.
2. **The invoice number regex dropped its own prefix.** It grouped only the
   tail, so `INV-2024-0042` came out as `2024-0042`. Field F1 was 0.444.
3. **`total_amount` had no label for "amount paid" or "net pay"**, so it
   missed on every receipt and salary slip. Recall was 0.5.
4. **`invoice_number` had no "PO No" label**, so purchase orders returned
   nothing.

Metadata micro F1 went 0.884 to 0.970 after fixing these.

**Read that number with suspicion.** It is 10 synthetic documents whose
templates I wrote. It measures that the plumbing works, not that the system
is accurate. One known miss is left unfixed on purpose: `buyer_name` on the
purchase order returns "Globex Corporation Warehouse" where gold says
"Globex Corporation". Chasing it would be tuning to a single fake document.

## Schema decisions

The literal example in the spec was the minimum. These fields were added
because the prose asked for them:

| Field | Why |
| --- | --- |
| `filename` | Spec: "Record the original filename separately" |
| `metadata[].normalizer_version` | Spec: "Version the normalizers and record the version in the output" |
| `metadata[].currency` | Spec: "currency to decimal with a separate currency code" |
| `text.page_methods` | Spec: hybrid documents must "Record which" pages went which way |
| `schema_version` | So a harness can reject records it does not understand |

Two enums were widened for the same reason:

- `MetadataMethod` gained `regex_format`. The spec demands that a validated
  checksum match and an unvalidated format match be kept apart ("they are
  different things"), and one enum value could not express both.
- `ExtractionMethod` gained `none`, for documents that produced no text at all.

Models are `extra="forbid"`, so an unexpected key fails validation loudly.

## Component notes

### Extraction

Routes per page, never OCRs a page that already has text. A PDF can be hybrid.
The threshold is `extraction.native_char_threshold` (default 100 chars after
stripping whitespace).

OCR sits behind `ocr.OCREngine`. Tesseract ships as the default and a `null`
engine exists for corpora that need none. Register another with
`ocr.register_engine`. Nothing downstream knows which engine ran.

OCR page text is assembled from the word boxes, so OCR word offsets are exact
by construction. Native PDF words are mapped back onto the page text by a
sequential scan and get `null` offsets when they cannot be placed. On the test
corpus that mapping is 83/83.

`extraction_confidence`: native pages take a flat high score, OCR pages take
the engine's per word confidence averaged and weighted by word length. The
document score blends pages weighted by how much text each contributed.

### Classification

Cascade, in order:

1. Rule specificity >= `rule_short_circuit` (0.9) wins outright, skips the model
2. Model probability >= `model_min_confidence` (0.45) wins
3. Rule specificity >= `rule_min_confidence` (0.7), only if the model declined or is absent
4. Otherwise `unknown`, with the score distribution kept intact

Step 3 exists so a fresh install with no trained model is still useful.

Never forces a guess. On junk input it returns `unknown` and keeps the scores.

Model is word ngrams (1,2) unioned with `char_wb` ngrams (3,5), into
`CalibratedClassifierCV(LinearSVC, method="sigmoid")`. Char ngrams are there
for OCR noise. `shuffle=False` on the CV splitter so refits are reproducible.

### Metadata

Strategies run in the order listed per field, first hit wins. Candidates are
ranked validated first, then confidence, then earliest offset.

Real checksums, all unit tested:

| Field | Check |
| --- | --- |
| PAN | Format plus holder type code in position 4 |
| Aadhaar | Verhoeff, plus leading digit not 0 or 1 |
| GSTIN | State code, embedded PAN, mandatory Z, base 36 check digit |
| IFSC | Format plus mandatory zero in position 5 |
| Card | Luhn |
| Dates | Parse plus configurable year range |
| Amounts | Decimal plus range sanity |

Note: the spec's example PAN `ABCDE1234F` is rejected, correctly. `D` is not a
valid holder type code. The allowed set is configurable per field via
`validator_args.entity_types` if a corpus needs it widened.

IFSC has no published checksum, so `validated: true` there means structurally
conformant, not checksum verified.

Fields may declare `classes:` to limit which document classes they run on.
This was added after invoice date and party name fields were firing on ID
cards and contracts. Documents classified `unknown` still get every field, so
recall is not lost where it matters most.

### Tagging

Three sources merged per tag, highest confidence wins, ties broken by source
priority derived > model > keyword. Then gated on the per tag threshold.

Derived rules read the metadata output and document level facts. They are the
strongest signal because they sit on validated data. Conditions support
`field` / `present` / `validated` / `min_confidence` / `value_matches`, and
`doc` with `gt` / `lt` / `gte` / `lte` / `eq`.

`tune-thresholds` fits each tag's F1 optimal cutoff and writes it back into
`tags.yaml`. Ties go to the lower threshold, which favours recall on rare tags.

## Determinism

Everything is byte reproducible except per stage timings, which by definition
vary. Two knobs, both tested:

- Default (`deterministic_timings: false`): real timings recorded, output is
  byte identical in every other field
- `deterministic_timings: true`: timings zeroed, output byte identical

Ordering is stable because input paths are sorted and the worker pool uses
`imap`, which preserves input order. Bbox sidecars pin the gzip mtime.

## Known gaps

1. **There is still no real data.** Every number in this file is from 10
   synthetic documents. The harness is validated, the accuracy is not.
2. **Models are trained on synthetic data only.** 42 generated samples across
   7 classes. Calibrated probabilities are conservative as a result, which is
   why a real bank statement scored 0.28 and fell to `unknown`.
   `model_min_confidence` needs retuning on real data. This is now the
   biggest open risk.
3. **No positional templates registered.** `templates: []`. The strategy is
   implemented and reads from bboxes, it just has nothing to match yet.
4. **DOCX has no page boundaries.** Treated as one canonical page, since
   pagination needs a renderer.
5. **Anchor confidences cluster around 0.95.** Fine for ranking, not yet
   calibrated against real accuracy.
6. **OCR is only verified on clean synthetic scans.** Real scans bring skew,
   noise, and multi column layouts. Expect the 0.99 similarity to fall.
7. **Only the `eng` language pack is installed.** Add more via the Tesseract
   installer if the corpus needs them, then set `extraction.ocr.lang`.
8. **Span level scoring has never run.** Gold has no character offsets
   because synthetic gold cannot honestly provide them. This matters: span
   accuracy is what a layout model would be judged on.
9. **No table or line item extraction at all.** Completely absent, and the
   largest functional gap in the system.

## Config reference

`pipeline.yaml` holds engine settings only, no domain vocabulary.
`classes.yaml`, `fields.yaml` and `tags.yaml` hold all the vocabulary. No
class, field, tag or pattern is hardcoded in any module, and
`test_a_new_document_class_needs_only_a_yaml_edit` enforces that.
