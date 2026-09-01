# Project state

Last updated: 2026-09-01 (eval harness, knowledge graph v0)

## What this is

A baseline document processing pipeline. Cheap, deterministic, classical ML
only. Its job is to be the floor a neural approach must beat, and to lock down
an output schema so an evaluation harness can score any implementation
interchangeably.

## Status

All four components are built, wired, and tested. Evaluation harness
is in, plus a knowledge graph v0 spike. 210 tests pass.

| Component | State | Notes |
| --- | --- | --- |
| Text extraction | Done | All three routes verified against a real engine: native, ocr, hybrid. |
| Classification | Done | Rule layer plus calibrated TF-IDF SVM, both verified. |
| Metadata | Done | All three strategies implemented. Positional has no templates registered yet. |
| Tagging | Done | Keyword, model, and derived rules all firing. Threshold tuning verified. |
| CLI | Done | run, train, tune-thresholds, profile, eval, compare, graph all exercised. |
| Eval harness | Done | Schema driven, does not import the pipeline. |
| Knowledge graph | v0 spike | Verified identifiers only. Deliberately narrow. |

## Verified behaviour

- 10 document synthetic corpus processes end to end, 46 metadata fields, 0 offset mismatches
- Tesseract 5.4.0 installed and driving the OCR route for real
- Byte identical output across reruns and across 1 vs 3 vs 4 workers
- Resume skips already processed doc_ids and picks up only new files
- A corrupt PDF records an error and the batch still completes
- Missing Tesseract degrades to a recorded error, not a crash
- A brand new document class defined only in YAML classifies with no code change
- OCR output is reproducible, so byte determinism holds with OCR in the loop
- Baseline scored against gold: metadata micro F1 0.970, classification macro
  F1 1.0 at 0.9 coverage, tagging micro F1 1.0, validated precision 1.0
- Graph builds 2 organisations and 2 accounts from verified identifiers only,
  and finds the receipt to invoice chains and both duplicate pairs

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

## Knowledge graph v0

`baseline graph`, in `graph.py`, config in `config/graph.yaml`. A separate
pass over the results file. It never touches the document schema, so the
record contract stays frozen and the graph can be rebuilt at any time.

The single rule that defines v0: **only fields whose format proves them may
create or merge an entity.** Names are recorded as attributes but never used
to join. Fuzzy name matching is where a graph fills with plausible looking
wrong edges, and once they are in nobody can tell which ones to trust.

That constraint pays for itself. Characters 3 to 12 of a GSTIN are the holder
PAN, so a document carrying only a GSTIN and one carrying only a PAN resolve
to the same organisation with zero guessing. Both are verified, so the join
is exact.

Nodes: Document, Organization (keyed on PAN), Account (keyed on IFSC plus
number, because the same number at two banks is two accounts).
Edges: mentions_org, issued_by, billed_to, pays_to, held_by, references,
duplicate_of.

Queries it answers: organisation exposure, document reference chains,
duplicate candidates, orphan references.

### Three wrong edge classes the spike caught

All three were the graph confidently asserting something false, which is
exactly the failure mode that makes graphs untrustworthy.

1. **One identity collected every role.** A document naming a vendor and a
   buyer but carrying one verified GSTIN attached both names to that one
   organisation, so the seller node held the buyer name. Fixed with a greedy
   matching where an identity holds at most one role per document.
2. **A GSTIN and a PAN for the same company counted as two identities**, which
   let one party claim two roles anyway. Identities are now collapsed to their
   canonical PAN before matching.
3. **A receipt quoting an invoice number was called a duplicate of it.**
   Duplicates are now scoped within a class, which correctly splits the two
   invoice variants and the two receipt variants into separate groups.

### Where it abstains

Character proximity is weak evidence for who owns an identifier. On the
purchase order, "Ship To" sits 11 characters nearer the seller GSTIN than
"Vendor Name" does, purely by layout. So a winner must beat the runner up by
`role_min_margin` characters, currently 60. The purchase order draws no role
edge at all. The organisation is still recorded as mentioned, we simply do
not claim to know its role.

That is the right trade for v0. A missing edge is recoverable, a wrong one
poisons every query built on it.

### What it is not

Precision oriented and therefore thin. Organisations without a verified tax
ID do not exist in this graph at all: Globex appears in five documents and
has no node, because nothing in the corpus proves its identity. Fixing that
needs either better extraction or name resolution, and name resolution is
explicitly out of scope for v0.

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
10. **The graph only sees organisations that carry a verified tax ID.**
    Everyone else is invisible to it. That is deliberate for v0, but it means
    graph coverage is capped by metadata quality, which is capped by having
    no real data.

## Config reference

`pipeline.yaml` holds engine settings only, no domain vocabulary.
`classes.yaml`, `fields.yaml`, `tags.yaml` and `graph.yaml` hold all the
vocabulary and rules. No class, field, tag or pattern is hardcoded in any
module, and `test_a_new_document_class_needs_only_a_yaml_edit` enforces that.

| File | Holds |
| --- | --- |
| `pipeline.yaml` | thresholds, OCR engine and binary, worker settings |
| `classes.yaml` | document taxonomy and classification rule patterns |
| `fields.yaml` | field definitions, patterns, anchors, validators, normalisers |
| `tags.yaml` | tag vocabulary, patterns, thresholds, derived rules |
| `graph.yaml` | which fields may key an entity, role and duplicate rules |

## Repository state

Four commits on `main`. Working tree clean. Generated artefacts are gitignored
and rebuilt by:

```
python tests/make_corpus.py corpus
cd tests && python make_annotations.py ../annotations.jsonl && python make_gold.py ../gold.jsonl && cd ..
python -m baseline run --input corpus --output results.jsonl --workers 4
python -m baseline eval --gold gold.jsonl --pred results.jsonl
python -m baseline graph --input results.jsonl --report graph.md
```
