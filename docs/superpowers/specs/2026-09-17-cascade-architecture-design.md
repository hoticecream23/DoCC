# Cascade: architecture design

Date: 2026-09-17
Status: draft, awaiting review

## What this is

`cascade` is our own document processing approach, built beside `baseline/`.
The baseline exists to be the floor every approach is scored against. Cascade
is the approach we intend to compete with.

The name describes the design: every document enters the cheapest stage that
can handle it, and only what a stage cannot resolve with confidence moves on to
a more expensive one.

## Goals

1. Beat the baseline on the evaluation harness, on the same output contract, on
   a test set it has never seen.
2. Run on CPU with a small footprint. The deployment server is not this laptop,
   so every cost is measured in a form that transfers to other hardware.
3. Get better as documents are processed and reviewed, through a knowledge
   graph and a review loop, without the graph learning its own mistakes.
4. Abstain rather than guess. A wrong answer delivered with confidence costs
   more than a document sent to a human.

## Constraints

- **Local models only.** No document content leaves the machine.
- **CPU first.** A GPU exists on the development laptop (RTX 3070 Ti mobile) but
  the design must not depend on one. GPU paths are later, optional, and decided
  by measured need.
- **Permissive licences only.** Every model's licence is checked before it is
  adopted. Non commercial weights, which some popular layout models ship with,
  are excluded.
- **The output contract holds.** Cascade writes `DocumentRecord` JSONL from
  `baseline/schema.py`, so `baseline eval` and `baseline compare` score it
  without modification. Every metadata value keeps character offsets that slice
  back to the canonical text.

## Non goals

- Arabic extraction. Every Yaumi field is also printed in English. Revisit only
  if a required field is Arabic only.
- A generative model in the main path. See "Deferred: VLM reader".
- Replacing the baseline. It stays as the fixed yardstick.

## Pipeline

```
files
  -> 1. ingest router        native text, quality gate, OCR; word boxes
  -> 2. document grouper      join files that are pages of one document
  -> 3. deterministic layer   checksummed IDs (proven), formats (candidates)
  -> 4. classify              rules -> small encoder + linear head
  -> 5. extract fields        rules and anchors propose -> LiLT tags and scores
  -> 6. knowledge graph       look up, check consistency, adjust confidence
  -> 7. calibrate and decide  accept / abstain / review
  -> DocumentRecord JSONL + provenance sidecar
review workbook -> gold -> retrain stages 4 and 5
```

Each stage reads the record so far and returns it with its own additions. Each
stage can abstain. A stage never overwrites a value a stronger stage proved:
a checksummed ID is never replaced by a model output.

### 1. Ingest router

Decides how text is read, per page.

- A page with a text layer is read natively, then scored by the text quality
  gate (`baseline/textquality.py`). If the score marks it corrupt, the page is
  OCR'd and the cleaner reading is kept. This is not optional on this corpus:
  all 25 client invoice PDFs carry a corrupt embedded OCR layer.
- A page with no text layer goes to OCR, with the pink paper preparation that
  already exists in `baseline/extract.py`.
- The OCR engine is pluggable behind `baseline/ocr.py`'s engine interface.
  Which engine is the default is decided by experiment 1a.
- Output: canonical text, pages, word boxes with character offsets.

Reuses: `baseline/extract.py`, `baseline/textquality.py`, `baseline/scan.py`.

### 2. Document grouper

Joins files that are pages of one document before anything classifies them.

Yaumi invoices are often two image files, and the second page carries no
header, so on its own it always abstains. A group is formed when files share a
document number (for Yaumi, `CASH # <n>` or `CREDIT # <n>`, which also appears
in the file name) and nothing contradicts the join. A group is processed as
one multi page document.

A file joins a group only on an exact identifier match. No fuzzy joining.

Open point for the spec review: the record contract has one record per file.
Two options, to be settled in the plan:
(a) keep one record per file, and have the grouper only share context, so page
two inherits the class and header fields of page one;
(b) emit one record per group. Option (a) needs no contract change and is the
default.

### 3. Deterministic layer

Two tiers of evidence, kept apart everywhere downstream.

- **Proven:** values whose checksum passes. PAN, Aadhaar (Verhoeff), GSTIN,
  IFSC, card numbers (Luhn). Confidence 1.0. No later stage may contradict them.
- **Candidates:** values that match a format only. Dates, amounts, invoice
  numbers, case numbers. These feed stage 5 as candidates and anchors. They
  are never reported as proven.

A checksummed field whose checksum fails is not emitted at all. On the client
corpus every failing Aadhaar and card candidate was OCR noise.

Reuses: `baseline/metadata/validators.py`, `normalizers.py`, and the pattern
and anchor strategies in `fields.py`.

### 4. Classify

A cascade inside the stage.

1. **Rules.** The baseline's marker rules. On the labelled PDFs they answer 71
   of 72 and are never wrong when they answer. A rule at or above the short
   circuit specificity ends the stage.
2. **Small encoder plus linear head.** A small multilingual sentence encoder,
   tens to about a hundred million parameters, embeds the first page. A
   logistic regression over the embeddings predicts the class. It learns from
   few examples, which a fine-tuned model cannot.
3. **Class descriptions for classes with no examples.** Eight of the 17 classes
   have zero real documents. Each class gets a short written description, and a
   document is compared to those by embedding similarity. This is the only
   stage that can say anything about those classes, and it is allowed only to
   propose, never to accept on its own.

Below the calibrated threshold, the stage abstains with label `unknown`.

Whether the encoder earns its place over TF-IDF is experiment 1b.

### 5. Extract fields

Rules propose, the model decides.

- Rules and anchors from stage 3 produce candidate values with offsets.
- LiLT, a layout aware token classifier with an MIT licence, reads text plus
  word boxes and tags tokens with field labels. Paired with a multilingual text
  encoder, so the Arabic on Yaumi pages is noise it has seen rather than noise
  that breaks it.
- A field value is emitted from a LiLT span or from a rule candidate that LiLT
  agrees with. Both carry offsets into the canonical text, so the contract
  holds and span scoring works.
- Long documents are read with overlapping 512 token windows.
- Proven values from stage 3 are copied through untouched.

Training data comes from three sources:
1. **Weak labels from reviewed values.** Each reviewed field value is located in
   the document's word boxes and turned into token labels. A value that appears
   more than once, or cannot be found because OCR mangled it, is skipped rather
   than guessed.
2. **More labelled documents** from the team, chosen to cover gaps.
3. **Synthetic documents** from templates, for pretraining only. Synthetic data
   is never in any test set.

Whether LiLT is fast enough on CPU and learns anything from weak labels is
experiment 1c.

### 6. Knowledge graph

The graph is memory across documents. It is consulted during a run, not built
afterwards.

**Entities, per domain.**
- Invoices: organisation (keyed on a proven tax ID, PAN or GSTIN; a UAE TRN
  has no checksum, so it keys an organisation only once it is established
  by repetition under the write rules below),
  customer account (keyed on vendor plus customer number), product (keyed on
  vendor plus SKU).
- Judgments: case (keyed on court plus case number), court, judge, party,
  citation.

**What it does for a document.**
- **Lookup.** A customer number already bound to a name supplies or corrects
  that name. `Z00M MARKET` next to customer `9036029` becomes `ZOOM MARKET`.
- **Consistency.** A vendor's TRN must match every earlier invoice from that
  vendor. A line item's amount must equal quantity times price plus VAT. A
  judgment cannot be dated before a case it cites.
- **Confidence.** Agreement raises a field's confidence, conflict lowers it.
  The size of each adjustment is set in calibration, not by hand.

**Write rules. These exist to stop the graph learning its own mistakes.**
- A fact enters the graph only if it is proven by checksum, confirmed by a
  human in review, or observed identically on at least 3 documents that are
  not duplicates of each other and do not contradict it anywhere.
- Every fact records its source documents and how it was established.
- Facts established by repetition can be revoked by a human correction, and the
  revocation propagates to documents that relied on them.
- Names never merge two entities that proven IDs keep apart. This is the v0
  rule from `baseline/graph.py` and it stays.

**Evaluation rule.** When scoring, the graph may contain only facts from
training documents and from documents processed earlier in the same run. It
never contains facts from test gold. Otherwise the graph hands the model the
answers.

### 7. Calibrate and decide

- Every model score is calibrated. Temperature scaling for the encoder head and
  LiLT, and a per method mapping for rule and anchor confidences, which today
  sit near 0.95 whatever their real accuracy is.
- Calibration is fitted on a calibration split, never on the test set.
- Thresholds are chosen per field and per class to meet a target error rate,
  and the harness's coverage at 1, 5 and 10 percent error is the metric that
  sets them.
- Each value ends as accepted, abstained, or flagged for review.

Until there are enough reviewed documents to fit calibration, raw scores are
used and the report says so.

### Deferred: VLM reader

A quantized vision language model is not in the main path. It is too slow on
CPU for a small footprint, and a model that writes values directly cannot
provide character offsets and can invent values.

If the residual errors after phase 3 are dominated by pages that OCR cannot
read, a VLM is tested as a reader only: it transcribes the worst pages into
text, and stages 3 to 7 extract from that text as from any OCR output. Off by
default, GPU tested first.

## Output contract

- `DocumentRecord` JSONL, unchanged in shape.
- New method values, which is an additive schema change to 1.2.0:
  `ClassifyMethod.ENCODER`, `ClassifyMethod.DESCRIPTION`,
  `MetadataMethod.LAYOUT_MODEL`, `MetadataMethod.GRAPH`.
- **Provenance sidecar**, `<output>.provenance`, one line per document: which
  stage produced each value, what each stage cost, what the graph looked up and
  changed, and why a value was abstained. Kept out of the record so the record
  contract stays frozen for every approach, the same way word boxes live in
  the `.bboxes` sidecar.

## Evaluation protocol

This is phase 0 and it comes before any stage is built, because every later
decision depends on it.

**Data.**
- `Client_Documents`: 227 records, 208 distinct. Deduplicate by `doc_id`
  before splitting.
- `Classification_Doc`: 112 labelled records. It was used to write the rules,
  so it is training and calibration data, never test data.

**Splits.** One vendor is 140 of the 208 client documents, so a random split
puts Yaumi on both sides and measures memorisation. Two test views instead:

1. **Seen layouts.** Within each layout group, 30 percent of documents held out,
   split by document number so both pages of one invoice land on the same side.
2. **Unseen layouts.** Whole layout groups held out: at least one invoice
   vendor batch other than Yaumi, and a share of the court judgments.

Layout groups are assigned from the source folder and, for invoices, the
proven vendor tax ID. The assignment is written to a file and frozen, and the
split is fixed by that file, not by a random seed.

The test split is frozen once Karishma's review returns. Until then, a
provisional split is used and every number reported from it is marked
provisional.

**Reference text.** About 15 pages are transcribed by hand: Yaumi, NG L.L.C.,
SIDBI and Jindal. They give character error rate for OCR, which field scores
alone cannot.

**Metrics.** The harness's own, reported overall and per layout group:
classification macro F1 and coverage at error rates, metadata value and span
F1 per field, `validated_precision`, tagging, tables.

**Cost.** Measured per stage and reported beside every quality number:
- share of documents each stage handles, which does not depend on hardware;
- CPU seconds per page per stage, median and 95th percentile;
- peak resident memory per stage;
- model file sizes.

Laptop wall clock time is recorded but not used for decisions.

**Gate.** A change is kept only if `baseline compare --fail-on-regression`
passes on both test views and its cost stays inside budget.

Proposed budget, to confirm at review: median at most 2 CPU seconds per page
for a document the rules resolve, and 95th percentile at most 20 CPU seconds
per page for the full cascade.

## Phases

| Phase | Work | Exit condition |
| --- | --- | --- |
| 0. Measurement | Deduplication, frozen layout group splits, provenance and cost capture, weak token labels, 15 transcribed pages | Split file committed; baseline scored on both test views |
| 1. Experiments | 1a to 1d below, as throwaway code | A written keep or drop decision for each, with numbers |
| 2. Skeleton | `cascade/` with the staged runner, grouper, provenance, calibration hooks, built only from baseline parts | Scores identical to the baseline on both test views |
| 3. Add stages | Kept experiments become stages, one at a time | Each passes the gate |
| 4. Graph and review loop | Graph live in the run; reviewed corrections feed retraining | Scores rise across a second review round with no gate failure |
| Later | VLM reader or GPU paths | Only if residual error analysis calls for them |

### Experiments

Each experiment is throwaway code with one question and a decision rule set
before it runs.

**1a. OCR engine.** Tesseract (current) against PaddleOCR and docTR, CPU
builds, on the 15 transcribed pages and on field value F1.
Keep a new engine if character error rate falls by at least a fifth relative
to Tesseract and field value F1 does not drop, within the cost budget.

**1b. Classifier.** TF-IDF plus linear SVM against a small encoder plus logistic
regression, behind the same rules, plus class description matching for the
zero example classes.
Keep the encoder if coverage at 5 percent error rises on either test view with
no fall on the other, within the cost budget.

**1c. LiLT.** CPU inference cost per page, and a short fine tune on weak labels.
Keep if metadata value F1 rises over rules and anchors alone on the seen
layout view, and inference fits the budget. If it learns nothing from weak
labels at the current data volume, record the volume at which to retry.

**1d. Graph consistency.** On the Yaumi documents, count the field errors that
lookup and consistency checks would catch or correct, against reviewed gold.
Keep if the values it corrects or flags as wrong outnumber the correct values
it changes or flags by at least 3 to 1.

## Package layout

```
cascade/
  runner.py        runs the stages in order, records provenance and cost
  group.py         document grouper
  classify.py      rules, encoder head, class descriptions
  fields.py        candidates plus LiLT
  graph/           store, write rules, lookup, consistency checks
  calibrate.py     fitting and applying calibration
  cli.py           cascade run, cascade train, cascade calibrate
config/cascade/    stage config, kept apart from the baseline's config
```

`cascade` imports proven parts of `baseline` rather than copying them:
extraction, text quality, scanning, validators, normalizers, rules, schema. It
never modifies them. If a baseline part needs a change for cascade, that change
is made in `baseline` and must leave the baseline's own outputs identical.

## Risks

- **Gold volume is the binding constraint**, more than CPU. With the review
  pending, phases 1c and 4 may have to wait on labels.
- **One vendor dominates.** Numbers on the seen layout view mostly describe
  Yaumi. The unseen layout view is the honest one for new clients.
- **Weak labels inherit OCR errors.** A value OCR mangled cannot be located, so
  the weak label set is biased toward cleanly read documents.
- **Graph poisoning.** Mitigated by the write rules, and tested directly: a
  deliberately wrong fact injected into the graph must be revoked by one human
  correction.
- **Licences.** A model that is best on quality may be unusable commercially.
  Checked before any experiment adopts it.

## Open questions for review

1. Document grouping: one record per file sharing context (default), or one
   record per group?
2. The cost budget above.
3. Who transcribes the 15 reference pages, and when.
4. Which layout groups are held out whole for the unseen layout view.
