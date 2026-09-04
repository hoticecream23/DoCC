# Project state

Last updated: 2026-09-04 (taxonomy reconciled; text quality gate; rule layer
measured at 0.986 on the labelled PDFs)

## What this is

A baseline document processing pipeline. Cheap, deterministic, classical ML
only. Its job is to be the floor a neural approach must beat, and to lock down
an output schema so an evaluation harness can score any implementation
interchangeably.

## Status

All four components are built, wired, and tested. Evaluation harness
is in, plus a knowledge graph v0 spike and table extraction v0. The real
corpus has arrived, the taxonomy is reconciled against it, and the rule
layer is measured on it. 264 tests pass.

| Component | State | Notes |
| --- | --- | --- |
| Text extraction | Done | All three routes verified against a real engine: native, ocr, hybrid. |
| Classification | Done | Rule layer plus calibrated TF-IDF SVM, both verified. |
| Metadata | Done | All three strategies implemented. Positional has no templates registered yet. |
| Tagging | Done | Keyword, model, and derived rules all firing. Threshold tuning verified. |
| CLI | Done | run, train, tune-thresholds, profile, eval, compare, graph, tables all exercised. |
| Eval harness | Done | Schema driven, does not import the pipeline. Scores tables too. |
| Knowledge graph | v0 spike | Verified identifiers only. Deliberately narrow. |
| Table extraction | v0 | Two strategies over boxes and over text. Abstains rather than guess. |

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
- Tables come out of 4 of the 5 documents that have one, with cell precision
  1.0 and adjacency precision 1.0: not one wrong cell, not one wrong neighbour
- Every one of the 49 extracted cells slices the canonical text back exactly

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
- Tables: cell precision and recall over content, and adjacency precision and
  recall over structure. Scored only when the gold row carries a `tables` key
  and `--pred-tables` is passed

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

## Table extraction v0

`baseline tables`, in `tables.py`, config in `config/tables.yaml`. A separate
pass over the results file plus the bbox sidecar, on the graph's pattern. It
never touches the document record, so tables can be rebuilt from an old run
without re-extracting anything.

This closes what was the largest functional gap in the system. It is also the
first thing in the repo that reads the word boxes: the sidecar has been
captured since the first commit and until now nothing consumed it.

### Two strategies, config ordered, first hit wins

Same pattern as the metadata field strategies.

**geometry** groups words into lines by vertical overlap, splits each line
into cells wherever the horizontal gap exceeds about 1.6 space widths, then
clusters the cells into columns. The space width is estimated per page from
the 25th percentile of observed within line gaps, because those gaps are
bimodal: small ones are spaces inside a cell, large ones separate columns.
Nothing has to know the font.

Column clustering is **complete linkage** with a same row exclusion. A cell
joins a column only if it clears the overlap ratio against every cell already
there, and never if that column already holds a cell from its own row. Single
linkage chains two adjacent numeric columns into one through a header that
straddles both, which is precisely the failure this has to avoid.

**text_grid** finds the character columns that are blank on every line of a
band and treats runs of two or more as separators. It is the only signal
available for `.txt` and `.docx`, which carry no geometry whatsoever, and it
rescues documents whose alignment survived into the text but not into the
coordinates.

### Offsets

Every cell carries `char_start` / `char_end` in the same canonical space as
`metadata`. The cell text is the **slice of the canonical text**, not the
words joined back together, so the offsets verify by construction.

`TableRecord.verify_offsets()` runs at write time and mirrors
`DocumentRecord.verify_offsets()`. A cell whose words cannot all be placed, or
whose span would cross a line, gets null offsets rather than approximate ones.
All 49 cells on the corpus verify.

text_grid cells get geometry back afterwards by looking up sidecar words whose
spans sit inside the cell's span. Derived from offsets the cell already owns,
so nothing is guessed.

### The v0 rule, and where it abstains

The graph's rule again: **a wrong table is worse than a missing one.** Three
structural checks, and failing any of them refuses the band:

1. A cell that fits more than one column. Ambiguity is not resolved quietly.
2. More columns than the widest row has cells. The rows do not line up.
3. A column standing on fewer than two cells. That is a coincidence, not a
   column, and seeing one means the band is misaligned.

Every refusal is recorded in `diagnostics.abstained` with its reason and
printed in the report. An abstention is a finding, not a silence.

Two abstentions on the corpus, and both are correct:

- `bank_statement.pdf` under geometry. The corpus renders space aligned text
  in a proportional font, so the header drifts off its own data: `Debit` ends
  at x=206 while the figure under it starts at x=210, and `22000.00` overlaps
  the `Credit` column more than the `Debit` one. Geometry refuses, text_grid
  picks it up, and the result is exactly right.
- `invoice_hybrid.pdf` page 1, the scanned page, under geometry. OCR merged
  `Widget assembly` into one token and the numeric columns drift by 25pt down
  the page. The same table extracts perfectly from the native route. This is
  the honest finding: **the same content succeeds natively and is refused
  under OCR**, and it is why table recall is 0.8 and not 1.0.

### Scored

`baseline eval --pred-tables tables.jsonl`, against gold grids in
`tests/make_gold.py`.

| metric | value |
| --- | --- |
| gold tables | 5 |
| detection recall | 0.800 |
| cell precision | 1.000 |
| cell recall | 0.803 |
| adjacency precision | 1.000 |
| adjacency recall | 0.800 |

Two metrics because a table can be wrong two ways. Cell F1 asks whether the
values came out. Adjacency F1 asks whether each value ended up beside the
right neighbours, and it is the one that matters: a figure that slides one
column left is perfect content and a useless table. Adjacency is also the
standard structural metric because it does not need the predicted grid to
line up index for index with the gold one, so a missed header row does not
cascade into every later score.

**Precision is 1.0 on both.** Not one wrong cell, not one wrong neighbour, on
either strategy. Recall is 0.8 and the entire deficit is the one abstention.
That is the trade v0 is built to make.

Documents with no table carry `"tables": []` in gold rather than no key, so a
table invented out of nothing is counted against precision. Without that, four
of the ten documents would not have constrained precision at all.

Table metrics join `compare --fail-on-regression` only when both runs were
given a tables file, so a run without one is not read as a collapse to zero.

### What it is not

- No spanning cells. `row_span` and `col_span` exist in the schema and are
  always 1. A merged header cell will be read as belonging to one column.
- No ruling line detection. Columns come from whitespace and coordinates, not
  from drawn borders, so a densely ruled table with no gaps is invisible.
- No multi page table stitching. A table continuing over a page break is two
  tables.
- Header detection is a heuristic: a full first row carrying almost no digits.
  When it fails the table is emitted with an empty header rather than a wrong
  one, which is what happens on the salary slip, correctly.
- Real `.docx` tables come through `_extract_docx` as tab joined rows. A
  single tab is one character, so it never forms a two character separator and
  text_grid cannot see it. Nothing in the corpus exercises this, but real
  documents will hit it immediately. This is the first thing to fix.

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

`baseline tables` obeys the same knob and the same rule. Its per strategy
timings are the only field that varies between runs; with
`deterministic_timings: true` the output is byte identical. Column clustering
sorts cells left to right and breaks every tie explicitly, so the grid never
depends on the order words came out of the sidecar.

## Taxonomy reconciliation

Done. The client's 14 folder labels now map onto our class list in
`config/taxonomy.yaml`, and `tests/test_taxonomy.py` fails if the two drift.

The reconciliation is not the seven-new-classes job `HANDOFF.md` expected,
because the corpus does not support it. **All 46 files under the seven legal
labels are court judgments and orders downloaded from indiankanoon.org**,
named `<party>_vs_<party>_on_<date>.PDF`. Their labels describe what the firm
uses a document for on a matter, not what the document is. The corpus proves
this outright: four files sit under two labels at once, and three of those
four are different files that happen to share a name.

So one class was added, `court_document`, and all seven labels map to it as
`lossy: true`. Splitting it into seven would mean inventing markers that
cannot exist, and the graph and the workbook would then key on a distinction
nothing in the text supports.

`tax document` is also lossy: it holds tax invoices, a tax statement and study
notes on direct tax, so it is a subject area rather than a form type. It maps
to `tax_form` and some of it will correctly classify as `invoice`.

The four doubly labelled rows are recorded under `conflicts` rather than
deduplicated. All four collapse to `court_document`, so none of them can
change a class level score, but the file name is not a key over this set and
`doc_id` must stay content addressed.

### court_document markers, measured not guessed

Ten markers, all checked against the 72 text bearing PDFs in
`Classification_Doc` before being written: **46/46 court PDFs fire at or above
0.88, and 0/26 business PDFs fire any of them.**

Two of them exist for specific reasons:

- The indiankanoon download banner (`X vs Y on <date>`) is a **provenance**
  marker, not an intrinsic one. It sits at 0.8, below `rule_short_circuit`, so
  it can never carry a decision alone. A court PDF from any other source will
  not have it. `test_the_provenance_marker_alone_cannot_short_circuit` pins
  this.
- `petitioner` plus `respondent` sits at 0.93 specifically to clear the
  contract layer's 0.92 `hereinafter referred to as`, which judgments quote
  constantly. Before it was added, *Prakash Singh vs Union of India* tied at
  0.92 and classified as `contract`.

### Text layer corruption, and the quality gate

The largest finding of this work, and it was not on anyone's list.

**A native PDF text layer can itself be somebody else's bad OCR.** Every one of
the 25 client invoice PDFs carries one. PyMuPDF returns it as native text, the
extractor never routes the page to OCR, and nothing downstream learns the text
is wrong:

```
lnvoice 160219808        lnvoico No. 1OO11      Itrvoice NuDb€r
lnvolce Number: 0007242-lN                      12122t2025
```

`invoice\s*(?:no|number|#)` cannot match `lnvoice No`, so the document falls
to `unknown`. This is not the image half of the corpus. These are the files we
believed were the easy ones.

**Tesseract beats the embedded layer on every document tested.** Re-rendering
the page at 300 dpi and running Tesseract 5.4 lowered the corruption score on
12 of 12 documents, and recovered 9 of the 10 that had failed classification.
Mean corruption over the 25 invoices went 0.104 to 0.019. That was the go/no
go for building the gate at all, and it passed clearly.

`baseline/textquality.py` scores the corruption. It is lexicon free, because no
word list is installed and depending on one would be a new dependency for a
deterministic pipeline. Three signals, all facts about Latin script rather than
vocabulary:

| signal | what it counts |
| --- | --- |
| `l_for_i` | lowercase `l` before a consonant: `lnvoice`, `lnvolce`, `lAm` |
| `digit_letter` | lowercase letter touching a digit: `lnvoic6`, `12122t2025` |
| `case_shape` | letters neither all lower, all upper, nor capitalised: `NuDb` |

Separation on the real corpus is complete:

| population | n | min | median | max |
| --- | --- | --- | --- | --- |
| client invoices (corrupt) | 25 | 0.028 | 0.106 | 0.191 |
| court PDFs (clean) | 46 | 0.000 | 0.002 | 0.013 |

Threshold `0.02` sits in the gap: **25/25 invoices flagged, 0/46 court PDFs.**

Two refinements came out of false positives and both are load bearing:

1. **Tokens containing `.` or `/` are exempt from `case_shape`.** Legal
   citations are legitimately mixed case: `G.Jayachandran`, `Crl.A.259`,
   `W.P.No.26378/2023`. Before this, a clean judgment scored 0.043.
2. **Non Latin script is measured separately and kept out of the score.** A
   Karnataka judgment carrying Kannada in a legacy 8 bit font reads as accented
   Latin-1 and scored 0.233, the highest in the corpus, while being perfectly
   clean. It is also the one case re-OCR cannot fix, since only the `eng`
   language pack is installed. `non_latin_rate` reports it as its own finding.

The gate action is deliberately conservative: score the native text, and if it
is suspect, OCR the page and **keep whichever reading scores lower**. A re-OCR
that comes out worse is discarded, so the gate cannot degrade a page by its own
metric. `prefer()` implements this and is tested both ways.

**It ships disabled.** `pipeline.yaml -> extraction.quality_gate.enabled:
false`. Turning it on changes extraction output with no field in the record to
say that it happened, which would make two runs silently incomparable. Wiring
it in properly is a versioned schema change and is the next decision, not a
side effect of this work.

### First real classification number

Rule layer only, no model, on the 72 PDFs of the labelled set, scored against
the mapped labels. Four passes, each isolating one cause:

| pass | score | what changed |
| --- | --- | --- |
| as found | 52 / 72 = 0.722 | 8 unknown, 7 wrong to `purchase_order`, 4 wrong to `bank_statement` |
| rule bugs fixed | 53 / 72 = 0.736 | **every wrong answer became an abstention** |
| quality gate on | 66 / 72 = 0.917 | 13 corrupt text layers rescued by re-OCR |
| US vocabulary | **71 / 72 = 0.986** | invoice 24/25, court 46/46, PO 1/1 |

The one remaining miss is `1-8-2026Eoscar.pdf` at 0.6, below
`rule_min_confidence`. It abstains. **Nothing in the labelled PDF set is now
assigned a wrong class**, so rule layer precision is 1.0 and the deficit is
entirely recall, which is the trade the whole system is built to make.

Two caveats on that 0.986, both material:

- **The quality gate is disabled by default**, so a run today reproduces the
  0.736 line, not the 0.986 one. The number is real but it is conditional on
  turning the gate on.
- **The 40 image files in the labelled set are not in it.** They need OCR end
  to end and were out of scope here.

**Read this as a rule layer floor, not a system score.** No model ran.

### The decomposition, which was the point

Three causes were separated and fixed independently, and each was measured on
its own. Had they been fixed together, none of these numbers would exist.

1. **Rule bugs, 11 documents.** Markers firing on evidence that does not
   support their claim. A `PO Number` field label on an invoice scoring 0.8 as
   a purchase order. `balance` plus `credit` scoring 0.7 as a bank statement,
   where `credit` came from the addressee being named `1199 SEIU FEDERAL CREDIT
   UNION`. `due upon receipt`, a payment term, scoring 0.87 as a receipt. And
   `invoice\s*(?:no|number|#)`, where the trailing `` after `#` can
   never match when a space follows, so `Invoice # 135551` was unmatchable by
   a pattern written specifically to match it.
2. **Corrupted source text, 13 documents.** Above.
3. **Vocabulary gap, 5 documents.** The class vocabulary was written for Indian
   GST invoices. The corpus is US vendor billing: `Bill To`, `Amount Due`,
   `Net 30`, `Remit To`, `Make checks payable to`. Added and measured at zero
   false positives against the court set and every synthetic template.

## Known gaps

1. **The quality gate is built but disabled.** Enabling it needs somewhere in
   the record to say a page was rescued, and that is a versioned schema
   change. Until it lands, a run reproduces 0.736 on the labelled PDFs, not
   the 0.986 the gate makes possible. This is the highest value open item in
   the repo.

2. **Most numbers in this file are still from 10 synthetic documents.**
   Classification is now measured on the real labelled PDFs. Metadata,
   tagging, tables and the graph are not: no gold exists for them yet. See `HANDOFF.md` for
   what the real corpus actually contains, which differs from what this
   pipeline was configured for.
3. **Models are trained on synthetic data only** and have never been run on
   the real corpus. Every number here is the rule layer alone. 42 generated
   samples across 7 classes, no `court_document` among them, so the model
   cannot currently predict 41% of the labelled set at all. Retraining is
   blocked on nothing but time.
4. **Metadata extraction has no vocabulary for court documents.** The current
   field set returns 7 fields across all 46 court PDFs: 5 emails, 1 phone, 1
   account number. Everything substantive in `fields.yaml` is scoped to
   invoice, receipt and purchase order. Measured availability in the same 46:
   parties 44, judge or bench 39, citation 28, court name 25, judgment date
   20, case number 16, statute section 8. Citations and case numbers have
   strict formats, so they are format validatable the way GSTIN is. This is a
   `fields.yaml` job, not a modelling one, and it is the largest untouched
   gap in the system.

5. **The seven legal labels are unrecoverable.** They collapse to
   `court_document`, which is 41% of the labelled set. If the client needs
   those seven distinguished, it cannot come from the document text and needs
   either matter metadata or a human filing step. Raise it with them rather
   than discovering it downstream.
6. **No positional templates registered.** `templates: []`. The strategy is
   implemented and reads from bboxes, it just has nothing to match yet.
7. **DOCX has no page boundaries.** Treated as one canonical page, since
   pagination needs a renderer.
8. **Anchor confidences cluster around 0.95.** Fine for ranking, not yet
   calibrated against real accuracy.
9. **OCR is only verified on clean synthetic scans.** Real scans bring skew,
   noise, and multi column layouts. Expect the 0.99 similarity to fall.
10. **Only the `eng` language pack is installed.** Add more via the Tesseract
   installer if the corpus needs them, then set `extraction.ocr.lang`.
11. **Span level scoring has never run.** Gold has no character offsets
   because synthetic gold cannot honestly provide them. This matters: span
   accuracy is what a layout model would be judged on.
12. **Table extraction is v0 and thin.** No spanning cells, no ruling line
   detection, no multi page stitching, and real `.docx` tables arrive tab
   joined where text_grid cannot see them. Table gold is 5 grids off 5
   synthetic templates, so 1.0 precision means the checks hold on documents
   whose layout was written by the same person who wrote the checks.
13. **The graph only sees organisations that carry a verified tax ID.**
    Everyone else is invisible to it. That is deliberate for v0, but it means
    graph coverage is capped by metadata quality, which is capped by having
    no real data.
14. **Line items are extracted but nothing consumes them.** The graph has no
    line item nodes and the tagger's `has_line_items` still fires on keywords
    rather than on whether a table was actually found. Tables are a post pass
    and tags are computed in the pipeline, so wiring the two together needs a
    decision about ordering, not just a rule.
15. **Table geometry is only proven on clean native PDFs.** The one scanned
    table in the corpus is refused. Real scans will be worse, and geometry is
    the only strategy that can work on them, because OCR text is assembled
    from word boxes with single spaces and carries no column alignment at all.

## Config reference

`pipeline.yaml` holds engine settings only, no domain vocabulary.
`classes.yaml`, `fields.yaml`, `tags.yaml`, `graph.yaml` and `tables.yaml`
hold all the vocabulary and rules. No class, field, tag or pattern is
hardcoded in any module, and `test_a_new_document_class_needs_only_a_yaml_edit`
enforces that. `test_no_column_or_table_vocabulary_is_hardcoded` does the same
for `tables.py`.

| File | Holds |
| --- | --- |
| `pipeline.yaml` | thresholds, OCR engine and binary, worker settings |
| `classes.yaml` | document taxonomy and classification rule patterns |
| `fields.yaml` | field definitions, patterns, anchors, validators, normalisers |
| `tags.yaml` | tag vocabulary, patterns, thresholds, derived rules |
| `graph.yaml` | which fields may key an entity, role and duplicate rules |
| `tables.yaml` | strategy order, geometry tolerances, text grid thresholds |
| `taxonomy.yaml` | client folder labels to our class names, and what that mapping loses |

`pipeline.yaml` also carries `extraction.quality_gate`, which is engine
settings rather than vocabulary: the corruption threshold and whether the gate
is on.

## Repository state

Five commits on `main`. Working tree clean. Generated artefacts are gitignored
and rebuilt by:

```
python tests/make_corpus.py corpus
cd tests && python make_annotations.py ../annotations.jsonl && python make_gold.py ../gold.jsonl && cd ..
python -m baseline run --input corpus --output results.jsonl --workers 4
python -m baseline tables --input results.jsonl --output tables.jsonl --report tables.md
python -m baseline eval --gold gold.jsonl --pred results.jsonl --pred-tables tables.jsonl
python -m baseline graph --input results.jsonl --report graph.md
```
