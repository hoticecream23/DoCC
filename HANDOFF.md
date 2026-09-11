# Handoff

Read `PROJECT_STATE.md` for what exists and why. This file is only what to do
next.

Baseline is built, OCR is verified, the evaluation harness is in, the
knowledge graph v0 spike is done, and table extraction v0 is in.

**The real documents have arrived, and the system has now been measured on
them.** `PROJECT_STATE.md` now carries the real numbers too, but some of its
older sections are still from the 10 document synthetic corpus and say so
where they are. When the two files disagree, this one is newer.

## Start here

Steps 1 to 6 below are done and struck through; read them for the reasoning,
not for work. **The live work is steps 7, 8 and 9**, and the one number that
governs everything else is this:

> **Images score 0.525 against 0.972 for PDFs**, and 60.4% of the real corpus
> sits below 0.6 extraction confidence. Extraction quality is the ceiling on
> every other metric in the system.

One correction to that, added after the first full run over `Client_Documents`:
the ceiling is real, but **ID cards are not what causes it**. There are zero of
them in the 227 client documents, and the four in the labelled set are
specimens. Step 3 has the evidence. Do not build an ID route.

The corpus is **64.8% invoices and 24.7% abstentions**. Those 56 abstaining
documents are the highest value thing a human can label right now.

The review loop is finished and closed in all three directions, which is what
makes growing a real gold set possible now:

```bash
# the run is already done; client_results.jsonl is a full pass over all 227
python -m baseline export-workbook --input client_results.jsonl --output review.xlsx [--merge-from previous.xlsx]
# a human corrects review.xlsx
python -m baseline import-workbook --xlsx review.xlsx --root Client_Documents --output gold.jsonl
python -m baseline eval            --gold gold.jsonl --pred client_results.jsonl

# only if config changed since that run, which invalidates it:
python -m baseline run --input Client_Documents --output client_results.jsonl --workers 4 --no-resume
```

Export and import round trip losslessly, and `--merge-from` carries an
existing review onto a new run. See the index workbook section for the rules
that make that safe.

## The real corpus

`Client_Documents/` holds 259 files. `Classification_Doc/` holds a second,
smaller set filed into folders by document type, plus `Classification.csv`.
Both are gitignored and must stay that way.

Four things about it that change the plan, all of them worth knowing before
running anything:

1. **It is mostly images.** 166 JPG and 15 TIF out of 259, so roughly 70% of
   the corpus goes down the OCR route. Everything the state file says about
   OCR being verified only on clean synthetic scans is now the critical path
   rather than a footnote. Expect the 0.99 character similarity to fall hard.
2. ~~**21 files will be silently skipped.**~~ **The real count is 32, and it
   is no longer silent.** See step 4: 19 `.xlsx` evaluation-set files, 12
   macOS resource forks and 1 archive, each excluded against a named rule that
   `extract.py` records. 259 files in, 227 documents out, and the run logs the
   breakdown.
3. **The client taxonomy does not match ours**, and it is worse than a naming
   mismatch. Their folders name 14 types. Seven of those map cleanly onto our
   classes. The other seven are not document types at all: every file under
   them is a court judgment, and they describe what the firm does with it on a
   matter. Reconciled in `config/taxonomy.yaml`, see step 1. The wider point
   stands and matters more than the mapping: **this is a legal corpus, and the
   field and tag vocabulary was written against a financial one.** Half the
   labelled set is judgments from which `invoice_number` and `total_amount`
   mean nothing, and no field in `fields.yaml` extracts a case number, a court,
   a bench or a party.
4. **`Classification.csv` is free classification gold**, with a caveat found
   during the taxonomy work. It is `File_Name,Document_Type` over 112 rows,
   but only 108 distinct file names: four names appear twice under two labels,
   and three of those four are *different files* with the same name. So the
   file name is not a key over this set. Join on content addressed `doc_id`,
   and read the labels through `config/taxonomy.yaml` rather than raw.

## Do these in order

1. ~~**Reconcile the taxonomy.**~~ **Done, and not the way this file
   expected.** See PROJECT_STATE for the detail. The short version: the seven
   missing "classes" are not document types. Every one of the 46 files under
   them is a court judgment from indiankanoon.org, four of them filed under
   two labels at once. So one class was added, `court_document`, all seven
   labels map to it as lossy, and `config/taxonomy.yaml` is the written record
   both the gold builder and the workbook's Taxonomy tab must read.

   Two things follow that the rest of this file has not absorbed:

   - **41% of the labelled set cannot be scored at label level, only at class
     level.** Any classification number from `Classification.csv` must say so.
   - **Ask the client whether they need those seven distinguished.** If they
     do, it cannot come from the document text. It needs matter metadata or a
     human filing step, and that is a scope conversation, not a modelling one.

   The rule layer alone now gets 46/46 on court documents and 52/72 overall on
   the labelled PDFs. Every remaining miss is an invoice, which is step 4's
   problem, not the taxonomy's.

   **Eight banking classes have since been added from the POC 1 deck**, taking
   the class list from 9 to 17: `address_proof`, `kyc_form`,
   `customer_application`, `loan_application`, `sanction_letter`,
   `repayment_schedule`, `account_opening_form`, `account_closure_form`. The
   full mapping between the deck's taxonomy and ours is in
   `taxonomy_reconciliation.md`.

   **Read these as provisional.** Every other class in `classes.yaml` cites
   how many real documents its markers fire on. These cannot, because no
   corpus we hold contains a single instance of any of them. Only the negative
   is verified: they fire on 0 of the 112 labelled documents and 0 of the 227
   in `Client_Documents`.

   That verification was not free, and it is the reason to re-run it after any
   edit here. Two false positives were caught by it and both are the same bug
   this file already documents twice, under `purchase_order` and `receipt`:

   - `repayment_schedule` took `payment_agreement sample.jpg` off `contract`
     at 0.95, on a Payment Plan Agreement's own clause "as well as the
     repayment schedule, have been created". Fixed with determiner lookbehinds.
   - `sanction_letter` claimed `Axis_Memorandum to ALCO.docx` at 0.80. That is
     an internal ALCO credit memo *seeking* approval, and it carries rate of
     interest, tenure and moratorium because every credit note does. A sanction
     letter is the reply, not the request. Fixed by requiring the past
     participle, and demoted below `rule_min_confidence`.

   The lesson generalises: **a document citing a form is not that form.** With
   no corpus to tune against, only unambiguous self identification is allowed
   to decide any of these eight classes. `tests/test_taxonomy.py` holds both
   regressions plus a title case for each class.

2. ~~**Wire in the text quality gate and turn it on.**~~ **Done.** Schema
   1.1.0, `ExtractionMethod.ocr_rescued`, gate enabled. It rescued 44 pages
   across 28 documents on the labelled set. Original note kept below because
   the reasoning still applies to the next person who wants to change it.

   A native PDF text layer can be somebody else's bad OCR. All 25 client
   invoice PDFs carry one (`lnvoice`, `lnvoico No. 1OO11`, `Itrvoice NuDb€r`),
   the extractor trusts it, and nothing downstream ever learns. Tesseract beats
   it on 12 of 12 documents tested. `baseline/textquality.py` scores the
   corruption and separates the 25 corrupt invoices (0.028 to 0.191) from the
   46 clean court PDFs (0.000 to 0.013) with no overlap at threshold 0.02.

   Turning it on is worth **0.736 to 0.986** on the labelled PDFs.

   It ships `enabled: false` for one reason: **there is nowhere in the record
   to say a page was rescued.** Adding one is a versioned schema change, and
   without it two runs differ with no field explaining why. That decision is
   the blocker, not the code. Options, in the order I would consider them:

   - add `ExtractionMethod.ocr_rescued`, which keeps it inside `page_methods`
     where the routing story already lives
   - add a `text.quality` block carrying the score per page
   - record it only in `diagnostics`, which is cheapest and least useful

   Do not enable it without one of these. A silent accuracy jump that no stored
   record can explain is exactly the failure `schema_version` exists to prevent.

3. ~~**Improve OCR on real scanned images.**~~ **Partly done. Still the top
   item, but for a different reason than this file assumed.**

   The cause was not skew, noise or binarisation. It was **resolution**, and
   it was an asymmetry between the two routes rather than a hard OCR problem.
   A scanned PDF page is rendered by `page.get_pixmap(dpi=300)`, so it reaches
   Tesseract at 300 DPI. An image file was opened and passed straight through
   at whatever resolution it was saved at, and most of the corpus is far under
   300 DPI: 17 of the 40 labelled images have a short side below 640px. Mean
   OCR confidence on those was 0.508 against 0.813 for the rest, and 76% of
   them came back `unknown`. Pearson r between log short side and OCR
   confidence is 0.61.

   `extraction.ocr.min_short_side: 1000` now upscales an image below the
   target with LANCZOS before OCR. Measured on the labelled set:

   | | before | after |
   |---|---|---|
   | image accuracy | 0.450 | **0.525** |
   | image abstentions | 18/40 | **14/40** |
   | mean OCR confidence, images | 0.683 | **0.765** |
   | PDF accuracy | 0.972 | 0.972 (unchanged) |
   | overall accuracy | 0.784 | **0.811** |
   | macro F1 | 0.625 | **0.664** |
   | coverage | 0.821 | **0.857** |

   Three things worth knowing before anyone changes this:

   - **1000 is a knee, not a floor.** Targets of 1300 and 1600 were both
     worse: 1000 improved 22 images and regressed 3, 1600 improved 19 and
     regressed 6, some badly (`vendor contract template.png` 0.871 to 0.505).
     Enlarging an image that is already legible only softens its glyphs. Do
     not raise this number without re-running the sweep.
   - **Grayscale conversion does nothing and sometimes hurts.** Mean across
     the 40 was 0.689 against 0.688, and it cost three large scans real
     accuracy (`PostDapartment_Payslip.jpg` 0.765 to 0.660). It is not in the
     code. Do not add it back on the assumption that it must help.
   - **No single `psm` wins.** Swept 3, 4, 6, 11 and 12 over the images still
     failing after upscale. The best mode is per document and the spread is
     wide: `Sample_voter_id.jpg` wants 12 (0.268 to 0.546), `Tax_statement.jpg`
     and `Aadhar_card.webp` are both best at 3. Picking best-of-N by the
     engine's own confidence is a trap, because a mode that reads half the
     page can score higher on the half it read. If this is worth doing it
     needs an external quality signal, so `psm` stays at 3 for now.

   **A coordinate bug was found and fixed on the way past.** OCR word boxes
   were stored in the pixel space of the image the engine read, while
   `page_sizes` held the page's own units, and `metadata.py` normalises every
   box as `w.x0 / page_width`. For OCR'd PDF pages those two differed by
   `dpi/72`, so **24 of 29 OCR'd PDFs had over half their word boxes outside
   the declared page**, and every template region over a scanned page was
   reading against a box four times too large. `_ocr_words` now divides by the
   scale of the image that was actually read, at all three OCR call sites.
   That count is 0/29 now. This never showed up in the numbers above because
   no template currently fires on a scanned page, but it would have silently
   broken the layout work that is the whole point of step 7's offsets.

   **What is left, and it is still the largest gap in the system.** Images are
   0.525 against 0.972 for PDFs. The residue is dominated by ID cards, which
   are not page-like: Aadhaar, PAN, voter ID and passport are dense, coloured,
   and printed over patterned security backgrounds. Upscaling does not help
   them much and `psm` helps inconsistently. They likely want their own route
   rather than another global knob.

   **Do not build that route. The count was run and it is zero.** The
   suspicion in the paragraph above was right and then some: the labelled set
   does not over-represent ID cards, it is the only place they exist.

   | | ID cards |
   |---|---|
   | `Classification_Doc`, labelled set | 4, and all four are specimens |
   | `Client_Documents`, 227 documents | **0** |

   The four are `Aadhar_card.webp`, `pan_card.webp`,
   `sample-indian-passport-1.jpg` and `Sample_voter_id.jpg`. Sample images,
   near certainly dropped in to exercise the PAN and Aadhaar checksums. They
   are not client filing and nothing in `Client_Documents` resembles them.

   Two independent detectors agree, which matters because neither is good
   enough alone:

   - **The classifier finds 1 of the 4 known cards.** Only `pan_card.webp`
     classifies, at 0.93. The other three abstain. A zero from a detector with
     that recall proves nothing on its own, so do not cite the label counts
     for this.
   - **A keyword scan over the raw OCR text finds 3 of 4**, missing only
     `Sample_voter_id.jpg`, whose OCR is 100 characters of noise at 0.27
     extraction confidence. Run over all 227 it returns 8 documents, and all
     8 are court judgments *discussing* passports, voters and the Election
     Commission. Not one is a card.
   - **The blind spot where both fail is one file**, and it is
     `~$rdness_Failure_minimization_plan.docx`, a Word lock stub. No image in
     the 227 has the failed-OCR profile that hid the voter ID.

   So image accuracy at 0.525 is still real and still the ceiling, but
   whatever causes it is not ID cards. Re-diagnose it against
   `client_results.jsonl` before spending anything more on OCR.

4. ~~**Decide what happens to the 21 unsupported files.**~~ **Done, and the
   count was wrong in both directions.** Every one of the four categories was
   looked at rather than assumed, and `baseline/extract.py` now records the
   decision next to the rule that implements it.

   `Client_Documents/` holds 259 files. **227 are documents.** The other 32:

   | reason | files | what they actually are |
   |---|---|---|
   | `macos_resource_fork` | 12 | AppleDouble stubs |
   | `evaluation_set` | 19 | question answering gold, not documents |
   | `archive` | 1 | a zip that duplicates loose files |

   **One category is still missing, found on the full run.** A Word lock stub,
   `~$rdness_Failure_minimization_plan.docx`, is the only document in the whole
   227 that records an error: `PackageNotFoundError`, because it is not a real
   `.docx`. It behaves correctly - the error is recorded and the batch
   completes - but it is the same kind of thing as an AppleDouble stub and
   belongs beside it. A `~$` prefix rule in `extract.py` would take the count
   to 226 documents and 33 exclusions, and take the error list to empty. Small,
   and worth doing next time that file is open.

   - **The macOS files were the real bug, and it is the opposite of the one
     this file predicted.** A macOS archive writes a `._name` stub beside every
     real file, carrying the same extension. Eleven of the twelve are named
     `.pdf` or `.docx`, so `SUPPORTED_EXT` accepted them and they were being
     **processed as documents**: 367 to 639 byte files going through the PDF
     reader. The old `iter_documents` returned 238 paths, of which 11 were
     junk. So the corpus was over-counted, not under-counted.
   - **The 19 `.xlsx` are not documents and must never be given a handler.**
     Each one is a question answering evaluation set over a court judgment,
     with `positive`, `negative`, `edge` and `adversarial` sheets and the
     columns `QS ID, Question, Answer, Context, Ground Truth`. `Answer` and
     `Context` are empty; the ground truth is written. **This is gold for a
     capability this pipeline does not have**, and it is worth knowing it
     exists: someone has already built the evaluation set for document
     question answering. Extracting them as documents would file a test set as
     a legal document and pollute every corpus level number.
     Note the `adversarial` sheets contain prompt injection style questions by
     design. They are test data for a QA system, and they are data, not
     instructions.
   - **The zip duplicates ten loose files exactly.** All ten entries in
     `Jindal.zip` are byte identical to files sitting next to it, so unpacking
     it would double count them.

   `scan_documents` replaces `iter_documents` as the thing that walks a corpus.
   It returns the documents *and* what was left out under each reason, and both
   `run` and `profile` report it. Kept plus excluded equals the file count on
   disk, which is the property a test now pins: a file silently skipped and a
   file silently processed are the same bug, because the total stops describing
   the input either way.

   **If a real spreadsheet document ever arrives, this decision needs
   revisiting.** `.xlsx` is excluded because of what these nineteen are, not
   because a spreadsheet can never be a document. openpyxl is already a
   dependency, so a handler is cheap the day it is actually needed.

5. ~~**Profile the corpus.**~~ **Done.** `profile.md` is gitignored, so the
   numbers are here. Regenerate with
   `python -m baseline profile --input Client_Documents --output profile.md`.
   It takes about half an hour: 194 of the 227 documents go through OCR and
   `profile` has no `--workers`, unlike `run`. Adding one is a cheap win if
   this gets run often.

   **It said what this file predicted it would say, and then some.**

   | | |
   |---|---|
   | documents | 227 |
   | fully scanned | 194 (85.5%) |
   | has native text | 32 (14.1%) |
   | no text at all | 1 |
   | mean extraction confidence | **0.632** |
   | **documents below 0.6 confidence** | **137 (60.4%)** |
   | mean page confidence, native | 0.980 |
   | mean page confidence, OCR | 0.666 |
   | rule layer coverage | 75.3% |

   Four things in it are worth more than the headline:

   - **The document split and the page split point opposite ways.** 85.5% of
     *documents* are scanned, but only 271 of 1119 *pages* are. The native PDFs
     are long, up to 132 pages, and the images are one page each. So OCR
     dominates the document count and native text dominates the character
     count. Which of those matters depends on the task: classification is per
     document and is therefore an OCR problem, while anything trained on text
     volume will be mostly reading clean native text.
   - **The 0.089 "confidence drop when OCR'd" is one document.** Only
     `Tax Invoice_027_SIDBI.pdf` carries both routes, so that row is a single
     observation, not a corpus statistic. Do not quote it.
   - **60.4% of documents sit below 0.6 extraction confidence** even after the
     resolution fix in step 3. That is the same conclusion step 3 reached from
     the other end, and it is the number to move.
   - **`invoice` at 147 of 227 (64.8%) has not been verified and looks high.**
     The labelled set in `Classification_Doc/` is 41% court documents; this
     corpus reads as two thirds invoices off the rule layer alone. It may be
     true, since `Client_Documents/` is a different corpus, but the invoice
     markers were widened for US commercial vocabulary and an over firing rule
     would look exactly like this. **Check it before trusting any per class
     number from this corpus**: run `baseline run` over `Client_Documents`,
     then look at which marker fired on the documents classified `invoice` and
     read twenty of them.

   **The profile earned its keep by exposing a real bug, now fixed.**
   `invoice_number` had a 69.2% hit rate against a **1.3%** validation rate.
   The cause was the pattern itself: `(?:INV|BILL|PO)[/-]?[A-Z0-9/-]{3,20}`
   matches the word **`INVOICE`**, which is printed on nearly every invoice in
   the corpus. `INV` plus `OICE` clears the `{3,20}` tail. So do `POLICY`,
   `PORTAL`, `POSTAL`, `POWER`, `PORTION`, `BILLING`, `BILLABLE` and
   `INVOICES`. On the labelled set the pattern layer emitted 8 values and 7 of
   them were English words: `INVOICE` three times, `BILLING` twice, `PORTAL`,
   `PORTNON`, and one real `PO-001234`.

   The fix is a lookahead requiring a digit, since a real reference always has
   one. Metadata micro F1 on the reviewed gold went 0.4651 to **0.4762** with
   no regression anywhere, gated through `compare --fail-on-regression`.

   **Two similar gaps are still open, and both are PII.** `aadhaar` has 8 hits
   and **0 validated**; `card_number` has 6 hits and **0 validated**. The
   checksums are rejecting every one, which is the validators working, but the
   fields are still emitted unvalidated at 0.4 confidence. So the system is
   currently asserting Aadhaar and card numbers that are almost certainly OCR
   noise. Either tighten those patterns the way `invoice_number` was tightened,
   or stop emitting an unvalidated candidate for a checksummed field at all.
   The second is probably right: for a field that carries a checksum, failing
   it is strong evidence rather than weak.

   Also open: **1 document produced no text at all and 1 extract error.**
   Neither is identified in the profile output, which is a gap in the profile
   itself. It should name them.

6. ~~**Score classification.**~~ **Done, and it is reproducible.**

   ```bash
   python -m baseline import-labels --csv Classification_Doc/Classification.csv --root Classification_Doc --output gold_classification.jsonl
   python -m baseline run --input Classification_Doc --output results_classification.jsonl --workers 4
   python -m baseline eval --gold gold_classification.jsonl --pred results_classification.jsonl
   ```

   Accuracy 0.786, macro F1 0.625, coverage 0.821, 0.956 when it answers.
   PDFs 69/71, images 18/40.

   **`model_min_confidence` has still not been retuned and the model has still
   never been retrained.** It has no `court_document` class, so it cannot
   predict 41% of the labelled set. Retrain first, then take the threshold
   from the harness's coverage at 1, 5 and 10 percent error.

7. **Build gold for the rest. This is the live item.** Do not annotate cold,
   and do not hand write the sheet either: both tools now exist. Run the
   baseline, export the workbook, correct it, import it back. The commands are
   at the top of this file.

   Today's gold is **21 reviewed documents**, of which 20 join. The corpus is
   227. That gap is the work.

   **The run half of that loop is already done.** `client_results.jsonl` is a
   full pass over all 227 documents of `Client_Documents`, so start at
   `export-workbook` rather than re-running the pipeline. What it says the
   corpus is:

   | label | n | share |
   |---|---|---|
   | `invoice` | 147 | 64.8% |
   | `unknown` | 56 | 24.7% |
   | `court_document` | 17 | 7.5% |
   | `purchase_order` | 6 | 2.6% |
   | `tax_form` | 1 | 0.4% |

   Two thirds of the corpus is invoices, which is worth knowing before
   annotating: the review effort is mostly one document type. **The 24%
   abstention rate is now the largest single number in the system**, and those
   56 documents are where review pays for itself, because an abstention that a
   human labels becomes training data for step 9 while a correct invoice
   mostly confirms what the rules already knew.

   Offsets are no longer a problem: the exporter carries `CHAR_START` and
   `CHAR_END`, the importer keeps them on confirmed rows, and **span scoring
   has now run for the first time at micro F1 0.207 over 11 documents.** That
   is the number a layout model has to beat.

8. **Score the baseline for real.** Partly done, on 20 documents. The floor,
   written down, against human reviewed gold:

   | metric | value |
   |---|---|
   | classification accuracy | 0.700 |
   | **accuracy when answered** | **1.000** |
   | metadata value micro F1 | 0.476 |
   | span micro F1 | 0.207 (11 docs) |
   | tagging micro F1 | 0.533 |
   | `validated_precision` | 1.0 |

   Every classification error is an abstention; the system is not yet wrong
   when it answers. Redo this on a larger gold set from step 7 and treat the
   above as provisional until then.

9. **Retrain and retune.** `train`, then `tune-thresholds`, once there is
   enough corrected data to train on.

   **`models/classifier` is now stale against 17 classes.** It already had no
   `court_document` in its training set, so it could not predict 41% of the
   labelled corpus; it now also has none of the eight banking classes. Nothing
   breaks, because the rule layer carries all of them and a model below
   `model_min_confidence` counts as declining rather than disagreeing, which
   `test_a_quiet_model_does_not_veto_a_confident_rule` pins down. But do not
   retrain before step 7 has produced real volume: a model fitted on 21
   documents would be worse than the rules it would be overriding.

## Then the new approach

Do not upgrade everything uniformly. The error surface is lopsided:

- **Leave the checksummed IDs alone.** PAN, GSTIN, Aadhaar, IFSC, cards. A
  model cannot beat a checksum.
- **Classification is close to done at the rule layer, and a model has still
  never run on real data.** 71/72 on the labelled PDFs with rules alone, and
  the one miss abstains rather than errs. Retrain before assuming a model adds
  anything: the current one has no `court_document` in its training set at all,
  so it cannot predict 41% of the labelled corpus.
- **Metadata on court documents is the real open gap.** 7 fields across 46
  judgments today. Parties, court, bench, citation, case number and judgment
  date are all sitting there at 16 to 44 hits out of 46. See PROJECT_STATE.
- **Anchor fields are the weak point, now with evidence.** `total_amount` is
  7 of the 11 remaining misses against reviewed gold, and they are not near
  misses: `4532321.00` wanted against `5.00` produced. Amounts, dates in roles
  and party names are brittle to layout and carry confidences near 0.95 that
  mean nothing.
- **A checksummed field should probably not emit an unvalidated candidate.**
  On the real corpus `aadhaar` has 8 hits and 0 validated, `card_number` 6 and
  0. The checksums are doing their job; the pipeline emits the candidates
  anyway at 0.4 confidence, so the system asserts PII that is almost certainly
  OCR noise. `invoice_number` was the same shape and was fixed by tightening
  its pattern. See step 5.
- **Tables exist now but only just.** v0 is precision first and refuses
  anything it cannot read cleanly. On real documents expect the abstention
  rate to be the number that moves, not the error rate.

So the target is layout aware token classification over text plus bounding
boxes, and better table structure. `tables.py` is the first consumer of the
word boxes, and it is a classical geometric reading of them. A layout model
should be measured against it with the same two numbers, not instead of them.

Keep the rule and checksum layer as a hard prior over any model. Best of
both, and it stays interpretable.

Gate every change with:

```bash
python -m baseline compare --gold gold.jsonl --a baseline.jsonl --b new.jsonl --fail-on-regression
```

Pass `--a-tables` and `--b-tables` as well when tables are in scope, or the
table metrics sit the gate out.

## Tables, where v0 leaves off

v0 is built and scored. `baseline tables`, then `baseline eval --pred-tables`.
See PROJECT_STATE for the design, the two strategies, and the two abstentions.

Precision is 1.0 on both cell content and adjacency. Recall is 0.8, and the
whole deficit is one refused table. So the ceiling is recall, exactly as it is
for the graph, and for the same reason: v0 refuses what it cannot read.

In rough order of value once there is real data:

1. **A tab delimited strategy.** Real `.docx` tables come through
   `_extract_docx` as tab joined rows. A single tab never forms a two
   character separator, so text_grid cannot see them and geometry has no boxes
   to work with. Nothing in the synthetic corpus hits this and every real
   `.docx` with a table will. Cheapest real win available.
2. **Ruling lines.** Columns currently come only from whitespace and
   coordinates. A densely ruled table with no gaps between its cells is
   invisible. PyMuPDF exposes the drawing operators, so the lines are there
   for the taking.
3. **Spanning cells.** `row_span` and `col_span` are in the schema and always
   1. Merged header cells are common and currently get read into one column.
4. **Multi page stitching.** A table crossing a page break is two tables. The
   join wants a column signature match plus a repeated header.
5. **Re tune the geometry tolerances on real documents.**
   `column_overlap_ratio` at 0.4 and `cell_gap_spaces` at 1.6 were set against
   synthetic layouts, and the margin on the invoice is thinner than it looks:
   the header to data overlap ratios land at 0.45 and 0.47 against a 0.4 bar.
   Profile the real distribution before touching them.
6. **Feed line items into the graph and the tagger.** `has_line_items` still
   fires on keywords when a real table is now sitting right there, and the
   graph has no line item nodes. Both need a decision about ordering first,
   since tables are a post pass and tags are computed in the pipeline.

Do not soften the abstention checks to raise recall. They are the only reason
precision is 1.0. If a check is refusing something it should accept, fix the
check's reasoning, do not lower its bar.

## The index workbook

A separate track, run with Karishma, who is building the sheet. It is the
human facing mirror of `results.jsonl`: an Excel workbook that indexes every
document, its class, its extracted fields and its tags, so that non engineers
can review and correct the pipeline's output.

It matters technically because **this is how real gold gets built.** Correcting
an exported row is far faster than annotating from scratch, so the workbook is
the annotation tool, and the gold file is generated back out of it.

Five tabs: Documents, Metadata, Tags, Taxonomy, Line items. Column names match
the output schema exactly so rows can be exported in rather than retyped. The
full column spec, with types, ownership and allowed values, was written up and
sent to Karishma; ask Advay for the link.

Three decisions in it that must not be quietly undone:

1. **The full extracted text does not go in a cell.** `PAGE_SEP` contains a
   form feed, and every `char_start` / `char_end` is counted against the exact
   canonical string. Excel will strip the form feed, rewrite the line endings,
   or hit its 32,767 character cell limit, and every offset in the Metadata tab
   becomes silently wrong. The sheet carries `char_count` and `text_sha256` as
   a tripwire instead, and the JSONL stays the source of truth for text.
2. **A correction never overwrites a pipeline value.** Each tab carries a
   `*_corrected` column, a `verdict`, and `is_gold`. Overwriting in place
   destroys the record of what the pipeline said, and with it the ability to
   score the run at all.
3. **`regex_checksum` and `regex_format` stay separate in the sheet.**
   Collapsing them to "regex" reintroduces the first bug the eval harness ever
   caught, at the human layer where nothing will catch it again.

**The first reviewed sheet has arrived**, as
`Classification_Doc/Classification_Sheets.xlsx` inside the `ML_Documents/`
delivery. 21 documents, 36 metadata rows, 30 tag rows, all reviewed by a
human. The reviewer found real errors: 10 of the 21 classes were wrong, and 11
of 36 field values.

Open items:

- ~~**No importer back out of it into `gold.jsonl`.**~~ **Done.**
  `baseline import-workbook --xlsx FILE --root DIR --output gold.jsonl`.
  See `baseline/workbook/importer.py`. Four things in it are load bearing:

  - **`spurious` produces nothing.** The gold for a field the pipeline
    invented is its *absence*. Writing a row for it would turn every correctly
    rejected field into a false negative and flatter the score.
  - **Corrections are normalised through the field's own normalizer.** The
    reviewer types `742.7` and `12/22/2025`; the pipeline emits `742.70` and
    `2025-12-22`. Comparing them raw marks the pipeline wrong for being right,
    and it cost 0.09 of metadata F1 before this was in. Confirmed values go
    through it too, because the sheet's `NORMALIZED_VALUE` column is only as
    normalised as whoever filled it in. The normalizers are idempotent, tested
    over all 177 values the current run emits, so this cannot corrupt a value
    the pipeline already produced.
  - **Only confirmed rows carry offsets.** A corrected value has no span: the
    reviewer typed an answer, not a location.
  - **`DOC_ID` is forward filled.** It is written once per group, so 18 of 36
    metadata rows and 14 of 30 tag rows have a blank one. Without the fill,
    half the sheet orphans.

  An undeclared class, field or tag name is a hard failure that writes no gold,
  the same way `import-labels` refuses an unmapped label. On this sheet that
  fires once, on `Aadhar_card`, which is not a name in `fields.yaml`. It is
  almost certainly `aadhaar`, but almost certainly is not good enough and the
  fix is one cell.

- **First score against human reviewed gold.** With that one cell corrected in
  a scratch copy: classification accuracy 0.700 over 20 joinable documents,
  **accuracy when answered 1.000**, every error an abstention. Metadata value
  micro F1 0.465, tagging micro F1 0.533. `validated_precision` is **1.0**, and
  this is the first time it has been measurable on real data rather than null.

  **Span scoring has now run for the first time: micro F1 0.207 over the 11
  documents carrying offsets.** That number is the one the layout model has to
  beat, and it is low enough to be worth beating.

  `total_amount` is 7 of the 11 remaining misses, and they are not close
  (`4532321.00` against `5.00`). The anchor layer is picking the wrong number
  on real documents, exactly as this file predicted.

- **Three things for the sheet itself, none of them code.**

  - One row references `3af3c1b47e4af807`, a `Sample_voter_id.jpg` whose bytes
    are not in the corpus. Both copies on disk hash to `83304e91bf6833a4`. It
    was reviewed against a different file and cannot be scored until that file
    turns up.
  - An `aadhaar` was corrected to `XXXX XXXX 3208`. Redacting PII in the gold
    is reasonable but it makes the row unscoreable, since the normalizer
    reduces it to `3208`. Decide whether gold should hold real identifiers or
    the field should be excluded from scoring.
  - `buyer_name` reads `John doe` where the pipeline now emits `John Doe`. The
    sheet was exported from an older run, so some confirmed values have drifted
    from what the pipeline currently produces.

- ~~**No exporter from `results.jsonl` into the workbook.**~~ **Done.**
  `baseline export-workbook --input results.jsonl --output workbook.xlsx
  [--tables tables.jsonl]`. All five tabs, and the column headers are
  byte identical to the sheet Karishma built, checked tab by tab. It reads its
  columns from the same `columns.py` as the importer on purpose: the two have to agree on every
  column name, and a rename touching only one of them breaks the loop in
  silence.

  What it will not do, and why:

  - **The full text never goes in a cell.** `TEXT_PREVIEW` is 200 characters
    with whitespace collapsed; `TEXT_SHA256` and `CHAR_COUNT` are the tripwire.
  - **Every review column ships blank**, `IS_GOLD` above all. Pre-filling it
    would turn an unreviewed export into gold on the next import and invent a
    score out of nothing.
  - **It refuses to overwrite an existing file** without `--overwrite`. Nothing
    here can merge a review forward yet, so re-exporting over a reviewed sheet
    would destroy the expensive half of the work. Merging corrections forward
    across runs is the next thing this needs.
  - **A string beginning with `=` is written as text.** openpyxl types it as a
    formula otherwise, and OCR output is untrusted input.
  - `LANGUAGE` and `QUALITY` are left blank. The pipeline has no source for
    either, and filling them would be a guess wearing the pipeline's clothes.

- ~~**The Taxonomy tab is empty.**~~ **Done**, and generated rather than typed.
  35 rows out of `classes.yaml`, `fields.yaml` and `tags.yaml` on every export,
  so it cannot drift from the code.

- **The loop is closed and proved lossless.** Export the run, mark every row
  `OK`, import it back, and the gold that comes out scores 1.0 against the
  predictions that went in on classification, metadata and tagging. Two real
  defects were found by that test and would not have been found any other way:

  - A confirmed abstention was being written as a gold label of `unknown`.
    There is no such class. Agreeing the pipeline was right not to answer is
    not a label, and the document still has a real class nobody has recorded,
    so those rows now carry no classification gold and the report says how many
    are waiting on a `CLASS_CORRECTED`. On this run that is 17 of 112.
  - A field whose `normalized_value` is null was being dropped on import. The
    pipeline keeps such a field and the harness falls back to the raw value, so
    dropping it lost a real answer. It now keeps the raw value too.

- ~~**Merge a review forward.**~~ **Done.**
  `baseline export-workbook --input new_results.jsonl --output next.xlsx
  --merge-from reviewed.xlsx`. This is what makes the loop iterative rather
  than a one shot: without it a pipeline change threw away every review made
  against the previous run.

  **One distinction decides everything in it.** A **correction** is a fact
  about the *document* ("the total is 4044"), and stays true whatever the
  pipeline says next, so it always carries. A **confirmation** is a fact about
  a *prediction* ("what you said is right"), and means nothing once the
  prediction changes, so it carries only while the answer is the same.

  Getting that backwards is a silent data problem in both directions. Carry a
  confirmation onto a changed prediction and a brand new, unreviewed answer is
  marked human approved. Drop a correction because the prediction moved and the
  expensive half of the work is gone. `tests/test_workbook.py` has a test for
  each direction; the first one is the one that matters.

  Three consequences fall out of the same rule:

  - **A confirmed value the new run no longer produces comes back as a human
    `missing` row.** The confirmation was still a statement that the value
    belongs on the document, so it survives the prediction that carried it.
    Same for a confirmed tag.
  - **A `spurious` row the new run has stopped emitting is simply dropped.**
    The reviewer said it did not belong and the pipeline agrees now. Keeping it
    would ask the same question twice.
  - **A correction the pipeline has caught up with becomes a confirmation**,
    with a note saying what it used to say. The row stops asking to be
    corrected once there is nothing left to correct.

  Merging the real 21 document review onto the current run: 10 confirmations
  carried, 5 corrections carried, 22 field reviews and 22 tag reviews carried,
  6 field reviews and 3 tag reviews rescued as human rows, 5 spurious rows
  resolved by the pipeline, and 1 confirmation invalidated for re-review.
  **Four class corrections and two field corrections were "now agreed"** —
  the OCR resolution fix moved the pipeline onto the answer the reviewer had
  already written down.

  Gold out of the merged sheet is 19 rows against 21 from the original. Both
  losses are correct: one is the row whose content is not in the corpus, and
  one is the invalidated confirmation, which is waiting on a human rather than
  carrying a stale approval.

- **What this track still does not do.** The merge keys documents on `doc_id`,
  which is content addressed, so a document that is re-scanned or re-exported
  by the client is a new document and its review does not follow. That is the
  correct default and probably the permanent one, but it means a corpus refresh
  loses review in a way a pipeline change no longer does.

## Knowledge graph, where v0 leaves off

v0 is built and works. `baseline graph`. See PROJECT_STATE for the design and
the three wrong edge classes it caught.

The ceiling is coverage, not correctness. Only organisations carrying a
verified tax ID exist in the graph. Globex appears in five documents and has
no node because nothing proves its identity.

v1, once there is real data and better extraction:

- Name based entity resolution, but only as a **second pass that can never
  merge two nodes already distinguished by verified IDs**. Blocking on
  normalised name, then a similarity threshold tuned on real pairs.
- Confidence on every edge, so a query can ask for only the safe subgraph.
- Line items as nodes. Table extraction now exists, so this is unblocked and
  it is what makes spend analysis and duplicate detection actually good. The
  join is `TableRecord.doc_id` and the cell offsets, both of which are already
  in the same space as everything else.
- Temporal edges. An invoice dated after the receipt that settles it is a
  finding, not a fact.

Do not soften the v0 rule to raise coverage. Add a confidence field and keep
the proven edges distinguishable from the inferred ones.

## Do not break these

- `PAGE_SEP` in `schema.py`. Changing it invalidates every stored offset.
- The `extra="forbid"` schema models. Adding a field is a versioned change.
- Config over code. A class, field or tag name in a `.py` file is a bug.
- `evaluate.py` must never import the pipeline. The moment it does, it stops
  being able to score a competing implementation fairly.
- `tables.py` must stay a separate pass. The moment a table lands inside
  `DocumentRecord`, the record contract stops being frozen and every stored
  run needs re-extracting to gain a field.
- OCR word boxes and `page_sizes` must be in the same coordinate space.
  `metadata.py` normalises every box as `w.x0 / page_width`, so a box read off
  a rendered or upscaled image has to be divided back down first. `_ocr_words`
  is the only place that should build a `WordBox` from an `OCRWord`.
- Table cell offsets are null or exact, never approximate. `verify_offsets`
  on `TableRecord` enforces it, same as on the record.
- The workbook column names live in `baseline/workbook/columns.py` and are read by
  both directions. Exporting and importing must never carry separate copies of
  that list, or a rename desynchronises them silently.
- `scan_documents` must account for every file under the root. Kept plus
  excluded equals what is on disk. A file that is silently skipped and one that
  is silently processed are the same bug: the reported total stops describing
  the input.
- The `.xlsx` files in `Client_Documents/` are question answering evaluation
  sets, not documents. Do not add an XLSX handler to make them extractable.
- Merging a review forward must never carry a confirmation onto a prediction
  that has changed. A confirmation approves an answer, not a document, and
  moving it to a new answer marks unreviewed output as human approved. A
  correction is the opposite and must never be dropped because the prediction
  moved.
- `export-workbook` must never fill a review column. `IS_GOLD` set by the
  exporter turns an unreviewed sheet into gold on the next import.
- A workbook verdict of `spurious` must produce no gold row. It records that
  the field is not there, and inventing a row for it converts a correct
  rejection into a false negative.
- `ML_Documents/` is gitignored explicitly. It was previously covered only by
  accident, because `Client_Documents/` and `Classification_Doc/` carry no
  leading slash and so matched the copies nested inside it.
- `config/taxonomy.yaml` is the only place the client label mapping lives.
  Restating it in the gold builder or the workbook is how the two drift apart.
  `tests/test_taxonomy.py` fails if a label stops mapping to a declared class.
- The court `vs ... on <date>` marker is a provenance artefact of the
  indiankanoon download, not evidence of a court. It stays below
  `rule_short_circuit` so it can never decide a class on its own.
- `Client_Documents/` and `Classification_Doc/` are gitignored. They are real
  client material and must never be committed.
- `validated_precision` must stay 1.0. Anything less means a validator is
  claiming proof it does not have.

## Machine specific setting

`config/pipeline.yaml -> extraction.ocr.binary` is an absolute path to
`tesseract.exe` on this machine. Set it to null if tesseract is on PATH.
Only the `eng` language pack is installed.

## Gotchas worth knowing

PAN `ABCDE1234F` fails validation on purpose. `D` is not a real holder type
code. If real PANs are being rejected, widen
`fields.yaml -> pan -> validator_args.entity_types` rather than loosening the
validator.

One gold miss is left unfixed deliberately: `buyer_name` on the purchase
order. Fixing it would be tuning to a single synthetic document.
