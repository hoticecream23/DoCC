# Handoff

Read `PROJECT_STATE.md` for what exists and why. This file is only what to do
next.

Baseline is built, OCR is verified, the evaluation harness is in, the
knowledge graph v0 spike is done, and table extraction v0 is in.

**The real documents have arrived.** Every number in `PROJECT_STATE.md` is
still from the 10 document synthetic corpus, so the accuracy of this system on
real input is currently unknown. Closing that is the whole job now.

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
2. **21 files will be silently skipped.** 19 `.xlsx`, one `.zip`, and one
   macOS resource fork. `SUPPORTED_EXT` in `extract.py` does not include them,
   so `iter_documents` never picks them up and they appear in no error list.
   The run will report a total that quietly excludes them. Either add an XLSX
   handler or decide to drop them on purpose, but do not leave it implicit.
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
   rather than another global knob. Before adding one, check how many of the
   259 real files are actually ID cards; the labelled set may over-represent
   them.

4. **Decide what happens to the 21 unsupported files** before the first real
   run, so the document count in the report means what it appears to mean.
   Note these are 19 `.xlsx`, one `.zip` and one macOS resource fork: they have
   no text layer, so they are unrelated to the corruption problem above.

5. **Profile the corpus.**
   `python -m baseline profile --input Client_Documents --output profile.md`.
   The native versus scanned split, the per field pattern hit rates, and the
   rule layer coverage decide where the effort goes. Expect this to say that
   OCR quality is the dominant variable.

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

7. **Build gold for the rest.** Do not annotate cold. Run the baseline, then
   correct its output, which is far faster. Format is in `tests/make_gold.py`.
   **Include character offsets this time.** Synthetic gold could not provide
   them honestly, so span scoring has never run, and span accuracy is exactly
   what a layout model gets judged on.

8. **Score the baseline for real.**
   `python -m baseline eval --gold gold.jsonl --pred results.jsonl --pred-tables tables.jsonl`.
   That number is the floor. Write it down. Everything from here is measured
   against it.

9. **Retrain and retune.** `train`, then `tune-thresholds`, once there is
   enough corrected data to train on.

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
- **Anchor fields are the weak point.** Amounts, dates in roles, party names.
  Brittle to layout, and confidences near 0.95 that mean nothing.
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

Open items:

- **The Taxonomy tab is empty on purpose**, waiting on a generated CSV. It
  should be exported from `classes.yaml`, `fields.yaml` and `tags.yaml` rather
  than typed, or it drifts from the code within a month. The exporter is about
  40 lines and has not been written yet. Do this after the taxonomy
  reconciliation above, not before, or it will need regenerating anyway.
- **No exporter from `results.jsonl` into the workbook exists yet.** Right now
  the sheet would have to be filled by hand, which defeats the point.
- **No importer back out of it into `gold.jsonl` exists yet.** That is the
  piece that actually turns review effort into a score, and it is the one to
  build first once the sheet has real rows in it.

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
