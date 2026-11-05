# Round 2 audit: tables, graph, workbook

Read-only audit. No repo code changed. review.xlsx was not opened. Corpus-only run; Client_Documents was not touched.

## 1. Tables

**Scores.** Table scoring exists (`baseline/evaluate/table_scores.py`, PROJECT_STATE "Scored"): 5 gold tables, detection recall 0.800, cell precision 1.000, cell recall 0.803, adjacency precision 1.000, adjacency recall 0.800. The gold is 5 hand-built grids in `tests/make_gold.py` on the 10-document synthetic corpus. There is no table score for the client corpus or SROIE (SROIE receipts carry no tables). `report.md` has no table section. The "precision 1.0" claim rests on 4 predicted tables and 49 cells.

**What comes out** (`python -m baseline tables --input results.jsonl`, written to scratchpad `r2b\tables.jsonl` and `r2b\tables.md`; the output matches the committed tables.md, so it is reproducible):
- invoice_native.pdf: 1 table, geometry, 3 rows x 4 cols (header Qty/Description/Rate/Amount + 2 item rows), fill 1.0.
- purchase_order.pdf: 1 table, geometry, 2 rows x 4 cols (header + 1 item row).
- bank_statement.pdf: text_grid 5x5 (geometry abstained: "6 columns but widest row holds 5").
- salary_slip.docx: text_grid 5x2, no header.
- invoice_hybrid.pdf page 1 (scanned): abstained (OCR merged "Widget assembly", numeric columns drift). 2 abstentions in total.

**Strategies.** tab_delimited (docx rows joined by tabs; run of 2+ lines with equal tab count). geometry (words -> lines by vertical overlap -> cells split at gap > 1.6 space widths -> complete-linkage column clustering). text_grid (character columns blank on every line of a band). Abstain checks (geometry.py ~L265-285): a cell fits 2+ columns; more columns than the widest row's cell count; a column with fewer than 2 cells. Gate: min_rows 2, min_cols 2, min_cells 6.

**Biggest weakness: line items never become records.** A table is a bag of text cells with offsets. Nothing maps rows to description/qty/rate/amount, nothing checks them, and nothing consumes them. Evidence: PROJECT_STATE known gap 14 ("Line items are extracted but nothing consumes them"); the graph has no line-item nodes; the tagger's `has_line_items` still fires on keywords; the only consumer is the xlsx "Line items" tab, which has no verdict column. Header detection is also weak (a heuristic; none on the salary slip). Second weakness: scans abstain. The only scanned table in the corpus is refused, and 5 gold tables are too few to show how common that is on real scans.

## 2. Graph

Built in `baseline/graph.py` (549 lines). Nodes: Document, Organization (keyed on PAN, GSTIN collapsed to PAN), Account (IFSC + number). Edges: mentions_org, issued_by, billed_to, pays_to, held_by, references, duplicate_of. Current graph.md: 10 docs, 2 orgs, 2 accounts, 21 edges (5 mentions_org, 6 pays_to, 4 references, 2 duplicate_of, 2 held_by, 2 issued_by). Both duplicate pairs and all 4 receipt-to-invoice chains are found; 0 orphan references.

Eval: `tests/test_graph.py` has 23 unit tests (determinism, no name-only orgs, role margin abstention, duplicates scoped by class, plus one run over real pipeline output). There is no gold and no precision/recall score for the graph. Consumers: none. Only `cli.py` calls `build`; nothing reads `graph.json`, `graph.dot` or `graph.md`.

Improvement: none worth doing now. A line-item link needs line-item records first (proposal 1). GSTIN dedupe is already done (via PAN). Names are deliberately not used, so Globex has no node. Verdict: stay a spike.

## 3. Workbook

**Export** (`exporter.py`, 239 lines): tabs Documents, Metadata, Tags, Taxonomy, Line items. Per field a reviewer sees FIELD, VALUE, NORMALIZED_VALUE, page and offsets, a raw numeric CONFIDENCE, METHOD, VALIDATED, then blank CORRECTED_VALUE / VERDICT / IS_GOLD / REVIEWED_BY / NOTES. Documents show CLASS_CONFIDENCE and RUNNER_UP. Confidence is a bare number. There is no conditional formatting, sort, freeze or filter, so low-confidence fields are not flagged and the reviewer must read every number. Export refuses to overwrite an existing file; `merge_from` carries an earlier review forward.

**Import** (`importer.py`, 427 lines): verdicts ok/wrong/missing/spurious become gold rows (`ok` keeps the pipeline value, `wrong`/`missing` use the corrected value normalised through the field's normalizer, `spurious` means gold is absence). Only rows with IS_GOLD set count. Output is gold rows only. It does not feed thresholds: `tune-thresholds` takes separate tag annotations, and the importer never touches confidence or thresholds. The Line items tab is export-only; the importer ignores it. Recall cannot be measured from the sheet, since rows exist only where the pipeline emitted something (render_summary says so).

**Frozen rule:** review.xlsx is out with a reviewer and must not be touched. Everything below applies to a new export path only.

**Calibration buckets as flags.** `BUCKETS` in `evaluate/scoring.py` are 0.5-0.7, 0.7-0.9 and 0.9-1.0, and the eval reports precision per bucket. In a new export (new path), add a REVIEW_PRIORITY column per field row: "high" if the row sits in a bucket whose measured precision is under a target (say 0.9), or the field is unvalidated, or confidence is below 0.5 (unbucketed); else "low". Add a header filter and sort. Reviewers then see the measurably unreliable buckets first. The bucket table is read from the last eval output as an input file.

## 4. Research (nothing installed)

For native PDFs, pdfplumber (MIT, `extract_tables` with text strategy over word boxes) and Camelot (lattice for ruled tables, stream for borderless; needs Ghostscript/OpenCV) are the usual choices. Neither handles scans. I did not fetch their docs, so check details before adopting. The search returned these leads. img2table (open source, OCR output plus spatial heuristics; reported good on borderless tables, weaker on broken multi-column layouts) is a candidate dependency for scans. The arXiv paper [Design and Implementation of an OCR-Powered Pipeline for Table Extraction from Invoices](https://arxiv.org/pdf/2507.07029) describes OpenCV line removal, row contours and per-row Tesseract. Patent text (e.g. [US 12417649](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12417649)) describes anchoring rows on amount-like tokens, dropping words far from any amount, then finding columns. That heuristic needs no dependency and fits the existing geometry code. I did not verify invoice2data or docTR heuristics. Dependencies: pdfplumber, Camelot, img2table (all new); amount-anchored rows (none).

## Ranked proposals

1. **Line-item records.** Map table rows to description/qty/rate/amount using header words in `config/tables.yaml`; check qty*rate=amount and the sum against the total field. Gain: line items become checkable data and get a first consumer. Measure: row F1 on the gold grids plus the share of rows whose arithmetic holds. Effort: medium (~150 lines). Risk: low, additive; abstain when no header is found. Files: new `baseline/tables/line_items.py`, `config/tables.yaml`, `tests/test_tables.py`.
2. **Real table gold.** 20-30 client-style invoices and POs, hand-gridded by a human (not from the client machine by Claude), outside the frozen workbook. Gain: replaces the 5-table score. Measure: detection recall and adjacency F1 on client-like docs. Effort: medium, human work. Risk: low. Files: `tests/make_gold.py` or a new gold file.
3. **Amount-anchored row fallback for OCR.** When geometry abstains, anchor rows on numeric tokens and cluster columns on their x. Gain: may recover the refused scan (recall 0.80 -> up to 1.0 on the 5 tables; real effect unknown). Measure: `eval --pred-tables`; precision must stay 1.0. Effort: medium. Risk: medium (wrong grids); keep the 3 structural checks. Files: `baseline/tables/geometry.py`, `config/tables.yaml`.
4. **REVIEW_PRIORITY column in a new review export** (section 3). Gain: reviewer time goes to the bad buckets. Measure: share of wrong verdicts inside flagged rows. Effort: low. Risk: low if written to a new path only. Files: `baseline/workbook/exporter.py`, `columns.py`.
5. **Wire `has_line_items` to a real table.** Gain: removes a keyword guess. Measure: tagging F1 on gold. Effort: low-medium (needs an ordering decision, tables are a post pass). Risk: low. Files: tagger module, `baseline/cli.py`. Leave the graph as a spike.
