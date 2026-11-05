# Round 1: layout-aware field extraction (audit, no code changed)

## 1. Map

Sidecar and boxes
- `BBoxSidecar` (doc_id, page_sizes, words): baseline/schema.py:197. `WordBox` has page, text, x0..y1, conf, char_start, char_end (schema.py ~185-195).
- Built in baseline/pipeline.py:176 from `res.words` and `res.page_sizes`; written by `write_sidecar` pipeline.py:211, to `<output>.bboxes` (pipeline.py:262). One `<doc_id>.json.gz` per doc.
- Filled when `extraction.capture_bboxes` is true (default). Native PDF: `page.get_text("words")` at extract.py:271-282; char offsets come from `_map_words_to_offsets` (extract.py:87), a find()-with-cursor guess, null when it cannot place a word. Scanned PDF / OCR rescue: extract.py:264, 302. Image: extract.py:404-406. OCR words carry exact offsets (ocr.py:100-127, built while the OCR text is assembled). Offsets are rebased per page at extract.py:449-456.
- Plain `.txt`/DOCX input: no words at all. The SROIE `text` route therefore has an empty sidecar.

Geometry helpers, baseline/tables/geometry.py (all take word dicts with x0,y0,x1,y1,text)
- `_percentile` :20, `_median` :31: numeric helpers.
- `_overlap` :35, `_overlap_ratio` :40: 1-D interval overlap (used for both y-row and x-column tests).
- `_Cell` :54, `_Line` :66: dataclasses for a run of words and a text line.
- `_group_lines` :76: words to lines by vertical overlap, sorted left to right.
- `_estimate_space_width` :102, `_segment_cells` :119, `_make_cell` :133: split a line into cells at wide gaps.
- `_build_columns` :149: cluster cells into columns by x overlap.
- `_geometry_bands` :197, `detect_geometry` :233: table detection per page.
- `_cell_from_words` :300: builds a TableCell with offsets sliced from `full`.

Does fields.py have box access? Yes, partly. `extract_fields` already receives `words` and `page_sizes` (fields.py:297-304; call at pipeline.py:136). Only the `positional` strategy uses them (`_positional_candidates` fields.py:172): fixed normalised regions from registered templates, offsets = min/max char_start/char_end of words inside, value re-sliced from `full` (fields.py:218-222). `_anchor_candidates` (fields.py:91-168) is text-only: label `find`, then regex in a char window; `direction` today is `after|before|both` over text. No field in config/fields.yaml uses a layout strategy for label-value pairing.

## 2. Opportunity (no gold opened)

File used: scratchpad/b1/sroie_b1.jsonl (post-50b5c92; 233 invoice_number reads). Label regex: invoice no|inv no|receipt no|bill no|doc no (361 docs).

| field | docs with label | emitted | label present, none emitted |
|---|---|---|---|
| invoice_number (text route) | 183 | 233 total reads | 14 |
| total_amount (label total, amount due, grand total) | 355 | 345 | 10 |

Side datapoint, image/OCR route (datasets/sroie/test/pred_images.jsonl, which has real boxes): 134 docs with an invoice label, 110 with a number, 24 without.

Sample of 10 of the 14 invoice_number misses (judged from text only):
- 7 have a blank label and no number anywhere in the text (X51005763999, 6619842, 5677337, 7339160, 6619328, 6619790, 6334766). The value is not in the OCR text. Boxes cannot help.
- 1 absent too (5764154, "INV NO. :" then QTY).
- 1 box-fixable: X51005806695. "INVOICE NO. :" then RECEIPT DATE and CASHIER, and `RCP180228-0101-77106` sits a few lines below. A same-row test finds it; text pairing cannot, because the label block is interleaved.
- 1 text-fixable: X51006334139, `RECEIPT NO.[00519738]`. The value regex has no `[` in its prefix class. One-char regex fix.
So layout would add about 1 in 10 (1-2 of 14) here. The "11 docs far from label" in PLAN-audit mostly have no value in the text at all (OCR line order dropped it, or it is a barcode). Layout cannot recover what the text lacks.

All 10 total_amount misses viewed:
- 6 are `TOTAL INCLUDES GST x%` followed by the amount on the next line (5447842/41/59/51, 9447842, 5806718). Text-fixable by a label variant. No boxes needed.
- 2 are stacked blocks (`TOTAL SALES (INCLUSIVE GST) RM / ROUNDING ... / CASH RM / CHANGE RM` then values 98.89 ...; X51005764161, 6619782). Column pairing would find the first value; a text stacked-block rule could too.
- 1 is `TL :RM` then 33.80 (5433548): alias fix.
- 1 has only SUBTOTAL, no total (6619509): absent.

Conclusion: on SROIE text the gain from boxes is about 3 docs of 361; most gain is cheap text fixes. The real layout payoff is the OCR image route and client scans/forms, where text order is column-scrambled.

## 3. Smallest layout-aware mechanism

Add `anchor.direction: right_of_label | below_label` (new values beside after/before/both). In `_anchor_candidates` the label `find` loop stays as is. When direction is a layout one and words exist, map the label span [pos, label_end) to its words (those with char_start >= pos and char_end <= label_end; first/last word give the label box). Then:
- right_of_label: take words whose y-range has `_overlap_ratio` >= 0.5 with the label box, x0 >= label x1 - eps, same page, sorted by x0. Join the first digit-bearing token (or run until a wide gap) as the value.
- below_label: words with x-overlap >= 0.4 (`_overlap_ratio`) and y0 >= label y1, nearest first, within a max gap of 3 line heights; first token matching `value_regex`.
The value span = (min char_start, max char_end) of the chosen words, and the value is re-sliced from `full` (same contract as fields.py:218-222), then run through the existing `value_regex`, `reject_value`, validator and confidence code. Distance for confidence decay = box distance in line heights, scaled to the window.

Reuse: `_overlap`/`_overlap_ratio` (geometry.py:35,40) made public or imported; `_group_lines` (:76) to get the label's row; the Candidate/validator/normalizer path in fields.py unchanged. The `words` argument must be threaded into `_anchor_candidates` (today only `_positional_candidates` gets it; call sites fields.py:264 and :337).

Offsets: every word has char_start/char_end; the value is always `full[start:end]`, so the existing contract check holds. Words with null offsets are skipped as value words. A word whose offsets were mis-mapped (native `_map_words_to_offsets` first-occurrence fallback, extract.py:98) fails a per-word `full[start:end] == word.text` guard and is dropped.

Empty sidecar (text/DOCX, or capture off): the layout branch returns [] and the strategy list falls through to the text rule. Preferred config: boxes tried only after the text rule misses (`strategies: [anchor, anchor_layout, pattern]` or an anchor flag). Then nothing changes where text already works, so regression risk is near zero.

## 4. Research

invoice2data is text plus regex per vendor template (YAML keywords and per-field regex, pdftotext/tesseract input); it has no box geometry, so it is the same class as our current anchor rules and needs no model ([github](https://github.com/m3nu/invoice2data), [pypi](https://pypi.org/project/invoice2data/)). Layout-model approaches (LayoutLMv3, BROS, LAMBERT; Hugging Face SROIE checkpoints) take words plus 2D boxes and tag tokens; they need a trained model and torch/transformers, and an SROIE fine-tune degrades on other layouts ([BROS](https://arxiv.org/pdf/2108.04539), [LAMBERT](https://arxiv.org/pdf/2002.08087), [LayoutLMv3 SROIE card](https://huggingface.co/Tanishq71/sroie-layoutlmv3)). Graph methods treat each box as a node with relative-coordinate edges ([paper](https://arxiv.org/pdf/2009.11746)): model needed. Rule-based geometric systems use key-phrase "RightOf" relations to pick the value ([USPTO 9552516](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/9552516)): this is our proposed heuristic and needs no model. LMDX uses LLMs over layout-tagged text ([paper](https://arxiv.org/pdf/2309.10952)): model needed. The searches did not return docTR KIE docs or the SROIE competition heuristics; those are not verified. Takeaway: the heuristic route is plain rules over boxes; anything learned adds a dependency.

## 5. Proposals (ranked)

1. Text-only fixes first. Add `TOTAL INCLUDES GST`, a `TL` alias, `[` in the invoice_number value prefix; a stacked-block rule for total_amount. Gain: about 8 of 10 total_amount misses and 1 invoice_number on SROIE text (361 docs, up to about +2% total_amount recall). Effort: small. Risk: low to moderate (shared regex; run SROIE and client suites). Files: config/fields.yaml, tests.
2. `right_of_label | below_label` anchor direction as a fallback after the text rule (section 3). Gain: about 1 of 14 on SROIE text (cannot run there: no boxes); likely more on the 24 image-route misses and client forms/scans. Effort: medium (about 80 lines plus tests). Risk: low, since it fires only when the text rule misses. Files: baseline/metadata/fields.py, baseline/tables/geometry.py (export `_overlap_ratio`), config/fields.yaml, tests.
3. Two-line vendor names: use `_group_lines` on the top of page 0 to join a second line with similar left x and height, with a company-suffix check (SDN BHD). Gain: fixes truncation (docs/parts/audit.md:43); unmeasured, no gold. Boxes only, so image/PDF. Effort: medium. Risk: moderate (could absorb the address line). Files: fields.py, fields.yaml.
4. Measure before building: run the image-route SROIE with a prototype of #2 and compare with the text rule on the 24 misses. Effort: small. Risk: none. This decides whether #2 pays off.

Caveat: label-regex counts are approximate (emitted counts include other labels such as inv#, cash #). "Miss" means label present and no field of that name emitted.
