# Round 1: OCR quality on scanned images

## Current state (file:line)

| Item | Setting | Where |
|---|---|---|
| Engine call | `image_to_data`, DICT output, `lang` default eng, timeout 120 s | baseline/ocr.py:81-87 |
| psm | `--psm {opts.psm}` default 3; no `--oem`, no `-c` options, no dpi hint | baseline/ocr.py:80; config/pipeline.yaml:16 |
| Image upscale | LANCZOS to short side 1000 px, only if smaller | baseline/extract.py:119-144; pipeline.yaml `min_short_side: 1000` |
| Pink paper | red channel only plus psm 6, gated on median RGB | baseline/extract.py:147-178; pipeline.yaml `pink_paper` |
| PDF route | render at dpi 300, no other prep | baseline/extract.py:212-240, 335 |
| Word confidence | kept as 0..1 (negative clamped to 0); no word filtering; mean used for page confidence | baseline/ocr.py:_assemble; extract.py:404 |
| Image path | `_extract_image`: upscale, pink gate, OCR | baseline/extract.py:384-410 |

NOT done: deskew, orientation detection (OSD), binarisation (Otsu/adaptive/Sauvola), despeckle, contrast normalisation, border removal, DPI metadata, per-document psm choice, extra language packs, low-confidence word filtering, `--oem` choice. HANDOFF.md (line 230-270) records grayscale as a no-op and psm sweep as per-document, with best-of-N by engine confidence called a trap.

## The 40 labelled images

They are the image files under `Classification_Doc/<class>/` (jpg, png, webp; 6 class folders). Accuracy 0.450 to 0.525 (HANDOFF.md:246) comes from `python -m baseline run` then `eval`/`compare` against gold.jsonl. I did not open gold. Many are tiny: 17 of 40 are under 640 px short side (HANDOFF.md:236), and the 8 sampled range from 387x516 to 2550x2756 px. So "300 dpi" is not a meaningful target for most; they hold little text pixel detail.

## Measured variants (8 images, scratch only)

Script: scratchpad/r1ocr/v.py. Files: Sample_voter_id, Tax_statement, Aadhar_card, pan_card, PostDapartment_Payslip, Receipt, Equipment_po, payment-receipt-template. All variants include the current 1000 px upscale unless noted. OpenCV (`cv2`) is NOT installed; PIL, numpy, pytesseract only.

| Variant | Mean word conf | Mean words |
|---|---|---|
| base (upscale 1000, psm 3) | 0.627 | 107 |
| autocontrast + Otsu binarise | 0.591 | 106 |
| psm 4 | 0.605 | 104 |
| psm 6 | 0.563 | 156 |
| long edge to 3300 px (instead of 1000) | 0.584 | 136 |
| long edge 3300 + Otsu | 0.591 | 154 |

Per file: Otsu helped pan_card (0.538 to 0.603) and Sample_voter_id (0.340 to 0.365) but hurt Receipt (0.601 to 0.386) and Tax_statement (0.481 to 0.400). psm 6 helped the payslip (0.731 to 0.764) and voter id but hurt receipt template (0.945 to 0.688) and PO (0.821 to 0.678). 3300 px upscale helped Aadhar (0.558 to 0.612) and pan_card (0.538 to 0.641) and hurt Receipt and receipt template. Classification per variant NOT measured: it needs the full harness run against gold, which I did not run. Mean confidence is also a weak proxy (more words at lower confidence can mean more text found).

Result: no global variant beats the current base. This matches HANDOFF.md. Gains are per-document, with opposite signs.

## Research

Tesseract's own docs (summarised at [DeepWiki tessdoc](https://deepwiki.com/tesseract-ocr/tessdoc/6-improving-ocr-quality)) say it wants about 300 dpi, and that 5.0 added adaptive Otsu and Sauvola thresholding, tunable with `thresholding_method` (note: Tesseract already binarises internally; external binarisation often hurts clean or coloured input, as measured above). Common advice also covers deskew via `minAreaRect` (OpenCV) and border removal ([Nutrient guide](https://www.nutrient.io/blog/tesseract-python-guide/)); vendor accuracy percentages lack methodology, so do not trust them. OCRmyPDF applies, in fixed order, rotate, background removal, deskew, then optional unpaper clean ([docs](https://ocrmypdf.readthedocs.io/en/latest/advanced.html)); rotate and deskew rely on Tesseract OSD. From background knowledge, not verified in this search: docTR and PaddleOCR use neural text detection plus an orientation/angle classifier, and PaddleOCR offers UVDoc unwarping. New dependencies: deskew via OpenCV or scikit-image (new); unpaper (external binary); docTR/PaddleOCR (torch or paddle, large, and a change of engine behind the existing `OCREngine` interface in baseline/ocr.py).

## Ranked proposals (none touches classification abstention rules)

1. **Per-document psm/prep selection using an external quality signal** (for example, a cheap keyword-hit or dictionary-word ratio on the OCR text, not engine confidence). Try psm 3 then 6 (or 12 for ID cards) only when page confidence is below 0.6. Gain: moderate, maybe 3-6 of the 14 abstained images; the per-file data above shows swings of +/-0.1 to 0.3. Effort: medium. Regression risk: medium (gate on low confidence and compare via `baseline compare`). Files: baseline/extract.py (`_extract_image`), baseline/ocr.py, config/pipeline.yaml, tests.
2. **Route ID cards separately** (HANDOFF says residue is dominated by Aadhaar, PAN, voter ID, passport): a prep branch gated on a cheap signal (small size, high colour variance) using psm 11/12 and 3300 px upscale plus Otsu, which helped pan_card and Aadhar here. Gain: up to 4 images. Effort: medium. Risk: low if gated. Files: baseline/extract.py, pipeline.yaml.
3. **Orientation and deskew via Tesseract OSD** (`image_to_osd`, no new dependency) and rotate by 90/180 when script confidence is high; add small-angle deskew only if scikit-image or OpenCV is approved. Gain: unknown, no skewed file found in the 8 sampled; needs a scan of all 40 for rotation first. Effort: low for OSD. Risk: low. Files: baseline/extract.py.
4. **Confidence filtering of words** (drop words below conf 0.3 with length 1 or non-alphanumeric) before the text reaches the classifier, to cut OCR noise (e.g. the 8 noise aadhaar candidates seen before). Gain: small, mostly precision of field extraction. Effort: low. Risk: low-medium; keep word boxes. Files: baseline/ocr.py `_assemble`, baseline/extract.py.
5. **Swap in a neural engine (docTR or PaddleOCR) behind `OCREngine`** for scanned images only. Gain: largest potential for low-res, patterned backgrounds; needs a new heavy dependency and GPU/CPU cost. Effort: high. Risk: medium. Files: baseline/ocr.py (registry), requirements, pipeline.yaml. Out of scope without approval.

Not recommended: global Otsu, global psm 6, larger upscale (all measured worse on mean confidence).
