# Handoff

Read `PROJECT_STATE.md` for what exists and why. This file is only what to do
next. Baseline is built and passing, OCR is installed and verified, so
everything here is about pointing it at real data.

## Do these in order

1. **Profile the real corpus before anything else.**
   `python -m baseline profile --input <real_dir> --output profile.md`.
   Read the native vs scanned split, the per field pattern hit rates, and the
   rule layer coverage. Those three numbers decide where the effort goes.

2. **Tune `extraction.native_char_threshold`.**
   Default 100 chars is a guess. The profile will show whether real PDFs with
   thin text layers are being wrongly kept as native.

3. **Retrain on real annotations.**
   Annotations are JSONL with `path` (or inline `text`), `label`, `tags`.
   `python -m baseline train --annotations real.jsonl --output-model models`.
   Then lower or raise `classification.model_min_confidence`, currently 0.45.
   Synthetic training made it too conservative to trust.

4. **Run `tune-thresholds` on real tag data** and let it write back to
   `config/tags.yaml`. Use `--dry-run` first to see what it wants to change.

5. **Expect OCR quality to drop on real scans.** It is verified only against
   clean synthetic images, where it hits 0.99 character similarity. Real skew
   and noise will hurt. If it does, try raising `extraction.ocr.dpi` from 300
   and check whether `psm` 3 suits your layouts, or 4 or 6 for tables.

## Then, if the numbers justify it

- Register positional templates in `fields.yaml` for any high volume vendor
  layout. `templates: []` today. The strategy is written and reads bboxes.
- Widen `classes.yaml` for whatever the profile shows landing in `unknown`.
- Calibrate anchor confidences against measured accuracy. They currently
  cluster near 0.95, which is fine for ranking but not meaningful as a
  probability.

## Do not break these

- `PAGE_SEP` in `schema.py`. Changing it invalidates every stored offset.
- The `extra="forbid"` schema models. Adding a field is a deliberate,
  versioned change, not a casual one.
- Config over code. If you find yourself typing a class, field or tag name
  into a `.py` file, it belongs in YAML instead.

## Machine specific setting

`config/pipeline.yaml -> extraction.ocr.binary` is an absolute path to
`tesseract.exe` on this machine. The Windows installer does not add it to
PATH. If you move the repo, either fix that path, set it to null and put
tesseract on PATH, or the OCR route records an error and native extraction
carries on regardless.

Only the `eng` language pack is installed.

## Gotcha worth knowing

PAN `ABCDE1234F` fails validation on purpose. `D` is not a real holder type
code. If real data has PANs being rejected, widen
`fields.yaml -> pan -> validator_args.entity_types` rather than loosening
the validator.
