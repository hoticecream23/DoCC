# Handoff

Read `PROJECT_STATE.md` for what exists and why. This file is only what to do
next.

Baseline is built, OCR is verified, the evaluation harness is in, and the
knowledge graph v0 spike is done. The blocker now is that there is no real
data, so every score is from 10 synthetic documents.

## Do these in order

1. **Get real documents.** Nothing below is meaningful without them. Even
   200 to 300 is enough to start. This is the only real blocker.

2. **Profile them before anything else.**
   `python -m baseline profile --input <real_dir> --output profile.md`.
   The native vs scanned split, the per field pattern hit rates, and the rule
   layer coverage decide where the effort goes.

3. **Build real gold.** Do not annotate cold. Run the baseline, then correct
   its output, which is far faster. Format is in `tests/make_gold.py`.
   **Include character offsets this time.** Synthetic gold could not provide
   them honestly, so span scoring has never run, and span accuracy is exactly
   what a layout model gets judged on.

4. **Score the baseline for real.**
   `python -m baseline eval --gold gold.jsonl --pred results.jsonl`.
   That number is the floor. Write it down. Everything from here is measured
   against it.

5. **Retrain and retune.** `train`, then `tune-thresholds`. Then revisit
   `classification.model_min_confidence`, currently 0.45 and far too
   conservative because it was fitted on synthetic data.

## Then the new approach

Do not upgrade everything uniformly. The error surface is lopsided:

- **Leave the checksummed IDs alone.** PAN, GSTIN, Aadhaar, IFSC, cards. A
  model cannot beat a checksum.
- **Classification gains will be small.** Rules plus TF-IDF is already strong.
- **Anchor fields are the weak point.** Amounts, dates in roles, party names.
  Brittle to layout, and confidences near 0.95 that mean nothing.
- **Tables and line items are missing entirely.** Biggest functional gap.

So the target is layout aware token classification over text plus bounding
boxes, and table structure. The pipeline already captures word boxes into
the sidecar and has never used them. That was the setup for this.

Keep the rule and checksum layer as a hard prior over any model. Best of
both, and it stays interpretable.

Gate every change with:

```bash
python -m baseline compare --gold gold.jsonl --a baseline.jsonl --b new.jsonl --fail-on-regression
```

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
- Line items as nodes, once table extraction exists. That is what makes
  spend analysis and duplicate detection actually good.
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
