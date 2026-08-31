# Baseline document processing pipeline

A deliberately cheap, deterministic document pipeline. No fine-tuned
transformers, no LLM calls. It exists to establish the floor that any neural
approach has to beat, and to fix an output schema that stays identical when
smarter components replace these ones.

## Install

```bash
pip install -r requirements.txt
```

Tesseract is optional but needed for scanned documents. Without it, native
text extraction still works and scanned pages record an error instead of
silently producing nothing.

```bash
winget install UB-Mannheim.TesseractOCR
```

The Windows installer does not add tesseract to PATH, so point the config at
it: `config/pipeline.yaml -> extraction.ocr.binary`. Set it to null to use
PATH instead. Currently installed and verified with Tesseract 5.4.0 (eng).

## Commands

Characterise a corpus before anything else. No models, just measurement.

```bash
python -m baseline profile --input corpus --output profile.md
```

Process documents into JSONL. Resumable, parallel, order stable.

```bash
python -m baseline run --input corpus --output results.jsonl --workers 4
```

Fit and persist the classifier and tagger from annotations.

```bash
python -m baseline train --annotations annotations.jsonl --output-model models
```

Fit per tag thresholds and write them back into `config/tags.yaml`.

```bash
python -m baseline tune-thresholds --annotations annotations.jsonl --predictions results.jsonl
```

## Try it without real data

```bash
python tests/make_corpus.py corpus
```

```bash
cd tests && python make_annotations.py ../annotations.jsonl && cd ..
```

## Layout

```
baseline/
  schema.py     pydantic models, the output contract
  extract.py    text extraction, OCR routing, bbox capture
  ocr.py        pluggable OCR engine interface, Tesseract by default
  rules.py      shared marker matching for classify and tagging
  classify.py   rule layer plus calibrated TF-IDF SVM
  metadata.py   field extraction, validators, normalizers
  tagging.py    multi label tagging and threshold tuning
  pipeline.py   orchestration, batch running
  profile.py    corpus characterisation
  cli.py        entry points
config/
  pipeline.yaml engine settings and thresholds
  classes.yaml  document taxonomy and rule patterns
  fields.yaml   field definitions, patterns, anchors, normalizers
  tags.yaml     tag vocabulary, patterns, thresholds, derived rules
```

## Config over code

Adding a document class, a field, or a tag is a YAML edit. No class name,
field name, tag name or pattern appears in any module. There is a test that
enforces this by defining a brand new class in a temporary config and
checking it classifies.

## Tests

```bash
python -m pytest tests -q
```

Covers every checksum validator with known good and known bad values, plus
the algorithmic guarantees (Verhoeff and Luhn are asserted to catch every
single digit error and every adjacent transposition), and offset correctness
on multi page synthetic documents. OCR tests skip themselves when no engine
is installed, so the suite passes either way.

See `PROJECT_STATE.md` for design decisions and current status, `HANDOFF.md`
for what to do next.
