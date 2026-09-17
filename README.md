# Baseline document pipeline

Takes a folder of documents and turns each one into a single JSON record:
the text, what kind of document it is, the fields found in it, and a set of
tags.

It uses only cheap, classical methods. Regex, checksums, TF-IDF, linear
SVMs, Tesseract. No fine-tuned transformers and no LLM calls.

The point is to set a floor. Before spending anything on a neural approach,
you should know what plain regex and a linear model already get you. The
output schema is fixed, so when you do swap in a smarter model later, the
same evaluation harness scores it without any changes.

## Quick start

```bash
pip install -e .
```

Make some sample documents to play with:

```bash
python tests/make_corpus.py corpus
```

Look at the corpus before doing anything else. This runs no models, it just
measures what you have:

```bash
python -m baseline profile --input corpus --output profile.md
```

Process everything into JSONL:

```bash
python -m baseline run --input corpus --output results.jsonl --workers 4
```

Build the hand written gold and training annotations for that corpus:

```bash
cd tests && python make_annotations.py ../annotations.jsonl && python make_gold.py ../gold.jsonl && cd ..
```

Score a run against ground truth:

```bash
python -m baseline eval --gold gold.jsonl --pred results.jsonl --output report.md
```

Compare two runs and see what got better or worse:

```bash
python -m baseline compare --gold gold.jsonl --a baseline.jsonl --b new.jsonl
```

Pull the tables out:

```bash
python -m baseline tables --input results.jsonl --output tables.jsonl --report tables.md
```

Build a knowledge graph across the documents:

```bash
python -m baseline graph --input results.jsonl --report graph.md --dot graph.dot
```

## What comes out

One JSON object per document, one per line. Trimmed for readability:

```json
{
  "doc_id": "4f28266c42cdeb99",
  "filename": "invoice_hybrid.pdf",
  "text": {
    "method": "hybrid",
    "page_methods": ["native", "ocr"],
    "char_count": 601,
    "extraction_confidence": 0.9387
  },
  "classification": {
    "label": "invoice",
    "confidence": 0.95,
    "method": "rule"
  },
  "metadata": [
    {
      "field": "gstin",
      "value": "27AAPFU0939F1ZV",
      "normalized_value": "27AAPFU0939F1ZV",
      "page": 0,
      "char_start": 86,
      "char_end": 101,
      "confidence": 1.0,
      "method": "regex_checksum",
      "validated": true
    }
  ],
  "tags": [
    {"tag": "gst_applicable", "confidence": 0.98, "method": "derived"}
  ]
}
```

`doc_id` is a hash of the file contents, so the same document always gets
the same id no matter what it is called or where it lives.

Every `char_start` and `char_end` points into the document text. You can
always slice the text with them and get the value back. This is checked on
every write.

## How it works

**Text.** PDFs are checked page by page. If a page already has a text layer
it is read directly. Only pages without one go to OCR. A document can be
half and half, and the record says which pages went which way. DOCX and TXT
are read directly. Images always go to OCR.

**Classification.** First a set of rules from `config/classes.yaml`. If a
strong rule matches, that is the answer. Otherwise a TF-IDF model decides.
If neither is confident, the label is `unknown`. It will not guess.

**Metadata.** Each field is found by regex, or by looking near a label word
like "Invoice No", or by reading a fixed spot on a known template. IDs are
checked with real checksums, so a PAN, Aadhaar, GSTIN, IFSC or card number
that does not add up is marked `validated: false` instead of being trusted.

**Tags.** Separate from classification. A document has one type but can have
many tags. Tags come from keywords, a model, or from the metadata itself. A
validated GSTIN is much stronger evidence than the word "GST" appearing
somewhere, and the tag records which source it came from.

## Adding things

Adding a document type, a field, or a tag means editing YAML. You do not
touch any Python.

| Want to add | Edit |
| --- | --- |
| A document type | `config/classes.yaml` |
| A field to extract | `config/fields.yaml` |
| A tag | `config/tags.yaml` |
| A setting or threshold | `config/pipeline.yaml` |
| A graph entity or edge rule | `config/graph.yaml` |
| A table detection threshold | `config/tables.yaml` |

There is a test that proves this. It defines a brand new document type in a
temporary config file and checks it works with no code changes.

## OCR

Tesseract is only needed for scanned documents. Without it, everything else
still works and scanned pages record an error rather than failing silently.

```bash
winget install UB-Mannheim.TesseractOCR
```

The Windows installer does not put Tesseract on your PATH. Point the config
at it instead, in `config/pipeline.yaml`:

```yaml
extraction:
  ocr:
    binary: "C:/Program Files/Tesseract-OCR/tesseract.exe"
```

Set that to `null` if Tesseract is already on your PATH.

## Training

The rules work on their own. Models are optional and make it better.

Annotations are JSONL, one per line, with a file path (or the text inline),
a label, and tags:

```json
{"path": "docs/inv1.pdf", "label": "invoice", "tags": ["gst_applicable"]}
```

```bash
python -m baseline train --annotations annotations.jsonl --output-model models
```

Then fit the per tag cutoffs and write them back into the config:

```bash
python -m baseline tune-thresholds --annotations annotations.jsonl --predictions results.jsonl
```

Add `--dry-run` to see what it would change first.

## Scoring

`eval` reads any JSONL that matches the schema. It does not import the
pipeline and does not care how a record was made, so the baseline and
anything you build later are scored by identical code.

Gold files are JSONL with a path or doc_id, a label, tags, and the fields
that should have been found:

```json
{"path": "docs/inv1.pdf", "label": "invoice",
 "tags": ["gst_applicable"],
 "metadata": [{"field": "gstin", "normalized_value": "27AAPFU0939F1ZV"}]}
```

Metadata is scored two ways. Value scoring asks whether the answer was
right. Span scoring asks whether it pointed at the right place in the text.
Span scoring needs character offsets in the gold file and is skipped when
they are absent.

Classification reports coverage at a fixed error rate, not just accuracy.
A system allowed to say `unknown` should be judged on how much it can answer
while staying under an error budget.

There is also a `validated` precision figure. Any field the pipeline marked
validated claims its format proved it, so that number should be exactly 1.0.
Anything less means a validator is lying.

Tables are scored too, when the gold row carries a `tables` key and you pass
`--pred-tables`. Gold is just the grid a reader would copy off the page:

```json
{"path": "docs/inv1.pdf", "label": "invoice",
 "tables": [{"page": 1, "cells": [["Qty", "Rate"], ["2", "10.00"]]}]}
```

Two numbers come back. Cell F1 asks whether the values came out at all.
Adjacency F1 asks whether each value ended up beside the right neighbours,
which is the question that matters: a figure that slides one column left is
perfect content and a useless table. A document with no table should carry
`"tables": []` rather than no key, so that a table invented out of nothing is
counted against precision.

`compare` scores two files against the same gold and lists regressions. Pass
`--fail-on-regression` to make it exit non zero, which makes it usable as a
CI gate. Table metrics join the gate only when both runs were given a tables
file, so a run without one is not read as a collapse to zero.

## Tables

`tables` is a separate pass over the results file, like the graph. It never
touches the document record, so tables can be rebuilt from an old run without
re-extracting anything.

Two strategies run in the order set in `config/tables.yaml`, first hit wins:

- **geometry** reads the word boxes in the bbox sidecar. Columns come out of
  where the words actually sit on the page. This is the right signal for real
  PDFs and scans.
- **text_grid** reads runs of whitespace that are blank on every line of a
  band. It is the only signal available for `.txt` and `.docx`, which carry no
  geometry at all, and it rescues text whose alignment survived into the text
  layer but not into the coordinates.

Every cell carries character offsets into the same canonical text as the
metadata, verified at write time by the same rule: the offsets must slice the
text back to the cell exactly, or they are null rather than approximate.

The v0 rule is the graph's rule: **a wrong table is worse than a missing one.**
A band whose columns do not line up is refused, and the refusal is recorded
with its reason in `diagnostics.abstained` and in the report. If a table you
expected is not there, the report says why.

## Knowledge graph

`graph` is a separate pass over the results file. It never touches the
document schema, so the record contract stays frozen.

The rule for v0 is that **only fields whose format proves them may create or
merge an entity**. Names are recorded but never used to join, because fuzzy
name matching is how a graph quietly fills with wrong edges.

That constraint buys a real join for free. Characters 3 to 12 of a GSTIN are
the holder PAN, so a document carrying only a GSTIN and one carrying only a
PAN resolve to the same organisation with no guessing.

It answers:

- which organisations appear, and in how many documents
- which documents reference which, so invoice to receipt chains fall out
- duplicate candidates, same reference and same total within a class
- orphan references, a document citing something no document owns

Where the evidence is ambiguous it declines. If two party names compete for
one verified identifier and neither is clearly closer, no role edge is drawn.
The organisation is still recorded as mentioned.

`--dot` writes graphviz source:

```bash
dot -Tpng graph.dot -o graph.png
```

## Other things worth knowing

Runs are resumable. If the output file already has a document, it is
skipped, so you can stop and restart a long job.

The same input gives the same output, byte for byte, whether you run with
one worker or eight. Timings are the only thing that varies, and there is a
config flag to pin those too.

A broken file never kills the run. The error is recorded on that document
and the batch carries on.

Logs are JSON on stderr, not print statements.

## Tests

```bash
python -m pytest tests -q
```

338 tests. Every checksum is tested against known good and known bad values.
Verhoeff and Luhn are also checked to catch every single digit error and
every swapped pair of digits, which is what those algorithms promise. Offsets
are tested across multi page documents and across documents where half the
pages are OCR.

OCR tests skip themselves if Tesseract is not installed, so the suite passes
either way.

## Layout

```
baseline/
  schema.py      the output format, the contract
  extract.py     text extraction and page routing
  scan.py        which files in a corpus are documents
  ocr.py         OCR engine interface, Tesseract by default
  rules.py       keyword and regex matching, shared
  classify.py    document type
  metadata/      fields, checksums, normalisers
  tagging.py     tags and threshold tuning
  pipeline.py    ties it together, batch runs
  profile.py     corpus measurement
  evaluate/      scoring, imports nothing from the pipeline
  graph.py       knowledge graph over a results file
  tables/        table extraction over a results file
  workbook/      the review sheet, out and back in
  cli.py         commands
config/          all the vocabulary and settings
tests/
```

`PROJECT_STATE.md` has the design decisions and current status.
`HANDOFF.md` has what to do next.
