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
pip install -r requirements.txt
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

161 tests. Every checksum is tested against known good and known bad values.
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
  ocr.py         OCR engine interface, Tesseract by default
  rules.py       keyword and regex matching, shared
  classify.py    document type
  metadata.py    fields, checksums, normalisers
  tagging.py     tags and threshold tuning
  pipeline.py    ties it together, batch runs
  profile.py     corpus measurement
  cli.py         commands
config/          all the vocabulary and settings
tests/
```

`PROJECT_STATE.md` has the design decisions and current status.
`HANDOFF.md` has what to do next.
