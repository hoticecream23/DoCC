# PLAN-demo: a local demo UI for the baseline pipeline

Date: 2026-11-05. Branch: `small-fixes`. Purpose: show a document going through the pipeline live, and a written walkthrough of how the system works.

## Decisions

- Stack: Python stdlib `http.server` + one static HTML page with vanilla JS. No new dependency, no build step, works offline. (FastAPI is installed but its upload path needs python-multipart, which is not.)
- Scope: baseline pipeline only (extract, classify, metadata, tagging). Tables and graph are post passes and stay out of the demo. Cascade v2 needs a VLM server and stays out; the doc explains it.
- Bind 127.0.0.1:8765 only. Uploads are capped at 25 MB and written to a temp dir.

## API contract (both agents build to this)

| Route | Request | Response |
| --- | --- | --- |
| `GET /` | | `demo/index.html` |
| `GET /api/corpus` | | JSON list of file names in `corpus/` |
| `POST /api/run?corpus=<name>` | no body | run result (below) |
| `POST /api/run?filename=<name>` | raw file bytes as body | run result (below) |

Run result, 200: `{"record": <DocumentRecord.model_dump(mode="json")>, "pages": [{"index": 0, "png_base64": "...", "width": W, "height": H}], "total_ms": float}`. PDFs render at 110 dpi via PyMuPDF; images via PIL; `.txt` / `.docx` return `pages: []` and the page shows the text instead. Errors: `{"error": "..."}` with 400 (bad input) or 500.

DocumentRecord fields the page uses: `classification.{label,confidence,all_scores,method}`, `metadata[].{field,value,normalized_value,page,char_start,char_end,confidence,validated,method}`, `tags[].{tag,confidence,method}`, `text.{full,method,extraction_confidence}`, `diagnostics.{page_count,is_scanned,timings_ms,errors}`.

## Tasks

| # | Task | Files | Done check |
| --- | --- | --- | --- |
| D1 | Server | demo/__init__.py, demo/__main__.py, demo/server.py, tests/test_demo_server.py | `python -m demo` serves; test posts corpus/invoice_native.pdf and gets a record with a label |
| D2 | Page | demo/index.html | loads against the contract; shows pages, class, fields with span highlight, tags, timings |
| D3 | Walkthrough doc | docs/How_It_Works.md | reads top to bottom for a demo audience; numbers match docs/parts/audit.md |
| D4 | Browser verification | .claude/launch.json | orchestrator runs the demo in the in-app browser on two corpus files |

## Status log

- 2026-11-05: plan written, D1-D3 dispatched in parallel.
- 2026-11-05: D2 1d1da75, D3 72b9068 + 5239f82 (cascade status corrected: a real run happened 2026-10-22, numbers from session notes), D1 8421356 (520 tests). D4: verified in the in-app browser on invoice_native.pdf (native, 13 fields, 7 tags), receipt_scanned.pdf (OCR, 1.1 s extract), salary_slip.docx (no page image, text shown); no console errors. In progress: Host/Origin check on the server after a security review flag.
- 2026-11-05: 900f791 Host/Origin check (522 tests). Re-verified in the browser after restart on purchase_order.pdf. **Demo complete.** Run: `python -m demo`, open http://127.0.0.1:8765. Walkthrough: docs/How_It_Works.md.
