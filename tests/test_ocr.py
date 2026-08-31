"""OCR route: routing, confidence, offsets, hybrid documents.

Skipped wholesale when no OCR engine is installed, so the suite still passes
on a machine that only has native PDFs to deal with.
"""

import difflib

import pytest

from baseline.extract import extract
from baseline.ocr import OCRWord, get_engine, ocr_confidence
from baseline.pipeline import process_document, run_batch
from baseline.schema import ExtractionMethod


def _engine_ready(cfg) -> bool:
    opts = cfg.extract_opts().get("ocr", {})
    engine = get_engine(opts.get("engine", "tesseract"))
    engine.configure(opts)
    return engine.available()


@pytest.fixture(scope="module", autouse=True)
def _require_ocr(cfg):
    if not _engine_ready(cfg):
        pytest.skip("no OCR engine available")


# ---------------------------------------------------------------- routing

def test_scanned_pdf_routes_to_ocr(corpus, cfg):
    res = extract(corpus / "receipt_scanned.pdf", cfg)
    assert res.method == ExtractionMethod.OCR
    assert res.page_methods == [ExtractionMethod.OCR]
    assert res.is_scanned is True
    assert res.has_native_text is False
    assert not res.errors


def test_scanned_pdf_actually_produces_text(corpus, cfg):
    res = extract(corpus / "receipt_scanned.pdf", cfg)
    body = res.pages[0]
    assert len(body) > 150
    assert "RECEIPT" in body.upper()
    assert "HDFC0001234" in body


def test_native_pdf_is_still_never_ocred(corpus, cfg):
    """The whole point of routing. Installing OCR must not change this."""
    res = extract(corpus / "invoice_native.pdf", cfg)
    assert ExtractionMethod.OCR not in res.page_methods


# ---------------------------------------------------------------- hybrid

def test_hybrid_document_routes_each_page_separately(corpus, cfg):
    res = extract(corpus / "invoice_hybrid.pdf", cfg)
    assert res.method == ExtractionMethod.HYBRID
    assert res.page_methods == [ExtractionMethod.NATIVE, ExtractionMethod.OCR]
    assert res.has_native_text is True
    assert res.is_scanned is False


def test_hybrid_offsets_survive_the_route_change(corpus, cfg):
    """Page 0 is native, page 1 is OCR. Offsets must hold across that seam."""
    record, _ = process_document(corpus / "invoice_hybrid.pdf", cfg)
    assert record.verify_offsets() == []
    pages_used = {m.page for m in record.metadata}
    assert pages_used == {0, 1}, "expected fields from both routes"
    for m in record.metadata:
        assert record.text.full[m.char_start:m.char_end] == m.value


def test_hybrid_word_boxes_from_both_routes_are_canonical(corpus, cfg):
    record, sidecar = process_document(corpus / "invoice_hybrid.pdf", cfg)
    pages = {w.page for w in sidecar.words}
    assert pages == {0, 1}
    for w in sidecar.words:
        if w.char_start is not None:
            assert record.text.full[w.char_start:w.char_end] == w.text


def test_hybrid_confidence_sits_between_the_two_routes(corpus, cfg):
    res = extract(corpus / "invoice_hybrid.pdf", cfg)
    native_conf = float(cfg.extract_opts().get("native_confidence", 0.98))
    ocr_page = [
        c for m, c in zip(res.page_methods, res.page_confidences)
        if m == ExtractionMethod.OCR
    ][0]
    assert ocr_page < res.confidence() < native_conf


# ---------------------------------------------------------------- confidence

def test_ocr_confidence_is_lower_than_native(corpus, cfg):
    ocr = extract(corpus / "receipt_scanned.pdf", cfg).confidence()
    native = extract(corpus / "receipt.txt", cfg).confidence()
    assert 0.0 < ocr < native


def test_ocr_confidence_is_length_weighted():
    """A long confident word must outweigh a short unconfident one."""
    words = [
        OCRWord("commencement", 0, 0, 1, 1, 0.99, 0, 12),
        OCRWord("x", 0, 0, 1, 1, 0.10, 13, 14),
    ]
    conf = ocr_confidence(words)
    assert conf > 0.9
    assert conf < 0.99


def test_ocr_confidence_of_nothing_is_zero():
    assert ocr_confidence([]) == 0.0


# ---------------------------------------------------------------- quality

def test_ocr_recovers_the_same_fields_as_the_native_route(corpus, cfg):
    """Same content, two routes. Extracted values must agree."""
    native, _ = process_document(corpus / "receipt.txt", cfg)
    ocr, _ = process_document(corpus / "receipt_scanned.pdf", cfg)

    fa = {m.field: m.normalized_value for m in native.metadata}
    fb = {m.field: m.normalized_value for m in ocr.metadata}
    shared = set(fa) & set(fb)
    assert "ifsc" in shared and "account_number" in shared
    for k in shared:
        assert fa[k] == fb[k], f"{k} disagrees between routes"


def test_ocr_text_is_close_to_the_native_text(corpus, cfg):
    native = " ".join(extract(corpus / "receipt.txt", cfg).pages[0].split())
    ocr = " ".join(extract(corpus / "receipt_scanned.pdf", cfg).pages[0].split())
    assert difflib.SequenceMatcher(None, native, ocr).ratio() > 0.95


def test_checksum_validation_still_works_on_ocred_text(corpus, cfg):
    record, _ = process_document(corpus / "receipt_scanned.pdf", cfg)
    ifsc = [m for m in record.metadata if m.field == "ifsc"]
    assert ifsc and ifsc[0].validated is True


def test_scanned_document_still_classifies(corpus, cfg):
    record, _ = process_document(corpus / "receipt_scanned.pdf", cfg)
    assert record.classification.label == "receipt"


# ---------------------------------------------------------------- determinism

def test_ocr_output_is_reproducible(corpus, cfg):
    a = extract(corpus / "receipt_scanned.pdf", cfg)
    b = extract(corpus / "receipt_scanned.pdf", cfg)
    assert a.pages == b.pages
    assert a.page_confidences == b.page_confidences
    assert [(w.text, w.char_start) for w in a.words] == [
        (w.text, w.char_start) for w in b.words
    ]


def test_batch_with_ocr_is_byte_identical_across_worker_counts(corpus, cfg, tmp_path):
    import shutil

    import yaml

    cfgdir = tmp_path / "cfg"
    cfgdir.mkdir()
    for name in ("classes.yaml", "fields.yaml", "tags.yaml"):
        shutil.copyfile(cfg.config_dir / name, cfgdir / name)
    pipe = yaml.safe_load((cfg.config_dir / "pipeline.yaml").read_text(encoding="utf-8"))
    pipe.setdefault("runtime", {})["deterministic_timings"] = True
    (cfgdir / "pipeline.yaml").write_text(yaml.safe_dump(pipe), encoding="utf-8")

    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    run_batch(corpus, a, config_dir=str(cfgdir), workers=1)
    run_batch(corpus, b, config_dir=str(cfgdir), workers=3)
    assert a.read_bytes() == b.read_bytes()


# ---------------------------------------------------------------- engine interface

def test_engine_is_pluggable_and_null_engine_is_never_available():
    null = get_engine("null")
    assert null.available() is False
    assert null.name == "null"


def test_unknown_engine_name_fails_loudly():
    with pytest.raises(KeyError):
        get_engine("definitely_not_an_engine")
