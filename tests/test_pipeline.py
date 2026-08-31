"""End to end behaviour: ids, routing, schema, determinism, resilience."""

import json
import shutil

import pytest
import yaml

from baseline.classify import classify
from baseline.config import load_config
from baseline.extract import extract, iter_documents
from baseline.pipeline import dumps, existing_doc_ids, process_document, run_batch
from baseline.schema import DocumentRecord, ExtractionMethod, compute_doc_id
from baseline.tagging import apply_tags, derived_scores


# ---------------------------------------------------------------- doc_id

def test_doc_id_is_content_addressed_not_path_addressed(corpus, tmp_path):
    src = corpus / "contract.txt"
    moved = tmp_path / "totally_different_name.txt"
    shutil.copyfile(src, moved)
    assert compute_doc_id(src) == compute_doc_id(moved)


def test_doc_id_changes_when_content_changes(corpus, tmp_path):
    src = corpus / "contract.txt"
    edited = tmp_path / "edited.txt"
    edited.write_text(src.read_text(encoding="utf-8") + "x", encoding="utf-8")
    assert compute_doc_id(src) != compute_doc_id(edited)


def test_doc_id_is_16_hex_chars(corpus):
    did = compute_doc_id(corpus / "contract.txt")
    assert len(did) == 16
    int(did, 16)


# ---------------------------------------------------------------- routing

def test_native_pdf_is_never_sent_to_ocr(corpus, cfg):
    res = extract(corpus / "invoice_native.pdf", cfg)
    assert res.method == ExtractionMethod.NATIVE
    assert res.page_methods == [ExtractionMethod.NATIVE, ExtractionMethod.NATIVE]
    assert res.has_native_text is True
    assert res.is_scanned is False
    assert not res.errors


def test_docx_takes_the_direct_path_with_no_ocr(corpus, cfg):
    res = extract(corpus / "salary_slip.docx", cfg)
    assert res.method == ExtractionMethod.NATIVE
    assert "Net Pay" in res.pages[0]
    assert not res.errors


def test_text_file_extracts_natively(corpus, cfg):
    res = extract(corpus / "contract.txt", cfg)
    assert res.method == ExtractionMethod.NATIVE
    assert res.page_count == 1


def test_pages_never_contain_the_page_separator(corpus, cfg):
    from baseline.schema import PAGE_SEP

    for p in iter_documents(corpus):
        for page in extract(p, cfg).pages:
            assert PAGE_SEP not in page


def test_unsupported_extension_is_reported_not_raised(tmp_path, cfg):
    odd = tmp_path / "thing.xyz"
    odd.write_text("hello", encoding="utf-8")
    res = extract(odd, cfg)
    assert res.errors and "unsupported" in res.errors[0]
    assert res.pages == [""]


def test_native_word_boxes_are_captured(corpus, cfg):
    res = extract(corpus / "invoice_native.pdf", cfg)
    assert len(res.words) > 20
    assert all(w.x1 >= w.x0 and w.y1 >= w.y0 for w in res.words)


# ---------------------------------------------------------------- schema

def test_record_validates_and_round_trips(corpus, cfg):
    record, _ = process_document(corpus / "invoice_native.pdf", cfg)
    blob = json.loads(dumps(record))
    again = DocumentRecord.model_validate(blob)
    assert again.doc_id == record.doc_id
    assert again.text.full == record.text.full
    assert len(again.metadata) == len(record.metadata)


def test_record_rejects_unknown_fields(corpus, cfg):
    record, _ = process_document(corpus / "invoice_native.pdf", cfg)
    blob = json.loads(dumps(record))
    blob["surprise"] = 1
    with pytest.raises(Exception):
        DocumentRecord.model_validate(blob)


def test_text_full_is_consistent_with_pages(corpus, cfg):
    for p in iter_documents(corpus):
        record, _ = process_document(p, cfg)
        record.text.check_consistent()
        assert record.text.char_count == len(record.text.full)


def test_offsets_verify_for_every_document_in_the_corpus(corpus, cfg):
    for p in iter_documents(corpus):
        record, _ = process_document(p, cfg)
        assert record.verify_offsets() == []


def test_word_boxes_use_canonical_offsets(corpus, cfg):
    record, sidecar = process_document(corpus / "invoice_native.pdf", cfg)
    mapped = [w for w in sidecar.words if w.char_start is not None]
    assert mapped
    for w in mapped:
        assert record.text.full[w.char_start:w.char_end] == w.text


# ---------------------------------------------------------------- resilience

def test_a_broken_pdf_does_not_kill_the_run(corpus, cfg):
    record, _ = process_document(corpus / "broken.pdf", cfg)
    assert record.diagnostics.errors, "the failure should be recorded"
    assert record.classification.label == "unknown"
    assert record.text.full == ""


def test_batch_completes_despite_a_broken_file(corpus, tmp_path):
    out = tmp_path / "results.jsonl"
    stats = run_batch(corpus, out, workers=1)
    assert stats["written"] == stats["found"]
    assert stats["failed"] == 0
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == stats["found"]


def test_every_stage_is_timed(corpus, cfg):
    record, _ = process_document(corpus / "invoice_native.pdf", cfg)
    assert set(record.diagnostics.timings_ms) == {"extract", "classify", "metadata", "tag"}


# ---------------------------------------------------------------- determinism

def _config_with_fixed_timings(cfg, tmp_path):
    """Copy the config and zero the timings, the one field that varies."""
    cfgdir = tmp_path / "detcfg"
    cfgdir.mkdir()
    for name in ("classes.yaml", "fields.yaml", "tags.yaml"):
        shutil.copyfile(cfg.config_dir / name, cfgdir / name)
    pipe = yaml.safe_load((cfg.config_dir / "pipeline.yaml").read_text(encoding="utf-8"))
    pipe.setdefault("runtime", {})["deterministic_timings"] = True
    (cfgdir / "pipeline.yaml").write_text(yaml.safe_dump(pipe), encoding="utf-8")
    return str(cfgdir)


def _strip_timings(path):
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        rec["diagnostics"]["timings_ms"] = {}
        out.append(json.dumps(rec, sort_keys=True))
    return out


def test_same_input_gives_byte_identical_output(corpus, cfg, tmp_path):
    """With timings pinned, two runs are identical byte for byte."""
    cfgdir = _config_with_fixed_timings(cfg, tmp_path)
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    run_batch(corpus, a, config_dir=cfgdir, workers=1)
    run_batch(corpus, b, config_dir=cfgdir, workers=1)
    assert a.read_bytes() == b.read_bytes()


def test_parallel_output_is_byte_identical_to_serial(corpus, cfg, tmp_path):
    cfgdir = _config_with_fixed_timings(cfg, tmp_path)
    a, b = tmp_path / "serial.jsonl", tmp_path / "parallel.jsonl"
    run_batch(corpus, a, config_dir=cfgdir, workers=1)
    run_batch(corpus, b, config_dir=cfgdir, workers=3)
    assert a.read_bytes() == b.read_bytes()


def test_default_config_is_deterministic_apart_from_timings(corpus, tmp_path):
    """Real timings are on by default, so only that field may differ."""
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    run_batch(corpus, a, workers=1)
    run_batch(corpus, b, workers=3)
    assert _strip_timings(a) == _strip_timings(b)


def test_timings_are_recorded_by_default(corpus, tmp_path):
    out = tmp_path / "t.jsonl"
    run_batch(corpus, out, workers=1)
    recs = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    non_zero = [r for r in recs if any(v > 0 for v in r["diagnostics"]["timings_ms"].values())]
    assert non_zero, "default config should record real per stage timings"


# ---------------------------------------------------------------- resume

def test_resume_skips_documents_already_written(corpus, tmp_path):
    out = tmp_path / "results.jsonl"
    first = run_batch(corpus, out, workers=1)
    second = run_batch(corpus, out, workers=1)
    assert second["skipped"] == first["found"]
    assert second["written"] == 0


def test_resume_picks_up_only_the_new_document(corpus, tmp_path):
    out = tmp_path / "results.jsonl"
    run_batch(corpus, out, workers=1)
    (corpus / "extra.txt").write_text("PAYMENT RECEIPT\nReceipt No: R-1\n", encoding="utf-8")
    try:
        stats = run_batch(corpus, out, workers=1)
        assert stats["written"] == 1
        assert len(existing_doc_ids(out)) == stats["found"]
    finally:
        (corpus / "extra.txt").unlink()


def test_no_resume_reprocesses_everything(corpus, tmp_path):
    out = tmp_path / "results.jsonl"
    run_batch(corpus, out, workers=1)
    stats = run_batch(corpus, out, workers=1, resume=False)
    assert stats["skipped"] == 0
    assert stats["written"] == stats["found"]


# ---------------------------------------------------------------- config over code

def test_a_new_document_class_needs_only_a_yaml_edit(tmp_path, cfg):
    """Nothing in the modules may know the class list."""
    cfgdir = tmp_path / "config"
    cfgdir.mkdir()
    for name in ("pipeline.yaml", "fields.yaml", "tags.yaml"):
        shutil.copyfile(cfg.config_dir / name, cfgdir / name)

    classes = {
        "version": 1,
        "classes": [
            {
                "name": "veterinary_report",
                "markers": [
                    {"kind": "regex", "pattern": r"\bveterinary\s+report\b",
                     "scope": "first_page", "specificity": 0.97}
                ],
            }
        ],
    }
    (cfgdir / "classes.yaml").write_text(yaml.safe_dump(classes), encoding="utf-8")

    custom = load_config(cfgdir)
    text = "VETERINARY REPORT\nPatient: a cat.\n"
    result = classify(text, [text], custom, model=None)
    assert result.label == "veterinary_report"
    assert result.method.value == "rule"


def test_classifier_never_forces_a_guess(cfg):
    junk = "qq ww ee rr tt yy uu ii oo pp"
    result = classify(junk, [junk], cfg, model=None)
    assert result.label == "unknown"


# ---------------------------------------------------------------- tagging

def test_derived_tag_beats_a_keyword(corpus, cfg):
    record, _ = process_document(corpus / "invoice_native.pdf", cfg)
    by_tag = {t.tag: t for t in record.tags}
    assert by_tag["gst_applicable"].method.value == "derived"
    assert by_tag["gst_applicable"].confidence > 0.9


def test_document_level_derived_tags_fire(corpus, cfg):
    record, _ = process_document(corpus / "invoice_native.pdf", cfg)
    assert "multi_page" in {t.tag for t in record.tags}

    single, _ = process_document(corpus / "contract.txt", cfg)
    assert "multi_page" not in {t.tag for t in single.tags}


def test_derived_rule_needs_a_validated_field(cfg):
    from baseline.schema import MetadataField, MetadataMethod

    unvalidated = [
        MetadataField(
            field="gstin", value="27AAPFU0939F1ZX", normalized_value="27AAPFU0939F1ZX",
            page=0, char_start=0, char_end=15, confidence=0.5,
            method=MetadataMethod.REGEX_FORMAT, validated=False,
        )
    ]
    assert "gst_applicable" not in derived_scores(unvalidated, {}, cfg)

    validated = unvalidated[0].model_copy(update={"validated": True})
    assert "gst_applicable" in derived_scores([validated], {}, cfg)


def test_tags_are_multi_label_not_exclusive(corpus, cfg):
    record, _ = process_document(corpus / "invoice_native.pdf", cfg)
    assert len(record.tags) >= 3
    assert len({t.tag for t in record.tags}) == len(record.tags)


def test_tags_respect_per_tag_thresholds(cfg):
    text = "cgst sgst igst gstin place of supply"
    tags = apply_tags(text, [text], cfg, metadata=[], facts={}, model=None)
    assert "gst_applicable" in {t.tag for t in tags}
