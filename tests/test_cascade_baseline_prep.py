import gzip
import json

from baseline.schema import (
    SCHEMA_VERSION,
    BBoxSidecar,
    ClassifyMethod,
    MetadataField,
    MetadataMethod,
    WordBox,
)


def test_schema_version_is_1_2_0():
    assert SCHEMA_VERSION == "1.2.0"


def test_vlm_methods_round_trip_through_json():
    m = MetadataField(
        field="x",
        value="a",
        normalized_value="a",
        page=0,
        char_start=0,
        char_end=1,
        confidence=0.5,
        method=MetadataMethod.VLM_RULE_AGREE,
        validated=False,
    )
    back = MetadataField.model_validate(m.model_dump(mode="json"))
    assert back.method is MetadataMethod.VLM_RULE_AGREE
    assert MetadataMethod("vlm") is MetadataMethod.VLM
    assert ClassifyMethod("vlm") is ClassifyMethod.VLM


def test_write_sidecar_is_public_and_skips_a_sidecar_with_no_words(tmp_path):
    from baseline.pipeline import write_sidecar

    doc_id = "0" * 16
    write_sidecar(BBoxSidecar(doc_id=doc_id), tmp_path / "b")
    assert not (tmp_path / "b").exists()

    sc = BBoxSidecar(
        doc_id=doc_id,
        page_sizes=[(10.0, 10.0)],
        words=[WordBox(page=0, text="a", x0=0, y0=0, x1=1, y1=1)],
    )
    write_sidecar(sc, tmp_path / "b")
    with gzip.open(tmp_path / "b" / f"{doc_id}.json.gz") as fh:
        assert json.loads(fh.read())["doc_id"] == doc_id


def test_tag_facts_matches_what_the_pipeline_feeds_the_tagger(corpus, cfg):
    from baseline.pipeline import process_document, tag_facts

    record, _ = process_document(corpus / "invoice_native.pdf", cfg)
    facts = tag_facts(record.text, record.diagnostics.page_count, record.diagnostics.is_scanned)
    assert facts == {
        "page_count": record.diagnostics.page_count,
        "char_count": record.text.char_count,
        "extraction_confidence": record.text.extraction_confidence,
        "is_scanned": record.diagnostics.is_scanned,
        "method": record.text.method.value,
    }
