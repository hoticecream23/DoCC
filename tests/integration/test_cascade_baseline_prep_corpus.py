"""Cascade prep against a real pipeline record."""


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
