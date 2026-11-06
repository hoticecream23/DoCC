"""The text quality gate on a real corpus document."""


def test_a_clean_native_pdf_is_not_rescued(corpus, cfg):
    """The gate must be invisible on documents whose text layer is fine."""
    from baseline.extract import extract
    from baseline.schema import ExtractionMethod

    res = extract(corpus / "invoice_native.pdf", cfg)
    assert ExtractionMethod.OCR_RESCUED not in res.page_methods
    assert res.method == ExtractionMethod.NATIVE
