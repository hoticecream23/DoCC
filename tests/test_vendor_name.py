from baseline.metadata.fields import extract_fields


def _vendor(cfg, text):
    fields, _ = extract_fields(text, [text], cfg, "receipt")
    return [m.value for m in fields if m.field == "vendor_name"]


def test_plain_receipt_header_is_the_vendor(cfg):
    text = "SWC ENTERPRISE SDN BHD\n(1125830-V)\nNO. 5-7\nTEL : 03-6057 1377\nTOTAL 12.00\n"
    assert _vendor(cfg, text) == ["SWC ENTERPRISE SDN BHD"]


def test_header_after_a_tax_invoice_banner(cfg):
    text = "* TAX INVOICE *\nUNIHAKKA INTERNATIONAL SDN BHD\n(867388-U)\nTOTAL 8.50\n"
    assert _vendor(cfg, text) == ["UNIHAKKA INTERNATIONAL SDN BHD"]


def test_a_date_on_line_one_is_skipped(cfg):
    text = "07/03/2018 18:22\nPASAR MINI JIN SENG\nTEL : 07-3874904\nTOTAL 4.20\n"
    assert _vendor(cfg, text) == ["PASAR MINI JIN SENG"]


def test_vendor_regex_is_linear_on_repeated_keywords(cfg):
    """Three skipped lines full of "invoice" used to backtrack for minutes."""
    import time

    text = ("invoice " * 700 + chr(10)) * 3 + "SHOP" + chr(10)
    t = time.perf_counter()
    _vendor(cfg, text)
    assert time.perf_counter() - t < 1.0


def test_a_name_wrapped_onto_a_second_capital_line_is_joined(cfg):
    text = "3180301\nSECURE PARKING\nCORPORATION S/B\nRIVERWALK VILLAGE\nTOTAL 5.00\n"
    assert [" ".join(v.split()) for v in _vendor(cfg, text)] == ["SECURE PARKING CORPORATION S/B"]
    # An address or a tel line under the name is not part of it.
    assert _vendor(cfg, "SHOP ONE\nJALAN BESAR\nTOTAL 1.00\n") == ["SHOP ONE"]
    assert _vendor(cfg, "SHOP ONE\nNO. 5-7\nTOTAL 1.00\n") == ["SHOP ONE"]
    # A name that already ends in its company suffix is complete.
    assert _vendor(cfg, "SHOP ONE SDN BHD\nIOI MALL\nTOTAL 1.00\n") == ["SHOP ONE SDN BHD"]


def test_vendor_regex_is_linear_on_wrapped_capital_lines(cfg):
    import time

    for text in (
        ("invoice " * 700 + chr(10)) * 3 + "SHOP" + chr(10) + "A" * 39 + chr(10),
        "A " * 40 + chr(10) + "B" * 5000 + chr(10) + "TOTAL",
        ("AB" * 40 + chr(10)) * 300 + "TOTAL",
    ):
        t = time.perf_counter()
        _vendor(cfg, text)
        assert time.perf_counter() - t < 1.0
