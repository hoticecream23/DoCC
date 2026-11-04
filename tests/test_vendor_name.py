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
