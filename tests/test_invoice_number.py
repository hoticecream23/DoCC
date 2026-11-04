from baseline.metadata.fields import extract_fields


def _inv(cfg, text):
    fields, _ = extract_fields(text, [text], cfg, "receipt")
    return [m.value for m in fields if m.field == "invoice_number"]


def test_invoice_no_label_reads_the_number_on_the_next_line(cfg):
    assert _inv(cfg, "TAX INVOICE\nINVOICE NO\n: PEGIV-1030765\nDATE\n: 15/01/2019\n") == ["PEGIV-1030765"]


def test_receipt_hash_and_doc_no_labels_are_read(cfg):
    assert _inv(cfg, "RECEIPT #: CS00031383 DATE: 07/12/2017\n") == ["CS00031383"]
    assert _inv(cfg, "DOC NO.\n: CS00035862\nDATE: 13/03/2018\n") == ["CS00035862"]


def test_an_empty_label_does_not_read_the_next_label_or_its_value(cfg):
    # The number was left blank: neither DATE nor the date under it is the number.
    assert _inv(cfg, "INVOICE NO\nDATE\n: 06/04/2018\n") == []
    assert _inv(cfg, "INV NO. :\nTel: 03-6273 2163\n") == []


def test_a_customer_order_label_is_not_this_documents_number(cfg):
    # LPO NO ends in "po no", a label, but names the customer's purchase order.
    assert _inv(cfg, "Invoice Date: 17-06-2020\nLPO No : 6332\n") == []


def test_a_payment_receipt_carries_the_invoice_it_settles(cfg):
    # The graph links receipt to invoice on this value, so the quoted invoice
    # outranks the receipt's own number.
    text = "PAYMENT RECEIPT\nReceipt No: RCP-2024-118\nPaid in full against Invoice INV-2024-0042\n"
    assert _inv(cfg, text) == ["INV-2024-0042"]
