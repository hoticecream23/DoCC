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


def test_stacked_labels_read_the_first_value_under_them(cfg):
    # Flattened text puts the labels in a column and their values after it;
    # the first value belongs to the first label.
    assert _inv(cfg, "DOC NO.\nCASHIER\nSALESPERSON :\n: CS00013065\n: USER\n") == ["CS00013065"]
    assert _inv(cfg, "DOC NO. :\nCASHIER :\nSALESPERSON:\nSO00050251\nUSER\n") == ["SO00050251"]
    # No colon to pair label and value: a barcode under the labels is not it.
    assert _inv(cfg, "INVOICE NO\nDATE\nCOUNTER\n6928835430158 HOOK-3015\n") == []


def test_stacked_labels_never_take_a_date_as_the_number(cfg):
    # The number was left blank, so the first value in the block is the date.
    assert _inv(cfg, "INV NO:\nDATE :\n16/12/2017\nTOTAL QTY:\n") == []
    assert _inv(cfg, "BILL NO:\nCASHIER:\nBILL DT:\n12/01/2017 08:44:46 PM\n") == []


def test_a_bracketed_receipt_number_is_read(cfg):
    assert _inv(cfg, "TAX INVOICE\nRECEIPT NO.[00519738]\n05/01/17 10:55\n") == ["00519738"]
