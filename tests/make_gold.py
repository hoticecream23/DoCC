"""Ground truth for the synthetic corpus.

Written by hand from the templates in make_corpus.py. This is what a careful
annotator would mark, deliberately NOT what the baseline currently produces.
The gaps between the two are the point.

Gold carries values, not character offsets. Span level gold needs real
annotation, so span scoring stays dark until there is real data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from make_corpus import RECEIPT

INVOICE_FIELDS = [
    {"field": "vendor_name", "normalized_value": "Acme Traders Private Limited"},
    {"field": "buyer_name", "normalized_value": "Globex Corporation"},
    {"field": "gstin", "normalized_value": "27AAPFU0939F1ZV"},
    {"field": "pan", "normalized_value": "AAPFU0939F"},
    {"field": "email", "normalized_value": "billing@acmetraders.example"},
    {"field": "phone", "normalized_value": "9876543210"},
    {"field": "invoice_number", "normalized_value": "INV-2024-0042"},
    {"field": "invoice_date", "normalized_value": "2024-03-14"},
    {"field": "due_date", "normalized_value": "2024-04-13"},
    {"field": "tax_amount", "normalized_value": "1800.00"},
    {"field": "total_amount", "normalized_value": "11800.00"},
    {"field": "account_number", "normalized_value": "502010123456789"},
    {"field": "ifsc", "normalized_value": "HDFC0001234"},
]
INVOICE_TAGS = [
    "gst_applicable", "has_line_items", "payment_pending", "signed",
    "contains_pii", "bank_details_present", "multi_page",
]

RECEIPT_FIELDS = [
    # The receipt quotes the invoice it settles.
    {"field": "invoice_number", "normalized_value": "INV-2024-0042"},
    {"field": "invoice_date", "normalized_value": "2024-03-20"},
    {"field": "total_amount", "normalized_value": "11800.00"},
    {"field": "account_number", "normalized_value": "502010123456789"},
    {"field": "ifsc", "normalized_value": "HDFC0001234"},
]
RECEIPT_TAGS = ["payment_made", "bank_details_present", "signed"]

GOLD = [
    {
        "path": "corpus/invoice_native.pdf",
        "label": "invoice",
        "tags": INVOICE_TAGS,
        "metadata": INVOICE_FIELDS,
    },
    {
        "path": "corpus/invoice_hybrid.pdf",
        "label": "invoice",
        "tags": INVOICE_TAGS,
        "metadata": INVOICE_FIELDS,
    },
    {
        "path": "corpus/receipt.txt",
        "label": "receipt",
        "tags": RECEIPT_TAGS,
        "metadata": RECEIPT_FIELDS,
    },
    {
        # Same content as receipt.txt but scanned, so it also gives us a CER.
        "path": "corpus/receipt_scanned.pdf",
        "label": "receipt",
        "tags": RECEIPT_TAGS,
        "metadata": RECEIPT_FIELDS,
        "text": RECEIPT,
    },
    {
        "path": "corpus/bank_statement.pdf",
        "label": "bank_statement",
        "tags": ["bank_details_present"],
        "metadata": [
            {"field": "account_number", "normalized_value": "502010123456789"},
            {"field": "ifsc", "normalized_value": "HDFC0001234"},
        ],
    },
    {
        "path": "corpus/purchase_order.pdf",
        "label": "purchase_order",
        "tags": ["gst_applicable", "has_line_items"],
        "metadata": [
            {"field": "vendor_name", "normalized_value": "Acme Traders Private Limited"},
            {"field": "buyer_name", "normalized_value": "Globex Corporation"},
            {"field": "gstin", "normalized_value": "27AAPFU0939F1ZV"},
            {"field": "invoice_number", "normalized_value": "PO-2024-7781"},
            {"field": "invoice_date", "normalized_value": "2024-03-02"},
            {"field": "total_amount", "normalized_value": "5000.00"},
        ],
    },
    {
        "path": "corpus/salary_slip.docx",
        "label": "salary_slip",
        "tags": ["contains_pii", "bank_details_present"],
        "metadata": [
            {"field": "pan", "normalized_value": "AAAPZ1234C"},
            {"field": "total_amount", "normalized_value": "55800.00"},
            {"field": "account_number", "normalized_value": "911010098765432"},
            {"field": "ifsc", "normalized_value": "UTIB0000123"},
        ],
    },
    {
        "path": "corpus/contract.txt",
        "label": "contract",
        "tags": ["signed", "foreign_currency"],
        "metadata": [],
    },
    {
        "path": "corpus/id_card.txt",
        "label": "id_document",
        "tags": ["contains_pii"],
        "metadata": [{"field": "pan", "normalized_value": "AAAPZ1234C"}],
    },
    {
        # Unreadable. The correct behaviour is to abstain, so unknown is right.
        "path": "corpus/broken.pdf",
        "label": "unknown",
        "tags": [],
        "metadata": [],
    },
]


def build(out_file: str | Path) -> Path:
    out = Path(out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        for row in GOLD:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out


if __name__ == "__main__":
    target = build(sys.argv[1] if len(sys.argv) > 1 else "gold.jsonl")
    fields = sum(len(r["metadata"]) for r in GOLD)
    tags = sum(len(r["tags"]) for r in GOLD)
    print(f"wrote {len(GOLD)} gold documents, {fields} fields, {tags} tags to {target}")
