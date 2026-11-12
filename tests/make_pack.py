"""A tiny client pack over the synthetic corpus, with made-up class and field names."""

from __future__ import annotations

import csv
from pathlib import Path

import yaml

CABINET = "Acme Finance"
SLUG = "acme_finance"
CLASSES = [
    {"name": "Supplier Bill", "aliases": ["Tax Invoice"], "has_samples": True},
    {"name": "Payment Advice", "aliases": ["Payment Receipt"], "has_samples": True},
    {"name": "Buying Order", "aliases": ["Purchase Order"], "has_samples": True},
    {"name": "other", "aliases": [], "has_samples": False},
]


def _field(name, dtype, role, origin="document", date_format=None):
    return {"name": name, "dtype": dtype, "date_format": date_format, "role": role, "origin": origin}


FIELDS = {
    "Supplier Bill": [
        _field("Vendor Code", "text", "required", origin="erp"),
        _field("Invoice Date", "date", "optional", date_format="DD/MM/YYYY"),
        _field("Invoice No", "text", "key"),
    ],
    "Payment Advice": [_field("Receipt No", "text", "key")],  # no extractor maps it: anchored on its own label
    "Buying Order": [_field("PO No", "text", "key")],
}
# (doc_id, file in the corpus, label, key_gold)
MANIFEST = [
    ("a000000000000001", "invoice_native.pdf", "Supplier Bill", "INV-2024-0042"),
    ("a000000000000002", "receipt.txt", "Payment Advice", "RCP-2024-118"),
    ("a000000000000003", "purchase_order.pdf", "Buying Order", "PO-2024-7781"),
    ("a000000000000004", "contract.txt", "", ""),
]
HEADER = ["doc_id", "rel_path", "in_zip", "case_id", "label", "label_source", "key_gold", "pages", "size_kb"]


def build(pack_dir: str | Path) -> Path:
    """Write the pack and return its directory."""
    root = Path(pack_dir)
    cab = root / SLUG
    cab.mkdir(parents=True, exist_ok=True)
    info = {
        "version": "1",
        "created_at": "2026-01-01T00:00:00",
        "share_root": "",
        "cabinets": [{"slug": SLUG, "name": CABINET, "documents": len(MANIFEST), "types_with_samples": 3}],
    }
    (root / "pack.yaml").write_text(yaml.safe_dump(info), encoding="utf-8")
    (cab / "taxonomy.yaml").write_text(yaml.safe_dump({"classes": CLASSES}), encoding="utf-8")
    (cab / "fields.yaml").write_text(yaml.safe_dump(FIELDS), encoding="utf-8")
    with open(cab / "manifest.csv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=HEADER)
        writer.writeheader()
        for doc_id, rel, label, key in MANIFEST:
            source = "folder" if label else "none"
            writer.writerow(
                {
                    "doc_id": doc_id,
                    "rel_path": rel,
                    "label": label,
                    "label_source": source,
                    "key_gold": key,
                    "size_kb": 1,
                }
            )
    return root
