"""Generate a small labelled annotation set for training demos and tests.

Deterministic variants of the corpus templates. Real training data replaces
this wholesale, the point here is to exercise train and tune-thresholds.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from make_corpus import (
    BANK_STATEMENT,
    CONTRACT,
    ID_CARD,
    INVOICE_P1,
    INVOICE_P2,
    PURCHASE_ORDER,
    RECEIPT,
    SALARY,
)

VENDORS = ["Acme Traders", "Initech Supplies", "Umbrella Retail", "Stark Components",
           "Wayne Logistics", "Cyberdyne Parts"]
CITIES = ["Mumbai", "Pune", "Chennai", "Kolkata", "Jaipur", "Indore"]

# label -> (template text, tags that are true for it)
TEMPLATES = {
    "invoice": (INVOICE_P1 + "\n" + INVOICE_P2,
                ["gst_applicable", "has_line_items", "payment_pending", "signed",
                 "contains_pii", "bank_details_present"]),
    "receipt": (RECEIPT, ["payment_made", "bank_details_present", "signed"]),
    "salary_slip": (SALARY, ["contains_pii", "bank_details_present"]),
    "bank_statement": (BANK_STATEMENT, ["bank_details_present"]),
    "purchase_order": (PURCHASE_ORDER, ["gst_applicable", "has_line_items"]),
    "contract": (CONTRACT, ["signed", "foreign_currency"]),
    "id_document": (ID_CARD, ["contains_pii"]),
}


def build(out_file: str | Path, variants: int = 6) -> Path:
    rows = []
    for label, (body, tags) in sorted(TEMPLATES.items()):
        for i in range(variants):
            text = (
                body.replace("Acme Traders", VENDORS[i % len(VENDORS)])
                .replace("Maharashtra", CITIES[i % len(CITIES)])
                .replace("2024-0042", f"2024-{1000 + i * 7}")
                .replace("11,800.00", f"{11 + i},{800 + i}.00")
            )
            rows.append(
                {
                    "doc_id": f"{label}-{i:02d}",
                    "path": f"synthetic/{label}_{i:02d}.txt",
                    "label": label,
                    "tags": sorted(tags),
                    "text": text,
                }
            )
    out = Path(out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out


if __name__ == "__main__":
    target = build(sys.argv[1] if len(sys.argv) > 1 else "annotations.jsonl")
    n = sum(1 for _ in open(target, encoding="utf-8"))
    print(f"wrote {n} annotations to {target}")
