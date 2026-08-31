"""Generate a deterministic synthetic corpus.

Used by the tests and handy for a demo run. Same bytes every time, so doc_ids
are stable across machines.
"""

from __future__ import annotations

import sys
from pathlib import Path

INVOICE_P1 = """TAX INVOICE

Acme Traders Private Limited
Sold By: Acme Traders Private Limited
GSTIN: 27AAPFU0939F1ZV
PAN: AAPFU0939F
Email: billing@acmetraders.example
Phone: 9876543210

Invoice No: INV-2024-0042
Invoice Date: 14/03/2024
Due Date: 13/04/2024
Place of Supply: Maharashtra

Bill To: Globex Corporation
"""

INVOICE_P2 = """Qty  Description              Rate       Amount
2    Widget assembly          2500.00    5000.00
1    Freight and handling     5000.00    5000.00

HSN: 8479
CGST 9%: 900.00
SGST 9%: 900.00
Total Tax: 1800.00
Grand Total: Rs. 11,800.00

Amount Due by the date above. Authorised Signatory
Bank: HDFC Bank
Account Number: 502010123456789
IFSC: HDFC0001234
"""

RECEIPT = """PAYMENT RECEIPT

Receipt No: RCP-2024-118
Date: 20/03/2024

Received with thanks from Globex Corporation
Amount Paid: Rs. 11,800.00
Paid in full against Invoice INV-2024-0042

Mode: NEFT
Account Number: 502010123456789
IFSC: HDFC0001234

Authorised Signatory
"""

SALARY = """SALARY SLIP
Month: March 2024

Employee Name: Priya Sharma
Employee ID: EMP-2291
PAN: AAAPZ1234C
UAN: 100234567890

Basic Pay        45000.00
HRA              18000.00
Gross            63000.00
Deductions        7200.00
Net Pay          55800.00

Account Number: 911010098765432
IFSC: UTIB0000123
"""

BANK_STATEMENT = """STATEMENT OF ACCOUNT

Bank: HDFC Bank
Account Number: 502010123456789
IFSC: HDFC0001234
Period: 01/03/2024 to 31/03/2024

Date        Description            Debit      Credit     Balance
01/03/2024  Opening Balance                              125000.00
05/03/2024  NEFT Globex                       11800.00   136800.00
12/03/2024  Vendor payment         22000.00              114800.00
31/03/2024  Closing Balance                              114800.00
"""

PURCHASE_ORDER = """PURCHASE ORDER

PO No: PO-2024-7781
Date: 02/03/2024

Vendor Name: Acme Traders Private Limited
GSTIN: 27AAPFU0939F1ZV

Ship To: Globex Corporation Warehouse 4
Delivery Date: 20/03/2024

Qty  Description         Rate      Amount
2    Widget assembly     2500.00   5000.00

Total Amount: 5000.00
"""

CONTRACT = """SERVICE AGREEMENT

This Agreement is made on 01/01/2024 between Acme Traders Private Limited,
hereinafter referred to as the Service Provider, and Globex Corporation,
hereinafter referred to as the Client.

WHEREAS the parties wish to record the terms and conditions of their
engagement, and WHEREAS the Client requires the services described herein.

1. Term. This Agreement runs for twelve months.
2. Fees. The Client shall pay USD 45000 annually.
3. Indemnity. Each party indemnifies the other against third party claims.

IN WITNESS WHEREOF the parties have executed this Agreement.

For and on behalf of Acme Traders Private Limited
Authorised Signatory
"""

ID_CARD = """INCOME TAX DEPARTMENT
GOVERNMENT OF INDIA

Permanent Account Number
AAAPZ1234C

Name: PRIYA SHARMA
Date of Birth: 11/07/1991
"""


def _pdf(path: Path, pages: list[str]) -> None:
    import fitz

    doc = fitz.open()
    for body in pages:
        page = doc.new_page(width=595, height=842)
        page.insert_text((56, 72), body, fontsize=10, fontname="helv")
    # Strip metadata so the bytes do not carry a timestamp.
    doc.set_metadata({})
    doc.save(str(path), garbage=4, deflate=True)
    doc.close()


def _scanned_pdf(path: Path, body: str) -> None:
    """A page with no text layer at all, to exercise the OCR route."""
    import fitz
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (1240, 1754), "white")
    draw = ImageDraw.Draw(img)
    y = 60
    for line in body.splitlines():
        draw.text((70, y), line, fill="black")
        y += 26
    tmp = path.with_suffix(".tmp.png")
    img.save(tmp)

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_image(page.rect, filename=str(tmp))
    doc.set_metadata({})
    doc.save(str(path), garbage=4, deflate=True)
    doc.close()
    tmp.unlink()


def _hybrid_pdf(path: Path, native_body: str, scanned_body: str) -> None:
    """Page 0 has a text layer, page 1 is an image. Exercises the hybrid route."""
    import fitz
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (1240, 1754), "white")
    draw = ImageDraw.Draw(img)
    y = 60
    for line in scanned_body.splitlines():
        draw.text((70, y), line, fill="black")
        y += 26
    tmp = path.with_suffix(".tmp.png")
    img.save(tmp)

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((56, 72), native_body, fontsize=10, fontname="helv")
    page2 = doc.new_page(width=595, height=842)
    page2.insert_image(page2.rect, filename=str(tmp))
    doc.set_metadata({})
    doc.save(str(path), garbage=4, deflate=True)
    doc.close()
    tmp.unlink()


def _docx(path: Path, body: str) -> None:
    import docx

    d = docx.Document()
    for line in body.splitlines():
        d.add_paragraph(line)
    d.save(str(path))


def build(out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    _pdf(out / "invoice_native.pdf", [INVOICE_P1, INVOICE_P2])
    _pdf(out / "bank_statement.pdf", [BANK_STATEMENT])
    _pdf(out / "purchase_order.pdf", [PURCHASE_ORDER])
    _scanned_pdf(out / "receipt_scanned.pdf", RECEIPT)
    _hybrid_pdf(out / "invoice_hybrid.pdf", INVOICE_P1, INVOICE_P2)
    _docx(out / "salary_slip.docx", SALARY)
    (out / "contract.txt").write_text(CONTRACT, encoding="utf-8", newline="\n")
    (out / "id_card.txt").write_text(ID_CARD, encoding="utf-8", newline="\n")
    (out / "receipt.txt").write_text(RECEIPT, encoding="utf-8", newline="\n")
    # Deliberately broken, to prove one bad file does not kill the batch.
    (out / "broken.pdf").write_bytes(b"%PDF-1.4\nthis is not a pdf at all\n")
    return out


if __name__ == "__main__":
    target = build(sys.argv[1] if len(sys.argv) > 1 else "corpus")
    for p in sorted(target.iterdir()):
        print(f"{p.name:26} {p.stat().st_size:>8} bytes")
