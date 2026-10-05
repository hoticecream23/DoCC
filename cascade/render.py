"""Page images for the model, one per page the extractor read.

An image file gives frame 0 only, because that is the only frame the
baseline extractor reads, and the model must see the page the text came from.
"""

from __future__ import annotations

import io
from pathlib import Path

from baseline.extract import IMAGE_EXT, PDF_EXT


def _png(img, max_long_edge: int) -> bytes:
    from PIL import Image

    img = img.convert("RGB")
    long_edge = max(img.size)
    if long_edge > max_long_edge:
        k = max_long_edge / long_edge
        size = (max(1, round(img.width * k)), max(1, round(img.height * k)))
        img = img.resize(size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _pdf(path: Path, max_pages: int, dpi: int, max_long_edge: int) -> list[bytes]:
    import fitz
    from PIL import Image

    out = []
    with fitz.open(path) as doc:
        for i in range(min(doc.page_count, max_pages)):
            pix = doc[i].get_pixmap(dpi=dpi, alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            out.append(_png(img, max_long_edge))
    return out


def render_pages(path: Path, max_pages: int, dpi: int, max_long_edge: int) -> list[bytes]:
    ext = Path(path).suffix.lower()
    if ext in PDF_EXT:
        return _pdf(Path(path), max_pages, dpi, max_long_edge)
    if ext in IMAGE_EXT:
        from PIL import Image

        with Image.open(path) as img:
            return [_png(img, max_long_edge)]
    return []
