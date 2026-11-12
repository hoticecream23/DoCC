"""Page images for the model, one per page the extractor read.

An image file gives one image per frame, as the baseline extractor reads
every frame of a multi-page TIFF, and the model must see the pages the text came from.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING

from baseline.extract import IMAGE_EXT, PDF_EXT

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage


def _png(img: PILImage, max_long_edge: int) -> bytes:
    from PIL import Image

    img = img.convert("RGB")
    long_edge = max(img.size)
    if long_edge > max_long_edge:
        k = max_long_edge / long_edge
        size = (max(1, round(img.width * k)), max(1, round(img.height * k)))
        img = img.resize(size, Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _pdf(path: Path, max_pages: int, dpi: int, max_long_edge: int) -> list[bytes]:
    import fitz
    from PIL import Image

    out: list[bytes] = []
    with fitz.open(path) as doc:
        for i in range(min(doc.page_count, max_pages)):
            pix = doc[i].get_pixmap(dpi=dpi, alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            out.append(_png(img, max_long_edge))
    return out


def render_pages(path: Path, max_pages: int, dpi: int, max_long_edge: int) -> list[bytes]:
    """Render the first pages of a PDF or image file as PNG.

    Args:
        path: The document.
        max_pages: Pages to render at most.
        dpi: Resolution for PDF pages.
        max_long_edge: Larger pages are scaled down to this many pixels.

    Returns:
        PNG bytes per page; empty for a file type with no image, such as text.
    """
    ext = Path(path).suffix.lower()
    if ext in PDF_EXT:
        return _pdf(Path(path), max_pages, dpi, max_long_edge)
    if ext in IMAGE_EXT:
        from PIL import Image, ImageSequence

        with Image.open(path) as img:
            frames = ImageSequence.Iterator(img)
            return [_png(frame, max_long_edge) for _, frame in zip(range(max_pages), frames, strict=False)]
    return []
