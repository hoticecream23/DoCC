"""Text extraction with per page routing.

Rule: never OCR a page that already has a text layer. A document may be
hybrid, and we record which pages went which way.
"""

from __future__ import annotations

import os
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Config
from .logging_setup import get_logger
from .ocr import OCRPage, get_engine, ocr_confidence
from .textquality import prefer, score_text
from .schema import ExtractionMethod, WordBox, page_spans

log = get_logger(__name__)

PDF_EXT = {".pdf"}
DOCX_EXT = {".docx"}
TEXT_EXT = {".txt", ".md"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
SUPPORTED_EXT = PDF_EXT | DOCX_EXT | TEXT_EXT | IMAGE_EXT


@dataclass
class ExtractionResult:
    pages: list[str] = field(default_factory=list)
    page_methods: list[ExtractionMethod] = field(default_factory=list)
    page_confidences: list[float] = field(default_factory=list)
    page_sizes: list[tuple[float, float]] = field(default_factory=list)
    words: list[WordBox] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def method(self) -> ExtractionMethod:
        kinds = {m for m in self.page_methods if m != ExtractionMethod.NONE}
        if not kinds:
            return ExtractionMethod.NONE
        if kinds == {ExtractionMethod.NATIVE}:
            return ExtractionMethod.NATIVE
        # A rescued page is an OCR page whose provenance we kept. At document
        # level it aggregates as OCR, and page_methods keeps the detail.
        if kinds <= {ExtractionMethod.OCR, ExtractionMethod.OCR_RESCUED}:
            return ExtractionMethod.OCR
        return ExtractionMethod.HYBRID

    @property
    def has_native_text(self) -> bool:
        return ExtractionMethod.NATIVE in self.page_methods

    @property
    def is_scanned(self) -> bool:
        # No text layer anywhere in the document. page_methods has the detail.
        return self.method == ExtractionMethod.OCR

    def confidence(self) -> float:
        """Blend page confidences weighted by how much text each page gave us."""
        num = den = 0.0
        for text, conf in zip(self.pages, self.page_confidences):
            w = float(len(text))
            num += conf * w
            den += w
        if den == 0:
            return 0.0
        return round(num / den, 4)


def _sanitize(text: str) -> str:
    """Normalise line endings and kill form feeds. Pages must not hold PAGE_SEP."""
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n")


def _sanitize_fixed_width(text: str) -> str:
    """Same intent, length preserving, so precomputed word offsets stay valid."""
    return text.replace("\r", "\n").replace("\f", "\n")


def _map_words_to_offsets(page_text: str, raw_words: list) -> list:
    """Best effort char offsets for native PDF words.

    Walks the page text in reading order. Yields (-1, -1) for words it cannot
    place, which the caller stores as null rather than guessing.
    """
    out = []
    cursor = 0
    for w in raw_words:
        tok = w[4]
        if not tok:
            out.append((-1, -1))
            continue
        idx = page_text.find(tok, cursor)
        if idx < 0:
            idx = page_text.find(tok)
        if idx < 0:
            out.append((-1, -1))
            continue
        out.append((idx, idx + len(tok)))
        cursor = idx + len(tok)
    return out


def _ocr_image(image: Any, opts: dict) -> OCRPage:
    engine = get_engine(opts.get("engine", "tesseract"))
    engine.configure(opts)
    if not engine.available():
        raise RuntimeError("ocr engine " + repr(engine.name) + " not available")
    page = engine.recognize(image, opts)
    if ocr_confidence(page.words) >= OSD_BELOW:
        return page
    # ponytail: one OSD guess, one re-read, keep the higher mean confidence.
    # OSD's orientation_conf is ~1.5 on a clean 90-degree scan, so the floor is
    # low and the confidence comparison does the real filtering. No deskew.
    try:
        deg, conf = engine.orientation(image, opts)
    except Exception as exc:  # too few characters, timeout: keep the first read
        log.debug("osd failed", extra={"error": str(exc)})
        return page
    if deg not in (90, 180, 270) or conf < OSD_MIN_CONF:
        return page
    turned = engine.recognize(image.rotate(-deg, expand=True), opts)
    if ocr_confidence(turned.words) <= ocr_confidence(page.words):
        return page
    log.debug("ocr page rotated", extra={"rotated_degrees": deg, "osd_conf": conf})
    for w in turned.words:  # boxes back into the unrotated image's frame
        w.x0, w.y0, w.x1, w.y1 = _unrotate(w, deg, image.width, image.height)
    return turned


OSD_BELOW = 0.6  # first-pass mean word confidence that triggers OSD
OSD_MIN_CONF = 1.0  # tesseract orientation_conf floor


def _unrotate(w, deg: int, width: int, height: int) -> tuple[float, float, float, float]:
    """A box read off the image turned `deg` clockwise, in the original's pixels."""
    if deg == 180:
        return width - w.x1, height - w.y1, width - w.x0, height - w.y0
    if deg == 90:
        return w.y0, height - w.x1, w.y1, height - w.x0
    return width - w.y1, w.x0, width - w.y0, w.x1  # 270


def _upscale_for_ocr(image: Any, opts: dict) -> tuple[Any, float]:
    """Enlarge a small image so its text is large enough for the engine.

    Tesseract wants roughly 300 DPI. The PDF route renders pages at that DPI,
    but an image file arrives at whatever resolution it happens to be saved
    at, and most of the client corpus is well under it. Nothing rescaled it,
    so the image half of the corpus was being read at a resolution the PDF
    half never sees.

    Measured on the 40 labelled images: raising the short side to 1000px moves
    mean OCR confidence from 0.689 to 0.766 and cuts the documents scoring
    under 0.6 from 15 to 8. Larger targets were worse on both counts, because
    enlarging an image that is already legible only softens its glyphs.

    Returns the image to read and the factor it was scaled by, so the caller
    can put the word boxes back into the original coordinate space.
    """
    target = int(opts.get("min_short_side", 0) or 0)
    short = min(int(image.width), int(image.height))
    if target <= 0 or short <= 0 or short >= target:
        return image, 1.0
    from PIL import Image

    k = target / short
    resized = image.resize((round(image.width * k), round(image.height * k)), Image.LANCZOS)
    return resized, k


def _prepare_pink_paper(image: Any, opts: dict) -> tuple[Any, dict]:
    """On pink paper, read the red channel only, as one block of text.

    Most of the client corpus is Yaumi invoices: blue print on pink paper, with
    a big red logo printed behind the totals. In the red channel the paper and
    the logo both turn near white and the blue ink stays dark. psm 6 keeps each
    label on the same line as its value, where psm 3 splits them into columns.

    Measured on 32 Yaumi scans: mean OCR confidence 0.503 -> 0.632, delivery
    date read 11 -> 17. Applied to every image it cost the labelled set 2 of 40
    images and 1 PDF, so it only runs when the paper itself is pink.

    Returns the image to read and the OCR options to read it with.
    """
    rule = opts.get("pink_paper") or {}
    if not rule:
        return image, opts
    from PIL import ImageStat

    rgb = image.convert("RGB")
    small = rgb.copy()
    small.thumbnail((200, 200))
    red, green, blue = ImageStat.Stat(small).median  # the median pixel is the paper
    # ponytail: thresholds fit one scan batch. Yaumi sits at 36 to 45, the
    # palest other pink paper at 32. Re-measure when a new tinted batch arrives.
    is_pink = (
        red - green >= rule["min_red_over_green"]
        and abs(green - blue) <= rule["max_green_blue_gap"]
    )
    if not is_pink:
        return image, opts
    return rgb.getchannel("R"), {**opts, "psm": rule["psm"]}


def _ocr_words(words: list, pno: int, scale: float = 1.0) -> list[WordBox]:
    """OCR words as WordBoxes in the page's own coordinate space.

    The engine reads a rendered or enlarged image, so it returns boxes in that
    image's pixels. page_sizes is in the page's own units and metadata.py
    normalises boxes by it, so the two have to agree. scale is what the image
    handed to the engine was multiplied by relative to those units.
    """
    if scale <= 0:
        scale = 1.0
    return [
        WordBox(
            page=pno,
            text=w.text,
            x0=w.x0 / scale,
            y0=w.y0 / scale,
            x1=w.x1 / scale,
            y1=w.y1 / scale,
            conf=w.conf,
            char_start=w.char_start,
            char_end=w.char_end,
        )
        for w in words
    ]


# ponytail: fixed pool per document; with --workers N this is N x threads
# tesseract processes. Set extraction.ocr.threads if that oversubscribes.
OCR_THREADS = min(4, os.cpu_count() or 1)


def _render(page, dpi: int):
    """Page to an RGB PIL image, straight from the pixmap buffer."""
    from PIL import Image

    pix = page.get_pixmap(dpi=dpi, alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def _submit_ocr(pool, page, dpi: int, opts: dict) -> Future:
    try:
        img = _render(page, dpi)
    except Exception as exc:
        failed: Future = Future()
        failed.set_exception(exc)
        return failed
    return pool.submit(_ocr_image, img, opts)


def _extract_pdf(path: Path, cfg: Config, res: ExtractionResult) -> None:
    import fitz

    opts = cfg.extract_opts()
    threshold = int(opts.get("native_char_threshold", 100))
    ocr_opts = dict(opts.get("ocr", {}))
    capture = bool(opts.get("capture_bboxes", True))
    native_conf = float(opts.get("native_confidence", 0.98))
    gate = dict(opts.get("quality_gate", {}))

    dpi = int(ocr_opts.get("dpi", 300))
    threads = int(ocr_opts.get("threads", OCR_THREADS))
    strip = opts.get("strip_before_count", True)

    with fitz.open(path) as doc, ThreadPoolExecutor(threads) as pool:
        natives = [_sanitize(doc.load_page(p).get_text("text") or "") for p in range(doc.page_count)]
        scanned = iter([p for p, t in enumerate(natives) if len(t.strip() if strip else t) < threshold])
        pending: dict[int, Future] = {}

        for pno in range(doc.page_count):
            page = doc.load_page(pno)
            res.page_sizes.append((float(page.rect.width), float(page.rect.height)))
            native = natives[pno]

            if len(native.strip() if strip else native) >= threshold:
                # The text layer may itself be somebody else's bad OCR. When it
                # scores as substituted, re-read the page properly and keep
                # whichever reading is cleaner. See textquality.py.
                rescued = _rescue_page(page, native, ocr_opts, gate, pno, capture)
                if rescued is not None:
                    text, words, conf = rescued
                    res.pages.append(text)
                    res.page_methods.append(ExtractionMethod.OCR_RESCUED)
                    res.page_confidences.append(conf)
                    res.words.extend(words)
                    continue

                res.pages.append(native)
                res.page_methods.append(ExtractionMethod.NATIVE)
                res.page_confidences.append(native_conf)
                if capture:
                    raw = sorted(page.get_text("words"), key=lambda w: (w[5], w[6], w[7]))
                    for w, off in zip(raw, _map_words_to_offsets(native, raw)):
                        res.words.append(
                            WordBox(
                                page=pno,
                                text=w[4],
                                x0=float(w[0]),
                                y0=float(w[1]),
                                x1=float(w[2]),
                                y1=float(w[3]),
                                conf=native_conf,
                                char_start=off[0] if off[0] >= 0 else None,
                                char_end=off[1] if off[1] >= 0 else None,
                            )
                        )
                continue

            # Thin or empty text layer, so treat the page as scanned.
            # Keep up to `threads` scanned pages in flight. Rendering stays on
            # this thread because PyMuPDF is not thread safe; Tesseract runs
            # as a subprocess, so the pool threads overlap for real.
            while len(pending) < threads and (nxt := next(scanned, None)) is not None:
                pending[nxt] = _submit_ocr(pool, doc.load_page(nxt), dpi, ocr_opts)
            try:
                ocr = pending.pop(pno).result()
                res.pages.append(_sanitize_fixed_width(ocr.text))
                res.page_methods.append(ExtractionMethod.OCR)
                res.page_confidences.append(ocr_confidence(ocr.words))
                if capture:
                    # The pixmap is dpi/72 times the size of the page rect that
                    # went into page_sizes, so undo that before storing.
                    res.words.extend(_ocr_words(ocr.words, pno, dpi / 72.0))
            except Exception as exc:
                res.errors.append("ocr failed on page " + str(pno) + ": " + str(exc))
                # Keep the thin native text rather than dropping the page entirely.
                res.pages.append(native)
                res.page_methods.append(
                    ExtractionMethod.NATIVE if native.strip() else ExtractionMethod.NONE
                )
                res.page_confidences.append(native_conf if native.strip() else 0.0)


def _rescue_page(page, native, ocr_opts, gate, pno, capture):
    """Re-read a page whose text layer looks substituted. None means keep native.

    A PDF can carry a text layer that was itself produced by a scanner's OCR,
    and PyMuPDF hands it over as native text, so nothing downstream learns it
    is wrong. This scores it and, when it looks substituted, renders the page
    and reads it properly.

    Returning None is the normal outcome and covers three cases: the gate is
    off, the text is fine, or OCR came out no cleaner. That last case is the
    important one. The gate keeps whichever reading scores lower, so it cannot
    make a page worse by its own metric.
    """
    if not gate.get("enabled", False):
        return None
    quality = score_text(native)
    if quality.tokens < int(gate.get("min_tokens", 40)):
        return None
    if not quality.is_suspect(float(gate.get("threshold", 0.02))):
        return None

    try:
        dpi = int(ocr_opts.get("dpi", 300))
        ocr = _ocr_image(_render(page, dpi), ocr_opts)
    except Exception as exc:
        # A failed rescue is not a failed page. Keep the native text and say so.
        log.warning("quality gate could not re-read page",
                    extra={"page": pno, "error": str(exc)})
        return None

    text = _sanitize_fixed_width(ocr.text)
    chosen, used_ocr = prefer(native, text)
    if not used_ocr:
        return None

    log.info("page rescued by the quality gate",
             extra={"page": pno,
                    "native_score": quality.score,
                    "ocr_score": score_text(chosen).score})
    words = _ocr_words(ocr.words, pno, dpi / 72.0) if capture else []
    return chosen, words, ocr_confidence(ocr.words)


def _extract_docx(path: Path, cfg: Config, res: ExtractionResult) -> None:
    import docx

    d = docx.Document(str(path))
    parts = [p.text for p in d.paragraphs]
    for table in d.tables:
        for row in table.rows:
            parts.append("\t".join(c.text.strip() for c in row.cells))
    text = _sanitize("\n".join(parts))
    # docx has no page concept without rendering, so it is one canonical page.
    res.pages.append(text)
    res.page_methods.append(ExtractionMethod.NATIVE if text.strip() else ExtractionMethod.NONE)
    res.page_confidences.append(
        float(cfg.extract_opts().get("native_confidence", 0.98)) if text.strip() else 0.0
    )
    res.page_sizes.append((0.0, 0.0))


def _extract_text(path: Path, cfg: Config, res: ExtractionResult) -> None:
    text = _sanitize(path.read_text(encoding="utf-8", errors="replace"))
    res.pages.append(text)
    res.page_methods.append(ExtractionMethod.NATIVE if text.strip() else ExtractionMethod.NONE)
    res.page_confidences.append(
        float(cfg.extract_opts().get("native_confidence", 0.98)) if text.strip() else 0.0
    )
    res.page_sizes.append((0.0, 0.0))


def _extract_image(path: Path, cfg: Config, res: ExtractionResult) -> None:
    from PIL import Image

    opts = cfg.extract_opts()
    ocr_opts = dict(opts.get("ocr", {}))
    with Image.open(path) as img:
        res.page_sizes.append((float(img.width), float(img.height)))
        try:
            for_ocr, scale = _upscale_for_ocr(img, ocr_opts)
            for_ocr, read_opts = _prepare_pink_paper(for_ocr, ocr_opts)
            ocr = _ocr_image(for_ocr, read_opts)
        except Exception as exc:
            res.errors.append("ocr failed: " + str(exc))
            res.pages.append("")
            res.page_methods.append(ExtractionMethod.NONE)
            res.page_confidences.append(0.0)
            return
    res.pages.append(_sanitize_fixed_width(ocr.text))
    res.page_methods.append(ExtractionMethod.OCR)
    res.page_confidences.append(ocr_confidence(ocr.words))
    if opts.get("capture_bboxes", True):
        # page_sizes holds the file's own pixels, so undo any OCR upscale.
        res.words.extend(_ocr_words(ocr.words, 0, scale))


_HANDLERS = [
    (PDF_EXT, _extract_pdf),
    (DOCX_EXT, _extract_docx),
    (TEXT_EXT, _extract_text),
    (IMAGE_EXT, _extract_image),
]


def extract(path, cfg: Config) -> ExtractionResult:
    """One file in, page texts plus word boxes out. Never raises on content."""
    p = Path(path)
    res = ExtractionResult()
    ext = p.suffix.lower()
    for exts, handler in _HANDLERS:
        if ext in exts:
            try:
                handler(p, cfg, res)
            except Exception as exc:
                log.exception("extraction failed", extra={"path": str(p)})
                res.errors.append("extract: " + type(exc).__name__ + ": " + str(exc))
            break
    else:
        res.errors.append("unsupported extension " + repr(ext))

    if not res.pages:
        res.pages.append("")
        res.page_methods.append(ExtractionMethod.NONE)
        res.page_confidences.append(0.0)
        if not res.page_sizes:
            res.page_sizes.append((0.0, 0.0))
    return res


def to_canonical_offsets(res: ExtractionResult) -> None:
    """Shift word offsets from page local into canonical text space, in place.

    Keeps one offset convention across the whole record. Run once, right
    after extraction, before anything reads the boxes.
    """
    spans = page_spans(res.pages)
    for w in res.words:
        if w.char_start is None or w.char_end is None:
            continue
        if w.page >= len(spans):
            w.char_start = w.char_end = None
            continue
        base = spans[w.page][0]
        w.char_start += base
        w.char_end += base
