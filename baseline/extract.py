"""Text extraction with per page routing.

Rule: never OCR a page that already has a text layer. A document may be
hybrid, and we record which pages went which way.
"""

from __future__ import annotations

import io
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
    return engine.recognize(image, opts)


def _extract_pdf(path: Path, cfg: Config, res: ExtractionResult) -> None:
    import fitz

    opts = cfg.extract_opts()
    threshold = int(opts.get("native_char_threshold", 100))
    ocr_opts = dict(opts.get("ocr", {}))
    capture = bool(opts.get("capture_bboxes", True))
    native_conf = float(opts.get("native_confidence", 0.98))
    gate = dict(opts.get("quality_gate", {}))

    with fitz.open(path) as doc:
        for pno in range(doc.page_count):
            page = doc.load_page(pno)
            res.page_sizes.append((float(page.rect.width), float(page.rect.height)))
            native = _sanitize(page.get_text("text") or "")
            probe = native.strip() if opts.get("strip_before_count", True) else native

            if len(probe) >= threshold:
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
            try:
                from PIL import Image

                pix = page.get_pixmap(dpi=int(ocr_opts.get("dpi", 300)))
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                ocr = _ocr_image(img, ocr_opts)
                res.pages.append(_sanitize_fixed_width(ocr.text))
                res.page_methods.append(ExtractionMethod.OCR)
                res.page_confidences.append(ocr_confidence(ocr.words))
                if capture:
                    for w in ocr.words:
                        res.words.append(
                            WordBox(
                                page=pno,
                                text=w.text,
                                x0=w.x0,
                                y0=w.y0,
                                x1=w.x1,
                                y1=w.y1,
                                conf=w.conf,
                                char_start=w.char_start,
                                char_end=w.char_end,
                            )
                        )
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
        from PIL import Image

        pix = page.get_pixmap(dpi=int(ocr_opts.get("dpi", 300)))
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        ocr = _ocr_image(img, ocr_opts)
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
    words = []
    if capture:
        for w in ocr.words:
            words.append(
                WordBox(
                    page=pno,
                    text=w.text,
                    x0=w.x0,
                    y0=w.y0,
                    x1=w.x1,
                    y1=w.y1,
                    conf=w.conf,
                    char_start=w.char_start,
                    char_end=w.char_end,
                )
            )
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
            ocr = _ocr_image(img, ocr_opts)
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
        for w in ocr.words:
            res.words.append(
                WordBox(
                    page=0,
                    text=w.text,
                    x0=w.x0,
                    y0=w.y0,
                    x1=w.x1,
                    y1=w.y1,
                    conf=w.conf,
                    char_start=w.char_start,
                    char_end=w.char_end,
                )
            )


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


def iter_documents(root) -> list:
    """Every supported file under root, sorted so runs are reproducible."""
    r = Path(root)
    if r.is_file():
        return [r]
    return sorted(
        (p for p in r.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXT),
        key=lambda p: str(p).lower(),
    )
