"""OCR behind an interface. Nothing downstream may import an engine directly.

Swapping Tesseract for something else is a config edit plus a registry entry.
"""

from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class OCRWord:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    conf: float  # 0..1
    char_start: int
    char_end: int


@dataclass
class OCRPage:
    text: str = ""
    words: list[OCRWord] = field(default_factory=list)
    engine: str = "none"


class OCREngine(ABC):
    name: str = "abstract"

    def configure(self, opts: dict[str, Any]) -> None:
        """Apply engine settings before use. Default is a no op."""

    @abstractmethod
    def available(self) -> bool:
        """Can this engine actually run right now."""

    @abstractmethod
    def recognize(self, image: Any, opts: dict[str, Any]) -> OCRPage:
        """Take a PIL image, return page text plus word boxes with offsets."""


class TesseractEngine(OCREngine):
    """Default engine. Classical, no neural net, no network."""

    name = "tesseract"

    @functools.cached_property
    def _pytesseract(self):
        import pytesseract

        return pytesseract

    def configure(self, opts: dict[str, Any]) -> None:
        # Windows installers do not put tesseract on PATH, so allow an
        # explicit binary. Null in config means "just use PATH".
        binary = opts.get("binary")
        if binary:
            self._pytesseract.pytesseract.tesseract_cmd = str(binary)

    def available(self) -> bool:
        try:
            self._pytesseract.get_tesseract_version()
            return True
        except Exception as exc:
            log.warning("tesseract unavailable", extra={"error": str(exc)})
            return False

    def recognize(self, image: Any, opts: dict[str, Any]) -> OCRPage:
        pt = self._pytesseract
        cfg = f"--psm {int(opts.get('psm', 3))}"
        data = pt.image_to_data(
            image,
            lang=opts.get("lang", "eng"),
            config=cfg,
            output_type=pt.Output.DICT,
            timeout=opts.get("timeout_s", 120),
        )
        return _assemble(data, self.name)


def _assemble(data: dict[str, list], engine: str) -> OCRPage:
    """Build page text from word boxes so offsets are exact by construction."""
    n = len(data.get("text", []))
    lines: dict[tuple[int, int, int], list[int]] = {}
    for i in range(n):
        if int(data["level"][i]) != 5:
            continue
        if not str(data["text"][i]).strip():
            continue
        key = (int(data["block_num"][i]), int(data["par_num"][i]), int(data["line_num"][i]))
        lines.setdefault(key, []).append(i)

    parts: list[str] = []
    words: list[OCRWord] = []
    cursor = 0
    for li, key in enumerate(sorted(lines)):
        if li:
            parts.append("\n")
            cursor += 1
        for wi, i in enumerate(sorted(lines[key], key=lambda j: int(data["word_num"][j]))):
            if wi:
                parts.append(" ")
                cursor += 1
            tok = str(data["text"][i]).strip()
            start = cursor
            parts.append(tok)
            cursor += len(tok)
            raw_conf = float(data["conf"][i])
            words.append(
                OCRWord(
                    text=tok,
                    x0=float(data["left"][i]),
                    y0=float(data["top"][i]),
                    x1=float(data["left"][i]) + float(data["width"][i]),
                    y1=float(data["top"][i]) + float(data["height"][i]),
                    conf=max(0.0, raw_conf) / 100.0,
                    char_start=start,
                    char_end=cursor,
                )
            )
    return OCRPage(text="".join(parts), words=words, engine=engine)


class NullEngine(OCREngine):
    """Always unavailable. Lets a corpus of native PDFs run with no OCR installed."""

    name = "null"

    def available(self) -> bool:
        return False

    def recognize(self, image: Any, opts: dict[str, Any]) -> OCRPage:
        return OCRPage(engine=self.name)


_REGISTRY: dict[str, type[OCREngine]] = {
    "tesseract": TesseractEngine,
    "null": NullEngine,
}


@functools.lru_cache(maxsize=8)
def get_engine(name: str) -> OCREngine:
    if name not in _REGISTRY:
        raise KeyError(f"unknown ocr engine {name!r}, have {sorted(_REGISTRY)}")
    return _REGISTRY[name]()


def ocr_confidence(words: list[OCRWord]) -> float:
    """Mean per-word confidence, weighted by word length.

    Long words carry more signal than stray one char noise.
    """
    num = 0.0
    den = 0.0
    for w in words:
        weight = float(len(w.text))
        num += w.conf * weight
        den += weight
    return round(num / den, 4) if den else 0.0
