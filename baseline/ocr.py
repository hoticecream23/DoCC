"""OCR behind an interface. Nothing downstream may import an engine directly.

Swapping Tesseract for something else is a config edit plus a registry entry.
"""

from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

from .errors import summarize_exception
from .logging_setup import get_logger

log = get_logger(__name__)

TESSERACT_WORD_LEVEL = 5  # image_to_data level: 1 page, 2 block, 3 para, 4 line, 5 word


@dataclass
class OCRWord:
    """One recognised word, its box in pixels, and its span in the page text."""

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
    """Page text plus the word boxes it was built from."""

    text: str = ""
    words: list[OCRWord] = field(default_factory=list)
    engine: str = "none"


class OCREngine(ABC):
    """Interface every OCR engine implements."""

    name: str = "abstract"

    def configure(self, opts: dict[str, Any]) -> None:  # noqa: B027 - optional hook, a no op unless overridden
        """Apply engine settings before use. Default is a no op.

        Args:
            opts: Engine options from config.
        """

    @abstractmethod
    def available(self) -> bool:
        """Can this engine actually run right now.

        Returns:
            True when the engine can run.
        """

    @abstractmethod
    def recognize(self, image: Any, opts: dict[str, Any]) -> OCRPage:
        """Take a PIL image, return page text plus word boxes with offsets.

        Args:
            image: A PIL image.
            opts: Engine options from config.

        Returns:
            The recognised page.
        """

    def orientation(self, image: Any, opts: dict[str, Any]) -> tuple[int, float]:
        """Clockwise degrees that would set the page upright, and confidence.

        Args:
            image: A PIL image.
            opts: Engine options from config.

        Returns:
            (degrees, confidence). The default is (0, 0.0).
        """
        return 0, 0.0


class TesseractEngine(OCREngine):
    """Default engine. Classical, no neural net, no network."""

    name = "tesseract"

    @functools.cached_property
    def _pytesseract(self) -> ModuleType:
        """The pytesseract module, imported on first use."""
        import pytesseract

        module: ModuleType = pytesseract
        return module

    def configure(self, opts: dict[str, Any]) -> None:
        """Point pytesseract at an explicit binary when config names one.

        Args:
            opts: Engine options; "binary" is the tesseract executable path.
        """
        # Windows installers do not put tesseract on PATH, so allow an
        # explicit binary. Null in config means "just use PATH". A configured
        # path that does not exist on this machine (the Windows path in
        # config/pipeline.yaml on a Linux CI runner) also falls back to PATH.
        binary = opts.get("binary")
        if binary and Path(binary).is_file():
            self._pytesseract.pytesseract.tesseract_cmd = str(binary)
        elif binary:
            log.debug("configured tesseract binary missing, using PATH", extra={"binary": str(binary)})

    def available(self) -> bool:
        """Check that the tesseract binary runs.

        Returns:
            True when tesseract reports a version.
        """
        try:
            self._pytesseract.get_tesseract_version()
            return True
        except (ImportError, OSError, RuntimeError, ValueError) as exc:  # not installed, not on PATH, bad version
            log.warning("tesseract unavailable", extra={"error": summarize_exception(exc)})
            return False

    def recognize(self, image: Any, opts: dict[str, Any]) -> OCRPage:
        """Run tesseract word-level OCR.

        Args:
            image: A PIL image.
            opts: psm, lang and timeout_s.

        Returns:
            The recognised page.
        """
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

    def orientation(self, image: Any, opts: dict[str, Any]) -> tuple[int, float]:
        """Run tesseract OSD.

        Args:
            image: A PIL image.
            opts: timeout_s.

        Returns:
            (clockwise degrees to rotate, OSD confidence).
        """
        pt = self._pytesseract
        osd = pt.image_to_osd(image, output_type=pt.Output.DICT, timeout=opts.get("timeout_s", 120))
        return int(osd["rotate"]), float(osd["orientation_conf"])


def _assemble(data: dict[str, list[Any]], engine: str) -> OCRPage:
    """Build page text from word boxes so offsets are exact by construction."""
    n = len(data.get("text", []))
    lines: dict[tuple[int, int, int], list[int]] = {}
    for i in range(n):
        if int(data["level"][i]) != TESSERACT_WORD_LEVEL:
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
        """Never available.

        Returns:
            False.
        """
        return False

    def recognize(self, image: Any, opts: dict[str, Any]) -> OCRPage:
        """Return an empty page.

        Args:
            image: Ignored.
            opts: Ignored.

        Returns:
            An empty OCRPage.
        """
        return OCRPage(engine=self.name)


_REGISTRY: dict[str, type[OCREngine]] = {
    "tesseract": TesseractEngine,
    "null": NullEngine,
}


@functools.lru_cache(maxsize=8)
def get_engine(name: str) -> OCREngine:
    """Return the cached engine instance registered under name.

    Args:
        name: Registry key, such as "tesseract" or "null".

    Returns:
        The engine.

    Raises:
        KeyError: No engine has that name.
    """
    if name not in _REGISTRY:
        raise KeyError(f"unknown ocr engine {name!r}, have {sorted(_REGISTRY)}")
    return _REGISTRY[name]()


def ocr_confidence(words: list[OCRWord]) -> float:
    """Mean per-word confidence, weighted by word length.

    Long words carry more signal than stray one char noise.

    Args:
        words: Recognised words.

    Returns:
        Confidence in 0..1; 0.0 when there are no characters.
    """
    num = 0.0
    den = 0.0
    for w in words:
        weight = float(len(w.text))
        num += w.conf * weight
        den += weight
    return round(num / den, 4) if den else 0.0
