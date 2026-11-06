"""TesseractEngine.configure honours a configured binary only when it exists."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from baseline.ocr import TesseractEngine


def _engine_with_fake_module() -> tuple[TesseractEngine, SimpleNamespace]:
    fake = SimpleNamespace(pytesseract=SimpleNamespace(tesseract_cmd="tesseract"))
    engine = TesseractEngine()
    engine.__dict__["_pytesseract"] = fake  # bypass the cached_property import
    return engine, fake


def test_missing_binary_falls_back_to_path():
    engine, fake = _engine_with_fake_module()
    engine.configure({"binary": "/nonexistent/dir/tesseract"})
    assert fake.pytesseract.tesseract_cmd == "tesseract"


def test_existing_binary_is_used(tmp_path: Path):
    binary = tmp_path / "tesseract"
    binary.write_bytes(b"")
    engine, fake = _engine_with_fake_module()
    engine.configure({"binary": str(binary)})
    assert fake.pytesseract.tesseract_cmd == str(binary)


def test_null_binary_leaves_path_default():
    engine, fake = _engine_with_fake_module()
    engine.configure({"binary": None})
    assert fake.pytesseract.tesseract_cmd == "tesseract"
