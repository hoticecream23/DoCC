"""Shared exception types for baseline, cascade and demo.

Every domain error subclasses DoccError. Some also subclass a builtin
(ValueError, RuntimeError) because callers already catch those.
"""

from __future__ import annotations


class DoccError(Exception):
    """Base class for all DoCC domain errors."""


class ModelIntegrityError(DoccError, ValueError):
    """Model file missing from, or not matching, MANIFEST.sha256."""


class OffsetMismatch(DoccError, ValueError):  # noqa: N818 - established public name
    """Raised in strict mode when an offset does not slice back to its value."""


class VLMError(DoccError, RuntimeError):
    """The VLM server could not give an answer."""


class VLMOutputError(DoccError, ValueError):
    """The model's answer is not the shape the schema asked for."""


class BadInput(DoccError, ValueError):  # noqa: N818 - established public name
    """A demo request is malformed. The server answers 400."""


class Unavailable(DoccError, RuntimeError):  # noqa: N818 - established public name
    """A demo backend is not available. The server answers 503."""


class ConfigError(DoccError):
    """A required setting is missing or invalid."""


def summarize_exception(exc: BaseException, limit: int = 200) -> str:
    """Return the exception type and a truncated message, safe to log.

    Parser, OCR and VLM messages can embed document text, so the message is
    cut to ``limit`` characters.

    Args:
        exc: The exception to describe.
        limit: Maximum number of message characters kept.

    Returns:
        A string like ``"ValueError: first 200 chars..."``.
    """
    msg = str(exc)
    if len(msg) > limit:
        msg = msg[:limit] + "..."
    return f"{type(exc).__name__}: {msg}"
