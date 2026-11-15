"""Shared exception types for baseline, cascade and demo.

Every domain error subclasses DoccError. Some also subclass a builtin
(ValueError, RuntimeError) because callers already catch those.
"""

from __future__ import annotations

import sqlite3


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


class DocumentInOtherCabinet(DoccError, sqlite3.IntegrityError):  # noqa: N818 - names the refused state
    """An upsert would move a stored document to another cabinet; the store refused it.

    Also an sqlite3.IntegrityError, so store hooks that log sqlite3.Error and
    carry on handle it without a change. Carries ids only.

    Attributes:
        doc_id: The document.
        stored_cabinet_id: The cabinet that holds it.
        cabinet_id: The cabinet the refused write named.
    """

    def __init__(self, doc_id: str, stored_cabinet_id: str, cabinet_id: str) -> None:
        """Keep the three ids.

        Args:
            doc_id: The document.
            stored_cabinet_id: The cabinet that holds it.
            cabinet_id: The cabinet the refused write named.
        """
        super().__init__(f"document {doc_id} is stored under cabinet {stored_cabinet_id}, not {cabinet_id}")
        self.doc_id = doc_id
        self.stored_cabinet_id = stored_cabinet_id
        self.cabinet_id = cabinet_id


class PackRegistrationError(DoccError, ValueError):
    """A client pack cannot be registered in the store; the message names types and fields, never values."""


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
