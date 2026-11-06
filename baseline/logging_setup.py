"""Structured JSON logging. No prints anywhere in the package."""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()) | {"message", "asctime", "taskName"}


class JsonFormatter(logging.Formatter):
    """One JSON object per record. Every ``extra=`` key is emitted as is.

    Because extras are not filtered, callers must pass only metadata: ids and
    paths (doc_id, path, dir, output), stage names (stage, strategy,
    normalizer, validator, page), counts and scores, config values, and
    ``error`` built with ``baseline.errors.summarize_exception``. Never pass
    document text or a raw ``str(exc)`` from a parser, OCR or VLM call.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Render one record as a JSON line.

        Args:
            record: The log record.

        Returns:
            The JSON text.
        """
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Anything passed via extra= lands here.
        for k, v in record.__dict__.items():
            if k not in _RESERVED:
                payload[k] = v
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(level: str = "INFO") -> None:
    """Send every log record to stderr as JSON, replacing existing handlers.

    Args:
        level: Root level name; unknown names fall back to INFO.
    """
    # Windows consoles default to cp1252 and would blow up on rupee signs.
    stream = sys.stderr
    reconfigure_error = None
    if hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError) as exc:  # io.UnsupportedOperation is both
            reconfigure_error = exc  # logged once the handler exists
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    if reconfigure_error is not None:
        logging.getLogger(__name__).debug("stderr not reconfigured to utf-8", extra={"error": repr(reconfigure_error)})


def get_logger(name: str) -> logging.Logger:
    """Return the named logger.

    Args:
        name: Logger name, normally __name__.

    Returns:
        The logger.
    """
    return logging.getLogger(name)
