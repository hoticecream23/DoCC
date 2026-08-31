"""Baseline document processing pipeline.

Cheap, deterministic, classical. This is the floor a neural approach must beat.
"""

from .schema import PAGE_SEP, SCHEMA_VERSION, DocumentRecord

__all__ = ["DocumentRecord", "PAGE_SEP", "SCHEMA_VERSION"]
__version__ = "0.1.0"
