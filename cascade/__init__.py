"""Cascade v2: a vision language model reads each page, and every value it
returns is located in the canonical text before it may be emitted.

Writes the same DocumentRecord JSONL as the baseline, so `baseline eval` and
`baseline compare` score it unchanged. See
docs/superpowers/specs/2026-10-05-cascade-v2-vlm-grounding-design.md.
"""
