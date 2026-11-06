"""What the model is asked, and the only shape of answer accepted back.

Every class and field name, and every description, comes from classes.yaml
and fields.yaml. A new document type for a new client is a YAML edit.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any, TypedDict

from baseline.config import Config
from baseline.errors import VLMOutputError  # noqa: F401 - re-exported

_INSTRUCTIONS = """You read business and legal documents and return JSON.

Return the document's class and the value of every field you can find.

Rules:
- Copy each value exactly as it is printed, character for character. Do not
  reformat dates or amounts, do not correct spelling, do not translate.
- When the value appears in the page text provided, copy it as it appears there.
- Return a value only if it is printed on the document. If a field is not
  there, return an empty list for it. Never return an empty string. Never guess.
- Return the exact printed string of the one value that fits the field's
  definition, not a value of a nearby or similar field.
- Use "{unknown}" as the class when the document is none of the classes below.
"""


# What each field is not, where the model was seen to confuse it.
_NOT = {
    "account_number": "Not a GSTIN, PAN, IFSC, phone, customer, reference or invoice number, nor a date.",
    "invoice_number": "The document's own number, not a PO, reference or order number of another document.",
    "total_amount": "The grand total, not a subtotal, a line amount or a tax line.",
    "tax_amount": "The total tax, not one line item's tax.",
    "vendor_name": "Who issues the document and is paid, not the customer.",
    "buyer_name": "Who is billed and pays, not the issuer.",
}


# Single-value fields a page often prints several of. Capped at one item, the
# model joins them into one string or drops the right one, so the merge picks.
_OFTEN_SEVERAL = ("email", "phone")


class TextPart(TypedDict):
    """A text part of a user message."""

    type: str
    text: str


class ImageURL(TypedDict):
    """Where an image part points: here always a data: URL."""

    url: str


class ImagePart(TypedDict):
    """An image part of a user message."""

    type: str
    image_url: ImageURL


class ChatMessage(TypedDict):
    """One OpenAI style chat message."""

    role: str
    content: str | list[TextPart | ImagePart]


@dataclass(frozen=True)
class VLMAnswer:
    """The model's answer after parse_answer checked its shape."""

    label: str
    fields: dict[str, list[str]]


def _unknown(cfg: Config) -> str:
    return str(cfg.classify_opts().get("unknown_label", "unknown"))


def _field_names(cfg: Config) -> list[str]:
    return [f["name"] for f in cfg.field_defs if f.get("name")]


def check_descriptions(cfg: Config) -> None:
    """Fail before the first document if the prompt would have a blank line.

    Args:
        cfg: The loaded config.

    Raises:
        ValueError: A class or field has no description.
    """
    missing = [f"class {c.get('name')}" for c in cfg.classes.get("classes", []) if not c.get("description")]
    missing += [f"field {f.get('name')}" for f in cfg.field_defs if not f.get("description")]
    if missing:
        raise ValueError("no description in config for: " + ", ".join(missing))


def response_schema(cfg: Config) -> dict[str, Any]:
    """Build the JSON schema the server must constrain the answer to.

    Args:
        cfg: The loaded config; classes and fields come from it.

    Returns:
        A JSON schema object.
    """
    names = _field_names(cfg)
    return {
        "type": "object",
        "properties": {
            "class": {"type": "string", "enum": [*cfg.class_names, _unknown(cfg)]},
            "fields": {
                "type": "object",
                "properties": {
                    f["name"]: {"type": "array", "items": {"type": "string"}}
                    | ({} if f.get("multi") or f["name"] in _OFTEN_SEVERAL else {"maxItems": 1})
                    for f in cfg.field_defs
                    if f.get("name")
                },
                "required": names,
                "additionalProperties": False,
            },
        },
        "required": ["class", "fields"],
        "additionalProperties": False,
    }


def _system_text(cfg: Config) -> str:
    lines = [_INSTRUCTIONS.format(unknown=_unknown(cfg)), "Classes:"]
    for c in cfg.classes.get("classes", []):
        lines.append(f"- {c['name']}: {c.get('description', '')}")
    lines.append("")
    lines.append("Fields:")
    for f in cfg.field_defs:
        scope = ", ".join(f.get("classes") or []) or "any class"
        lines.append(
            f"- {f['name']} (applies to: {scope}): {f.get('description', '')} {_NOT.get(f['name'], '')}".rstrip()
        )
    return "\n".join(lines)


def build_messages(cfg: Config, pages: list[str], images: list[bytes], max_text_chars: int) -> list[ChatMessage]:
    """Build the system and user messages for one document.

    Args:
        cfg: The loaded config.
        pages: Page text, one string per page sent.
        images: PNG bytes, one per page sent.
        max_text_chars: Total page text budget across pages.

    Returns:
        The system message, then one user message with text and images.
    """
    content: list[TextPart | ImagePart] = []
    budget = max_text_chars
    parts = []
    for i, page in enumerate(pages):
        if budget <= 0:
            break
        chunk = page[:budget]
        budget -= len(chunk)
        parts.append(f"--- page {i + 1} ---\n{chunk}")
    if parts:
        content.append(
            {
                "type": "text",
                "text": "Page text read from the document by OCR or from its text layer:\n\n" + "\n\n".join(parts),
            }
        )
    for img in images:
        url = "data:image/png;base64," + base64.b64encode(img).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": url}})
    return [
        {"role": "system", "content": _system_text(cfg)},
        {"role": "user", "content": content},
    ]


def parse_answer(content: str, cfg: Config) -> VLMAnswer:
    """Parse and check the model's answer.

    Args:
        content: The raw message content from the server.
        cfg: The loaded config, for the declared classes and fields.

    Returns:
        The answer, with blank and repeated values removed.

    Raises:
        VLMOutputError: The answer is not JSON or not the declared shape.
    """
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise VLMOutputError(f"answer is not JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise VLMOutputError("answer is not an object")
    extra = set(data) - {"class", "fields"}
    if extra:
        raise VLMOutputError(f"answer has unexpected keys: {sorted(extra)}")

    label = data.get("class")
    allowed = set(cfg.class_names) | {_unknown(cfg)}
    if not isinstance(label, str) or label not in allowed:
        raise VLMOutputError(f"class {label!r} is not a declared class")

    if "fields" not in data:
        raise VLMOutputError("answer has no fields")
    raw = data["fields"]
    if not isinstance(raw, dict):
        raise VLMOutputError("fields is not an object")
    undeclared = set(raw) - set(_field_names(cfg))
    if undeclared:
        raise VLMOutputError(f"undeclared fields: {sorted(undeclared)}")

    fields: dict[str, list[str]] = {}
    for name, values in raw.items():
        if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
            raise VLMOutputError(f"{name}: expected a list of strings")
        kept = list(dict.fromkeys(v.strip() for v in values if v.strip()))
        if kept:
            fields[name] = kept
    missing = set(_field_names(cfg)) - set(raw)
    if missing:
        raise VLMOutputError(f"missing fields: {sorted(missing)}")
    return VLMAnswer(label=label, fields=fields)
