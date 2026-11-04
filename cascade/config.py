"""cascade.yaml, validated when it is loaded.

A misspelt key or an out of range value fails the run before the first
document, not halfway through a batch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, model_validator

from baseline.config import DEFAULT_CONFIG_DIR
from baseline.schema import Confidence, Strict


class VLMSettings(Strict):
    base_url: str
    # Null until a deployment chooses one. VLMClient refuses to start without it.
    model: str | None = None
    api_key_env: str | None = None
    timeout_s: float = Field(gt=0)
    retries: int = Field(ge=0, le=10)
    backoff_s: float = Field(ge=0)
    max_tokens: int = Field(gt=0)
    max_response_bytes: int = Field(default=8 * 1024 * 1024, gt=0)
    allowed_hosts: list[str] = Field(default_factory=list)
    # Server specific request fields, such as the switch that turns off a
    # model's thinking. Never overrides the fields VLMClient sets itself.
    extra_body: dict[str, Any] = Field(default_factory=dict)


class RenderSettings(Strict):
    dpi: int = Field(gt=0, le=600)
    max_long_edge: int = Field(gt=0)
    max_pages: int = Field(gt=0)
    max_text_chars: int = Field(ge=0)


class ConfidenceTiers(Strict):
    agree: Confidence
    grounded: Confidence
    ambiguous: Confidence
    vlm_class: Confidence

    @model_validator(mode="after")
    def _ordered(self) -> ConfidenceTiers:
        if not (self.agree >= self.grounded >= self.ambiguous):
            raise ValueError("confidence tiers must satisfy agree >= grounded >= ambiguous")
        return self


class CascadeConfig(Strict):
    version: int
    vlm: VLMSettings
    render: RenderSettings
    confidence: ConfidenceTiers


def load_cascade_config(config_dir: str | Path | None = None) -> CascadeConfig:
    d = Path(config_dir) if config_dir else DEFAULT_CONFIG_DIR
    with open(d / "cascade.yaml", "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return CascadeConfig.model_validate(data or {})
