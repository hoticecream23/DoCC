"""cascade.yaml, validated when it is loaded.

A misspelt key or an out of range value fails the run before the first
document, not halfway through a batch.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from baseline.config import DEFAULT_CONFIG_DIR, read_yaml
from baseline.errors import ConfigError
from baseline.schema import Confidence, Strict


class VLMSettings(Strict):
    """How to reach the model server and how far to trust it."""

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
    # HTTP codes worth another attempt. 400 is not: it means the prompt is too big.
    retry_statuses: frozenset[int] = frozenset({429, 500, 502, 503, 504})
    # Server specific request fields, such as the switch that turns off a
    # model's thinking. Never overrides the fields VLMClient sets itself.
    extra_body: dict[str, Any] = Field(default_factory=dict)
    # "text" sends the extracted page text and no page images, for a model that cannot read images.
    input: Literal["image", "text"] = "image"


class RenderSettings(Strict):
    """How pages are turned into images and text for the model."""

    dpi: int = Field(gt=0, le=600)
    max_long_edge: int = Field(gt=0)
    max_pages: int = Field(gt=0)
    max_text_chars: int = Field(ge=0)


class ConfidenceTiers(Strict):
    """Confidence given to a value by how it was confirmed."""

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
    """The whole of cascade.yaml."""

    version: int
    vlm: VLMSettings
    render: RenderSettings
    confidence: ConfidenceTiers


def load_cascade_config(config_dir: str | Path | None = None) -> CascadeConfig:
    """Read and validate cascade.yaml.

    Args:
        config_dir: Directory holding cascade.yaml; None uses ./config.

    Returns:
        The validated config.

    Raises:
        pydantic.ValidationError: A key is unknown or a value is out of range.
    """
    d = Path(config_dir) if config_dir else DEFAULT_CONFIG_DIR
    return CascadeConfig.model_validate(read_yaml(d / "cascade.yaml"))


def vlm_settings(ccfg: CascadeConfig) -> VLMSettings:
    """Return the VLM settings: cascade.yaml, then env overrides.

    The model may be unset here; require_model checks it when cascade runs.

    Args:
        ccfg: The cascade config.

    Returns:
        The settings with DOCC_VLM_BASE_URL and DOCC_VLM_MODEL applied.
    """
    over = {
        k: v
        for k, v in (("base_url", os.environ.get("DOCC_VLM_BASE_URL")), ("model", os.environ.get("DOCC_VLM_MODEL")))
        if v
    }
    return ccfg.vlm.model_copy(update=over)


def require_model(settings: VLMSettings) -> None:
    """Refuse cascade mode without a model or endpoint. There is no default.

    Args:
        settings: The VLM settings.

    Raises:
        ConfigError: No model is set in cascade.yaml vlm.model or DOCC_VLM_MODEL,
            or no endpoint is set.
    """
    if not settings.model:
        raise ConfigError("cascade needs a VLM model: set DOCC_VLM_MODEL or vlm.model in config/cascade.yaml")
    if not settings.base_url:
        raise ConfigError("cascade needs a VLM endpoint: set DOCC_VLM_BASE_URL or vlm.base_url in config/cascade.yaml")
