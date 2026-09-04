"""YAML config loading.

Rule: no document class, field name, tag name or pattern appears in any module.
If you need a new one, edit YAML. If you cannot, the code is wrong.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing config file: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping at top level")
    return data


@dataclass(frozen=True)
class Config:
    """All YAML, parsed once, passed down. Workers get a copy per process."""

    config_dir: Path
    pipeline: dict[str, Any] = field(default_factory=dict)
    classes: dict[str, Any] = field(default_factory=dict)
    fields: dict[str, Any] = field(default_factory=dict)
    tags: dict[str, Any] = field(default_factory=dict)
    graph: dict[str, Any] = field(default_factory=dict)
    tables: dict[str, Any] = field(default_factory=dict)
    taxonomy: dict[str, Any] = field(default_factory=dict)

    # -- convenience accessors, all pure lookups --

    @property
    def class_names(self) -> list[str]:
        return [c["name"] for c in self.classes.get("classes", [])]

    @property
    def client_label_map(self) -> dict[str, str | None]:
        """Client folder label to our class name. Null where we refuse to map."""
        return {m["label"]: m.get("class") for m in self.taxonomy.get("mappings", [])}

    @property
    def lossy_client_labels(self) -> list[str]:
        """Labels our class list cannot reproduce. Score these collapsed, not raw."""
        return [m["label"] for m in self.taxonomy.get("mappings", []) if m.get("lossy")]

    @property
    def field_defs(self) -> list[dict[str, Any]]:
        return self.fields.get("fields", [])

    @property
    def templates(self) -> list[dict[str, Any]]:
        return self.fields.get("templates", []) or []

    @property
    def tag_defs(self) -> list[dict[str, Any]]:
        return self.tags.get("tags", [])

    @property
    def derived_tag_rules(self) -> list[dict[str, Any]]:
        return self.tags.get("derived", []) or []

    @property
    def normalizer_version(self) -> str:
        return str(self.fields.get("normalizer_version", "1.0.0"))

    def extract_opts(self) -> dict[str, Any]:
        return self.pipeline.get("extraction", {})

    def classify_opts(self) -> dict[str, Any]:
        return self.pipeline.get("classification", {})

    def tagging_opts(self) -> dict[str, Any]:
        return self.pipeline.get("tagging", {})

    def metadata_opts(self) -> dict[str, Any]:
        return self.pipeline.get("metadata", {})

    def table_opts(self) -> dict[str, Any]:
        return self.tables

    def dump(self) -> dict[str, Any]:
        return {
            "pipeline": self.pipeline,
            "classes": self.classes,
            "fields": self.fields,
            "tags": self.tags,
            "graph": self.graph,
            "tables": self.tables,
            "taxonomy": self.taxonomy,
        }


def _load_optional(path: Path) -> dict[str, Any]:
    """Config that a deployment may simply not have yet."""
    return _load(path) if path.exists() else {}


def load_config(config_dir: str | Path | None = None) -> Config:
    d = Path(config_dir) if config_dir else DEFAULT_CONFIG_DIR
    d = d.resolve()
    return Config(
        config_dir=d,
        pipeline=_load(d / "pipeline.yaml"),
        classes=_load(d / "classes.yaml"),
        fields=_load(d / "fields.yaml"),
        tags=_load(d / "tags.yaml"),
        graph=_load_optional(d / "graph.yaml"),
        tables=_load_optional(d / "tables.yaml"),
        taxonomy=_load_optional(d / "taxonomy.yaml"),
    )


@functools.lru_cache(maxsize=4)
def load_config_cached(config_dir: str | None = None) -> Config:
    """Cached per process. Workers call this once, not once per document."""
    return load_config(config_dir)


def save_yaml(path: str | Path, data: dict[str, Any]) -> None:
    """Write YAML back out. Used by tune-thresholds."""
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True, width=100)
