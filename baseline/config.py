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

from .io import atomic_open

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def read_yaml(path: Path) -> Any:
    """The one yaml.safe_load for config files. Empty file gives {}.

    Args:
        path: YAML file.

    Returns:
        The parsed document, or {} for an empty file.
    """
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return {} if data is None else data


def _load(path: Path) -> dict[str, Any]:
    """Read a required YAML mapping; raise if missing or not a mapping."""
    if not path.exists():
        raise FileNotFoundError(f"missing config file: {path}")
    data = read_yaml(path)
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
        """Declared document class names, in classes.yaml order."""
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
        """Metadata field definitions from fields.yaml."""
        defs: list[dict[str, Any]] = self.fields.get("fields", [])
        return defs

    @property
    def templates(self) -> list[dict[str, Any]]:
        """Layout templates from fields.yaml."""
        defs: list[dict[str, Any]] = self.fields.get("templates", []) or []
        return defs

    @property
    def tag_defs(self) -> list[dict[str, Any]]:
        """Tag definitions from tags.yaml."""
        defs: list[dict[str, Any]] = self.tags.get("tags", [])
        return defs

    @property
    def derived_tag_rules(self) -> list[dict[str, Any]]:
        """Derived-tag rules from tags.yaml."""
        defs: list[dict[str, Any]] = self.tags.get("derived", []) or []
        return defs

    @property
    def normalizer_version(self) -> str:
        """Version string of the metadata normalizer rules."""
        return str(self.fields.get("normalizer_version", "1.0.0"))

    def _section(self, key: str) -> dict[str, Any]:
        """One top-level section of pipeline.yaml, or {}."""
        opts: dict[str, Any] = self.pipeline.get(key, {})
        return opts

    def extract_opts(self) -> dict[str, Any]:
        """Extraction options.

        Returns:
            The "extraction" section of pipeline.yaml.
        """
        return self._section("extraction")

    def classify_opts(self) -> dict[str, Any]:
        """Classification options.

        Returns:
            The "classification" section of pipeline.yaml.
        """
        return self._section("classification")

    def tagging_opts(self) -> dict[str, Any]:
        """Tagging options.

        Returns:
            The "tagging" section of pipeline.yaml.
        """
        return self._section("tagging")

    def metadata_opts(self) -> dict[str, Any]:
        """Metadata options.

        Returns:
            The "metadata" section of pipeline.yaml.
        """
        return self._section("metadata")

    def table_opts(self) -> dict[str, Any]:
        """Table options.

        Returns:
            The whole of tables.yaml.
        """
        return self.tables

    def dump(self) -> dict[str, Any]:
        """Every loaded YAML file, keyed by name.

        Returns:
            {file stem: parsed content}.
        """
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
    """Load every YAML file from a config directory.

    Args:
        config_dir: Config directory; None for DEFAULT_CONFIG_DIR.

    Returns:
        The Config.

    Raises:
        FileNotFoundError: A required file is missing.
        ValueError: A file does not hold a mapping at top level.
    """
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
    """Cached per process. Workers call this once, not once per document.

    Args:
        config_dir: Config directory; None for the default.

    Returns:
        The shared Config.
    """
    return load_config(config_dir)


def save_yaml(path: str | Path, data: dict[str, Any]) -> None:
    """Write YAML back out. Used by tune-thresholds.

    Args:
        path: Output file.
        data: Mapping to write.
    """
    with atomic_open(path) as fh:
        yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True, width=100)
