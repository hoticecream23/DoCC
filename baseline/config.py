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
from pydantic import ConfigDict, Field, ValidationError

from .errors import ConfigError
from .io import atomic_open
from .schema import DEFAULT_CABINET, Strict

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
# SQLite store file name, beside the results file, when db.path is not set.
DEFAULT_DB_NAME = "docc.sqlite"


class _Section(Strict):
    """A checked pipeline.yaml section: unknown keys and wrong types fail the load."""

    model_config = ConfigDict(extra="forbid", strict=True)


class ExtractionSettings(_Section):
    """pipeline.yaml `extraction`. Only the keys are checked; extract.py reads the values."""

    native_char_threshold: int = Field(default=100, ge=0)
    strip_before_count: bool = True
    ocr: dict[str, Any] = Field(default_factory=dict)
    quality_gate: dict[str, Any] = Field(default_factory=dict)
    native_confidence: float = Field(default=0.98, ge=0.0, le=1.0)
    # Pages read from one file; null reads every page. Pack mode defaults it to 500.
    max_pages: int | None = Field(default=None, gt=0)
    # OCR every frame of a multi-frame image; false reads frame 0 only. Pack mode defaults it to true.
    multi_frame_images: bool = False
    capture_bboxes: bool = True


class ClientPackSettings(_Section):
    """pipeline.yaml `client_pack`. A relative dir resolves against the config directory."""

    dir: str | None = None
    cabinet: str | None = None
    slug: str | None = None  # set by packmode.pack_config, never by hand


_CHECKED_SECTIONS: dict[str, type[_Section]] = {"extraction": ExtractionSettings, "client_pack": ClientPackSettings}


def _check_pipeline(data: dict[str, Any], path: Path) -> None:
    """Validate the checked sections of pipeline.yaml.

    Args:
        data: The parsed pipeline.yaml.
        path: Its path, for the error message.

    Raises:
        ConfigError: A section has an unknown key or a value of the wrong type.
    """
    for key, model in _CHECKED_SECTIONS.items():
        try:
            model.model_validate(data.get(key) or {})
        except ValidationError as e:
            problems = "; ".join(f"{key}.{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors())
            raise ConfigError(f"{path}: {problems}") from e


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
        opts: dict[str, Any] = self.pipeline.get(key) or {}  # a null section means its defaults
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

    @property
    def cabinet_id(self) -> str:
        """Cabinet stamped on every record a run writes; "default" when unset."""
        return str(self.pipeline.get("cabinet_id") or DEFAULT_CABINET)

    def db_path(self, results: str | Path) -> Path | None:
        """Where the SQLite store sits for a results file, or None when disabled.

        Args:
            results: The results JSONL; a relative db.path resolves beside it.

        Returns:
            The database path; None when db.path is set to null.
        """
        opts = self._section("db")
        raw = opts.get("path", DEFAULT_DB_NAME) if isinstance(opts, dict) else DEFAULT_DB_NAME
        if raw is None:
            return None
        return Path(results).parent / str(raw)

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
        ConfigError: pipeline.yaml extraction or client_pack has an unknown or mistyped key.
    """
    d = Path(config_dir) if config_dir else DEFAULT_CONFIG_DIR
    d = d.resolve()
    pipeline = _load(d / "pipeline.yaml")
    _check_pipeline(pipeline, d / "pipeline.yaml")
    for key in _CHECKED_SECTIONS:
        if key in pipeline and pipeline[key] is None:
            pipeline[key] = {}  # "extraction:" with nothing under it means the defaults
    return Config(
        config_dir=d,
        pipeline=pipeline,
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
