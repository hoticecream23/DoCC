"""fields.yaml of a client pack: field specs from the schema rows, merged spellings and field_overrides.yaml.

``clientpack_build`` calls these for each type of a cabinet. Rows are the
plain dicts ``csv.DictReader`` yields from schema.csv.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .clientpack import FIELD_OVERRIDES_FILE, ClientPackError, FieldSpec, read_yaml
from .clientpack_rules import field_origin, norm, parse_dtype, parse_role, parse_source

UNRECOGNISED_ROLE = "optional"
ROLE_RANK = {"optional": 0, "required": 1, "key": 2}  # on a merge conflict the higher role wins

Row = dict[str, str]
RawOverrides = dict[str, dict[str, dict[str, Any]]]  # type -> field -> FieldSpec values, as written


@dataclass
class FieldNotes:
    """What field specs of one cabinet raised for REPORT.md."""

    bad_roles: list[tuple[str, str, str]] = field(default_factory=list)  # (type, field, raw field type)
    conflicts: list[tuple[str, str, str, str, str]] = field(default_factory=list)  # (type, field, what, kept, other)


def read_field_overrides(path: Path) -> dict[str, RawOverrides]:
    """field_overrides.yaml as {cabinet: {type: {field: values}}}, names as written.

    Args:
        path: The file; a missing file gives no overrides.

    Returns:
        The overrides per cabinet name.

    Raises:
        ClientPackError: The file is not nested mappings.
    """
    data = read_yaml(path) if path.exists() else None
    try:
        return {
            str(cab): {str(t): {str(f): dict(v or {}) for f, v in (fs or {}).items()} for t, fs in (ts or {}).items()}
            for cab, ts in (data or {}).items()
        }
    except (AttributeError, TypeError, ValueError) as e:
        raise ClientPackError(f"{FIELD_OVERRIDES_FILE} is not {{cabinet: {{type: {{field: {{...}}}}}}}}") from e


def unknown_overrides(raw: RawOverrides, types: dict[str, list[Row]]) -> list[str]:
    """Override entries whose type or field the cabinet's schema does not have.

    Args:
        raw: One cabinet's overrides, names as written.
        types: The cabinet's schema types and their rows.

    Returns:
        One line per unknown type or field, names as written.
    """
    known = {norm(t): {norm(r["index_field"]) for r in rows} for t, rows in types.items()}
    out = []
    for type_name, fields in raw.items():
        if norm(type_name) not in known:
            out.append(f"type `{type_name}`")
            continue
        out += [f"field `{type_name}` / `{f}`" for f in fields if norm(f) not in known[norm(type_name)]]
    return out


def _values(type_name: str, row: Row, notes: FieldNotes) -> tuple[dict[str, Any], bool]:
    """Spec values of one schema row, and whether its data type cell was filled."""
    name = row["index_field"]
    role = parse_role(row["field_type"])
    if role is None:
        notes.bad_roles.append((type_name, name, row["field_type"]))
    role = role or UNRECOGNISED_ROLE
    dtype, date_format = parse_dtype(row["data_type"])
    origin = field_origin(name, row["field_type"], parse_source(row["source"]), role)
    values = {"name": name, "dtype": dtype, "date_format": date_format, "role": role, "origin": origin}
    return values, bool(row["data_type"].strip())


def _merge(
    type_name: str, cur: dict[str, Any], new: dict[str, Any], typed: tuple[bool, bool], notes: FieldNotes
) -> None:
    """Fold a second definition of a field into the first: key > required > optional, filled values over blank."""
    conflict = notes.conflicts.append
    if cur["role"] != new["role"]:
        if ROLE_RANK[new["role"]] > ROLE_RANK[cur["role"]]:
            conflict((type_name, cur["name"], "role", new["role"], cur["role"]))
            cur["role"], cur["origin"] = new["role"], new["origin"]
        else:
            conflict((type_name, cur["name"], "role", cur["role"], new["role"]))
    if typed[1] and not typed[0]:
        cur["dtype"] = new["dtype"]
    elif typed[1] and cur["dtype"] != new["dtype"]:
        conflict((type_name, cur["name"], "dtype", cur["dtype"], new["dtype"]))
    if new["date_format"] and not cur["date_format"]:
        cur["date_format"] = new["date_format"]
    elif new["date_format"] and cur["date_format"] != new["date_format"]:
        conflict((type_name, cur["name"], "date_format", cur["date_format"], new["date_format"]))


def field_specs(type_name: str, rows: list[Row], raw: RawOverrides, notes: FieldNotes) -> list[FieldSpec]:
    """The field specs of one type, its spellings merged and field_overrides.yaml applied.

    Args:
        type_name: Schema type name.
        rows: Its schema rows, from every spelling of the type.
        raw: The cabinet's field_overrides.yaml entries.
        notes: Collects unrecognised roles and merge conflicts.

    Returns:
        One spec per normalised field name, in first-seen order.

    Raises:
        ClientPackError: An override value is outside the allowed set.
    """
    merged: dict[str, tuple[dict[str, Any], bool]] = {}
    for row in rows:
        values, typed = _values(type_name, row, notes)
        key = norm(values["name"])
        if key in merged:
            _merge(type_name, merged[key][0], values, (merged[key][1], typed), notes)
            merged[key] = (merged[key][0], merged[key][1] or typed)
        else:
            merged[key] = (values, typed)
    overrides = {norm(f): v for t, fields in raw.items() if norm(t) == norm(type_name) for f, v in fields.items()}
    try:
        return [FieldSpec.model_validate({**v, **overrides.get(k, {})}) for k, (v, _) in merged.items()]
    except ValidationError as e:
        raise ClientPackError(f"bad {FIELD_OVERRIDES_FILE} entry for {type_name}") from e
