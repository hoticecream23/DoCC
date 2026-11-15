"""Diagnostics notes, a null config section in pack mode, and the text-mode page plan."""

from dataclasses import replace
from pathlib import Path

import make_pack

from baseline import packmode
from baseline.config import load_config
from baseline.schema import Diagnostics
from cascade.prompt import plan_text

CONFIG = Path(__file__).resolve().parents[2] / "config"


def _diag(**extra):
    return Diagnostics(page_count=1, is_scanned=False, has_native_text=True, **extra)


def test_empty_notes_are_left_out_of_the_serialised_record():
    assert "notes" not in _diag().model_dump(mode="json")


def test_notes_are_serialised_when_present():
    assert _diag(notes=["read frame 1 of 3"]).model_dump(mode="json")["notes"] == ["read frame 1 of 3"]


def test_a_record_without_notes_still_loads():
    assert Diagnostics.model_validate({"page_count": 1, "is_scanned": False, "has_native_text": True}).notes == []


def test_a_null_extraction_section_gets_pack_defaults(tmp_path):
    cfg = load_config(CONFIG)
    cfg = replace(cfg, pipeline={**cfg.pipeline, "extraction": None, "metadata": None})
    pcfg = packmode.pack_config(cfg, make_pack.build(tmp_path), make_pack.CABINET)
    assert pcfg.extract_opts()["max_pages"] == packmode.PACK_MAX_PAGES


def test_a_tight_total_still_sends_the_last_page():
    plan = plan_text(["A" * 5000, "B" * 5000], 4000, 4000)
    assert (plan.included, plan.kept) == ([0, 1], [2000, 2000])
