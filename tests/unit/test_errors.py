"""Shared error hierarchy and the cascade model check."""

import os

import pytest

from baseline import errors
from baseline.classify import ModelIntegrityError
from baseline.errors import ConfigError, DoccError, summarize_exception
from baseline.metadata.fields import OffsetMismatch
from cascade.config import load_cascade_config, require_model, vlm_settings
from cascade.prompt import VLMOutputError
from cascade.vlm import VLMBadRequest, VLMError
from demo.server import BadInput, Unavailable


@pytest.mark.parametrize(
    ("cls", "builtin"),
    [
        (ModelIntegrityError, ValueError),
        (OffsetMismatch, ValueError),
        (VLMError, RuntimeError),
        (VLMBadRequest, RuntimeError),
        (VLMOutputError, ValueError),
        (BadInput, ValueError),
        (Unavailable, RuntimeError),
        (ConfigError, Exception),
    ],
)
def test_every_domain_error_is_a_docc_error_and_keeps_its_builtin(cls, builtin):
    assert issubclass(cls, DoccError)
    assert issubclass(cls, builtin)


def test_old_import_paths_are_the_shared_classes():
    assert ModelIntegrityError is errors.ModelIntegrityError
    assert VLMError is errors.VLMError
    assert BadInput is errors.BadInput


def test_from_chaining_is_preserved():
    with pytest.raises(VLMOutputError) as info:
        try:
            raise ValueError("low level")
        except ValueError as e:
            raise VLMOutputError("answer is not JSON") from e
    assert isinstance(info.value.__cause__, ValueError)


def test_summarize_exception_truncates_and_names_the_type():
    out = summarize_exception(ValueError("x" * 500))
    assert out == "ValueError: " + "x" * 200 + "..."
    assert summarize_exception(KeyError("k"), limit=10) == "KeyError: 'k'"


def test_demo_has_no_default_model(monkeypatch):
    monkeypatch.delenv("DOCC_VLM_MODEL", raising=False)
    ccfg = load_cascade_config(None)
    if ccfg.vlm.model:
        pytest.skip("cascade.yaml sets a model")
    settings = vlm_settings(ccfg)
    assert settings.model is None
    with pytest.raises(ConfigError, match="DOCC_VLM_MODEL"):
        require_model(settings)


def test_without_path_cuts_every_spelling_of_the_path_to_the_name(tmp_path):
    p = tmp_path / "Acme Client" / "x.pdf"
    raw = str(p)
    forms = [raw, p.as_posix(), raw.replace("\\", "\\\\")] + ([raw.upper()] if os.name == "nt" else [])
    out = errors.without_path(" | ".join(forms), p)
    assert "acme client" not in out.lower()
    assert out == " | ".join(["x.pdf"] * len(forms))


def test_failed_extraction_logs_the_file_name_not_its_directory(tmp_path, caplog, cfg):
    from baseline.extract import extract

    folder = tmp_path / "Acme Client"
    folder.mkdir()
    bad = folder / "garbage.pdf"
    bad.write_bytes(b"not a pdf at all")
    with caplog.at_level("ERROR"):
        res = extract(bad, cfg)
    assert res.errors
    records = [r for r in caplog.records if r.getMessage() == "extraction failed"]
    assert records
    for record in records:
        dump = str(record.__dict__)
        assert "garbage.pdf" in dump
        assert str(folder) not in dump and folder.as_posix() not in dump
        assert str(folder).replace("\\", "\\\\") not in dump
        assert record.__dict__["file"] == "garbage.pdf"
