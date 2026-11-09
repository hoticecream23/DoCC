"""Shared error hierarchy and the cascade model check."""

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
