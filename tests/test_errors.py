"""Shared error hierarchy and the demo server's model check."""

import pytest

from baseline import errors
from baseline.classify import ModelIntegrityError
from baseline.errors import ConfigError, DoccError, summarize_exception
from baseline.metadata.fields import OffsetMismatch
from cascade.config import load_cascade_config
from cascade.prompt import VLMOutputError
from cascade.vlm import VLMBadRequest, VLMError
from demo import server
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
    settings = server.vlm_settings(ccfg)
    assert settings.model is None
    with pytest.raises(ConfigError, match="DOCC_VLM_MODEL"):
        server.require_model(settings)


def test_demo_without_a_model_serves_baseline_and_503s_cascade(monkeypatch):
    import json
    import threading
    import urllib.error
    import urllib.request
    from pathlib import Path

    monkeypatch.delenv("DOCC_VLM_MODEL", raising=False)
    if load_cascade_config(None).vlm.model:
        pytest.skip("cascade.yaml sets a model")
    srv = server.make_server(0, Path(__file__).resolve().parent.parent)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}/api/run"

    def post(url, data):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, data=data, method="POST"), timeout=120) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    try:
        assert post(base + "?filename=x.txt", b"Invoice No: 123")[0] == 200
        code, body = post(base + "?filename=x.txt&mode=cascade", b"Invoice No: 123")
        assert code == 503 and "DOCC_VLM_MODEL" in body["error"]
    finally:
        srv.shutdown()
        srv.server_close()
