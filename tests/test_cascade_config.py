import pytest
import yaml
from pydantic import ValidationError

from conftest import ROOT


def _write(tmp_path, data):
    (tmp_path / "cascade.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    return tmp_path


def _repo_data():
    return yaml.safe_load((ROOT / "config" / "cascade.yaml").read_text(encoding="utf-8"))


def test_repo_config_loads():
    from cascade.config import load_cascade_config

    c = load_cascade_config(ROOT / "config")
    assert c.vlm.base_url.startswith("http://127.0.0.1")
    assert c.vlm.model is None  # chosen per deployment, never defaulted
    assert c.confidence.agree >= c.confidence.grounded >= c.confidence.ambiguous


def test_a_misspelt_key_fails_at_load(tmp_path):
    from cascade.config import load_cascade_config

    data = _repo_data()
    data["vlm"]["timeout"] = 5  # should be timeout_s
    with pytest.raises(ValidationError):
        load_cascade_config(_write(tmp_path, data))


def test_confidence_tiers_must_be_ordered(tmp_path):
    from cascade.config import load_cascade_config

    data = _repo_data()
    data["confidence"]["ambiguous"] = 0.95
    with pytest.raises(ValidationError, match="agree >= grounded >= ambiguous"):
        load_cascade_config(_write(tmp_path, data))


def test_a_confidence_above_one_is_rejected(tmp_path):
    from cascade.config import load_cascade_config

    data = _repo_data()
    data["confidence"]["vlm_class"] = 1.5
    with pytest.raises(ValidationError):
        load_cascade_config(_write(tmp_path, data))
