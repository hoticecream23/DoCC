import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
# Let tests import the package and the corpus helpers without installing.
for p in (str(ROOT), str(ROOT / "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from baseline.config import load_config  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config(ROOT / "config")


@pytest.fixture(scope="session")
def corpus(tmp_path_factory):
    from make_corpus import build

    return build(tmp_path_factory.mktemp("corpus"))


def _serial_run(tmp_path_factory, corpus, config_dir):
    from types import SimpleNamespace

    from baseline.pipeline import run_batch

    out = tmp_path_factory.mktemp("serial") / "serial.jsonl"
    stats = run_batch(corpus, out, config_dir=config_dir, workers=1)
    return SimpleNamespace(out=out, stats=stats, config_dir=config_dir)


@pytest.fixture(scope="session")
def fixed_timings_config(cfg, tmp_path_factory):
    """Copy the config and zero the timings, the one field that varies."""
    import shutil

    import yaml

    cfgdir = tmp_path_factory.mktemp("detcfg")
    for name in ("classes.yaml", "fields.yaml", "tags.yaml"):
        shutil.copyfile(cfg.config_dir / name, cfgdir / name)
    pipe = yaml.safe_load((cfg.config_dir / "pipeline.yaml").read_text(encoding="utf-8"))
    pipe.setdefault("runtime", {})["deterministic_timings"] = True
    (cfgdir / "pipeline.yaml").write_text(yaml.safe_dump(pipe), encoding="utf-8")
    return str(cfgdir)


@pytest.fixture(scope="session")
def serial_run(tmp_path_factory, corpus, fixed_timings_config):
    """One serial batch over the corpus with timings pinned. Read only: copy before writing."""
    return _serial_run(tmp_path_factory, corpus, fixed_timings_config)


@pytest.fixture(scope="session")
def default_serial_run(tmp_path_factory, corpus):
    """One serial batch over the corpus with the default config. Read only."""
    return _serial_run(tmp_path_factory, corpus, None)
