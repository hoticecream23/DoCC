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
