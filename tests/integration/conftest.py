"""Mark every test in this folder as integration, so `-m "not integration"` skips them."""

from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


def pytest_collection_modifyitems(config, items):
    for item in items:
        if HERE in Path(item.fspath).resolve().parents:
            item.add_marker(pytest.mark.integration)
