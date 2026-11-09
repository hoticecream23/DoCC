import pytest

pytest.importorskip("fastapi")

import api.__main__ as entry  # noqa: E402


def test_main_accepts_a_config_file_and_binds_loopback(monkeypatch, cfg):
    seen = {}
    monkeypatch.setattr(entry, "setup_logging", lambda: None)
    monkeypatch.setattr(entry, "create_app", lambda config_dir, workers: ("app", config_dir, workers))
    monkeypatch.setattr(entry.uvicorn, "run", lambda app, host, port, log_config: seen.update(app=app, host=host))
    entry.main(["--config", str(cfg.config_dir / "pipeline.yaml"), "--workers", "2"])
    assert seen == {"app": ("app", cfg.config_dir, 2), "host": "127.0.0.1"}
