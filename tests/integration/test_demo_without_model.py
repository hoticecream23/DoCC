"""The demo server with no VLM model configured."""

import pytest

from cascade.config import load_cascade_config
from demo import server


def test_demo_without_a_model_serves_baseline_and_503s_cascade(monkeypatch):
    import json
    import threading
    import urllib.error
    import urllib.request
    from pathlib import Path

    monkeypatch.delenv("DOCC_VLM_MODEL", raising=False)
    if load_cascade_config(None).vlm.model:
        pytest.skip("cascade.yaml sets a model")
    srv = server.make_server(0, Path(__file__).resolve().parents[2])
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}/api/run"

    def post(url, data):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, data=data, method="POST"), timeout=10) as r:
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
