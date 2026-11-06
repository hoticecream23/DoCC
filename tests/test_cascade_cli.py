import json

from conftest import ROOT
from vlm_stub import chat, serve


def test_run_against_a_local_server_writes_records_and_provenance(corpus, tmp_path):
    from baseline.config import load_config
    from cascade.cli import main

    out = tmp_path / "out.jsonl"
    names = [f["name"] for f in load_config(ROOT / "config").field_defs]
    answer = chat(json.dumps({"class": "unknown", "fields": {n: [] for n in names}}))
    with serve([(200, answer, 0, None)]) as (url, reqs):
        rc = main(
            [
                "--config",
                str(ROOT / "config"),
                "run",
                "--input",
                str(corpus),
                "--output",
                str(out),
                "--model",
                "m",
                "--base-url",
                url,
            ]
        )
    assert rc == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    provs = (tmp_path / "out.jsonl.provenance").read_text(encoding="utf-8").splitlines()
    assert lines and len(lines) == len(provs)
    assert reqs and all(r["body"]["model"] == "m" for r in reqs)


def test_a_remote_server_is_refused_before_any_document_is_read(corpus, tmp_path):
    from cascade.cli import main

    out = tmp_path / "out.jsonl"
    rc = main(
        [
            "--config",
            str(ROOT / "config"),
            "run",
            "--input",
            str(corpus),
            "--output",
            str(out),
            "--model",
            "m",
            "--base-url",
            "http://192.0.2.1:8000/v1",
        ]
    )
    assert rc == 1
    assert not out.exists()


def test_no_model_configured_is_refused(corpus, tmp_path):
    from cascade.cli import main

    rc = main(
        ["--config", str(ROOT / "config"), "run", "--input", str(corpus), "--output", str(tmp_path / "out.jsonl")]
    )
    assert rc == 1
