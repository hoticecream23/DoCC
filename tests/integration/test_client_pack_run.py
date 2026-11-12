"""Pack-mode runs: the cabinet's classes and fields in place of the global ones, end to end."""

import json
from pathlib import Path

import make_pack
import pytest
from vlm_stub import chat, serve

from baseline import packmode
from baseline.clientpack import materialize
from baseline.clientrun import REPORT_FILE, RESULTS_FILE, client_run
from baseline.config import load_config
from baseline.evaluate.client_eval import score_cabinet
from baseline.pipeline import run_batch
from cascade.config import load_cascade_config
from cascade.prompt import response_schema
from cascade.runner import process_document
from cascade.vlm import VLMClient

CONFIG = Path(__file__).resolve().parents[2] / "config"
PACK_CLASSES = {c["name"] for c in make_pack.CLASSES}
FIELD_NAMES = ["Invoice No", "Invoice Date", "Receipt No", "PO No"]  # key first, the erp field left out
# The prior reads "Tax Invoice" in the header; the document is a payment advice.
MISLEADING = "TAX INVOICE copy\n" + "x " * 400 + "\nReceipt No: RCP-77-001\nPaid against the bill above.\n"


@pytest.fixture(scope="module")
def pack(tmp_path_factory):
    return make_pack.build(tmp_path_factory.mktemp("pack"))


@pytest.fixture(scope="module")
def pack_run(pack, corpus, tmp_path_factory):
    work = tmp_path_factory.mktemp("work")
    materialize(pack, make_pack.CABINET, corpus, work)
    out = work / make_pack.SLUG / "results.jsonl"
    run_batch(work / make_pack.SLUG, out, client_pack=(str(pack), make_pack.CABINET))
    return out, [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]


@pytest.fixture()
def pack_cfg(pack):
    return packmode.pack_config(load_config(CONFIG), pack, make_pack.CABINET)


def _answer(label, **fields):
    return json.dumps({"class": label, "fields": {n: fields.get(n.replace(" ", "_"), []) for n in FIELD_NAMES}})


def _cascade(pack_cfg, path, answer, vlm_input="image"):
    ccfg = load_cascade_config(CONFIG)
    ccfg = ccfg.model_copy(update={"vlm": ccfg.vlm.model_copy(update={"input": vlm_input})})
    with serve([(200, chat(answer), 0, None)]) as (url, requests):
        backend = VLMClient(ccfg.vlm.model_copy(update={"base_url": url, "model": "stub", "retries": 0}))
        return process_document(path, pack_cfg, ccfg, backend), requests


def test_pack_run_labels_every_record_with_a_pack_class(pack_run):
    _, records = pack_run
    assert {r["classification"]["label"] for r in records} <= PACK_CLASSES


def test_pack_run_stamps_the_cabinet_slug(pack_run):
    _, records = pack_run
    assert {r["cabinet_id"] for r in records} == {make_pack.SLUG}


def test_pack_run_files_the_key_under_the_client_field_name(pack_run):
    _, records = pack_run
    keys = {r["filename"]: {m["field"]: m["value"] for m in r["metadata"]} for r in records}
    assert keys["a000000000000002.txt"]["Receipt No"] == "RCP-2024-118"


def test_client_eval_joins_a_pack_run_by_file_stem(pack, pack_run):
    out, _ = pack_run
    score = score_cabinet(pack, make_pack.CABINET, out)
    assert (score.missing, score.correct, score.key_exact) == (0, 3, 3)


def test_erp_fields_are_not_asked_of_the_model(pack_cfg):
    asked = response_schema(pack_cfg)["properties"]["fields"]["required"]
    assert asked == FIELD_NAMES


def test_the_model_overrides_a_wrong_prior(pack_cfg, tmp_path):
    doc = tmp_path / "advice.txt"
    doc.write_text(MISLEADING, encoding="utf-8")
    result, _ = _cascade(pack_cfg, doc, _answer("Payment Advice", Receipt_No=["RCP-77-001"]))
    assert result.provenance["prior"] and result.record.classification.label == "Payment Advice"


def test_a_grounded_key_from_the_model_is_kept(pack_cfg, tmp_path):
    doc = tmp_path / "advice.txt"
    doc.write_text(MISLEADING, encoding="utf-8")
    result, _ = _cascade(pack_cfg, doc, _answer("Payment Advice", Receipt_No=["RCP-77-001"]))
    assert [(m.field, m.value) for m in result.record.metadata] == [("Receipt No", "RCP-77-001")]


def test_an_ungrounded_key_from_the_model_is_refused(pack_cfg, tmp_path):
    doc = tmp_path / "advice.txt"
    doc.write_text(MISLEADING, encoding="utf-8")
    result, _ = _cascade(pack_cfg, doc, _answer("Payment Advice", Receipt_No=["RCP-99-999"]))
    assert not [m for m in result.record.metadata if m.field == "Receipt No"]


def test_text_input_sends_no_page_images(pack_cfg, corpus):
    _, requests = _cascade(pack_cfg, corpus / "invoice_native.pdf", _answer("Supplier Bill"), vlm_input="text")
    parts = requests[0]["body"]["messages"][1]["content"]
    assert [p["type"] for p in parts] == ["text"]


def test_client_run_without_a_model_scores_the_baseline(pack, corpus, tmp_path, monkeypatch):
    monkeypatch.delenv("DOCC_VLM_MODEL", raising=False)
    score = client_run(pack, make_pack.CABINET, corpus, tmp_path / "work", config_dir=str(CONFIG))
    report = tmp_path / "work" / make_pack.SLUG / REPORT_FILE
    assert score is not None and score.accuracy > 0 and report.is_file()


def test_client_run_with_a_model_writes_cascade_provenance(pack, corpus, tmp_path, monkeypatch):
    with serve([(200, chat(_answer("other")), 0, None)]) as (url, _):
        monkeypatch.setenv("DOCC_VLM_BASE_URL", url)
        monkeypatch.setenv("DOCC_VLM_MODEL", "stub")
        client_run(pack, make_pack.CABINET, corpus, tmp_path / "work", config_dir=str(CONFIG))
    assert (tmp_path / "work" / make_pack.SLUG / f"{RESULTS_FILE}.provenance").is_file()


def test_client_run_dry_run_writes_nothing(pack, corpus, tmp_path):
    assert client_run(pack, make_pack.CABINET, corpus, tmp_path / "work", dry_run=True) is None
    assert not (tmp_path / "work").exists()


def test_baseline_cli_client_run_dry_run_writes_nothing(pack, corpus, tmp_path):
    from baseline.cli import main

    argv = ["client-run", "--pack", str(pack), "--cabinet", make_pack.CABINET, "--share", str(corpus)]
    assert main([*argv, "--work", str(tmp_path / "work"), "--dry-run"]) == 0
    assert not (tmp_path / "work").exists()


def test_baseline_cli_run_takes_a_client_pack(pack, pack_run, tmp_path):
    from baseline.cli import main

    out = tmp_path / "cli.jsonl"
    argv = ["run", "--input", str(pack_run[0].parent), "--output", str(out)]
    assert main([*argv, "--client-pack", str(pack), "--cabinet", make_pack.CABINET]) == 0
    assert {json.loads(line)["cabinet_id"] for line in out.read_text(encoding="utf-8").splitlines()} == {make_pack.SLUG}


def test_cascade_cli_run_takes_a_client_pack(pack, pack_run, tmp_path):
    from cascade.cli import main

    out = tmp_path / "cascade.jsonl"
    with serve([(200, chat(_answer("Buying Order")), 0, None)]) as (url, _):
        argv = ["--config", str(CONFIG), "run", "--input", str(pack_run[0].parent), "--output", str(out)]
        argv += ["--base-url", url, "--model", "stub", "--client-pack", str(pack), "--cabinet", make_pack.CABINET]
        assert main(argv) == 0
    labels = {json.loads(line)["classification"]["label"] for line in out.read_text(encoding="utf-8").splitlines()}
    assert labels == {"Buying Order"}
