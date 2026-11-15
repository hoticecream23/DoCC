"""Pack-mode cascade batches: resume by manifest document, duplicates, cabinet switches, CLI cabinet and dry run."""

import json
from pathlib import Path

import make_pack
import pytest

from baseline import packmode
from baseline.clientpack import ClientPackError, materialize
from baseline.config import load_config

CONFIG = Path(__file__).resolve().parents[2] / "config"
# A second manifest document with the same bytes as the first.
TWIN = ("a000000000000005", "invoice_native.pdf", "Supplier Bill", "INV-2024-0042")


class OtherBackend:
    """Answers "other" with no fields, whatever it is sent."""

    def complete(self, messages, schema):
        from cascade.vlm import VLMReply

        names = schema["properties"]["fields"]["required"]
        return VLMReply(json.dumps({"class": "other", "fields": {n: [] for n in names}}), "fake", "stop")


@pytest.fixture(scope="module")
def twin_pack(corpus, tmp_path_factory):
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(make_pack, "MANIFEST", [*make_pack.MANIFEST, TWIN])
        pack = make_pack.build(tmp_path_factory.mktemp("pack"))
    work = tmp_path_factory.mktemp("work")
    materialize(pack, make_pack.CABINET, corpus, work)
    (work / make_pack.SLUG / "notes.txt").write_text("not in the manifest", encoding="utf-8")
    return pack, work / make_pack.SLUG


@pytest.fixture(scope="module")
def pcfg(twin_pack):
    return packmode.pack_config(load_config(CONFIG), twin_pack[0], make_pack.CABINET)


@pytest.fixture(scope="module")
def ccfg():
    from cascade.config import load_cascade_config

    return load_cascade_config(CONFIG)


def _run(input_dir, out, cfg, ccfg):
    from cascade.runner import run_batch

    return run_batch(input_dir, out, cfg, ccfg, OtherBackend())


def _stems(out):
    return [Path(json.loads(line)["filename"]).stem for line in out.read_text(encoding="utf-8").splitlines()]


def test_a_byte_identical_twin_is_not_skipped_on_resume(twin_pack, pcfg, ccfg, tmp_path):
    out = tmp_path / "out.jsonl"
    _run(twin_pack[1], out, pcfg, ccfg)
    lines = out.read_text(encoding="utf-8").splitlines(keepends=True)
    out.write_text("".join(line for line in lines if TWIN[0] not in line), encoding="utf-8", newline="")
    _run(twin_pack[1], out, pcfg, ccfg)
    assert {"a000000000000001", TWIN[0]} <= set(_stems(out))


def test_a_restart_after_a_crash_neither_duplicates_nor_skips(twin_pack, pcfg, ccfg, tmp_path):
    out = tmp_path / "out.jsonl"
    first = _run(twin_pack[1], out, pcfg, ccfg)
    lines = out.read_text(encoding="utf-8").splitlines(keepends=True)
    out.write_text("".join(lines[:-2]) + lines[-2][:20], encoding="utf-8", newline="")
    again = _run(twin_pack[1], out, pcfg, ccfg)
    stems = _stems(out)
    assert again["written"] == 2 and sorted(stems) == sorted(set(stems)) and len(stems) == first["written"]


def test_resuming_into_another_cabinets_output_is_refused(twin_pack, pcfg, ccfg, tmp_path):
    out = tmp_path / "out.jsonl"
    out.write_text(json.dumps({"doc_id": "x", "cabinet_id": "elsewhere", "filename": "x.pdf"}) + "\n", encoding="utf-8")
    with pytest.raises(ClientPackError, match="elsewhere"):
        _run(twin_pack[1], out, pcfg, ccfg)


def _configured(tmp_path, pack):
    """A config directory whose pipeline.yaml names the pack with a cabinet that does not exist."""
    for src in CONFIG.glob("*.yaml"):
        (tmp_path / src.name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    text = (tmp_path / "pipeline.yaml").read_text(encoding="utf-8")
    text = text.replace("client_pack:\n  dir: null\n  cabinet: null", f"client_pack:\n  dir: '{pack}'\n  cabinet: nope")
    (tmp_path / "pipeline.yaml").write_text(text, encoding="utf-8")
    return tmp_path


def _counting(monkeypatch):
    from cascade import cli

    seen = {}
    monkeypatch.setattr(cli, "report_dry_run", lambda outputs, n: seen.update(n=n) or 0)
    return seen


def test_cli_cabinet_overrides_the_configured_cabinet(twin_pack, tmp_path, monkeypatch):
    from cascade import cli

    seen = _counting(monkeypatch)
    (tmp_path / "cfg").mkdir()
    cdir = _configured(tmp_path / "cfg", twin_pack[0])
    argv = ["--config", str(cdir), "run", "--input", str(twin_pack[1]), "--output", str(tmp_path / "o.jsonl")]
    # The configured cabinet "nope" does not exist, so this passes only if --cabinet replaced it.
    assert cli.main([*argv, "--cabinet", make_pack.CABINET, "--dry-run"]) == 0
    assert seen["n"] == len(make_pack.MANIFEST) + 1


def test_pack_dry_run_counts_manifest_documents_not_files(twin_pack, tmp_path, monkeypatch):
    from cascade import cli

    seen = _counting(monkeypatch)
    argv = ["run", "--input", str(twin_pack[1]), "--output", str(tmp_path / "o.jsonl"), "--dry-run"]
    assert cli.main([*argv, "--client-pack", str(twin_pack[0]), "--cabinet", make_pack.CABINET]) == 0
    assert seen["n"] == len(make_pack.MANIFEST) + 1
