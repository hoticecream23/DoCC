"""Pack-mode baseline batches resume by manifest document, so byte-identical twins each get a record."""

import json
import logging
from pathlib import Path

import make_pack
import pytest

from baseline.clientpack import materialize
from baseline.pipeline import run_batch

# A second manifest document with the same bytes as the first.
TWIN = ("a000000000000005", "invoice_native.pdf", "Supplier Bill", "INV-2024-0042")


@pytest.fixture(scope="module")
def twin_pack(corpus, tmp_path_factory):
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(make_pack, "MANIFEST", [*make_pack.MANIFEST, TWIN])
        pack = make_pack.build(tmp_path_factory.mktemp("pack"))
    work = tmp_path_factory.mktemp("work")
    materialize(pack, make_pack.CABINET, corpus, work)
    return pack, work / make_pack.SLUG


def _run(twin_pack, out):
    return run_batch(twin_pack[1], out, client_pack=(str(twin_pack[0]), make_pack.CABINET))


def _stems(out):
    return [Path(json.loads(line)["filename"]).stem for line in out.read_text(encoding="utf-8").splitlines()]


def test_a_byte_identical_twin_is_not_skipped_on_resume(twin_pack, tmp_path):
    out = tmp_path / "out.jsonl"
    _run(twin_pack, out)
    lines = out.read_text(encoding="utf-8").splitlines(keepends=True)
    out.write_text("".join(line for line in lines if TWIN[0] not in line), encoding="utf-8", newline="")
    _run(twin_pack, out)
    assert {"a000000000000001", TWIN[0]} <= set(_stems(out))


def test_the_store_takes_a_twin_without_an_error(twin_pack, tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        _run(twin_pack, tmp_path / "out.jsonl")
    store_noise = ("db write failed", "db unavailable", "another cabinet")
    assert not [r for r in caplog.records if any(s in r.getMessage() for s in store_noise)]


def test_a_restart_after_a_crash_neither_duplicates_nor_skips(twin_pack, tmp_path):
    out = tmp_path / "out.jsonl"
    first = _run(twin_pack, out)
    lines = out.read_text(encoding="utf-8").splitlines(keepends=True)
    out.write_text("".join(lines[:-2]) + lines[-2][:20], encoding="utf-8", newline="")
    again = _run(twin_pack, out)
    stems = _stems(out)
    assert again["written"] == 2 and sorted(stems) == sorted(set(stems)) and len(stems) == first["written"]
