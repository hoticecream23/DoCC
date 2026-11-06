"""Knowledge graph over a real pipeline run."""

import json

from baseline.graph import build_graph, find_chains


def test_graph_over_the_real_pipeline_output(cfg, corpus, tmp_path):
    """Build a run, then a graph over it, with no hand written records."""
    from baseline.pipeline import run_batch

    out = tmp_path / "results.jsonl"
    run_batch(corpus, out, workers=1)
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    g, stats = build_graph(records, cfg.graph)

    assert stats["documents"] == len(records)
    # Acme appears via GSTIN on the invoices and the purchase order.
    orgs = {n.attrs["pan"] for n in g.nodes if n.kind == "organization"}
    assert "AAPFU0939F" in orgs
    # The receipts quote the invoice, so the chain must exist.
    assert any(c["from_class"] == "receipt" and c["to_class"] == "invoice" for c in find_chains(g))
    # Every organisation node came from a verified identifier.
    for n in g.nodes:
        if n.kind == "organization":
            assert len(n.attrs["pan"]) == 10
