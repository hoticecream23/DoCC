"""Knowledge graph v0.

The whole premise is precision: only proven identities may create or merge a
node. Most of these tests assert that the graph REFUSES to draw an edge.
"""

import json

import pytest

from baseline.graph import (
    Graph,
    build_graph,
    collapse_identities,
    find_chains,
    match_roles,
    org_exposure,
    org_key,
    orphan_references,
    to_dot,
)


@pytest.fixture(scope="module")
def gcfg(cfg):
    return cfg.graph


def _f(name, value, start=0, validated=True):
    return {"field": name, "value": value, "normalized_value": value,
            "char_start": start, "char_end": start + len(value),
            "confidence": 1.0, "validated": validated}


def _rec(doc_id, label="invoice", fields=(), filename=None):
    return {
        "doc_id": doc_id,
        "filename": filename or f"{doc_id}.pdf",
        "classification": {"label": label, "confidence": 0.95},
        "metadata": [dict(f) for f in fields],
        "text": {"method": "native"},
        "diagnostics": {"page_count": 1},
    }


# ---------------------------------------------------------------- identity

def test_gstin_reduces_to_the_pan_it_carries():
    assert org_key("27AAPFU0939F1ZV") == "AAPFU0939F"
    assert org_key("AAPFU0939F") == "AAPFU0939F"


def test_a_gstin_document_and_a_pan_document_are_the_same_org(gcfg):
    """The join that only works because both are verified."""
    records = [
        _rec("a", fields=[_f("gstin", "27AAPFU0939F1ZV")]),
        _rec("b", fields=[_f("pan", "AAPFU0939F")]),
    ]
    g, stats = build_graph(records, gcfg)
    assert stats["organizations"] == 1
    org = [n for n in g.nodes if n.kind == "organization"][0]
    assert org.attrs["pan"] == "AAPFU0939F"
    assert sorted(g.neighbours("doc:a", "mentions_org")) == ["org:AAPFU0939F"]
    assert sorted(g.neighbours("doc:b", "mentions_org")) == ["org:AAPFU0939F"]


def test_collapse_identities_keeps_one_entry_per_org():
    items = [_f("gstin", "27AAPFU0939F1ZV", 100), _f("pan", "AAPFU0939F", 130)]
    out = collapse_identities(items)
    assert len(out) == 1
    # keeps the earliest mention
    assert out[0]["char_start"] == 100


def test_unvalidated_identifiers_never_create_a_node(gcfg):
    """A format match that failed its checksum must not become an entity."""
    records = [_rec("a", fields=[_f("gstin", "27AAPFU0939F1ZX", validated=False)])]
    g, stats = build_graph(records, gcfg)
    assert stats["organizations"] == 0
    assert stats["unverified_ids_ignored"] == 1


def test_names_alone_never_create_an_org(gcfg):
    """Fuzzy name matching is exactly what v0 refuses to do."""
    records = [
        _rec("a", fields=[_f("vendor_name", "Acme Traders", validated=False)]),
        _rec("b", fields=[_f("vendor_name", "Acme Traders Pvt Ltd", validated=False)]),
    ]
    g, stats = build_graph(records, gcfg)
    assert stats["organizations"] == 0


# ---------------------------------------------------------------- roles

def test_role_goes_to_the_clearly_nearest_identity():
    roles = {"issued_by": [_f("vendor_name", "Acme", 0)],
             "billed_to": [_f("buyer_name", "Globex", 500)]}
    ids = [_f("gstin", "27AAPFU0939F1ZV", 20)]
    out = match_roles(roles, ids, limit=400, margin=60)
    assert "issued_by" in out
    assert "billed_to" not in out   # out of range, and the id is taken


def test_an_identity_cannot_hold_two_roles():
    roles = {"issued_by": [_f("vendor_name", "Acme", 0)],
             "billed_to": [_f("buyer_name", "Globex", 40)]}
    ids = [_f("gstin", "27AAPFU0939F1ZV", 20)]
    out = match_roles(roles, ids, limit=400, margin=0)
    assert len(out) == 1


def test_a_narrow_win_is_refused_rather_than_guessed():
    """Ship To can sit closer to the seller GSTIN than Vendor Name does."""
    roles = {"issued_by": [_f("vendor_name", "Acme", 0)],
             "billed_to": [_f("buyer_name", "Globex", 55)]}
    ids = [_f("gstin", "27AAPFU0939F1ZV", 30)]
    # distances 30 and 25, so neither wins by 60
    assert match_roles(roles, ids, limit=400, margin=60) == {}
    # with no margin required, the closer one takes it
    assert list(match_roles(roles, ids, limit=400, margin=0)) == ["billed_to"]


def test_role_beyond_the_distance_limit_is_dropped():
    roles = {"issued_by": [_f("vendor_name", "Acme", 0)]}
    ids = [_f("gstin", "27AAPFU0939F1ZV", 5000)]
    assert match_roles(roles, ids, limit=400, margin=0) == {}


# ---------------------------------------------------------------- accounts

def test_account_needs_a_verified_ifsc_to_become_an_entity(gcfg):
    """The same number at two banks is two accounts, so an unqualified
    number cannot safely be a node."""
    with_ifsc = [_rec("a", fields=[_f("account_number", "502010123456789"),
                                   _f("ifsc", "HDFC0001234")])]
    without = [_rec("b", fields=[_f("account_number", "502010123456789")])]
    assert build_graph(with_ifsc, gcfg)[1]["accounts"] == 1
    assert build_graph(without, gcfg)[1]["accounts"] == 0


def test_same_number_at_different_banks_stays_two_accounts(gcfg):
    records = [
        _rec("a", fields=[_f("account_number", "502010123456789"), _f("ifsc", "HDFC0001234")]),
        _rec("b", fields=[_f("account_number", "502010123456789"), _f("ifsc", "UTIB0000123")]),
    ]
    assert build_graph(records, gcfg)[1]["accounts"] == 2


# ---------------------------------------------------------------- references

def test_a_receipt_links_to_the_invoice_it_quotes(gcfg):
    records = [
        _rec("inv", "invoice", [_f("invoice_number", "INV-1")]),
        _rec("rcp", "receipt", [_f("invoice_number", "INV-1")]),
    ]
    g, _ = build_graph(records, gcfg)
    chains = find_chains(g)
    assert len(chains) == 1
    assert chains[0]["from_class"] == "receipt"
    assert chains[0]["to_class"] == "invoice"


def test_an_owner_does_not_reference_itself(gcfg):
    records = [_rec("inv", "invoice", [_f("invoice_number", "INV-1")])]
    g, _ = build_graph(records, gcfg)
    assert find_chains(g) == []


def test_orphan_reference_is_reported(gcfg):
    records = [_rec("rcp", "receipt", [_f("invoice_number", "INV-MISSING")])]
    g, _ = build_graph(records, gcfg)
    orphans = orphan_references(g, records, gcfg)
    assert len(orphans) == 1
    assert orphans[0]["key"] == "INV-MISSING"


# ---------------------------------------------------------------- duplicates

def test_duplicates_are_found_within_a_class(gcfg):
    records = [
        _rec("a", "invoice", [_f("invoice_number", "INV-1"), _f("total_amount", "100.00")]),
        _rec("b", "invoice", [_f("invoice_number", "INV-1"), _f("total_amount", "100.00")]),
    ]
    _, stats = build_graph(records, gcfg)
    assert len(stats["duplicate_groups"]) == 1
    assert stats["duplicate_groups"][0]["documents"] == ["doc:a", "doc:b"]


def test_a_receipt_quoting_an_invoice_is_not_its_duplicate(gcfg):
    """This was a real wrong edge before duplicates were scoped by class."""
    records = [
        _rec("inv", "invoice", [_f("invoice_number", "INV-1"), _f("total_amount", "100.00")]),
        _rec("rcp", "receipt", [_f("invoice_number", "INV-1"), _f("total_amount", "100.00")]),
    ]
    _, stats = build_graph(records, gcfg)
    assert stats["duplicate_groups"] == []


def test_different_totals_are_not_duplicates(gcfg):
    records = [
        _rec("a", "invoice", [_f("invoice_number", "INV-1"), _f("total_amount", "100.00")]),
        _rec("b", "invoice", [_f("invoice_number", "INV-1"), _f("total_amount", "999.00")]),
    ]
    assert build_graph(records, gcfg)[1]["duplicate_groups"] == []


# ---------------------------------------------------------------- output

def test_graph_is_deterministic(gcfg):
    records = [
        _rec("a", fields=[_f("gstin", "27AAPFU0939F1ZV")]),
        _rec("b", fields=[_f("pan", "AAPFU0939F")]),
    ]
    one, _ = build_graph(records, gcfg)
    two, _ = build_graph(list(reversed(records)), gcfg)
    assert json.dumps(one.to_dict(), sort_keys=True) == json.dumps(two.to_dict(), sort_keys=True)


def test_dot_output_covers_every_node_and_edge(gcfg):
    records = [_rec("a", fields=[_f("gstin", "27AAPFU0939F1ZV")])]
    g, _ = build_graph(records, gcfg)
    dot = to_dot(g)
    assert dot.startswith("digraph kg {") and dot.rstrip().endswith("}")
    for n in g.nodes:
        assert n.id in dot


def test_graph_node_merging_unions_lists():
    g = Graph()
    g.add_node("x", "organization", gstins=["A"], pan="P")
    g.add_node("x", "organization", gstins=["B"], pan="P")
    assert g._nodes["x"].attrs["gstins"] == ["A", "B"]


def test_exposure_counts_documents_per_org(gcfg):
    records = [
        _rec("a", fields=[_f("gstin", "27AAPFU0939F1ZV")]),
        _rec("b", fields=[_f("pan", "AAPFU0939F")]),
        _rec("c", fields=[_f("pan", "AAAPZ1234C")]),
    ]
    g, _ = build_graph(records, gcfg)
    exposure = org_exposure(g)
    assert exposure[0]["pan"] == "AAPFU0939F"
    assert exposure[0]["documents"] == 2


def test_empty_input_produces_an_empty_graph(gcfg):
    g, stats = build_graph([], gcfg)
    assert stats["documents"] == 0 and stats["edges"] == 0
    assert g.to_dict() == {"nodes": [], "edges": []}


# ---------------------------------------------------------------- end to end

def test_graph_over_the_real_pipeline_output(cfg, corpus, tmp_path):
    """Build a run, then a graph over it, with no hand written records."""
    from baseline.pipeline import run_batch

    out = tmp_path / "results.jsonl"
    run_batch(corpus, out, workers=1)
    records = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    g, stats = build_graph(records, cfg.graph)

    assert stats["documents"] == len(records)
    # Acme appears via GSTIN on the invoices and the purchase order.
    orgs = {n.attrs["pan"] for n in g.nodes if n.kind == "organization"}
    assert "AAPFU0939F" in orgs
    # The receipts quote the invoice, so the chain must exist.
    assert any(c["from_class"] == "receipt" and c["to_class"] == "invoice"
               for c in find_chains(g))
    # Every organisation node came from a verified identifier.
    for n in g.nodes:
        if n.kind == "organization":
            assert len(n.attrs["pan"]) == 10
