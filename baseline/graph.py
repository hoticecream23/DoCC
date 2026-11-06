"""Knowledge graph v0, built over the JSONL.

Over 400 lines: entity rules, merging and edge building share one narrow policy; splitting would scatter it.

Deliberately narrow. Only fields whose format proves them may create or merge
an entity, so every node here is one we can defend. Fuzzy name matching is
where graphs quietly fill with wrong edges, so v0 does not do it at all.

This never touches the document schema. It is a separate pass over records
that already exist. Rendering (DOT, markdown) lives in graph_report.py.
"""

from __future__ import annotations

import collections
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

from .graph_report import render_report, to_dot
from .io import atomic_write_text
from .logging_setup import get_logger
from .schema import read_jsonl

__all__ = [
    "Edge",
    "Graph",
    "Node",
    "build",
    "build_graph",
    "collapse_identities",
    "find_chains",
    "match_roles",
    "org_exposure",
    "org_key",
    "orphan_references",
    "render_report",
    "to_dot",
]

log = get_logger(__name__)

# Chars 3 to 12 of a GSTIN are the holder PAN.
GSTIN_PAN_SLICE = slice(2, 12)
GSTIN_LENGTH = 15

Field = dict[str, Any]
"""One metadata entry of a results record."""


@dataclass(frozen=True)
class Node:
    """A graph vertex: a document, organisation or account.

    Attributes:
        id: Unique id, prefixed by kind (doc:, org:, acct:).
        kind: document, organization or account.
        attrs: Free-form attributes.
    """

    id: str
    kind: str
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Edge:
    """A directed, typed graph edge.

    Attributes:
        src: Source node id.
        dst: Destination node id.
        type: Edge type, e.g. mentions_org or references.
        attrs: Free-form attributes.
    """

    src: str
    dst: str
    type: str
    attrs: dict[str, Any] = field(default_factory=dict)


class Graph:
    """A small, deterministic, in-memory directed multigraph."""

    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}
        self._edges: dict[tuple[str, str, str], Edge] = {}

    def add_node(self, node_id: str, kind: str, **attrs: Any) -> str:
        """Add a node, or merge attributes into an existing one.

        List and set attributes are unioned; scalars keep the first value seen.

        Args:
            node_id: Unique id.
            kind: Node kind; an existing node keeps its kind.
            **attrs: Attributes to set or merge.

        Returns:
            node_id.
        """
        cur = self._nodes.get(node_id)
        if cur is None:
            self._nodes[node_id] = Node(node_id, kind, dict(attrs))
            return node_id
        merged = dict(cur.attrs)
        for k, v in attrs.items():
            if isinstance(v, (list, set)) or isinstance(merged.get(k), (list, set)):
                have = set(merged.get(k) or [])
                have |= set(v if isinstance(v, (list, set)) else [v])
                merged[k] = sorted(have)
            else:
                merged.setdefault(k, v)
        self._nodes[node_id] = Node(node_id, cur.kind, merged)
        return node_id

    def add_edge(self, src: str, dst: str, etype: str, **attrs: Any) -> None:
        """Add an edge; a repeat of (src, dst, etype) is ignored.

        Args:
            src: Source node id.
            dst: Destination node id.
            etype: Edge type.
            **attrs: Edge attributes.
        """
        key = (src, dst, etype)
        if key in self._edges:
            return
        self._edges[key] = Edge(src, dst, etype, dict(attrs))

    @property
    def nodes(self) -> list[Node]:
        """All nodes, sorted by id."""
        return [self._nodes[k] for k in sorted(self._nodes)]

    @property
    def edges(self) -> list[Edge]:
        """All edges, sorted by (src, dst, type)."""
        return [self._edges[k] for k in sorted(self._edges)]

    def neighbours(self, node_id: str, etype: str | None = None) -> list[str]:
        """Destinations of node_id's outgoing edges.

        Args:
            node_id: Source node.
            etype: Only follow edges of this type; None for all.

        Returns:
            Sorted destination ids.
        """
        return sorted(e.dst for e in self.edges if e.src == node_id and (etype is None or e.type == etype))

    def to_dict(self) -> dict[str, list[dict[str, Any]]]:
        """JSON-ready nodes and edges, attributes flattened in.

        Returns:
            {"nodes": [...], "edges": [...]}.
        """
        return {
            "nodes": [{"id": n.id, "kind": n.kind, **n.attrs} for n in self.nodes],
            "edges": [{"src": e.src, "dst": e.dst, "type": e.type, **e.attrs} for e in self.edges],
        }


class DuplicateGroup(TypedDict):
    """Documents that agree on every match_on field."""

    key: list[str]
    documents: list[str]


class GraphStats(TypedDict):
    """Counts build_graph reports."""

    documents: int
    organizations: int
    accounts: int
    edges: int
    edges_by_type: dict[str, int]
    duplicate_groups: list[DuplicateGroup]
    unverified_ids_ignored: int


Chain = TypedDict("Chain", {"from": str, "from_class": str, "to": str, "to_class": str, "key": str})
"""One document citing another."""


class OrgExposure(TypedDict):
    """One verified organisation and the documents it appears in."""

    org: str
    pan: str
    gstins: list[str]
    names: list[str]
    documents: int
    files: list[str]


Orphan = TypedDict("Orphan", {"document": str, "class": str, "key": str})
"""A document citing a reference nothing in the corpus owns."""


class GraphBuild(TypedDict):
    """Everything build() produced."""

    graph: Graph
    stats: GraphStats
    chains: list[Chain]
    exposure: list[OrgExposure]
    orphans: list[Orphan]


# --------------------------------------------------------------------------
# building
# --------------------------------------------------------------------------


def _validated(record: dict[str, Any]) -> dict[str, list[Field]]:
    """Only fields the pipeline could actually prove. This is the whole trick."""
    out: dict[str, list[Field]] = collections.defaultdict(list)
    for m in record.get("metadata") or []:
        if m.get("validated"):
            out[m["field"]].append(m)
    return out


def _all_fields(record: dict[str, Any]) -> dict[str, list[Field]]:
    out: dict[str, list[Field]] = collections.defaultdict(list)
    for m in record.get("metadata") or []:
        out[m["field"]].append(m)
    return out


def _label(record: dict[str, Any]) -> str:
    return str((record.get("classification") or {}).get("label", "unknown"))


def org_key(value: str) -> str:
    """One organisation, one key. A GSTIN reduces to the PAN it carries.

    Args:
        value: A normalised GSTIN or PAN.

    Returns:
        The PAN.
    """
    return value[GSTIN_PAN_SLICE] if len(value) == GSTIN_LENGTH else value


def collapse_identities(id_items: list[Field]) -> list[Field]:
    """One entry per organisation.

    A document usually prints both the GSTIN and the PAN of the same company.
    Treating those as two identities lets one party claim two roles, which is
    how the vendor ends up holding the buyer name.

    Args:
        id_items: GSTIN and PAN metadata entries.

    Returns:
        The earliest entry per organisation, sorted by organisation key.
    """
    best: dict[str, Field] = {}
    for item in id_items:
        value = item.get("normalized_value") or ""
        if not value:
            continue
        key = org_key(value)
        cur = best.get(key)
        if cur is None or int(item.get("char_start", 0)) < int(cur.get("char_start", 0)):
            best[key] = item
    return [best[k] for k in sorted(best)]


RoleMatch = tuple[int, Field, Field]
"""Distance, the role entry, the identity entry."""


def _role_pairs(
    role_items: dict[str, list[Field]], id_items: list[Field], limit: int
) -> list[tuple[int, str, Field, Field]]:
    pairs = []
    for role, items in role_items.items():
        for role_item in items:
            for ident_item in id_items:
                dist = abs(int(ident_item.get("char_start", 0)) - int(role_item.get("char_start", 0)))
                if dist <= limit:
                    pairs.append((dist, role, role_item, ident_item))
    # Sort by distance, then by role and offset so ties resolve the same way twice.
    pairs.sort(key=lambda t: (t[0], t[1], int(t[2].get("char_start", 0))))
    return pairs


def match_roles(
    role_items: dict[str, list[Field]], id_items: list[Field], limit: int, margin: int = 0
) -> dict[str, RoleMatch]:
    """Assign roles to verified identities, at most one role per identity.

    Documents name several parties but often carry only one verified number.
    Letting every role attach to that number is how a vendor node ends up
    holding the buyer name, so this is a greedy closest first matching.

    Proximity alone is weak evidence. On a purchase order "Ship To" can sit
    closer to the seller GSTIN than "Vendor Name" does, purely by layout. So
    a winner must beat the runner up for the same identity by `margin`
    characters. If it does not, the document still records that the party was
    mentioned but claims no role. Abstaining beats a confident wrong edge.

    Args:
        role_items: Role (edge type) to the name entries for it.
        id_items: Verified identity entries, one per organisation.
        limit: Max character distance between name and identity.
        margin: How much closer the winner must be than a rival role.

    Returns:
        Role to (distance, role entry, identity entry).
    """
    pairs = _role_pairs(role_items, id_items, limit)
    out: dict[str, RoleMatch] = {}
    claimed_ids: set[str] = set()
    claimed_roles: set[str] = set()
    for idx, (dist, role, role_item, ident_item) in enumerate(pairs):
        ident = org_key(ident_item.get("normalized_value", ""))
        if ident in claimed_ids or role in claimed_roles:
            continue
        # Nearest competing claim on the same identity from a different role.
        rival = next(
            (
                other_dist
                for other_dist, other_role, _, other_ident in pairs[idx + 1 :]
                if other_role != role and org_key(other_ident.get("normalized_value", "")) == ident
            ),
            None,
        )
        claimed_ids.add(ident)
        if rival is not None and (rival - dist) < margin:
            # Too close to call. Record nothing rather than guess.
            continue
        claimed_roles.add(role)
        out[role] = (dist, role_item, ident_item)
    return out


def build_graph(records: list[dict[str, Any]], gcfg: dict[str, Any]) -> tuple[Graph, GraphStats]:
    """Build the graph from result records.

    Args:
        records: Results JSONL rows.
        gcfg: The graph.yaml config.

    Returns:
        The graph and its counts.
    """
    g = Graph()
    skipped_unverified = 0
    for rec in records:
        if rec.get("doc_id"):
            skipped_unverified += _add_document(g, rec, gcfg)
    _add_references(g, records, gcfg.get("references", {}))
    duplicates = _add_duplicates(g, records, gcfg.get("duplicates", {}))

    stats: GraphStats = {
        "documents": sum(1 for n in g.nodes if n.kind == "document"),
        "organizations": sum(1 for n in g.nodes if n.kind == "organization"),
        "accounts": sum(1 for n in g.nodes if n.kind == "account"),
        "edges": len(g.edges),
        "edges_by_type": dict(sorted(collections.Counter(e.type for e in g.edges).items())),
        "duplicate_groups": duplicates,
        "unverified_ids_ignored": skipped_unverified,
    }
    return g, stats


def _add_orgs(g: Graph, doc_node: str, vfields: dict[str, list[Field]]) -> list[str]:
    """Organisation nodes from verified GSTINs and PANs, one per PAN."""
    gstins = [m["normalized_value"] for m in vfields.get("gstin", []) if m.get("normalized_value")]
    pans = [m["normalized_value"] for m in vfields.get("pan", []) if m.get("normalized_value")]
    org_ids: list[str] = []
    seen_keys: set[str] = set()
    for source in (gstins, pans):
        for value in sorted(source):
            key = org_key(value)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            node = f"org:{key}"
            attrs: dict[str, Any] = {"pan": key}
            if len(value) == GSTIN_LENGTH:
                attrs["gstins"] = [value]
            g.add_node(node, "organization", **attrs)
            g.add_edge(doc_node, node, "mentions_org")
            org_ids.append(node)
    return org_ids


def _add_roles(
    g: Graph, doc_node: str, vfields: dict[str, list[Field]], afields: dict[str, list[Field]], gcfg: dict[str, Any]
) -> None:
    """Role edges (issued_by, billed_to, ...), at most one role per verified identity."""
    roles = gcfg.get("roles", {})
    id_items = collapse_identities(vfields.get("gstin", []) + vfields.get("pan", []))
    wanted = {etype: afields.get(fname, []) for etype, fname in sorted(roles.items()) if afields.get(fname)}
    if not (wanted and id_items):
        return
    limit = int(gcfg.get("role_max_distance", 400))
    margin = int(gcfg.get("role_min_margin", 0))
    for edge_type, (dist, role_item, id_item) in sorted(match_roles(wanted, id_items, limit, margin).items()):
        node = f"org:{org_key(id_item.get('normalized_value', ''))}"
        name = role_item.get("normalized_value", "")
        g.add_node(node, "organization", names=[name])
        g.add_edge(doc_node, node, edge_type, name=name, distance=dist)


def _add_accounts(
    g: Graph,
    doc_node: str,
    vfields: dict[str, list[Field]],
    afields: dict[str, list[Field]],
    acct_cfg: dict[str, Any],
    org_ids: list[str],
) -> None:
    """Account nodes, only when a verified IFSC qualifies the number."""
    for m in vfields.get("account_number", []) + afields.get("account_number", []):
        number = m.get("normalized_value")
        if not number:
            continue
        ifscs = [x.get("normalized_value") for x in vfields.get(acct_cfg.get("qualifier", "ifsc"), [])]
        ifsc = sorted([i for i in ifscs if i])[0] if any(ifscs) else None
        # Without a verified IFSC the same number at two banks would merge,
        # so an unqualified account is not an entity in v0.
        if not ifsc:
            continue
        node = f"acct:{ifsc}:{number}"
        g.add_node(node, "account", number=number, ifsc=ifsc)
        g.add_edge(doc_node, node, "pays_to")
        for org in org_ids:
            g.add_edge(node, org, "held_by")


def _add_document(g: Graph, rec: dict[str, Any], gcfg: dict[str, Any]) -> int:
    """One document, its verified organisations and their roles, and its accounts.

    Returns how many unverified identifiers the document carried. They are
    counted so the report can say what was refused, and never used.
    """
    vfields = _validated(rec)
    afields = _all_fields(rec)
    doc_node = f"doc:{rec.get('doc_id')}"
    g.add_node(
        doc_node,
        "document",
        filename=rec.get("filename", ""),
        label=_label(rec),
        page_count=(rec.get("diagnostics") or {}).get("page_count", 0),
        method=(rec.get("text") or {}).get("method", ""),
    )
    org_ids = _add_orgs(g, doc_node, vfields)
    _add_roles(g, doc_node, vfields, afields, gcfg)
    _add_accounts(g, doc_node, vfields, afields, gcfg.get("identity", {}).get("account", {}), org_ids)
    # Count what we deliberately refused to use.
    return sum(len(afields.get(name, [])) - len(vfields.get(name, [])) for name in ("gstin", "pan"))


def _add_references(g: Graph, records: list[dict[str, Any]], ref_cfg: dict[str, Any]) -> None:
    """Document to document edges: a document quoting a value another one owns."""
    ref_field = ref_cfg.get("field", "invoice_number")
    owners = set(ref_cfg.get("owner_classes") or [])
    edge_name = ref_cfg.get("edge", "references")

    owner_of: dict[str, list[str]] = collections.defaultdict(list)
    mentions: dict[str, list[str]] = collections.defaultdict(list)
    for rec in records:
        doc_id = rec.get("doc_id")
        if not doc_id:
            continue
        cls = _label(rec)
        for m in _all_fields(rec).get(ref_field, []):
            val = m.get("normalized_value")
            if not val:
                continue
            (owner_of if cls in owners else mentions)[val].append(f"doc:{doc_id}")

    for val, citers in sorted(mentions.items()):
        for target in sorted(owner_of.get(val, [])):
            for citer in sorted(citers):
                if citer != target:
                    g.add_edge(citer, target, edge_name, key=val)


def _duplicate_key(rec: dict[str, Any], dup_cfg: dict[str, Any]) -> tuple[str, ...] | None:
    """The values a duplicate must share, or None when the record lacks one."""
    fields = _all_fields(rec)
    same_class = dup_cfg.get("same_class", True)
    # A receipt quoting an invoice number is a reference, not a duplicate.
    key = [_label(rec)] if same_class else []
    for name in dup_cfg.get("match_on") or []:
        vals = sorted(m["normalized_value"] for m in fields.get(name, []) if m.get("normalized_value"))
        if not vals:
            return None
        key.append(vals[0])
    return tuple(key) if len(key) > (1 if same_class else 0) else None


def _add_duplicates(g: Graph, records: list[dict[str, Any]], dup_cfg: dict[str, Any]) -> list[DuplicateGroup]:
    """Same class documents that agree on every match_on field.

    Returns the groups, and links every pair inside each one.
    """
    buckets: dict[tuple[str, ...], list[str]] = collections.defaultdict(list)
    for rec in records:
        doc_id = rec.get("doc_id")
        if not doc_id:
            continue
        key = _duplicate_key(rec, dup_cfg)
        if key is not None:
            buckets[key].append(f"doc:{doc_id}")

    duplicates: list[DuplicateGroup] = []
    for key, docs in sorted(buckets.items()):
        if len(docs) > 1:
            duplicates.append({"key": list(key), "documents": sorted(docs)})
            for i, first in enumerate(sorted(docs)):
                for second in sorted(docs)[i + 1 :]:
                    g.add_edge(first, second, "duplicate_of", key=" | ".join(key))
    return duplicates


# --------------------------------------------------------------------------
# queries the graph exists to answer
# --------------------------------------------------------------------------


def find_chains(g: Graph) -> list[Chain]:
    """Documents that cite another document. Invoice to receipt, PO to invoice.

    Args:
        g: The graph.

    Returns:
        One entry per references edge, sorted by (from, to).
    """
    out: list[Chain] = []
    by_id = {n.id: n for n in g.nodes}
    for e in g.edges:
        if e.type != "references":
            continue
        src, dst = by_id.get(e.src), by_id.get(e.dst)
        if src and dst:
            out.append(
                {
                    "from": src.attrs.get("filename", e.src),
                    "from_class": src.attrs.get("label", ""),
                    "to": dst.attrs.get("filename", e.dst),
                    "to_class": dst.attrs.get("label", ""),
                    "key": e.attrs.get("key", ""),
                }
            )
    return sorted(out, key=lambda d: (d["from"], d["to"]))


def org_exposure(g: Graph) -> list[OrgExposure]:
    """How many documents each verified organisation appears in.

    Args:
        g: The graph.

    Returns:
        One entry per organisation, most documents first.
    """
    by_id = {n.id: n for n in g.nodes}
    doc_edges = ("mentions_org", "issued_by", "billed_to")
    out: list[OrgExposure] = []
    for node in g.nodes:
        if node.kind != "organization":
            continue
        docs = sorted({e.src for e in g.edges if e.dst == node.id and e.type in doc_edges})
        out.append(
            {
                "org": node.id,
                "pan": node.attrs.get("pan", ""),
                "gstins": node.attrs.get("gstins", []),
                "names": node.attrs.get("names", []),
                "documents": len(docs),
                "files": [by_id[d].attrs.get("filename", d) for d in docs if d in by_id],
            }
        )
    return sorted(out, key=lambda d: (-d["documents"], d["org"]))


def orphan_references(g: Graph, records: list[dict[str, Any]], gcfg: dict[str, Any]) -> list[Orphan]:
    """Documents citing a number that no document in the corpus owns.

    Args:
        g: The graph (unused; kept for a uniform query signature).
        records: Results JSONL rows.
        gcfg: The graph.yaml config.

    Returns:
        One entry per unresolved citation, sorted by (document, key).
    """
    ref_cfg = gcfg.get("references", {})
    ref_field = ref_cfg.get("field", "invoice_number")
    owners = set(ref_cfg.get("owner_classes") or [])
    owned = set()
    for rec in records:
        if _label(rec) in owners:
            for m in _all_fields(rec).get(ref_field, []):
                if m.get("normalized_value"):
                    owned.add(m["normalized_value"])

    out: list[Orphan] = []
    for rec in records:
        cls = _label(rec)
        if cls in owners:
            continue
        for m in _all_fields(rec).get(ref_field, []):
            val = m.get("normalized_value")
            if val and val not in owned:
                out.append({"document": rec.get("filename", ""), "class": cls, "key": val})
    return sorted(out, key=lambda d: (d["document"], d["key"]))


def build(
    input_path: str | Path,
    output_path: str | Path,
    gcfg: dict[str, Any],
    dot_path: str | Path | None = None,
    report_path: str | Path | None = None,
) -> GraphBuild:
    """Build the graph over a results file and write it out.

    Args:
        input_path: Results JSONL.
        output_path: Graph JSON to write.
        gcfg: The graph.yaml config.
        dot_path: Also write Graphviz source here.
        report_path: Also write a markdown report here.

    Returns:
        The graph, its stats and the query results.
    """
    records = read_jsonl(input_path)
    g, stats = build_graph(records, gcfg)
    chains = find_chains(g)
    exposure = org_exposure(g)
    orphans = orphan_references(g, records, gcfg)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {**g.to_dict(), "stats": stats}
    atomic_write_text(out, json.dumps(payload, indent=2, sort_keys=True))
    if dot_path:
        atomic_write_text(dot_path, to_dot(g))
    if report_path:
        atomic_write_text(report_path, render_report(stats, chains, exposure, orphans, str(input_path)))

    log.info("graph built", extra={k: v for k, v in stats.items() if k not in ("duplicate_groups", "edges_by_type")})
    return {"graph": g, "stats": stats, "chains": chains, "exposure": exposure, "orphans": orphans}
