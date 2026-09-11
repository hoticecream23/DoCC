"""Knowledge graph v0, built over the JSONL.

Deliberately narrow. Only fields whose format proves them may create or merge
an entity, so every node here is one we can defend. Fuzzy name matching is
where graphs quietly fill with wrong edges, so v0 does not do it at all.

This never touches the document schema. It is a separate pass over records
that already exist.
"""

from __future__ import annotations

import collections
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .logging_setup import get_logger
from .schema import read_jsonl

log = get_logger(__name__)

# Chars 3 to 12 of a GSTIN are the holder PAN.
GSTIN_PAN_SLICE = slice(2, 12)


@dataclass(frozen=True)
class Node:
    id: str
    kind: str
    attrs: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    type: str
    attrs: dict = field(default_factory=dict)


class Graph:
    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}
        self._edges: dict[tuple, Edge] = {}

    def add_node(self, node_id: str, kind: str, **attrs) -> str:
        cur = self._nodes.get(node_id)
        if cur is None:
            self._nodes[node_id] = Node(node_id, kind, dict(attrs))
            return node_id
        # Merge attributes. Sets union, scalars keep the first value seen.
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

    def add_edge(self, src: str, dst: str, etype: str, **attrs) -> None:
        key = (src, dst, etype)
        if key in self._edges:
            return
        self._edges[key] = Edge(src, dst, etype, dict(attrs))

    @property
    def nodes(self) -> list[Node]:
        return [self._nodes[k] for k in sorted(self._nodes)]

    @property
    def edges(self) -> list[Edge]:
        return [self._edges[k] for k in sorted(self._edges)]

    def neighbours(self, node_id: str, etype: str | None = None) -> list[str]:
        return sorted(
            e.dst for e in self.edges
            if e.src == node_id and (etype is None or e.type == etype)
        )

    def to_dict(self) -> dict:
        return {
            "nodes": [{"id": n.id, "kind": n.kind, **n.attrs} for n in self.nodes],
            "edges": [
                {"src": e.src, "dst": e.dst, "type": e.type, **e.attrs} for e in self.edges
            ],
        }


# --------------------------------------------------------------------------
# building
# --------------------------------------------------------------------------


def _validated(record: dict) -> dict[str, list[dict]]:
    """Only fields the pipeline could actually prove. This is the whole trick."""
    out: dict[str, list[dict]] = collections.defaultdict(list)
    for m in record.get("metadata") or []:
        if m.get("validated"):
            out[m["field"]].append(m)
    return out


def _all_fields(record: dict) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = collections.defaultdict(list)
    for m in record.get("metadata") or []:
        out[m["field"]].append(m)
    return out


def org_key(value: str) -> str:
    """One organisation, one key. A GSTIN reduces to the PAN it carries."""
    return value[GSTIN_PAN_SLICE] if len(value) == 15 else value


def collapse_identities(id_items: list[dict]) -> list[dict]:
    """One entry per organisation.

    A document usually prints both the GSTIN and the PAN of the same company.
    Treating those as two identities lets one party claim two roles, which is
    how the vendor ends up holding the buyer name.
    """
    best: dict[str, dict] = {}
    for item in id_items:
        value = item.get("normalized_value") or ""
        if not value:
            continue
        key = org_key(value)
        cur = best.get(key)
        if cur is None or int(item.get("char_start", 0)) < int(cur.get("char_start", 0)):
            best[key] = item
    return [best[k] for k in sorted(best)]


def match_roles(role_items: dict[str, list[dict]], id_items: list[dict],
                limit: int, margin: int = 0) -> dict:
    """Assign roles to verified identities, at most one role per identity.

    Documents name several parties but often carry only one verified number.
    Letting every role attach to that number is how a vendor node ends up
    holding the buyer name, so this is a greedy closest first matching.

    Proximity alone is weak evidence. On a purchase order "Ship To" can sit
    closer to the seller GSTIN than "Vendor Name" does, purely by layout. So
    a winner must beat the runner up for the same identity by `margin`
    characters. If it does not, the document still records that the party was
    mentioned but claims no role. Abstaining beats a confident wrong edge.
    """
    pairs = []
    for role, items in role_items.items():
        for r in items:
            for i in id_items:
                dist = abs(int(i.get("char_start", 0)) - int(r.get("char_start", 0)))
                if dist <= limit:
                    pairs.append((dist, role, r, i))
    # Sort by distance, then by role and offset so ties resolve the same way twice.
    pairs.sort(key=lambda t: (t[0], t[1], int(t[2].get("char_start", 0))))

    out: dict = {}
    claimed_ids: set = set()
    claimed_roles: set = set()
    for idx, (dist, role, r, i) in enumerate(pairs):
        ident = org_key(i.get("normalized_value", ""))
        if ident in claimed_ids or role in claimed_roles:
            continue
        # Nearest competing claim on the same identity from a different role.
        rival = next(
            (d for d, ro, _, ii in pairs[idx + 1:]
             if ro != role and org_key(ii.get("normalized_value", "")) == ident),
            None,
        )
        if rival is not None and (rival - dist) < margin:
            # Too close to call. Record nothing rather than guess.
            claimed_ids.add(ident)
            continue
        claimed_ids.add(ident)
        claimed_roles.add(role)
        out[role] = (dist, r, i)
    return out


def build_graph(records: list[dict], gcfg: dict) -> tuple[Graph, dict]:
    g = Graph()
    ident = gcfg.get("identity", {})
    acct_cfg = ident.get("account", {})
    roles = gcfg.get("roles", {})
    role_limit = int(gcfg.get("role_max_distance", 400))
    role_margin = int(gcfg.get("role_min_margin", 0))
    ref_cfg = gcfg.get("references", {})
    dup_cfg = gcfg.get("duplicates", {})

    skipped_unverified = 0
    doc_orgs: dict[str, list[str]] = {}

    for rec in records:
        doc_id = rec.get("doc_id")
        if not doc_id:
            continue
        cls = (rec.get("classification") or {}).get("label", "unknown")
        vfields = _validated(rec)
        afields = _all_fields(rec)

        doc_node = f"doc:{doc_id}"
        g.add_node(
            doc_node, "document",
            filename=rec.get("filename", ""),
            label=cls,
            page_count=(rec.get("diagnostics") or {}).get("page_count", 0),
            method=(rec.get("text") or {}).get("method", ""),
        )

        # ---- organisations, verified identity only ----
        gstins = [m["normalized_value"] for m in vfields.get("gstin", []) if m.get("normalized_value")]
        pans = [m["normalized_value"] for m in vfields.get("pan", []) if m.get("normalized_value")]
        org_ids: list[str] = []
        seen_keys: set[str] = set()
        for source in (gstins, pans):
            for value in sorted(source):
                key = value[GSTIN_PAN_SLICE] if len(value) == 15 else value
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                node = f"org:{key}"
                attrs: dict[str, Any] = {"pan": key}
                if len(value) == 15:
                    attrs["gstins"] = [value]
                g.add_node(node, "organization", **attrs)
                g.add_edge(doc_node, node, "mentions_org")
                org_ids.append(node)
        doc_orgs[doc_node] = org_ids

        # Count what we deliberately refused to use.
        for name in ("gstin", "pan"):
            skipped_unverified += len(afields.get(name, [])) - len(vfields.get(name, []))

        # ---- roles, one per verified identity ----
        id_items = collapse_identities(vfields.get("gstin", []) + vfields.get("pan", []))
        wanted = {etype: afields.get(fname, [])
                  for etype, fname in sorted(roles.items()) if afields.get(fname)}
        if wanted and id_items:
            for edge_type, (dist, role_item, id_item) in sorted(
                match_roles(wanted, id_items, role_limit, role_margin).items()
            ):
                value = id_item.get("normalized_value", "")
                key = value[GSTIN_PAN_SLICE] if len(value) == 15 else value
                g.add_node(f"org:{key}", "organization",
                           names=[role_item.get("normalized_value", "")])
                g.add_edge(doc_node, f"org:{key}", edge_type,
                           name=role_item.get("normalized_value", ""), distance=dist)

        # ---- accounts ----
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

    # ---- document to document references ----
    ref_field = ref_cfg.get("field", "invoice_number")
    owners = set(ref_cfg.get("owner_classes") or [])
    edge_name = ref_cfg.get("edge", "references")

    owner_of: dict[str, list[str]] = collections.defaultdict(list)
    mentions: dict[str, list[str]] = collections.defaultdict(list)
    for rec in records:
        doc_id = rec.get("doc_id")
        if not doc_id:
            continue
        cls = (rec.get("classification") or {}).get("label", "unknown")
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

    # ---- duplicate candidates ----
    match_on = dup_cfg.get("match_on") or []
    buckets: dict[tuple, list[str]] = collections.defaultdict(list)
    for rec in records:
        doc_id = rec.get("doc_id")
        if not doc_id:
            continue
        fields = _all_fields(rec)
        cls_here = (rec.get("classification") or {}).get("label", "unknown")
        # A receipt quoting an invoice number is a reference, not a duplicate.
        key = [cls_here] if dup_cfg.get("same_class", True) else []
        for name in match_on:
            vals = sorted(
                m.get("normalized_value") for m in fields.get(name, [])
                if m.get("normalized_value")
            )
            if not vals:
                key = []
                break
            key.append(vals[0])
        if len(key) > (1 if dup_cfg.get("same_class", True) else 0):
            buckets[tuple(key)].append(f"doc:{doc_id}")

    duplicates = []
    for key, docs in sorted(buckets.items()):
        if len(docs) > 1:
            duplicates.append({"key": list(key), "documents": sorted(docs)})
            for i, a in enumerate(sorted(docs)):
                for b in sorted(docs)[i + 1:]:
                    g.add_edge(a, b, "duplicate_of", key=" | ".join(key))

    stats = {
        "documents": sum(1 for n in g.nodes if n.kind == "document"),
        "organizations": sum(1 for n in g.nodes if n.kind == "organization"),
        "accounts": sum(1 for n in g.nodes if n.kind == "account"),
        "edges": len(g.edges),
        "edges_by_type": dict(sorted(collections.Counter(e.type for e in g.edges).items())),
        "duplicate_groups": duplicates,
        "unverified_ids_ignored": skipped_unverified,
    }
    return g, stats


# --------------------------------------------------------------------------
# queries the graph exists to answer
# --------------------------------------------------------------------------


def find_chains(g: Graph) -> list[dict]:
    """Documents that cite another document. Invoice to receipt, PO to invoice."""
    out = []
    by_id = {n.id: n for n in g.nodes}
    for e in g.edges:
        if e.type != "references":
            continue
        src, dst = by_id.get(e.src), by_id.get(e.dst)
        if src and dst:
            out.append({
                "from": src.attrs.get("filename", e.src),
                "from_class": src.attrs.get("label", ""),
                "to": dst.attrs.get("filename", e.dst),
                "to_class": dst.attrs.get("label", ""),
                "key": e.attrs.get("key", ""),
            })
    return sorted(out, key=lambda d: (d["from"], d["to"]))


def org_exposure(g: Graph) -> list[dict]:
    """How many documents each verified organisation appears in."""
    by_id = {n.id: n for n in g.nodes}
    counts: collections.Counter = collections.Counter()
    for e in g.edges:
        if e.type in ("mentions_org", "issued_by", "billed_to"):
            counts[e.dst] += 1 if e.type == "mentions_org" else 0
    out = []
    for node in g.nodes:
        if node.kind != "organization":
            continue
        docs = sorted({e.src for e in g.edges
                       if e.dst == node.id and e.type in ("mentions_org", "issued_by", "billed_to")})
        out.append({
            "org": node.id,
            "pan": node.attrs.get("pan", ""),
            "gstins": node.attrs.get("gstins", []),
            "names": node.attrs.get("names", []),
            "documents": len(docs),
            "files": [by_id[d].attrs.get("filename", d) for d in docs if d in by_id],
        })
    return sorted(out, key=lambda d: (-d["documents"], d["org"]))


def orphan_references(g: Graph, records: list[dict], gcfg: dict) -> list[dict]:
    """Documents citing a number that no document in the corpus owns."""
    ref_cfg = gcfg.get("references", {})
    ref_field = ref_cfg.get("field", "invoice_number")
    owners = set(ref_cfg.get("owner_classes") or [])
    owned = set()
    for rec in records:
        cls = (rec.get("classification") or {}).get("label", "unknown")
        if cls in owners:
            for m in _all_fields(rec).get(ref_field, []):
                if m.get("normalized_value"):
                    owned.add(m["normalized_value"])

    out = []
    for rec in records:
        cls = (rec.get("classification") or {}).get("label", "unknown")
        if cls in owners:
            continue
        for m in _all_fields(rec).get(ref_field, []):
            val = m.get("normalized_value")
            if val and val not in owned:
                out.append({"document": rec.get("filename", ""), "class": cls, "key": val})
    return sorted(out, key=lambda d: (d["document"], d["key"]))


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------


def to_dot(g: Graph) -> str:
    """Graphviz source, so the thing can actually be looked at."""
    shapes = {"document": "box", "organization": "ellipse", "account": "diamond"}
    L = ["digraph kg {", '  rankdir=LR;', '  node [fontname="Helvetica" fontsize=10];',
         '  edge [fontname="Helvetica" fontsize=8];']
    for n in g.nodes:
        if n.kind == "document":
            lab = f"{n.attrs.get('filename', n.id)}\\n{n.attrs.get('label', '')}"
        elif n.kind == "organization":
            names = n.attrs.get("names") or []
            lab = (names[0] if names else n.attrs.get("pan", n.id))
        else:
            lab = f"{n.attrs.get('ifsc', '')}\\n{n.attrs.get('number', '')}"
        L.append(f'  "{n.id}" [label="{lab}" shape={shapes.get(n.kind, "box")}];')
    for e in g.edges:
        L.append(f'  "{e.src}" -> "{e.dst}" [label="{e.type}"];')
    L.append("}")
    return "\n".join(L)


def render_report(g: Graph, stats: dict, chains, exposure, orphans, source: str) -> str:
    L: list[str] = []
    a = L.append
    a("# Knowledge graph v0")
    a("")
    a(f"Source: `{source}`")
    a("")
    a("Built only from fields whose format proves them. Names are recorded")
    a("but never used to merge entities, so every node here is defensible.")
    a("")
    a("| thing | count |")
    a("| --- | --- |")
    a(f"| documents | {stats['documents']} |")
    a(f"| organisations | {stats['organizations']} |")
    a(f"| accounts | {stats['accounts']} |")
    a(f"| edges | {stats['edges']} |")
    a("")
    a("| edge type | count |")
    a("| --- | --- |")
    for k, v in stats["edges_by_type"].items():
        a(f"| {k} | {v} |")
    a("")
    if stats["unverified_ids_ignored"]:
        a(f"Ignored {stats['unverified_ids_ignored']} unverified identifier(s) "
          "rather than risk a wrong merge.")
        a("")

    a("## Organisations")
    a("")
    a("| pan | gstins | names seen | documents |")
    a("| --- | --- | --- | --- |")
    for o in exposure:
        a(f"| {o['pan']} | {', '.join(o['gstins']) or 'none'} | "
          f"{', '.join(o['names']) or 'none'} | {o['documents']} |")
    a("")

    a("## Document chains")
    a("")
    if chains:
        a("| from | | to | via |")
        a("| --- | --- | --- | --- |")
        for c in chains:
            a(f"| {c['from']} ({c['from_class']}) | references | "
              f"{c['to']} ({c['to_class']}) | {c['key']} |")
    else:
        a("No document referenced another.")
    a("")

    a("## Duplicate candidates")
    a("")
    if stats["duplicate_groups"]:
        a("Same reference number and same total, different documents.")
        a("")
        for d in stats["duplicate_groups"]:
            a(f"- `{' | '.join(d['key'])}`: {', '.join(d['documents'])}")
    else:
        a("None found.")
    a("")

    a("## Orphan references")
    a("")
    if orphans:
        a("Documents citing a reference no document in this corpus owns.")
        a("")
        for o in orphans:
            a(f"- {o['document']} ({o['class']}) cites `{o['key']}`")
    else:
        a("None. Every cited reference resolves.")
    a("")
    return "\n".join(L)


def build(input_path, output_path, gcfg: dict, dot_path=None, report_path=None) -> dict:
    records = read_jsonl(input_path)

    g, stats = build_graph(records, gcfg)
    chains = find_chains(g)
    exposure = org_exposure(g)
    orphans = orphan_references(g, records, gcfg)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = g.to_dict()
    payload["stats"] = stats
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    if dot_path:
        Path(dot_path).write_text(to_dot(g), encoding="utf-8")
    if report_path:
        Path(report_path).write_text(
            render_report(g, stats, chains, exposure, orphans, str(input_path)),
            encoding="utf-8",
        )

    log.info("graph built", extra={k: v for k, v in stats.items()
                                  if k not in ("duplicate_groups", "edges_by_type")})
    return {"graph": g, "stats": stats, "chains": chains,
            "exposure": exposure, "orphans": orphans}
