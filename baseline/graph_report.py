"""Human-readable views of the knowledge graph: Graphviz source and a markdown report.

Kept apart from graph.py so construction and presentation change separately.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .graph import Chain, Graph, GraphStats, OrgExposure, Orphan


def _dot_label(kind: str, attrs: dict[str, Any], node_id: str) -> str:
    if kind == "document":
        return f"{attrs.get('filename', node_id)}\\n{attrs.get('label', '')}"
    if kind == "organization":
        names = attrs.get("names") or []
        return str(names[0] if names else attrs.get("pan", node_id))
    return f"{attrs.get('ifsc', '')}\\n{attrs.get('number', '')}"


def to_dot(g: Graph) -> str:
    """Graphviz source, so the thing can actually be looked at.

    Args:
        g: The graph.

    Returns:
        DOT source text.
    """
    shapes = {"document": "box", "organization": "ellipse", "account": "diamond"}
    lines = [
        "digraph kg {",
        "  rankdir=LR;",
        '  node [fontname="Helvetica" fontsize=10];',
        '  edge [fontname="Helvetica" fontsize=8];',
    ]
    for n in g.nodes:
        label = _dot_label(n.kind, n.attrs, n.id)
        lines.append(f'  "{n.id}" [label="{label}" shape={shapes.get(n.kind, "box")}];')
    for e in g.edges:
        lines.append(f'  "{e.src}" -> "{e.dst}" [label="{e.type}"];')
    lines.append("}")
    return "\n".join(lines)


def _summary_section(stats: GraphStats, source: str) -> list[str]:
    lines = [
        "# Knowledge graph v0",
        "",
        f"Source: `{source}`",
        "",
        "Built only from fields whose format proves them. Names are recorded",
        "but never used to merge entities, so every node here is defensible.",
        "",
        "| thing | count |",
        "| --- | --- |",
        f"| documents | {stats['documents']} |",
        f"| organisations | {stats['organizations']} |",
        f"| accounts | {stats['accounts']} |",
        f"| edges | {stats['edges']} |",
        "",
        "| edge type | count |",
        "| --- | --- |",
    ]
    lines += [f"| {k} | {v} |" for k, v in stats["edges_by_type"].items()]
    lines.append("")
    if stats["unverified_ids_ignored"]:
        lines.append(
            f"Ignored {stats['unverified_ids_ignored']} unverified identifier(s) " "rather than risk a wrong merge."
        )
        lines.append("")
    return lines


def _org_section(exposure: list[OrgExposure]) -> list[str]:
    lines = ["## Organisations", "", "| pan | gstins | names seen | documents |", "| --- | --- | --- | --- |"]
    for o in exposure:
        lines.append(
            f"| {o['pan']} | {', '.join(o['gstins']) or 'none'} | "
            f"{', '.join(o['names']) or 'none'} | {o['documents']} |"
        )
    lines.append("")
    return lines


def _chain_section(chains: list[Chain]) -> list[str]:
    lines = ["## Document chains", ""]
    if chains:
        lines += ["| from | | to | via |", "| --- | --- | --- | --- |"]
        for c in chains:
            lines.append(
                f"| {c['from']} ({c['from_class']}) | references | " f"{c['to']} ({c['to_class']}) | {c['key']} |"
            )
    else:
        lines.append("No document referenced another.")
    lines.append("")
    return lines


def _duplicate_section(stats: GraphStats) -> list[str]:
    lines = ["## Duplicate candidates", ""]
    if stats["duplicate_groups"]:
        lines += ["Same reference number and same total, different documents.", ""]
        for d in stats["duplicate_groups"]:
            lines.append(f"- `{' | '.join(d['key'])}`: {', '.join(d['documents'])}")
    else:
        lines.append("None found.")
    lines.append("")
    return lines


def _orphan_section(orphans: list[Orphan]) -> list[str]:
    lines = ["## Orphan references", ""]
    if orphans:
        lines += ["Documents citing a reference no document in this corpus owns.", ""]
        for o in orphans:
            lines.append(f"- {o['document']} ({o['class']}) cites `{o['key']}`")
    else:
        lines.append("None. Every cited reference resolves.")
    lines.append("")
    return lines


def render_report(
    stats: GraphStats,
    chains: list[Chain],
    exposure: list[OrgExposure],
    orphans: list[Orphan],
    source: str,
) -> str:
    """Markdown summary of a built graph.

    Args:
        stats: Counts from build_graph.
        chains: Output of find_chains.
        exposure: Output of org_exposure.
        orphans: Output of orphan_references.
        source: Results file the graph was built from, for the header.

    Returns:
        The report as markdown.
    """
    lines = _summary_section(stats, source)
    lines += _org_section(exposure)
    lines += _chain_section(chains)
    lines += _duplicate_section(stats)
    lines += _orphan_section(orphans)
    return "\n".join(lines)
