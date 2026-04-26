"""Export an audit evidence graph as JSON + Graphviz DOT."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

from .schemas import GraphEdge, GraphNode


def build(evidence, claims, findings, risks, mitigations, kpis, gates) -> dict:
    nodes: List[GraphNode] = []
    edges: List[GraphEdge] = []

    def add_node(kind, node_id, label, attrs=None):
        nodes.append(GraphNode(node_id=node_id, kind=kind, label=label,
                               attributes=attrs or {}))

    for e in evidence:
        add_node("file", e.item_id, e.path,
                 {"evidence_class": e.evidence_class, "type": e.type})
    for c in claims:
        add_node("claim", c.claim_id, c.classification + ": " + (c.text[:60] or ""),
                 {"source": c.source, "category": c.category})
    for f in findings:
        add_node("finding", f.finding_id, f.summary,
                 {"domain": f.domain, "severity": f.severity})
        # finding -> evidence edges
        for ep in f.evidence_paths:
            for ev in evidence:
                if ev.path == ep:
                    edges.append(GraphEdge(src=f.finding_id, dst=ev.item_id,
                                           relation="supports"))
                    break
    for r in risks:
        add_node("risk", r.risk_id, r.description[:80],
                 {"impact": r.impact, "likelihood": r.likelihood})
        edges.append(GraphEdge(src=r.risk_id, dst=r.finding_id, relation="requires"))
        for mid in r.mitigation_ids:
            edges.append(GraphEdge(src=mid, dst=r.risk_id, relation="mitigates"))
    for m in mitigations:
        add_node("mitigation", m.mitigation_id, m.description[:80],
                 {"effort": m.effort})
    for k in kpis:
        add_node("kpi", k.kpi_id, k.label,
                 {"classification": k.classification, "target": k.target})
    for g in gates:
        add_node("gate", g.gate_id, g.label,
                 {"tier": g.tier, "status": g.status})

    return {
        "nodes": [asdict(n) for n in nodes],
        "edges": [asdict(e) for e in edges],
    }


def write_outputs(graph: dict, out_dir: Path) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "audit_graph.json"
    dot_path = out_dir / "audit_graph.dot"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2, ensure_ascii=False)

    # Minimal dot: only node + edge lines, colored by kind.
    kind_color = {
        "file": "lightgray", "claim": "khaki", "finding": "lightpink",
        "risk": "salmon", "mitigation": "palegreen", "kpi": "lightblue",
        "gate": "gold",
    }
    with dot_path.open("w", encoding="utf-8") as f:
        f.write("digraph audit {\n")
        f.write('  rankdir=LR;\n  node [shape=box, style=filled];\n')
        for n in graph["nodes"]:
            color = kind_color.get(n["kind"], "white")
            label = n["label"].replace("\"", "'").replace("\n", " ")[:50]
            f.write(f'  "{n["node_id"]}" [label="{label}", fillcolor={color}];\n')
        for e in graph["edges"]:
            f.write(f'  "{e["src"]}" -> "{e["dst"]}" [label="{e["relation"]}"];\n')
        f.write("}\n")

    return {"audit_graph_json": str(json_path), "audit_graph_dot": str(dot_path)}
