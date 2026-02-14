"""
Graph Core — Main GraphOfThoughts Class
=======================================
The complete Graph-of-Thoughts reasoning engine composed from mixins.

Standing on Giants:
- Graph of Thoughts (Besta et al., 2024): "Solving elaborate problems with LLMs"
- Tree of Thoughts (Yao et al., 2023): Deliberate problem solving
- Chain of Thought (Wei et al., 2022): Step-by-step reasoning
- BIZRA ARTE Engine: Symbolic-neural bridge

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    THOUGHT GRAPH                            │
    │                                                             │
    │         [Hypothesis A]──────┐                               │
    │              │              │                               │
    │              ▼              ▼                               │
    │         [Evidence 1]   [Evidence 2]                         │
    │              │              │                               │
    │              └──────┬───────┘                               │
    │                     ▼                                       │
    │              [Synthesis]────────► [Conclusion]              │
    │                     │                   │                   │
    │                     ▼                   ▼                   │
    │              [Refinement]         [Validation]              │
    │                     │                   │                   │
    │                     └─────────┬─────────┘                   │
    │                               ▼                             │
    │                      [Final Answer]                         │
    │                         (SNR ≥ 0.95)                        │
    └─────────────────────────────────────────────────────────────┘

Reasoning Operations:
- GENERATE: Create new thought nodes
- AGGREGATE: Merge multiple thoughts into synthesis
- REFINE: Improve existing thoughts iteratively
- VALIDATE: Check thoughts against Ihsan constraints
- PRUNE: Remove low-SNR branches
- BACKTRACK: Return to promising unexplored paths
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import Any, Optional

from core.integration.constants import UNIFIED_IHSAN_THRESHOLD, UNIFIED_SNR_THRESHOLD
from core.proof_engine.canonical import hex_digest

from .graph_operations import GraphOperationsMixin
from .graph_reasoning import GraphReasoningMixin
from .graph_search import GraphSearchMixin
from .graph_types import (
    EdgeType,
    ReasoningStrategy,
    ThoughtEdge,
    ThoughtNode,
    ThoughtType,
)

logger = logging.getLogger(__name__)


class GraphOfThoughts(GraphOperationsMixin, GraphSearchMixin, GraphReasoningMixin):  # type: ignore[misc]
    """
    Graph-of-Thoughts Reasoning Engine.

    Implements networked reasoning where multiple thought branches
    can be explored, merged, refined, and validated in parallel.

    Key operations:
    1. GENERATE: Create new thought nodes from prompts/context
    2. AGGREGATE: Merge multiple thoughts into synthesis
    3. REFINE: Iteratively improve thought quality
    4. VALIDATE: Check against Ihsan constraints
    5. SCORE: Compute SNR for ranking
    6. PRUNE: Remove low-quality branches

    Usage:
        graph = GraphOfThoughts()
        root = graph.add_thought("What is the solution?", ThoughtType.QUESTION)

        # Generate hypotheses
        h1 = graph.generate("Hypothesis A", ThoughtType.HYPOTHESIS, parent=root)
        h2 = graph.generate("Hypothesis B", ThoughtType.HYPOTHESIS, parent=root)

        # Add evidence
        e1 = graph.generate("Evidence for A", ThoughtType.EVIDENCE, parent=h1)

        # Synthesize
        synth = graph.aggregate([h1, e1], "Combined conclusion")

        # Get best path
        best = graph.find_best_path(root.id)
    """

    def __init__(
        self,
        strategy: ReasoningStrategy = ReasoningStrategy.BEST_FIRST,
        max_depth: int = 10,
        beam_width: int = 5,
        snr_threshold: float = UNIFIED_SNR_THRESHOLD,
        ihsan_threshold: float = UNIFIED_IHSAN_THRESHOLD,
        inference_gateway: object | None = None,
    ):
        self.strategy = strategy
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.snr_threshold = snr_threshold
        self.ihsan_threshold = ihsan_threshold
        self._inference_gateway = inference_gateway

        # Graph structure
        self.nodes: dict[str, ThoughtNode] = {}
        self.edges: list[ThoughtEdge] = []
        self.adjacency: dict[str, list[str]] = defaultdict(list)  # node -> children
        self.reverse_adj: dict[str, list[str]] = defaultdict(list)  # node -> parents

        # Root nodes (entry points)
        self.roots: list[str] = []

        # Statistics
        self.stats = {
            "nodes_created": 0,
            "nodes_pruned": 0,
            "edges_created": 0,
            "refinements": 0,
            "aggregations": 0,
        }

    @property
    def thoughts(self) -> dict[str, ThoughtNode]:
        """Property alias for nodes dict."""
        return self.nodes

    def to_dict(self) -> dict:
        """Serialize graph for inspection."""
        result: dict[str, Any] = {
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges],
            "roots": self.roots,
            "stats": self.stats,
            "config": {
                "strategy": self.strategy.value,
                "max_depth": self.max_depth,
                "snr_threshold": self.snr_threshold,
                "ihsan_threshold": self.ihsan_threshold,
            },
            "graph_hash": self.compute_graph_hash(),
        }
        return result

    def compute_graph_hash(self) -> str:
        """Compute BLAKE3 hash of the canonical graph representation.

        Standing on: Merkle (1979) — content-addressed integrity
        for graph artifacts. Makes the entire reasoning graph
        independently verifiable as a first-class artifact.
        SEC-001: Uses BLAKE3 for Python-Rust interop parity.
        """
        # Canonical representation: sorted nodes by ID, sorted edges
        canonical_nodes = sorted(
            [
                {
                    "id": n.id,
                    "content_hash": n.content_hash,
                    "type": n.thought_type.value,
                    "snr": round(n.snr_score, 6),
                    "ihsan": round(n.ihsan_score, 6),
                    "depth": n.depth,
                }
                for n in self.nodes.values()
            ],
            key=lambda x: x["id"],
        )
        canonical_edges = sorted(
            [
                {
                    "source": e.source_id,
                    "target": e.target_id,
                    "type": e.edge_type.value,
                    "weight": round(e.weight, 6),
                }
                for e in self.edges
            ],
            key=lambda x: (x["source"], x["target"]),
        )
        canonical = json.dumps(
            {
                "nodes": canonical_nodes,
                "edges": canonical_edges,
                "roots": sorted(self.roots),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hex_digest(canonical)  # SEC-001: BLAKE3 for Rust interop

    def to_artifact(self, build_id: str = "", policy_version: str = "1.0.0") -> dict:
        """Produce a schema-validated artifact matching reasoning_graph.schema.json.

        Standing on: Besta (GoT, 2024) — graph artifacts are first-class,
        Merkle (1979) — content-addressed integrity for every node and the graph.

        Returns a dict conforming to the BIZRA reasoning_graph JSON schema,
        suitable for storage alongside receipts in the Evidence Ledger.
        """
        graph_hash = self.compute_graph_hash()

        # Build schema-compliant nodes
        artifact_nodes = []
        for n in self.nodes.values():
            node_dict: dict[str, Any] = {
                "id": n.id,
                "content": n.content[:500] if len(n.content) > 500 else n.content,
                "type": n.thought_type.value,
                "content_hash": n.content_hash,
                "confidence": round(n.confidence, 6),
                "snr": round(n.snr_score, 6),
                "ihsan": round(n.ihsan_score, 6),
                "depth": n.depth,
            }
            # Optional fields
            claim_tag = n.metadata.get("claim_tag")
            if claim_tag:
                node_dict["claim_tag"] = claim_tag
            evidence_hash = n.metadata.get("evidence_hash")
            if evidence_hash:
                node_dict["evidence_hash"] = evidence_hash
            failure_modes = n.metadata.get("failure_modes")
            if failure_modes:
                node_dict["failure_modes"] = failure_modes
            artifact_nodes.append(node_dict)

        # Build schema-compliant edges
        artifact_edges = []
        for e in self.edges:
            edge_dict: dict[str, Any] = {
                "source": e.source_id,
                "target": e.target_id,
                "type": e.edge_type.value,
                "weight": round(e.weight, 6),
            }
            artifact_edges.append(edge_dict)

        artifact: dict[str, Any] = {
            "nodes": artifact_nodes,
            "edges": artifact_edges,
            "roots": list(self.roots),
            "graph_hash": graph_hash,
            "stats": dict(self.stats),
            "config": {
                "strategy": self.strategy.value,
                "max_depth": self.max_depth,
                "snr_threshold": self.snr_threshold,
                "ihsan_threshold": self.ihsan_threshold,
            },
        }
        if build_id:
            artifact["build_id"] = build_id
        if policy_version:
            artifact["policy_version"] = policy_version

        return artifact

    @classmethod
    def from_artifact(cls, artifact: dict[str, Any]) -> "GraphOfThoughts":
        """Reconstruct a GraphOfThoughts from a previously exported artifact.

        Standing on: Besta (GoT, 2024) — graph artifacts as first-class,
        recoverable objects for audit replay and crash recovery.

        Args:
            artifact: dict produced by to_artifact() or to_dict().

        Returns:
            Reconstructed GraphOfThoughts with nodes, edges, and config.

        Raises:
            ValueError: If artifact is malformed or hash integrity fails.
        """
        config = artifact.get("config", {})
        strategy_str = config.get("strategy", ReasoningStrategy.BEST_FIRST.value)
        try:
            strategy = ReasoningStrategy(strategy_str)
        except ValueError:
            strategy = ReasoningStrategy.BEST_FIRST

        graph = cls(
            strategy=strategy,
            max_depth=config.get("max_depth", 10),
            snr_threshold=config.get("snr_threshold", 0.85),
            ihsan_threshold=config.get("ihsan_threshold", 0.95),
        )

        # Restore nodes
        for node_data in artifact.get("nodes", []):
            try:
                thought_type = ThoughtType(node_data.get("type", "reasoning"))
            except ValueError:
                thought_type = ThoughtType.REASONING

            node = ThoughtNode(
                id=node_data["id"],
                content=node_data.get("content", ""),
                thought_type=thought_type,
                confidence=node_data.get("confidence", 0.5),
                snr_score=node_data.get("snr", 0.5),
                depth=node_data.get("depth", 0),
            )
            # Restore optional metadata
            if "claim_tag" in node_data:
                node.metadata["claim_tag"] = node_data["claim_tag"]
            if "evidence_hash" in node_data:
                node.metadata["evidence_hash"] = node_data["evidence_hash"]
            if "failure_modes" in node_data:
                node.metadata["failure_modes"] = node_data["failure_modes"]

            graph.nodes[node.id] = node

        # Restore edges
        for edge_data in artifact.get("edges", []):
            try:
                edge_type = EdgeType(edge_data.get("type", "derives"))
            except ValueError:
                edge_type = EdgeType.DERIVES

            edge = ThoughtEdge(
                source_id=edge_data["source"],
                target_id=edge_data["target"],
                edge_type=edge_type,
                weight=edge_data.get("weight", 1.0),
            )
            graph.edges.append(edge)
            graph.adjacency[edge.source_id].append(edge.target_id)
            graph.reverse_adj[edge.target_id].append(edge.source_id)

        # Restore roots
        graph.roots = list(artifact.get("roots", []))

        # Restore stats
        if "stats" in artifact:
            graph.stats.update(artifact["stats"])

        # Verify integrity if stored hash is present
        stored_hash = artifact.get("graph_hash")
        if stored_hash:
            computed_hash = graph.compute_graph_hash()
            if computed_hash != stored_hash:
                raise ValueError(
                    f"Graph integrity check failed: "
                    f"stored={stored_hash[:16]}... computed={computed_hash[:16]}..."
                )

        return graph

    def sign_graph(self, private_key_hex: str) -> Optional[str]:
        """Sign the graph hash with Ed25519.

        Standing on: Bernstein (2011) — Ed25519 signatures
        for tamper-evident graph artifacts.

        Returns the hex-encoded signature, or None if signing fails.
        """
        try:
            from core.pci.crypto import sign_message

            graph_hash = self.compute_graph_hash()
            signature = sign_message(graph_hash, private_key_hex)
            return signature
        except Exception as e:
            logger.warning(f"Graph signing failed: {e}")
            return None

    def visualize_ascii(self) -> str:
        """Generate ASCII visualization of the graph."""
        lines = ["Graph of Thoughts", "=" * 50]

        def render_node(node_id: str, indent: int = 0) -> list[str]:
            if node_id not in self.nodes:
                return []

            node = self.nodes[node_id]
            prefix = "  " * indent
            symbol = {
                ThoughtType.QUESTION: "?",
                ThoughtType.HYPOTHESIS: "H",
                ThoughtType.EVIDENCE: "E",
                ThoughtType.REASONING: "R",
                ThoughtType.SYNTHESIS: "S",
                ThoughtType.REFINEMENT: "↑",
                ThoughtType.VALIDATION: "✓",
                ThoughtType.CONCLUSION: "★",
                ThoughtType.COUNTERPOINT: "⚡",
            }.get(node.thought_type, "•")

            result = [
                f"{prefix}[{symbol}] {node.content[:60]}... (SNR: {node.snr_score:.2f})"
            ]

            for child_id in self.adjacency.get(node_id, [])[:3]:  # Limit children
                result.extend(render_node(child_id, indent + 1))

            return result

        for root_id in self.roots[:3]:  # Limit roots
            lines.extend(render_node(root_id))

        lines.append("=" * 50)
        lines.append(f"Nodes: {len(self.nodes)} | Edges: {len(self.edges)}")

        return "\n".join(lines)


__all__ = [
    "GraphOfThoughts",
]
