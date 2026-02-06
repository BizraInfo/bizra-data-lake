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

import logging
from collections import defaultdict
from typing import Dict, List

from core.integration.constants import UNIFIED_IHSAN_THRESHOLD, UNIFIED_SNR_THRESHOLD

from .graph_operations import GraphOperationsMixin
from .graph_reasoning import GraphReasoningMixin
from .graph_search import GraphSearchMixin
from .graph_types import (
    ReasoningStrategy,
    ThoughtEdge,
    ThoughtNode,
    ThoughtType,
)

logger = logging.getLogger(__name__)


class GraphOfThoughts(GraphOperationsMixin, GraphSearchMixin, GraphReasoningMixin):
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
    ):
        self.strategy = strategy
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.snr_threshold = snr_threshold
        self.ihsan_threshold = ihsan_threshold

        # Graph structure
        self.nodes: Dict[str, ThoughtNode] = {}
        self.edges: List[ThoughtEdge] = []
        self.adjacency: Dict[str, List[str]] = defaultdict(list)  # node -> children
        self.reverse_adj: Dict[str, List[str]] = defaultdict(list)  # node -> parents

        # Root nodes (entry points)
        self.roots: List[str] = []

        # Statistics
        self.stats = {
            "nodes_created": 0,
            "nodes_pruned": 0,
            "edges_created": 0,
            "refinements": 0,
            "aggregations": 0,
        }

    @property
    def thoughts(self) -> Dict[str, ThoughtNode]:
        """Property alias for nodes dict."""
        return self.nodes

    def to_dict(self) -> dict:
        """Serialize graph for inspection."""
        return {
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
        }

    def visualize_ascii(self) -> str:
        """Generate ASCII visualization of the graph."""
        lines = ["Graph of Thoughts", "=" * 50]

        def render_node(node_id: str, indent: int = 0) -> List[str]:
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
