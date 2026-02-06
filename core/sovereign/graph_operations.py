"""
Graph Operations — Core Graph Manipulation Methods
===================================================
Core CRUD and manipulation operations for the Graph-of-Thoughts:
- add_thought, add_edge
- generate, aggregate, refine, validate
- score_node, prune

Standing on Giants:
- Graph of Thoughts (Besta et al., 2024): GoT operations framework
"""

from __future__ import annotations

import logging
import uuid
from typing import Callable, Dict, List, Optional, Tuple

from .graph_types import (
    EdgeType,
    ThoughtEdge,
    ThoughtNode,
    ThoughtType,
)

logger = logging.getLogger(__name__)


class GraphOperationsMixin:
    """
    Mixin providing core graph operations for GraphOfThoughts.

    This mixin implements the fundamental GoT operations:
    - GENERATE: Create new thought nodes
    - AGGREGATE: Merge multiple thoughts into synthesis
    - REFINE: Improve existing thoughts iteratively
    - VALIDATE: Check thoughts against Ihsan constraints
    - PRUNE: Remove low-SNR branches
    """

    # These attributes are defined in the main class
    nodes: Dict[str, ThoughtNode]
    edges: List[ThoughtEdge]
    adjacency: Dict[str, List[str]]
    reverse_adj: Dict[str, List[str]]
    roots: List[str]
    stats: Dict[str, int]
    snr_threshold: float
    ihsan_threshold: float

    def _generate_id(self) -> str:
        """Generate unique thought ID."""
        return f"thought_{uuid.uuid4().hex[:12]}"

    def add_thought(
        self,
        content: str,
        thought_type: ThoughtType,
        confidence: float = 0.5,
        metadata: Optional[Dict] = None,
        parent_id: Optional[str] = None,
        edge_type: EdgeType = EdgeType.DERIVES,
    ) -> ThoughtNode:
        """Add a new thought node to the graph."""
        node_id = self._generate_id()

        # Determine depth
        depth = 0
        if parent_id and parent_id in self.nodes:
            depth = self.nodes[parent_id].depth + 1

        node = ThoughtNode(
            id=node_id,
            content=content,
            thought_type=thought_type,
            confidence=confidence,
            depth=depth,
            metadata=metadata or {},
        )

        self.nodes[node_id] = node
        self.stats["nodes_created"] += 1

        # Track root nodes
        if parent_id is None:
            self.roots.append(node_id)
        else:
            # Create edge to parent
            self.add_edge(parent_id, node_id, edge_type)

        logger.debug(f"Added thought: {node_id} ({thought_type.value})")
        return node

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
        weight: float = 1.0,
        reasoning: str = "",
    ) -> ThoughtEdge:
        """Add an edge between thought nodes."""
        edge = ThoughtEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            weight=weight,
            reasoning=reasoning,
        )

        self.edges.append(edge)
        self.adjacency[source_id].append(target_id)
        self.reverse_adj[target_id].append(source_id)
        self.stats["edges_created"] += 1

        return edge

    def generate(
        self,
        content: str,
        thought_type: ThoughtType,
        parent: Optional[ThoughtNode] = None,
        edge_type: EdgeType = EdgeType.DERIVES,
        **kwargs,
    ) -> ThoughtNode:
        """Generate a new thought from parent context."""
        parent_id = parent.id if parent else None
        return self.add_thought(
            content=content,
            thought_type=thought_type,
            parent_id=parent_id,
            edge_type=edge_type,
            **kwargs,
        )

    def aggregate(
        self,
        thoughts: List[ThoughtNode],
        synthesis_content: str,
        aggregation_type: ThoughtType = ThoughtType.SYNTHESIS,
    ) -> ThoughtNode:
        """Aggregate multiple thoughts into a synthesis."""
        # Compute aggregated confidence
        avg_confidence = sum(t.confidence for t in thoughts) / len(thoughts)
        avg_snr = sum(t.snr_score for t in thoughts) / len(thoughts)
        max_depth = max(t.depth for t in thoughts)

        synth = ThoughtNode(
            id=self._generate_id(),
            content=synthesis_content,
            thought_type=aggregation_type,
            confidence=avg_confidence * 1.1,  # Synthesis bonus
            snr_score=avg_snr * 1.05,
            depth=max_depth + 1,
        )

        self.nodes[synth.id] = synth
        self.stats["aggregations"] += 1

        # Connect all source thoughts to synthesis
        for thought in thoughts:
            self.add_edge(
                thought.id,
                synth.id,
                EdgeType.SYNTHESIZES,
                weight=thought.confidence,
            )

        logger.debug(f"Aggregated {len(thoughts)} thoughts into {synth.id}")
        return synth

    def refine(
        self,
        thought: ThoughtNode,
        refined_content: str,
        improvement_score: float = 0.1,
    ) -> ThoughtNode:
        """Refine an existing thought with improved content."""
        refined = ThoughtNode(
            id=self._generate_id(),
            content=refined_content,
            thought_type=ThoughtType.REFINEMENT,
            confidence=min(thought.confidence + improvement_score, 1.0),
            snr_score=min(thought.snr_score + improvement_score, 1.0),
            depth=thought.depth + 1,
        )

        self.nodes[refined.id] = refined
        self.stats["refinements"] += 1

        self.add_edge(thought.id, refined.id, EdgeType.REFINES)

        logger.debug(f"Refined {thought.id} -> {refined.id}")
        return refined

    def validate(
        self,
        thought: ThoughtNode,
        validator_fn: Optional[Callable[[str], Tuple[bool, float]]] = None,
    ) -> ThoughtNode:
        """Validate a thought and create validation node."""
        # Default validation: check Ihsan threshold
        if validator_fn:
            is_valid, score = validator_fn(thought.content)
        else:
            is_valid = thought.ihsan_score >= self.ihsan_threshold
            score = thought.ihsan_score

        validation = ThoughtNode(
            id=self._generate_id(),
            content=f"Validation: {'PASS' if is_valid else 'FAIL'} (score: {score:.3f})",
            thought_type=ThoughtType.VALIDATION,
            confidence=score,
            snr_score=score,
            depth=thought.depth + 1,
            metadata={"validated_id": thought.id, "passed": is_valid},
        )

        self.nodes[validation.id] = validation
        self.add_edge(thought.id, validation.id, EdgeType.VALIDATES)

        return validation

    def score_node(self, node: ThoughtNode) -> float:
        """Compute comprehensive SNR score for a node."""
        # Base score from node properties
        base_score = (node.confidence + node.snr_score + node.ihsan_score) / 3

        # Depth penalty (deeper = less certain)
        depth_factor = 1.0 / (1.0 + 0.1 * node.depth)

        # Support factor (more supporting edges = higher score)
        support_count = sum(
            1
            for e in self.edges
            if e.target_id == node.id and e.edge_type == EdgeType.SUPPORTS
        )
        support_factor = 1.0 + 0.1 * support_count

        # Refutation penalty
        refute_count = sum(
            1
            for e in self.edges
            if e.target_id == node.id and e.edge_type == EdgeType.REFUTES
        )
        refute_factor = 1.0 / (1.0 + 0.2 * refute_count)

        final_score = base_score * depth_factor * support_factor * refute_factor
        node.snr_score = min(final_score, 1.0)

        return node.snr_score

    def prune(self, threshold: Optional[float] = None) -> int:
        """Remove low-SNR nodes from the graph."""
        threshold = threshold or self.snr_threshold
        pruned = 0

        nodes_to_remove = [
            node_id
            for node_id, node in self.nodes.items()
            if node.snr_score < threshold and node.thought_type != ThoughtType.QUESTION
        ]

        for node_id in nodes_to_remove:
            # Remove from adjacency
            for child in self.adjacency.get(node_id, []):
                if node_id in self.reverse_adj.get(child, []):
                    self.reverse_adj[child].remove(node_id)
            for parent in self.reverse_adj.get(node_id, []):
                if node_id in self.adjacency.get(parent, []):
                    self.adjacency[parent].remove(node_id)

            del self.nodes[node_id]
            if node_id in self.adjacency:
                del self.adjacency[node_id]
            if node_id in self.reverse_adj:
                del self.reverse_adj[node_id]
            pruned += 1

        # Remove orphaned edges
        self.edges = [
            e
            for e in self.edges
            if e.source_id in self.nodes and e.target_id in self.nodes
        ]

        self.stats["nodes_pruned"] += pruned
        logger.info(f"Pruned {pruned} low-SNR nodes")
        return pruned

    def clear(self):
        """Reset the graph to empty state."""
        self.nodes.clear()
        self.edges.clear()
        self.adjacency.clear()
        self.reverse_adj.clear()
        self.roots.clear()
        # Reset stats but keep structure
        self.stats = {
            "nodes_created": 0,
            "nodes_pruned": 0,
            "edges_created": 0,
            "refinements": 0,
            "aggregations": 0,
        }
        logger.debug("Graph cleared")


__all__ = [
    "GraphOperationsMixin",
]
