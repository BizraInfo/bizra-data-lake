"""
Graph Search — Search and Traversal Algorithms
===============================================
Search algorithms for navigating the thought graph:
- find_best_path: Best-first search for optimal reasoning paths
- backtrack: Return to promising unexplored nodes
- explore_with_backtrack: Iterative exploration with backtracking

Standing on Giants:
- Graph of Thoughts (Besta et al., 2024): BACKTRACK operation
- Best-first search: Dijkstra, A*
"""

from __future__ import annotations

import heapq
import logging
from typing import Dict, List, Optional

from .graph_types import (
    ThoughtNode,
    ThoughtEdge,
    ThoughtType,
    ReasoningPath,
)

logger = logging.getLogger(__name__)


class GraphSearchMixin:
    """
    Mixin providing search algorithms for GraphOfThoughts.

    This mixin implements the BACKTRACK operation from GoT,
    along with path-finding algorithms for navigating the
    reasoning graph.
    """

    # These attributes are defined in the main class
    nodes: Dict[str, ThoughtNode]
    edges: List[ThoughtEdge]
    adjacency: Dict[str, List[str]]
    reverse_adj: Dict[str, List[str]]
    roots: List[str]
    snr_threshold: float
    ihsan_threshold: float

    def find_best_path(
        self,
        start_id: Optional[str] = None,
        target_type: ThoughtType = ThoughtType.CONCLUSION,
    ) -> ReasoningPath:
        """Find the highest-SNR path through the graph."""
        if start_id is None:
            start_id = self.roots[0] if self.roots else None

        if not start_id or start_id not in self.nodes:
            return ReasoningPath(nodes=[], total_snr=0, total_confidence=0)

        # Best-first search using priority queue
        # Priority = negative SNR (for max-heap behavior)
        pq = [(-self.nodes[start_id].snr_score, [start_id])]
        visited = set()
        best_path = ReasoningPath(nodes=[start_id], total_snr=self.nodes[start_id].snr_score)

        while pq:
            neg_snr, path = heapq.heappop(pq)
            current_id = path[-1]

            if current_id in visited:
                continue
            visited.add(current_id)

            current_node = self.nodes[current_id]

            # Check if we've reached a conclusion
            if current_node.thought_type == target_type:
                total_snr = sum(self.nodes[n].snr_score for n in path)
                if total_snr > best_path.total_snr:
                    best_path = ReasoningPath(
                        nodes=path,
                        total_snr=total_snr,
                        total_confidence=sum(self.nodes[n].confidence for n in path),
                    )

            # Explore children
            for child_id in self.adjacency.get(current_id, []):
                if child_id not in visited and child_id in self.nodes:
                    child = self.nodes[child_id]
                    new_path = path + [child_id]
                    heapq.heappush(pq, (-child.snr_score, new_path))

        return best_path

    def get_conclusions(self, min_snr: float = 0.7) -> List[ThoughtNode]:
        """Get all conclusion nodes above SNR threshold."""
        return [
            node for node in self.nodes.values()
            if node.thought_type == ThoughtType.CONCLUSION
            and node.snr_score >= min_snr
        ]

    def get_frontier(self) -> List[ThoughtNode]:
        """Get leaf nodes (nodes with no children) - the current reasoning frontier."""
        # Find nodes that have no outgoing edges (no children)
        parent_ids = set(self.adjacency.keys())
        leaf_ids = [
            node_id for node_id in self.nodes.keys()
            if node_id not in parent_ids or not self.adjacency[node_id]
        ]
        return [self.nodes[nid] for nid in leaf_ids if nid in self.nodes]

    def get_leaves(self) -> List[ThoughtNode]:
        """Alias for get_frontier."""
        return self.get_frontier()

    def backtrack(self) -> Optional[ThoughtNode]:
        """
        Return to the highest-SNR unexplored frontier node.

        Per Besta et al. (2024), backtracking enables recovery from dead-ends
        by returning to promising unexplored branches in the reasoning graph.

        This is the 6th GoT operation, completing the reasoning framework:
        - GENERATE: Create new thought nodes
        - AGGREGATE: Merge multiple thoughts into synthesis
        - REFINE: Improve existing thoughts iteratively
        - VALIDATE: Check thoughts against Ihsan constraints
        - PRUNE: Remove low-SNR branches
        - BACKTRACK: Return to promising unexplored paths (THIS METHOD)

        Returns:
            The highest-SNR unexplored frontier node, or None if all paths
            have been explored or lead to conclusions.
        """
        frontier = self.get_frontier()

        # Filter for unexplored nodes:
        # - Not CONCLUSION (already terminal)
        # - Not VALIDATION (already checked)
        # - Has no children (is a true leaf, not just low-connectivity)
        unexplored = [
            node for node in frontier
            if node.thought_type not in (ThoughtType.CONCLUSION, ThoughtType.VALIDATION)
            and len(self.adjacency.get(node.id, [])) == 0
        ]

        if not unexplored:
            logger.debug("Backtrack: No unexplored frontier nodes available")
            return None

        # Return highest-SNR unexplored node
        best = max(unexplored, key=lambda n: n.snr_score)
        logger.info(
            f"Backtrack: Returning to '{best.content[:40]}...' "
            f"(SNR: {best.snr_score:.3f}, type: {best.thought_type.value})"
        )
        return best

    def explore_with_backtrack(
        self,
        max_iterations: int = 10,
        target_snr: float = None,
    ) -> Optional[ThoughtNode]:
        """
        Iteratively explore the graph with backtracking until target SNR is reached.

        Per Besta et al. (2024), combines best-first search with backtracking
        for robust exploration of complex reasoning spaces.

        Args:
            max_iterations: Maximum exploration iterations
            target_snr: Target SNR for conclusions (defaults to ihsan_threshold)

        Returns:
            Best conclusion node found, or None if exploration fails.
        """
        target_snr = target_snr or self.ihsan_threshold

        for i in range(max_iterations):
            # Try to find a high-quality conclusion
            conclusions = self.get_conclusions(min_snr=target_snr)
            if conclusions:
                best = max(conclusions, key=lambda n: n.snr_score)
                logger.info(f"Exploration succeeded at iteration {i+1}: SNR={best.snr_score:.3f}")
                return best

            # No good conclusion yet - backtrack and continue
            backtrack_node = self.backtrack()
            if backtrack_node is None:
                logger.warning("Exploration exhausted: no backtrack options")
                break

            # The caller should use backtrack_node to generate new thoughts
            # This method just identifies the node to explore from
            logger.debug(f"Iteration {i+1}: backtracking to explore from {backtrack_node.id}")

        # Return best available conclusion even if below threshold
        conclusions = self.get_conclusions(min_snr=0.0)
        if conclusions:
            return max(conclusions, key=lambda n: n.snr_score)
        return None


__all__ = [
    "GraphSearchMixin",
]
