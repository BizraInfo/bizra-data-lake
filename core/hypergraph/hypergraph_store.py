"""
HyperGraphStore -- in-memory N-ary graph with cosine similarity search.

Provides CRUD operations on nodes and hyperedges together with structural
queries (neighbors, cross-domain bridges) and embedding-based retrieval.

Standing on Giants: Berge (1973) -- Hypergraph theory, Shannon (1948) -- SNR
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

from core.integration.constants import SNR_THRESHOLD

from .hyperedge import (
    HyperEdge,
    HyperEdgeType,
    HyperGraphNode,
    generate_edge_id,
)

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors.

    Returns 0.0 when either vector has zero norm (avoids division by zero).
    """
    if np is None:
        raise ImportError(
            "numpy is required for embedding-based queries. "
            "Install it with: pip install numpy"
        )
    va, vb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    dot = float(np.dot(va, vb))
    norm = float(np.linalg.norm(va) * np.linalg.norm(vb))
    return dot / norm if norm > 0 else 0.0


# ---------------------------------------------------------------------------
# HyperGraphStore
# ---------------------------------------------------------------------------


class HyperGraphStore:
    """In-memory hypergraph with structural and vector queries.

    Usage::

        store = HyperGraphStore()
        store.add_node(HyperGraphNode("n1", "Node 1", "physics"))
        store.add_node(HyperGraphNode("n2", "Node 2", "physics"))
        store.add_hyperedge({"n1", "n2"}, HyperEdgeType.CONCEPT_CLUSTER, 0.9)
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, HyperGraphNode] = {}
        self._edges: Dict[str, HyperEdge] = {}
        self._node_to_edges: Dict[str, Set[str]] = defaultdict(set)

    # -- mutation -----------------------------------------------------------

    def add_node(self, node: HyperGraphNode) -> None:
        """Register a node in the graph.

        Args:
            node: The node to store.  Overwrites if *node_id* already exists.
        """
        self._nodes[node.node_id] = node

    def add_hyperedge(
        self,
        node_ids: Set[str],
        edge_type: HyperEdgeType,
        weight: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> HyperEdge:
        """Create a hyperedge connecting *node_ids*.

        Args:
            node_ids:  Set of two or more existing node identifiers.
            edge_type: Semantic category of the relationship.
            weight:    Strength of the edge in [0, 1].
            metadata:  Optional annotations.

        Returns:
            The newly created :class:`HyperEdge`.

        Raises:
            ValueError: If fewer than 2 node IDs are given.
            ValueError: If any referenced node does not exist in the store.
        """
        if len(node_ids) < 2:
            raise ValueError(
                f"A hyperedge requires at least 2 nodes, got {len(node_ids)}."
            )

        missing = node_ids - self._nodes.keys()
        if missing:
            raise ValueError(f"Node(s) not found in the store: {sorted(missing)}")

        edge_id = generate_edge_id(node_ids)
        edge = HyperEdge(
            edge_id=edge_id,
            node_ids=frozenset(node_ids),
            edge_type=edge_type,
            weight=weight,
            metadata=metadata or {},
        )
        self._edges[edge_id] = edge

        for nid in node_ids:
            self._node_to_edges[nid].add(edge_id)

        return edge

    # -- structural queries -------------------------------------------------

    def get_neighbors(self, node_id: str) -> Set[str]:
        """Return all node IDs reachable from *node_id* via any hyperedge.

        The result does **not** include *node_id* itself.

        Args:
            node_id: Source node.

        Returns:
            Set of neighbor node identifiers.
        """
        neighbors: Set[str] = set()
        for edge_id in self._node_to_edges.get(node_id, set()):
            edge = self._edges[edge_id]
            neighbors.update(edge.node_ids)
        neighbors.discard(node_id)
        return neighbors

    def get_hyperedges(
        self,
        node_id: str,
        edge_type: Optional[HyperEdgeType] = None,
    ) -> List[HyperEdge]:
        """Return hyperedges incident to *node_id*, optionally filtered.

        Args:
            node_id:   Node whose edges to retrieve.
            edge_type: If given, only return edges of this type.

        Returns:
            List of matching hyperedges.
        """
        edges: List[HyperEdge] = []
        for edge_id in self._node_to_edges.get(node_id, set()):
            edge = self._edges[edge_id]
            if edge_type is None or edge.edge_type == edge_type:
                edges.append(edge)
        return edges

    def get_cross_domain_bridges(self) -> List[HyperEdge]:
        """Return all edges of type :attr:`HyperEdgeType.CROSS_DOMAIN_BRIDGE`.

        Returns:
            List of cross-domain bridge hyperedges.
        """
        return [
            e
            for e in self._edges.values()
            if e.edge_type == HyperEdgeType.CROSS_DOMAIN_BRIDGE
        ]

    # -- embedding queries --------------------------------------------------

    def query_by_concept(
        self,
        concept_embedding: List[float],
        top_k: int = 10,
    ) -> List[tuple[HyperGraphNode, float]]:
        """Find nodes closest to *concept_embedding* via cosine similarity.

        Only nodes that have a non-``None`` embedding are considered.
        Results are filtered to those meeting the minimum SNR threshold
        imported from ``core.integration.constants``.

        Args:
            concept_embedding: Dense query vector.
            top_k:             Maximum results to return.

        Returns:
            List of (node, similarity) tuples sorted descending by score.
        """
        scored: List[tuple[HyperGraphNode, float]] = []
        for node in self._nodes.values():
            if node.embedding is None:
                continue
            sim = _cosine_similarity(concept_embedding, node.embedding)
            if sim >= SNR_THRESHOLD:
                scored.append((node, sim))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]

    # -- aggregate properties -----------------------------------------------

    @property
    def node_count(self) -> int:
        """Total number of nodes in the store."""
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        """Total number of hyperedges in the store."""
        return len(self._edges)

    @property
    def mean_cardinality(self) -> float:
        """Average number of nodes per hyperedge.

        Returns 0.0 when the store contains no edges.
        """
        if not self._edges:
            return 0.0
        total = sum(e.cardinality for e in self._edges.values())
        return total / len(self._edges)
