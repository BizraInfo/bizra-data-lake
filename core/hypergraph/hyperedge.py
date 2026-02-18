"""
HyperEdge types and data structures for N-ary graph connections.

A hyperedge generalises a standard graph edge to connect N >= 2 nodes
simultaneously.  This module provides the foundational data structures
used throughout the BIZRA hypergraph subsystem.

Standing on Giants: Berge (1973) -- Hypergraph theory
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Dict, FrozenSet, List, Optional


# ---------------------------------------------------------------------------
# HyperEdge type taxonomy
# ---------------------------------------------------------------------------


class HyperEdgeType(Enum):
    """Classification of N-ary relationships in the knowledge graph."""

    CONCEPT_CLUSTER = auto()  # N nodes share a concept
    CAUSAL_CHAIN = auto()  # Ordered cause->effect across N nodes
    CROSS_DOMAIN_BRIDGE = auto()  # N nodes from different domains share pattern
    TEMPORAL_COHORT = auto()  # N nodes active in same time window
    EVIDENCE_BUNDLE = auto()  # N evidence items supporting one claim


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HyperEdge:
    """An immutable N-ary edge connecting two or more nodes.

    Attributes:
        edge_id:    Deterministic SHA-256 digest of sorted node IDs.
        node_ids:   Frozen set of participating node identifiers.
        edge_type:  Semantic category from *HyperEdgeType*.
        weight:     Edge strength in [0, 1].
        metadata:   Arbitrary key-value annotations.
        created_at: ISO-8601 UTC timestamp of creation.
    """

    edge_id: str
    node_ids: FrozenSet[str]
    edge_type: HyperEdgeType
    weight: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )

    # -- derived properties --------------------------------------------------

    @property
    def cardinality(self) -> int:
        """Number of nodes participating in this hyperedge."""
        return len(self.node_ids)

    @property
    def is_pairwise(self) -> bool:
        """True when the hyperedge degenerates to a standard binary edge."""
        return self.cardinality == 2


@dataclass
class HyperGraphNode:
    """A node in the hypergraph, optionally carrying a vector embedding.

    Attributes:
        node_id:   Unique identifier.
        label:     Human-readable display name.
        domain:    Knowledge domain (e.g. 'physics', 'economics').
        embedding: Optional dense vector for similarity search.
        metadata:  Arbitrary key-value annotations.
    """

    node_id: str
    label: str
    domain: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def generate_edge_id(node_ids: FrozenSet[str] | set[str]) -> str:
    """Deterministic edge ID from sorted node identifiers.

    Uses the first 16 hex characters of the SHA-256 digest of the
    concatenated, sorted node IDs.

    Args:
        node_ids: Set of node identifiers participating in the edge.

    Returns:
        A 16-character hexadecimal string.
    """
    raw = "".join(sorted(node_ids)).encode()
    return hashlib.sha256(raw).hexdigest()[:16]
