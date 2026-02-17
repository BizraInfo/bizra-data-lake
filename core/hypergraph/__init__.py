"""
BIZRA HyperGraph -- N-ary knowledge graph with RAG fusion retrieval.

This package extends the standard pairwise graph model to hyperedges that
connect two or more nodes simultaneously, enabling richer representations
of concept clusters, causal chains, cross-domain bridges, and evidence
bundles.

Public API:
    HyperEdgeType       -- enum of supported edge categories
    HyperEdge           -- immutable N-ary edge dataclass
    HyperGraphNode      -- mutable node with optional embedding
    HyperGraphStore     -- in-memory graph with structural + vector queries
    HyperGraphRAGFusion -- triple-source retrieval (vector + keyword + graph)
    RetrievalResult     -- fused retrieval result dataclass
"""

from __future__ import annotations

from .hyperedge import HyperEdge, HyperEdgeType, HyperGraphNode
from .hypergraph_store import HyperGraphStore
from .rag_fusion import HyperGraphRAGFusion, RetrievalResult

__all__ = [
    "HyperEdgeType",
    "HyperEdge",
    "HyperGraphNode",
    "HyperGraphStore",
    "HyperGraphRAGFusion",
    "RetrievalResult",
]
