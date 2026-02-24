"""
Tests for core.hypergraph -- N-ary knowledge graph and RAG fusion.

Covers: edge creation, validation, neighbor queries, filtering, cosine
similarity search, RAG fusion without an external agent_db, and aggregate
properties.
"""

from __future__ import annotations

import pytest

from core.hypergraph import (
    HyperEdgeType,
    HyperGraphNode,
    HyperGraphRAGFusion,
    HyperGraphStore,
)
from core.hypergraph.rag_fusion import FUSION_WEIGHTS

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def store() -> HyperGraphStore:
    """Empty HyperGraphStore."""
    return HyperGraphStore()


@pytest.fixture()
def populated_store(store: HyperGraphStore) -> HyperGraphStore:
    """Store with 4 nodes and 2 edges across two domains."""
    store.add_node(HyperGraphNode("n1", "Alpha", "physics", embedding=[1.0, 0.0, 0.0]))
    store.add_node(HyperGraphNode("n2", "Beta", "physics", embedding=[0.9, 0.1, 0.0]))
    store.add_node(
        HyperGraphNode("n3", "Gamma", "economics", embedding=[0.0, 1.0, 0.0])
    )
    store.add_node(HyperGraphNode("n4", "Delta", "biology", embedding=[0.0, 0.0, 1.0]))

    store.add_hyperedge(
        {"n1", "n2", "n3"},
        HyperEdgeType.CROSS_DOMAIN_BRIDGE,
        weight=0.92,
    )
    store.add_hyperedge(
        {"n1", "n2"},
        HyperEdgeType.CONCEPT_CLUSTER,
        weight=0.88,
    )
    return store


# ---------------------------------------------------------------------------
# 1. test_hyperedge_connects_n_nodes
# ---------------------------------------------------------------------------


def test_hyperedge_connects_n_nodes(store: HyperGraphStore) -> None:
    """A hyperedge of 3 nodes has cardinality 3 and correct neighbors."""
    store.add_node(HyperGraphNode("a", "A", "math"))
    store.add_node(HyperGraphNode("b", "B", "math"))
    store.add_node(HyperGraphNode("c", "C", "math"))

    edge = store.add_hyperedge(
        {"a", "b", "c"},
        HyperEdgeType.CONCEPT_CLUSTER,
        weight=0.9,
    )

    assert edge.cardinality == 3
    assert store.get_neighbors("a") == {"b", "c"}
    assert store.get_neighbors("b") == {"a", "c"}
    assert store.get_neighbors("c") == {"a", "b"}


# ---------------------------------------------------------------------------
# 2. test_hyperedge_requires_minimum_2_nodes
# ---------------------------------------------------------------------------


def test_hyperedge_requires_minimum_2_nodes(store: HyperGraphStore) -> None:
    """Creating a hyperedge with < 2 nodes must raise ValueError."""
    store.add_node(HyperGraphNode("only", "Lonely", "math"))

    with pytest.raises(ValueError, match="at least 2 nodes"):
        store.add_hyperedge(
            {"only"},
            HyperEdgeType.CONCEPT_CLUSTER,
            weight=0.5,
        )


# ---------------------------------------------------------------------------
# 3. test_hyperedge_all_nodes_must_exist
# ---------------------------------------------------------------------------


def test_hyperedge_all_nodes_must_exist(store: HyperGraphStore) -> None:
    """Referencing a non-existent node in a hyperedge must raise ValueError."""
    store.add_node(HyperGraphNode("real", "Real", "physics"))

    with pytest.raises(ValueError, match="not found"):
        store.add_hyperedge(
            {"real", "ghost"},
            HyperEdgeType.EVIDENCE_BUNDLE,
            weight=0.7,
        )


# ---------------------------------------------------------------------------
# 4. test_get_neighbors_returns_correct_set
# ---------------------------------------------------------------------------


def test_get_neighbors_returns_correct_set(populated_store: HyperGraphStore) -> None:
    """Neighbors of n1 span both edges: {n2, n3}."""
    neighbors = populated_store.get_neighbors("n1")
    assert neighbors == {"n2", "n3"}


# ---------------------------------------------------------------------------
# 5. test_get_hyperedges_filters_by_type
# ---------------------------------------------------------------------------


def test_get_hyperedges_filters_by_type(populated_store: HyperGraphStore) -> None:
    """Filtering by CONCEPT_CLUSTER returns only the pairwise edge."""
    edges = populated_store.get_hyperedges(
        "n1",
        edge_type=HyperEdgeType.CONCEPT_CLUSTER,
    )
    assert len(edges) == 1
    assert edges[0].edge_type == HyperEdgeType.CONCEPT_CLUSTER
    assert edges[0].cardinality == 2

    all_edges = populated_store.get_hyperedges("n1")
    assert len(all_edges) == 2


# ---------------------------------------------------------------------------
# 6. test_get_cross_domain_bridges
# ---------------------------------------------------------------------------


def test_get_cross_domain_bridges(populated_store: HyperGraphStore) -> None:
    """Only the 3-node edge is a CROSS_DOMAIN_BRIDGE."""
    bridges = populated_store.get_cross_domain_bridges()
    assert len(bridges) == 1
    assert bridges[0].edge_type == HyperEdgeType.CROSS_DOMAIN_BRIDGE
    assert bridges[0].cardinality == 3


# ---------------------------------------------------------------------------
# 7. test_node_count_edge_count_properties
# ---------------------------------------------------------------------------


def test_node_count_edge_count_properties(populated_store: HyperGraphStore) -> None:
    """Verify node_count and edge_count reflect the fixture state."""
    assert populated_store.node_count == 4
    assert populated_store.edge_count == 2


# ---------------------------------------------------------------------------
# 8. test_mean_cardinality
# ---------------------------------------------------------------------------


def test_mean_cardinality(populated_store: HyperGraphStore) -> None:
    """Mean cardinality of (3-node edge, 2-node edge) should be 2.5."""
    assert populated_store.mean_cardinality == pytest.approx(2.5)


def test_mean_cardinality_empty(store: HyperGraphStore) -> None:
    """Empty store has mean_cardinality of 0.0."""
    assert store.mean_cardinality == 0.0


# ---------------------------------------------------------------------------
# 9. test_query_by_concept_cosine_similarity
# ---------------------------------------------------------------------------


def test_query_by_concept_cosine_similarity(populated_store: HyperGraphStore) -> None:
    """Querying with [1, 0, 0] should rank n1 and n2 highest."""
    results = populated_store.query_by_concept([1.0, 0.0, 0.0], top_k=4)

    # n1 has embedding [1,0,0] => similarity 1.0
    # n2 has embedding [0.9,0.1,0] => high similarity
    # n3 has embedding [0,1,0] => similarity 0.0
    # n4 has embedding [0,0,1] => similarity 0.0
    assert len(results) >= 1
    top_node, top_score = results[0]
    assert top_node.node_id == "n1"
    assert top_score == pytest.approx(1.0)

    if len(results) >= 2:
        second_node, second_score = results[1]
        assert second_node.node_id == "n2"
        assert second_score > 0.85


# ---------------------------------------------------------------------------
# 10. test_rag_fusion_without_agent_db
# ---------------------------------------------------------------------------


def test_rag_fusion_without_agent_db(populated_store: HyperGraphStore) -> None:
    """With no agent_db, only graph-hop can produce candidates.

    Since vector search returns nothing, seed_ids will be empty and graph
    expansion will also be empty.  The result list should therefore be empty.
    """
    fusion = HyperGraphRAGFusion(store=populated_store, agent_db=None)
    results = fusion.retrieve("test query", query_embedding=[1.0, 0.0, 0.0])
    # Without agent_db, no vector/keyword seeds exist, so graph expansion
    # has no seeds either.  The result must be an empty list.
    assert isinstance(results, list)
    assert len(results) == 0


# ---------------------------------------------------------------------------
# 11. test_rag_fusion_weights_sum_to_one
# ---------------------------------------------------------------------------


def test_rag_fusion_weights_sum_to_one() -> None:
    """The five fusion weights must sum to exactly 1.0."""
    total = sum(FUSION_WEIGHTS.values())
    assert total == pytest.approx(1.0)
    assert len(FUSION_WEIGHTS) == 5


# ---------------------------------------------------------------------------
# 12. test_hyperedge_is_pairwise
# ---------------------------------------------------------------------------


def test_hyperedge_is_pairwise(populated_store: HyperGraphStore) -> None:
    """A 2-node edge reports is_pairwise=True; a 3-node edge does not."""
    edges_n1 = populated_store.get_hyperedges("n1")

    pairwise = [e for e in edges_n1 if e.is_pairwise]
    non_pairwise = [e for e in edges_n1 if not e.is_pairwise]

    assert len(pairwise) == 1
    assert pairwise[0].cardinality == 2

    assert len(non_pairwise) == 1
    assert non_pairwise[0].cardinality == 3
