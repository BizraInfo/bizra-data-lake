"""
Tests for core.graph.semantic_layer — Dual-overlay topology separation.

Covers:
- EdgeClassification enum values
- ClassifiedEdge frozen dataclass behavior
- TopologyMetrics / GraphTopologyReport dataclass serialization
- DualOverlayGraph edge routing (structural, semantic, ambiguous)
- TopologyAnalyzer static methods with hand-verifiable small graphs
- SemanticLayerSeparator classify_edge heuristics and classify_from_json
- Edge cases: empty graph, single node, disconnected components
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Set

import pytest

from core.graph.semantic_layer import (
    SCALE_FREE_GAMMA_RANGE,
    SEMANTIC_EDGE_TYPES,
    SMALL_WORLD_SIGMA_THRESHOLD,
    STRUCTURAL_EDGE_TYPES,
    ClassifiedEdge,
    DualOverlayGraph,
    EdgeClassification,
    GraphTopologyReport,
    SemanticLayerSeparator,
    TopologyAnalyzer,
    TopologyMetrics,
    create_semantic_separator,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_adj(*edges: tuple) -> Dict[str, Dict[str, float]]:
    """Build an undirected adjacency dict from a list of (u, v) tuples."""
    adj: Dict[str, Dict[str, float]] = defaultdict(dict)
    for u, v in edges:
        adj[u][v] = 1.0
        adj[v][u] = 1.0
    return dict(adj)


def _nodes_from_adj(adj: Dict[str, Dict[str, float]]) -> Set[str]:
    nodes: Set[str] = set()
    for src, targets in adj.items():
        nodes.add(src)
        nodes.update(targets.keys())
    return nodes


# ===========================================================================
# 1. EdgeClassification enum
# ===========================================================================


class TestEdgeClassification:

    def test_values(self):
        assert EdgeClassification.STRUCTURAL.value == "structural"
        assert EdgeClassification.SEMANTIC.value == "semantic"
        assert EdgeClassification.AMBIGUOUS.value == "ambiguous"

    def test_is_str_enum(self):
        """EdgeClassification members are strings."""
        assert isinstance(EdgeClassification.STRUCTURAL, str)
        assert EdgeClassification.SEMANTIC == "semantic"

    def test_member_count(self):
        assert len(EdgeClassification) == 3


# ===========================================================================
# 2. ClassifiedEdge frozen dataclass
# ===========================================================================


class TestClassifiedEdge:

    def test_construction_defaults(self):
        edge = ClassifiedEdge(
            source="A",
            target="B",
            edge_type="PART_OF",
            classification=EdgeClassification.STRUCTURAL,
        )
        assert edge.source == "A"
        assert edge.target == "B"
        assert edge.weight == 1.0
        assert edge.metadata == {}

    def test_construction_with_metadata(self):
        meta = {"via_tag": "python"}
        edge = ClassifiedEdge(
            source="X",
            target="Y",
            edge_type="RELATES_TO",
            classification=EdgeClassification.SEMANTIC,
            weight=0.8,
            metadata=meta,
        )
        assert edge.weight == 0.8
        assert edge.metadata == {"via_tag": "python"}

    def test_frozen(self):
        edge = ClassifiedEdge(
            source="A",
            target="B",
            edge_type="PART_OF",
            classification=EdgeClassification.STRUCTURAL,
        )
        with pytest.raises(AttributeError):
            edge.weight = 2.0  # type: ignore[misc]


# ===========================================================================
# 3. TopologyMetrics & GraphTopologyReport serialization
# ===========================================================================


class TestTopologyMetrics:

    def test_defaults(self):
        m = TopologyMetrics()
        assert m.node_count == 0
        assert m.is_small_world is False
        assert m.is_scale_free is False

    def test_to_dict_roundtrip(self):
        m = TopologyMetrics(node_count=5, edge_count=4, avg_degree=1.6)
        d = m.to_dict()
        assert d["node_count"] == 5
        assert d["edge_count"] == 4
        assert d["avg_degree"] == 1.6
        # Every field should appear
        assert "is_small_world" in d
        assert "degree_entropy" in d


class TestGraphTopologyReport:

    def test_defaults(self):
        r = GraphTopologyReport()
        assert r.total_nodes == 0
        assert r.structural_topology is None
        assert r.edge_type_counts == {}
        assert r.bridge_nodes == []

    def test_to_dict_without_topology(self):
        r = GraphTopologyReport(total_nodes=10, total_edges=15)
        d = r.to_dict()
        assert d["total_nodes"] == 10
        # None topologies should not appear as keys at top level
        assert "structural_topology" not in d
        assert "semantic_topology" not in d

    def test_to_dict_with_topology(self):
        m = TopologyMetrics(node_count=10, edge_count=12)
        r = GraphTopologyReport(
            total_nodes=10,
            total_edges=12,
            structural_topology=m,
        )
        d = r.to_dict()
        assert "structural_topology" in d
        assert d["structural_topology"]["node_count"] == 10


# ===========================================================================
# 4. DualOverlayGraph
# ===========================================================================


class TestDualOverlayGraph:

    def test_empty_graph(self):
        g = DualOverlayGraph()
        assert g.node_count == 0
        assert g.structural_edge_count == 0
        assert g.semantic_edge_count == 0

    def test_add_node(self):
        g = DualOverlayGraph()
        g.add_node("A", "DOCUMENT")
        g.add_node("B", "CHUNK")
        assert g.node_count == 2

    def test_add_node_dedup(self):
        g = DualOverlayGraph()
        g.add_node("A", "DOCUMENT")
        g.add_node("A", "DOCUMENT")
        assert g.node_count == 1

    def test_structural_edge_routing(self):
        """Structural edges go only to structural adjacency."""
        g = DualOverlayGraph()
        edge = ClassifiedEdge("A", "B", "PART_OF", EdgeClassification.STRUCTURAL)
        g.add_edge(edge)

        assert g.structural_edge_count == 1
        assert g.semantic_edge_count == 0
        assert g.structural_degree("A") == 1
        assert g.semantic_degree("A") == 0
        assert g.structural_neighbors("A") == {"B"}
        assert g.semantic_neighbors("A") == set()

    def test_semantic_edge_routing(self):
        """Semantic edges go only to semantic adjacency."""
        g = DualOverlayGraph()
        edge = ClassifiedEdge("A", "B", "RELATES_TO", EdgeClassification.SEMANTIC)
        g.add_edge(edge)

        assert g.structural_edge_count == 0
        assert g.semantic_edge_count == 1
        assert g.semantic_degree("A") == 1
        assert g.structural_degree("A") == 0
        assert g.semantic_neighbors("A") == {"B"}

    def test_ambiguous_edge_routing(self):
        """Ambiguous edges go to BOTH adjacency lists."""
        g = DualOverlayGraph()
        edge = ClassifiedEdge("A", "B", "MYSTERY", EdgeClassification.AMBIGUOUS)
        g.add_edge(edge)

        # Ambiguous edges are not counted as structural or semantic
        assert g.structural_edge_count == 0
        assert g.semantic_edge_count == 0
        # But both adjacencies get the edge
        assert g.structural_degree("A") == 1
        assert g.semantic_degree("A") == 1

    def test_structural_edge_is_undirected(self):
        """Structural edges are stored bidirectionally."""
        g = DualOverlayGraph()
        g.add_edge(ClassifiedEdge("A", "B", "PART_OF", EdgeClassification.STRUCTURAL))
        assert g.structural_neighbors("A") == {"B"}
        assert g.structural_neighbors("B") == {"A"}

    def test_semantic_edge_is_undirected(self):
        """Semantic edges are stored bidirectionally."""
        g = DualOverlayGraph()
        g.add_edge(ClassifiedEdge("A", "B", "RELATES_TO", EdgeClassification.SEMANTIC))
        assert g.semantic_neighbors("A") == {"B"}
        assert g.semantic_neighbors("B") == {"A"}

    def test_ambiguous_edge_not_undirected_in_semantic_adj(self):
        """
        Ambiguous edges are stored directionally in both adjacencies
        (source->target only, not target->source) per the implementation.
        """
        g = DualOverlayGraph()
        g.add_edge(ClassifiedEdge("A", "B", "MYSTERY", EdgeClassification.AMBIGUOUS))
        # The implementation only adds source->target for ambiguous
        assert g.structural_degree("A") == 1
        assert g.structural_degree("B") == 0  # NOT bidirectional for ambiguous
        assert g.semantic_degree("A") == 1
        assert g.semantic_degree("B") == 0

    def test_edge_adds_nodes_implicitly(self):
        """Adding an edge also registers both endpoint nodes."""
        g = DualOverlayGraph()
        g.add_edge(ClassifiedEdge("X", "Y", "USES", EdgeClassification.SEMANTIC))
        assert g.node_count == 2

    def test_get_semantic_nodes(self):
        g = DualOverlayGraph()
        g.add_node("isolated", "NODE")
        g.add_edge(ClassifiedEdge("A", "B", "RELATES_TO", EdgeClassification.SEMANTIC))
        g.add_edge(ClassifiedEdge("C", "D", "PART_OF", EdgeClassification.STRUCTURAL))

        semantic_nodes = g.get_semantic_nodes()
        assert "A" in semantic_nodes
        assert "B" in semantic_nodes
        # C and D have no semantic edges
        assert "C" not in semantic_nodes
        assert "isolated" not in semantic_nodes

    def test_get_semantic_edges(self):
        g = DualOverlayGraph()
        g.add_edge(ClassifiedEdge("A", "B", "RELATES_TO", EdgeClassification.SEMANTIC))
        g.add_edge(ClassifiedEdge("C", "D", "PART_OF", EdgeClassification.STRUCTURAL))
        g.add_edge(ClassifiedEdge("E", "F", "USES", EdgeClassification.SEMANTIC))

        sem_edges = g.get_semantic_edges()
        assert len(sem_edges) == 2
        assert all(e.classification == EdgeClassification.SEMANTIC for e in sem_edges)

    def test_get_structural_edges(self):
        g = DualOverlayGraph()
        g.add_edge(ClassifiedEdge("A", "B", "PART_OF", EdgeClassification.STRUCTURAL))
        g.add_edge(ClassifiedEdge("C", "D", "USES", EdgeClassification.SEMANTIC))

        struct_edges = g.get_structural_edges()
        assert len(struct_edges) == 1
        assert struct_edges[0].edge_type == "PART_OF"

    def test_degree_nonexistent_node(self):
        g = DualOverlayGraph()
        assert g.structural_degree("ghost") == 0
        assert g.semantic_degree("ghost") == 0
        assert g.structural_neighbors("ghost") == set()
        assert g.semantic_neighbors("ghost") == set()


# ===========================================================================
# 5. TopologyAnalyzer
# ===========================================================================


class TestTopologyAnalyzerDegreeDistribution:

    def test_empty_graph(self):
        result = TopologyAnalyzer.compute_degree_distribution({}, set())
        assert result == {}

    def test_isolated_nodes(self):
        """Nodes with no edges have degree 0."""
        result = TopologyAnalyzer.compute_degree_distribution({}, {"A", "B", "C"})
        assert result == {0: 3}

    def test_triangle(self):
        """Three nodes forming a triangle: each has degree 2."""
        adj = _make_adj(("A", "B"), ("B", "C"), ("A", "C"))
        nodes = _nodes_from_adj(adj)
        result = TopologyAnalyzer.compute_degree_distribution(adj, nodes)
        assert result == {2: 3}

    def test_star(self):
        """Star graph: center has degree 3, leaves have degree 1."""
        adj = _make_adj(("H", "A"), ("H", "B"), ("H", "C"))
        nodes = _nodes_from_adj(adj)
        result = TopologyAnalyzer.compute_degree_distribution(adj, nodes)
        assert result[3] == 1  # hub
        assert result[1] == 3  # leaves


class TestTopologyAnalyzerClustering:

    def test_empty(self):
        assert TopologyAnalyzer.compute_clustering_coefficient({}) == 0.0

    def test_triangle_is_one(self):
        """A complete triangle has clustering coefficient 1.0."""
        adj = _make_adj(("A", "B"), ("B", "C"), ("A", "C"))
        cc = TopologyAnalyzer.compute_clustering_coefficient(dict(adj))
        assert cc == pytest.approx(1.0)

    def test_star_is_zero(self):
        """Star graph has zero clustering (no triangles among leaves)."""
        adj = _make_adj(("H", "A"), ("H", "B"), ("H", "C"))
        cc = TopologyAnalyzer.compute_clustering_coefficient(dict(adj))
        assert cc == pytest.approx(0.0)

    def test_single_edge(self):
        """A single edge: both nodes have degree 1 < 2, so CC = 0."""
        adj = _make_adj(("A", "B"))
        cc = TopologyAnalyzer.compute_clustering_coefficient(dict(adj))
        assert cc == pytest.approx(0.0)


class TestTopologyAnalyzerPathLength:

    def test_empty(self):
        assert TopologyAnalyzer.compute_avg_path_length({}) == 0.0

    def test_single_edge(self):
        adj = _make_adj(("A", "B"))
        apl = TopologyAnalyzer.compute_avg_path_length(dict(adj))
        # Only path is A->B and B->A, both length 1
        assert apl == pytest.approx(1.0)

    def test_path_graph_three_nodes(self):
        """A-B-C: paths are A->B=1, A->C=2, B->A=1, B->C=1, C->A=2, C->B=1."""
        adj = _make_adj(("A", "B"), ("B", "C"))
        apl = TopologyAnalyzer.compute_avg_path_length(dict(adj))
        # Average: (1+2+1+1+2+1)/6 = 8/6 = 1.333...
        assert apl == pytest.approx(8.0 / 6.0, abs=0.01)


class TestTopologyAnalyzerConnectedComponents:

    def test_empty(self):
        result = TopologyAnalyzer.compute_connected_components({}, set())
        assert result == []

    def test_single_component(self):
        adj = _make_adj(("A", "B"), ("B", "C"))
        nodes = _nodes_from_adj(adj)
        components = TopologyAnalyzer.compute_connected_components(dict(adj), nodes)
        assert len(components) == 1
        assert components[0] == {"A", "B", "C"}

    def test_two_components(self):
        adj = _make_adj(("A", "B"), ("C", "D"))
        nodes = _nodes_from_adj(adj)
        components = TopologyAnalyzer.compute_connected_components(dict(adj), nodes)
        assert len(components) == 2
        # Sorted by size descending; both size 2 so order may vary
        sizes = sorted([len(c) for c in components], reverse=True)
        assert sizes == [2, 2]

    def test_isolated_node_is_own_component(self):
        adj = _make_adj(("A", "B"))
        nodes = _nodes_from_adj(adj) | {"Z"}
        components = TopologyAnalyzer.compute_connected_components(dict(adj), nodes)
        assert len(components) == 2
        # Largest first
        assert len(components[0]) == 2
        assert len(components[1]) == 1
        assert components[1] == {"Z"}


class TestTopologyAnalyzerPowerLaw:

    def test_too_few_points(self):
        """Fewer than 3 data points returns (0, 0)."""
        gamma, r2 = TopologyAnalyzer.fit_power_law({1: 10, 2: 5})
        assert gamma == 0.0
        assert r2 == 0.0

    def test_degree_zero_filtered(self):
        """Degree 0 entries are filtered out."""
        gamma, r2 = TopologyAnalyzer.fit_power_law({0: 100, 1: 10, 2: 5})
        # Only 2 valid points after filtering degree 0 -> still too few
        assert gamma == 0.0
        assert r2 == 0.0

    def test_perfect_power_law(self):
        """P(k) = k^(-2): degree_counts = {1:100, 2:25, 4:6.25, ...}."""
        # Exact power law: count(k) = C * k^(-2)
        # Using integer approximations
        degree_counts = {1: 1000, 2: 250, 4: 62, 8: 15, 16: 4}
        gamma, r2 = TopologyAnalyzer.fit_power_law(degree_counts)
        # gamma should be close to 2.0
        assert 1.5 < gamma < 2.5
        assert r2 > 0.95


class TestTopologyAnalyzerShannonEntropy:

    def test_zero_nodes(self):
        entropy, max_ent = TopologyAnalyzer.shannon_degree_entropy({}, 0)
        assert entropy == 0.0
        assert max_ent == 0.0

    def test_uniform_distribution(self):
        """All nodes same degree: single bin, entropy = 0."""
        entropy, max_ent = TopologyAnalyzer.shannon_degree_entropy({2: 10}, 10)
        # One bin: p = 1.0, -1*log2(1) = 0
        assert entropy == pytest.approx(0.0)
        assert max_ent == pytest.approx(0.0)  # log2(1) = 0

    def test_two_equal_bins(self):
        """Two bins with equal count: entropy = 1 bit."""
        entropy, max_ent = TopologyAnalyzer.shannon_degree_entropy({1: 5, 2: 5}, 10)
        # p(1)=0.5, p(2)=0.5 -> H = -2*(0.5*log2(0.5)) = 1.0
        assert entropy == pytest.approx(1.0)
        assert max_ent == pytest.approx(1.0)  # log2(2) = 1


class TestTopologyAnalyzerAnalyze:

    def test_empty_graph(self):
        analyzer = TopologyAnalyzer()
        m = analyzer.analyze({}, set())
        assert m.node_count == 0
        assert m.edge_count == 0

    def test_triangle(self):
        """
        Triangle graph: 3 nodes, 3 edges.
        avg_degree = 2.0, density = 1.0, CC = 1.0.
        """
        adj = _make_adj(("A", "B"), ("B", "C"), ("A", "C"))
        nodes = _nodes_from_adj(adj)
        analyzer = TopologyAnalyzer()
        m = analyzer.analyze(dict(adj), nodes)

        assert m.node_count == 3
        assert m.edge_count == 3
        assert m.avg_degree == pytest.approx(2.0)
        assert m.max_degree == 2
        assert m.density == pytest.approx(1.0)
        assert m.avg_clustering == pytest.approx(1.0)
        assert m.num_components == 1
        assert m.largest_component_size == 3
        assert m.largest_component_fraction == pytest.approx(1.0)

    def test_disconnected(self):
        """Two disconnected edges: 2 components, no clustering."""
        adj = _make_adj(("A", "B"), ("C", "D"))
        nodes = _nodes_from_adj(adj)
        analyzer = TopologyAnalyzer()
        m = analyzer.analyze(dict(adj), nodes)

        assert m.node_count == 4
        assert m.edge_count == 2
        assert m.num_components == 2
        assert m.largest_component_size == 2


# ===========================================================================
# 6. SemanticLayerSeparator — classify_edge
# ===========================================================================


class TestClassifyEdge:

    @pytest.fixture()
    def separator(self) -> SemanticLayerSeparator:
        return SemanticLayerSeparator()

    # -- Known structural types --

    @pytest.mark.parametrize(
        "edge_type",
        [
            "PART_OF",
            "CONTAINS",
            "HAS_CHILD",
            "HAS_PARENT",
            "IN_DIRECTORY",
            "IN_FOLDER",
            "CHILD_OF",
            "PARENT_OF",
            "HAS_FILE",
            "BELONGS_TO",
        ],
    )
    def test_known_structural_types(self, separator, edge_type):
        assert separator.classify_edge(edge_type) == EdgeClassification.STRUCTURAL

    # -- Known semantic types --

    @pytest.mark.parametrize(
        "edge_type",
        [
            "RELATES_TO",
            "DEPENDS_ON",
            "REFERENCES",
            "SIMILAR_TO",
            "IMPLEMENTS",
            "EXTENDS",
            "USES",
            "IMPORTS",
            "CALLS",
            "INHERITS",
            "INSTANTIATES",
            "CONFIGURED_BY",
            "TESTED_BY",
            "VALIDATES",
            "CONTRADICTS",
            "SUPERSEDES",
            "COMPLEMENTS",
            "CO_OCCURS",
            "CAUSED_BY",
            "ENABLES",
            "CONSTRAINS",
            "DERIVES_FROM",
            "EXPORTS",
            "EMBEDS",
            "WRAPS",
            "DELEGATES_TO",
            "TRIGGERS",
            "MONITORS",
            "GATES",
            "PROVES",
            "DISPROVES",
        ],
    )
    def test_known_semantic_types(self, separator, edge_type):
        assert separator.classify_edge(edge_type) == EdgeClassification.SEMANTIC

    # -- Normalization: hyphens, spaces, lowercase --

    def test_normalize_hyphen(self, separator):
        assert separator.classify_edge("part-of") == EdgeClassification.STRUCTURAL

    def test_normalize_spaces(self, separator):
        assert separator.classify_edge("relates to") == EdgeClassification.SEMANTIC

    def test_normalize_mixed_case(self, separator):
        assert separator.classify_edge("Depends_On") == EdgeClassification.SEMANTIC

    # -- Heuristic fallback: containment signals --

    def test_heuristic_containment_part(self, separator):
        assert separator.classify_edge("IS_PART") == EdgeClassification.STRUCTURAL

    def test_heuristic_containment_child(self, separator):
        assert separator.classify_edge("MY_CHILD_NODE") == EdgeClassification.STRUCTURAL

    def test_heuristic_containment_folder(self, separator):
        assert separator.classify_edge("IN_SUB_FOLDER") == EdgeClassification.STRUCTURAL

    def test_heuristic_containment_file(self, separator):
        assert separator.classify_edge("LINKED_FILE") == EdgeClassification.STRUCTURAL

    def test_heuristic_containment_belong(self, separator):
        """'BELONG' signal triggers structural classification."""
        assert (
            separator.classify_edge("DOES_BELONG_HERE") == EdgeClassification.STRUCTURAL
        )

    # -- Heuristic fallback: relation signals --

    def test_heuristic_relation_relate(self, separator):
        assert separator.classify_edge("RELATE_VIA") == EdgeClassification.SEMANTIC

    def test_heuristic_relation_depend(self, separator):
        assert separator.classify_edge("SOFT_DEPEND") == EdgeClassification.SEMANTIC

    def test_heuristic_relation_refer(self, separator):
        assert separator.classify_edge("CROSS_REFER") == EdgeClassification.SEMANTIC

    def test_heuristic_relation_similar(self, separator):
        assert separator.classify_edge("VERY_SIMILAR") == EdgeClassification.SEMANTIC

    def test_heuristic_relation_impl(self, separator):
        assert separator.classify_edge("SOFT_IMPL") == EdgeClassification.SEMANTIC

    def test_heuristic_relation_use(self, separator):
        assert separator.classify_edge("REUSE") == EdgeClassification.SEMANTIC

    def test_heuristic_relation_call(self, separator):
        assert separator.classify_edge("CALLBACK") == EdgeClassification.SEMANTIC

    # -- Truly unknown -> AMBIGUOUS --

    def test_unknown_type(self, separator):
        assert separator.classify_edge("XYZZY_MAGIC") == EdgeClassification.AMBIGUOUS

    def test_empty_type(self, separator):
        assert separator.classify_edge("") == EdgeClassification.AMBIGUOUS

    # -- Custom types --

    def test_custom_structural(self):
        sep = SemanticLayerSeparator(custom_structural={"MOUNTED_ON"})
        assert sep.classify_edge("MOUNTED_ON") == EdgeClassification.STRUCTURAL

    def test_custom_semantic(self):
        sep = SemanticLayerSeparator(custom_semantic={"INSPIRED_BY"})
        assert sep.classify_edge("INSPIRED_BY") == EdgeClassification.SEMANTIC


# ===========================================================================
# 7. SemanticLayerSeparator — classify_from_json
# ===========================================================================


class TestClassifyFromJson:

    @pytest.fixture()
    def separator(self) -> SemanticLayerSeparator:
        return SemanticLayerSeparator()

    def test_empty_json(self, separator):
        graph = separator.classify_from_json({})
        assert graph.node_count == 0

    def test_nodes_only(self, separator):
        data = {
            "nodes": [
                {"id": "A", "type": "DOC"},
                {"id": "B", "type": "CHUNK"},
            ],
            "edges": [],
        }
        graph = separator.classify_from_json(data)
        assert graph.node_count == 2
        assert graph.structural_edge_count == 0
        assert graph.semantic_edge_count == 0

    def test_mixed_edges(self, separator):
        data = {
            "nodes": [
                {"id": "doc1", "type": "DOCUMENT"},
                {"id": "chunk1", "type": "CHUNK"},
                {"id": "chunk2", "type": "CHUNK"},
            ],
            "edges": [
                {"source": "chunk1", "target": "doc1", "type": "PART_OF"},
                {"source": "chunk2", "target": "doc1", "type": "PART_OF"},
                {"source": "chunk1", "target": "chunk2", "type": "RELATES_TO"},
            ],
        }
        graph = separator.classify_from_json(data)
        assert graph.node_count == 3
        assert graph.structural_edge_count == 2
        assert graph.semantic_edge_count == 1

    def test_alternative_field_names(self, separator):
        """The parser supports 'from'/'to' and 'name'/'label' alternatives."""
        data = {
            "nodes": [{"name": "N1", "label": "TYPE_A"}],
            "edges": [
                {"from": "N1", "to": "N2", "relationship": "USES", "score": 0.75},
            ],
        }
        graph = separator.classify_from_json(data)
        assert graph.node_count == 2  # N1 from node list + N2 from edge
        assert graph.semantic_edge_count == 1

        sem_edges = graph.get_semantic_edges()
        assert len(sem_edges) == 1
        assert sem_edges[0].source == "N1"
        assert sem_edges[0].target == "N2"
        assert sem_edges[0].weight == 0.75

    def test_metadata_passthrough(self, separator):
        """Extra edge fields beyond source/target/type/weight are stored as metadata."""
        data = {
            "nodes": [],
            "edges": [
                {
                    "source": "A",
                    "target": "B",
                    "type": "RELATES_TO",
                    "confidence": 0.95,
                    "provenance": "llm",
                },
            ],
        }
        graph = separator.classify_from_json(data)
        edge = graph.get_semantic_edges()[0]
        assert edge.metadata["confidence"] == 0.95
        assert edge.metadata["provenance"] == "llm"

    def test_ambiguous_edges_counted(self, separator):
        data = {
            "nodes": [],
            "edges": [
                {"source": "A", "target": "B", "type": "XYZZY_UNKNOWN"},
            ],
        }
        graph = separator.classify_from_json(data)
        assert graph.structural_edge_count == 0
        assert graph.semantic_edge_count == 0
        # Total edges = 1, but neither structural nor semantic
        assert len(graph.get_semantic_edges()) == 0
        assert len(graph.get_structural_edges()) == 0


# ===========================================================================
# 8. SemanticLayerSeparator — analyze_topology
# ===========================================================================


class TestAnalyzeTopology:

    def _build_sample_graph(self) -> DualOverlayGraph:
        """
        Build a small graph:
        - Structural: doc <- chunk1, doc <- chunk2, doc <- chunk3 (star)
        - Semantic: chunk1 -- chunk2 -- chunk3 -- chunk1 (triangle)
        """
        g = DualOverlayGraph()
        g.add_node("doc", "DOCUMENT")
        for i in range(1, 4):
            g.add_node(f"chunk{i}", "CHUNK")

        # Structural star
        for i in range(1, 4):
            g.add_edge(
                ClassifiedEdge(
                    f"chunk{i}", "doc", "PART_OF", EdgeClassification.STRUCTURAL
                )
            )

        # Semantic triangle
        g.add_edge(
            ClassifiedEdge(
                "chunk1", "chunk2", "RELATES_TO", EdgeClassification.SEMANTIC
            )
        )
        g.add_edge(
            ClassifiedEdge(
                "chunk2", "chunk3", "RELATES_TO", EdgeClassification.SEMANTIC
            )
        )
        g.add_edge(
            ClassifiedEdge(
                "chunk3", "chunk1", "RELATES_TO", EdgeClassification.SEMANTIC
            )
        )

        return g

    def test_report_edge_counts(self):
        graph = self._build_sample_graph()
        separator = SemanticLayerSeparator()
        report = separator.analyze_topology(graph)

        assert report.total_nodes == 4
        assert report.total_edges == 6  # 3 structural + 3 semantic
        assert report.structural_edges == 3
        assert report.semantic_edges == 3
        assert report.ambiguous_edges == 0

    def test_report_fractions(self):
        graph = self._build_sample_graph()
        separator = SemanticLayerSeparator()
        report = separator.analyze_topology(graph)

        assert report.structural_fraction == pytest.approx(0.5)
        assert report.semantic_fraction == pytest.approx(0.5)

    def test_report_has_timestamp(self):
        graph = self._build_sample_graph()
        separator = SemanticLayerSeparator()
        report = separator.analyze_topology(graph)

        assert report.timestamp.endswith("Z")
        assert len(report.timestamp) > 10

    def test_structural_topology_computed(self):
        graph = self._build_sample_graph()
        separator = SemanticLayerSeparator()
        report = separator.analyze_topology(graph)

        st = report.structural_topology
        assert st is not None
        assert st.node_count == 4  # All nodes in the graph
        assert st.edge_count == 3  # 3 structural edges

    def test_semantic_topology_computed(self):
        graph = self._build_sample_graph()
        separator = SemanticLayerSeparator()
        report = separator.analyze_topology(graph)

        se = report.semantic_topology
        assert se is not None
        # Semantic nodes: chunk1, chunk2, chunk3 (doc has no semantic edges)
        assert se.node_count == 3
        assert se.edge_count == 3  # triangle

    def test_combined_topology_computed(self):
        graph = self._build_sample_graph()
        separator = SemanticLayerSeparator()
        report = separator.analyze_topology(graph)

        ct = report.combined_topology
        assert ct is not None
        assert ct.node_count == 4

    def test_edge_type_counts(self):
        graph = self._build_sample_graph()
        separator = SemanticLayerSeparator()
        report = separator.analyze_topology(graph)

        assert report.edge_type_counts["PART_OF"] == 3
        assert report.edge_type_counts["RELATES_TO"] == 3

    def test_hub_nodes_identified(self):
        graph = self._build_sample_graph()
        separator = SemanticLayerSeparator()
        report = separator.analyze_topology(graph)

        # All 3 semantic nodes have degree 2 in semantic layer
        # top_n = max(10, 3//20) = 10, so all 3 are returned
        assert len(report.hub_nodes) == 3

    def test_snr_improvement(self):
        graph = self._build_sample_graph()
        separator = SemanticLayerSeparator()
        report = separator.analyze_topology(graph)

        # pre_separation_snr = max(0.3, 1.0 - 0.5) = 0.5
        assert report.pre_separation_snr == pytest.approx(0.5)
        # post should be computed and >= 0
        assert report.post_separation_snr > 0
        assert report.snr_improvement == pytest.approx(
            report.post_separation_snr - report.pre_separation_snr
        )

    def test_confidence_nonzero(self):
        graph = self._build_sample_graph()
        separator = SemanticLayerSeparator()
        report = separator.analyze_topology(graph)
        assert 0.0 < report.confidence <= 1.0

    def test_empty_graph_report(self):
        separator = SemanticLayerSeparator()
        graph = DualOverlayGraph()
        report = separator.analyze_topology(graph)

        assert report.total_nodes == 0
        assert report.total_edges == 0
        assert report.structural_fraction == 0.0
        assert report.semantic_fraction == 0.0

    def test_report_to_dict(self):
        graph = self._build_sample_graph()
        separator = SemanticLayerSeparator()
        report = separator.analyze_topology(graph)
        d = report.to_dict()

        assert isinstance(d, dict)
        assert "structural_topology" in d
        assert "semantic_topology" in d
        assert "combined_topology" in d
        assert d["total_nodes"] == 4


# ===========================================================================
# 9. Confidence computation edge cases
# ===========================================================================


class TestComputeConfidence:

    def test_high_confidence_many_semantic_no_ambiguous(self):
        """Lots of semantic edges, no ambiguous -> high factor 1."""
        separator = SemanticLayerSeparator()
        report = GraphTopologyReport(
            semantic_edges=200,
            ambiguous_edges=0,
            total_edges=300,
            structural_edges=100,
            structural_fraction=0.333,
        )
        # We need semantic_topology for factor 3
        report.semantic_topology = TopologyMetrics(
            is_small_world=True, small_world_sigma=2.0
        )
        confidence = separator._compute_confidence(report)
        # Factor 1: 1.0, Factor 2: 1.0, Factor 3: 1.0, Factor 4: 0.5 (struct_frac=0.333)
        assert confidence == pytest.approx((1.0 + 1.0 + 1.0 + 0.5) / 4.0)

    def test_low_confidence_few_semantic_many_ambiguous(self):
        separator = SemanticLayerSeparator()
        report = GraphTopologyReport(
            semantic_edges=5,
            ambiguous_edges=50,
            total_edges=100,
            structural_edges=45,
            structural_fraction=0.45,
        )
        report.semantic_topology = TopologyMetrics(
            is_small_world=False, small_world_sigma=0.1
        )
        confidence = separator._compute_confidence(report)
        # Factor 1: 0.3, Factor 2: max(0.3, 1 - 0.5*5) = max(0.3, -1.5) = 0.3
        # Factor 3: 0.4, Factor 4: 0.5
        assert confidence < 0.5


# ===========================================================================
# 10. Constants sanity checks
# ===========================================================================


class TestConstants:

    def test_structural_types_are_frozenset(self):
        assert isinstance(STRUCTURAL_EDGE_TYPES, frozenset)
        assert len(STRUCTURAL_EDGE_TYPES) == 10

    def test_semantic_types_are_frozenset(self):
        assert isinstance(SEMANTIC_EDGE_TYPES, frozenset)
        assert len(SEMANTIC_EDGE_TYPES) >= 27

    def test_no_overlap_between_structural_and_semantic(self):
        overlap = STRUCTURAL_EDGE_TYPES & SEMANTIC_EDGE_TYPES
        assert overlap == frozenset(), f"Unexpected overlap: {overlap}"

    def test_small_world_threshold(self):
        assert SMALL_WORLD_SIGMA_THRESHOLD == 1.0

    def test_scale_free_range(self):
        assert SCALE_FREE_GAMMA_RANGE == (2.0, 3.0)


# ===========================================================================
# 11. Factory function
# ===========================================================================


class TestCreateSemanticSeparator:

    def test_default(self):
        sep = create_semantic_separator()
        assert isinstance(sep, SemanticLayerSeparator)
        assert sep.classify_edge("PART_OF") == EdgeClassification.STRUCTURAL

    def test_with_custom_types(self):
        sep = create_semantic_separator(
            custom_structural={"HOSTED_ON"},
            custom_semantic={"LINKED_WITH"},
        )
        assert sep.classify_edge("HOSTED_ON") == EdgeClassification.STRUCTURAL
        assert sep.classify_edge("LINKED_WITH") == EdgeClassification.SEMANTIC
