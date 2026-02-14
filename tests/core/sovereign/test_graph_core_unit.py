"""
Tests for GraphOfThoughts — Core Unit Tests
=============================================
Comprehensive unit tests for the Graph-of-Thoughts reasoning engine,
covering graph creation, thought operations, hashing, artifact generation,
serialization, signing, and visualization.

Standing on Giants:
- Besta et al. (2024): Graph of Thoughts operations
- Merkle (1979): Content-addressed integrity
- Bernstein (2011): Ed25519 signatures
"""

import re

import pytest

from core.sovereign.graph_core import GraphOfThoughts
from core.sovereign.graph_types import (
    EdgeType,
    ReasoningStrategy,
    ThoughtEdge,
    ThoughtNode,
    ThoughtType,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def empty_graph() -> GraphOfThoughts:
    """Return a fresh, empty GraphOfThoughts instance with default config."""
    return GraphOfThoughts()


@pytest.fixture
def seeded_graph() -> GraphOfThoughts:
    """Return a graph pre-populated with a small reasoning tree.

    Structure:
        [Question] root
            |-- [Hypothesis] hyp_a
            |       |-- [Evidence] ev_a
            |-- [Hypothesis] hyp_b
    """
    g = GraphOfThoughts()
    root = g.add_thought("What causes X?", ThoughtType.QUESTION)
    hyp_a = g.generate("Hypothesis A explains X", ThoughtType.HYPOTHESIS, parent=root)
    g.generate("Evidence supporting A", ThoughtType.EVIDENCE, parent=hyp_a)
    g.generate("Hypothesis B explains X", ThoughtType.HYPOTHESIS, parent=root)
    return g


# ===========================================================================
# 1. TestGraphCreation
# ===========================================================================


class TestGraphCreation:
    """Tests for GraphOfThoughts instantiation and default configuration."""

    def test_empty_graph(self, empty_graph: GraphOfThoughts):
        """An empty graph has no nodes, edges, or roots."""
        assert len(empty_graph.nodes) == 0
        assert len(empty_graph.edges) == 0
        assert len(empty_graph.roots) == 0
        assert empty_graph.stats["nodes_created"] == 0
        assert empty_graph.stats["edges_created"] == 0

    def test_default_strategy(self, empty_graph: GraphOfThoughts):
        """Default reasoning strategy is BEST_FIRST."""
        assert empty_graph.strategy == ReasoningStrategy.BEST_FIRST

    def test_custom_config(self):
        """Custom thresholds and strategy are honoured."""
        g = GraphOfThoughts(
            strategy=ReasoningStrategy.DEPTH_FIRST,
            max_depth=20,
            beam_width=10,
            snr_threshold=0.90,
            ihsan_threshold=0.98,
        )
        assert g.strategy == ReasoningStrategy.DEPTH_FIRST
        assert g.max_depth == 20
        assert g.beam_width == 10
        assert g.snr_threshold == 0.90
        assert g.ihsan_threshold == 0.98

    def test_thoughts_alias(self, empty_graph: GraphOfThoughts):
        """The `thoughts` property is an alias for the `nodes` dict."""
        assert empty_graph.thoughts is empty_graph.nodes
        # After adding a node, both views reflect the change.
        node = empty_graph.add_thought("test", ThoughtType.QUESTION)
        assert node.id in empty_graph.thoughts
        assert empty_graph.thoughts[node.id] is empty_graph.nodes[node.id]


# ===========================================================================
# 2. TestThoughtOperations
# ===========================================================================


class TestThoughtOperations:
    """Tests for adding, generating, and aggregating thought nodes."""

    def test_add_thought_creates_node(self, empty_graph: GraphOfThoughts):
        """add_thought should insert a node into the graph and update stats."""
        node = empty_graph.add_thought("Initial question", ThoughtType.QUESTION)
        assert node.id in empty_graph.nodes
        assert node.content == "Initial question"
        assert node.thought_type == ThoughtType.QUESTION
        assert empty_graph.stats["nodes_created"] == 1

    def test_add_thought_with_parent(self, empty_graph: GraphOfThoughts):
        """A thought with a parent_id creates an edge and is not a root."""
        root = empty_graph.add_thought("Root", ThoughtType.QUESTION)
        child = empty_graph.add_thought(
            "Child hypothesis",
            ThoughtType.HYPOTHESIS,
            parent_id=root.id,
        )
        assert child.id not in empty_graph.roots
        assert root.id in empty_graph.roots
        assert len(empty_graph.edges) == 1
        edge = empty_graph.edges[0]
        assert edge.source_id == root.id
        assert edge.target_id == child.id

    def test_generate_creates_edge(self, empty_graph: GraphOfThoughts):
        """generate() with a parent ThoughtNode creates a DERIVES edge."""
        root = empty_graph.add_thought("Root Q", ThoughtType.QUESTION)
        child = empty_graph.generate(
            "Hypothesis from root",
            ThoughtType.HYPOTHESIS,
            parent=root,
        )
        assert child.id in empty_graph.nodes
        assert len(empty_graph.edges) == 1
        edge = empty_graph.edges[0]
        assert edge.source_id == root.id
        assert edge.target_id == child.id
        assert edge.edge_type == EdgeType.DERIVES

    def test_aggregate_merges_nodes(self, seeded_graph: GraphOfThoughts):
        """aggregate() produces a SYNTHESIS node with SYNTHESIZES edges."""
        # Collect the two hypothesis nodes.
        hypotheses = [
            n
            for n in seeded_graph.nodes.values()
            if n.thought_type == ThoughtType.HYPOTHESIS
        ]
        assert len(hypotheses) == 2, "seeded_graph fixture should have 2 hypotheses"

        synth = seeded_graph.aggregate(hypotheses, "Combined conclusion from A and B")
        assert synth.thought_type == ThoughtType.SYNTHESIS
        assert synth.id in seeded_graph.nodes

        # Each hypothesis should have a SYNTHESIZES edge to the synthesis node.
        synth_edges = [
            e
            for e in seeded_graph.edges
            if e.target_id == synth.id and e.edge_type == EdgeType.SYNTHESIZES
        ]
        assert len(synth_edges) == len(hypotheses)
        assert seeded_graph.stats["aggregations"] >= 1

    def test_node_depth_tracking(self, empty_graph: GraphOfThoughts):
        """Depth increments correctly across a chain of thoughts."""
        root = empty_graph.add_thought("Root", ThoughtType.QUESTION)
        assert root.depth == 0

        child = empty_graph.generate("Depth 1", ThoughtType.HYPOTHESIS, parent=root)
        assert child.depth == 1

        grandchild = empty_graph.generate("Depth 2", ThoughtType.EVIDENCE, parent=child)
        assert grandchild.depth == 2

    def test_root_tracking(self, empty_graph: GraphOfThoughts):
        """Only parentless nodes appear in the roots list."""
        r1 = empty_graph.add_thought("Root 1", ThoughtType.QUESTION)
        r2 = empty_graph.add_thought("Root 2", ThoughtType.QUESTION)
        _child = empty_graph.generate("Child", ThoughtType.HYPOTHESIS, parent=r1)

        assert r1.id in empty_graph.roots
        assert r2.id in empty_graph.roots
        assert _child.id not in empty_graph.roots
        assert len(empty_graph.roots) == 2


# ===========================================================================
# 3. TestGraphHash
# ===========================================================================


class TestGraphHash:
    """Tests for compute_graph_hash determinism and uniqueness."""

    def test_hash_deterministic(self, seeded_graph: GraphOfThoughts):
        """The same graph must produce the same hash every time."""
        first_hash = seeded_graph.compute_graph_hash()
        for _ in range(100):
            assert seeded_graph.compute_graph_hash() == first_hash

    def test_hash_changes_with_content(self, empty_graph: GraphOfThoughts):
        """Different content must yield different hashes."""
        empty_graph.add_thought("Content A", ThoughtType.QUESTION)
        hash_a = empty_graph.compute_graph_hash()

        # Build a second independent graph with different content.
        g2 = GraphOfThoughts()
        g2.add_thought("Content B", ThoughtType.QUESTION)
        hash_b = g2.compute_graph_hash()

        assert hash_a != hash_b

    def test_hash_is_sha256_hex(self, seeded_graph: GraphOfThoughts):
        """The hash must be a 64-character lowercase hex string (SHA-256)."""
        h = seeded_graph.compute_graph_hash()
        assert len(h) == 64
        assert re.fullmatch(r"[0-9a-f]{64}", h) is not None

    def test_empty_graph_hash(self, empty_graph: GraphOfThoughts):
        """An empty graph still produces a valid SHA-256 hash."""
        h = empty_graph.compute_graph_hash()
        assert len(h) == 64
        assert re.fullmatch(r"[0-9a-f]{64}", h) is not None


# ===========================================================================
# 4. TestGraphArtifact
# ===========================================================================


class TestGraphArtifact:
    """Tests for to_artifact() schema conformance."""

    def test_to_artifact_has_required_fields(self, seeded_graph: GraphOfThoughts):
        """Artifact dict must contain the core schema fields."""
        artifact = seeded_graph.to_artifact()
        required_keys = {"nodes", "edges", "roots", "graph_hash", "stats", "config"}
        assert required_keys.issubset(artifact.keys())

    def test_to_artifact_includes_build_id(self, seeded_graph: GraphOfThoughts):
        """When build_id is supplied it appears in the artifact."""
        artifact = seeded_graph.to_artifact(build_id="build-42")
        assert artifact["build_id"] == "build-42"

        # When build_id is empty it should NOT appear.
        artifact_no_build = seeded_graph.to_artifact(build_id="")
        assert "build_id" not in artifact_no_build

    def test_to_artifact_nodes_serialized(self, seeded_graph: GraphOfThoughts):
        """Each node in the artifact has the required node fields."""
        artifact = seeded_graph.to_artifact()
        assert len(artifact["nodes"]) == len(seeded_graph.nodes)
        node_required_keys = {"id", "content", "type", "content_hash", "confidence", "snr", "ihsan", "depth"}
        for node_dict in artifact["nodes"]:
            assert node_required_keys.issubset(node_dict.keys()), (
                f"Missing keys in node: {node_required_keys - node_dict.keys()}"
            )

    def test_to_artifact_edges_serialized(self, seeded_graph: GraphOfThoughts):
        """Each edge in the artifact has source, target, type, weight."""
        artifact = seeded_graph.to_artifact()
        assert len(artifact["edges"]) == len(seeded_graph.edges)
        edge_required_keys = {"source", "target", "type", "weight"}
        for edge_dict in artifact["edges"]:
            assert edge_required_keys.issubset(edge_dict.keys())

    def test_artifact_graph_hash_matches(self, seeded_graph: GraphOfThoughts):
        """The graph_hash in the artifact must match compute_graph_hash()."""
        artifact = seeded_graph.to_artifact()
        assert artifact["graph_hash"] == seeded_graph.compute_graph_hash()


# ===========================================================================
# 5. TestGraphSerialization
# ===========================================================================


class TestGraphSerialization:
    """Tests for to_dict() completeness."""

    def test_to_dict_complete(self, seeded_graph: GraphOfThoughts):
        """to_dict must include nodes, edges, roots, stats, config, graph_hash."""
        d = seeded_graph.to_dict()
        expected_keys = {"nodes", "edges", "roots", "stats", "config", "graph_hash"}
        assert expected_keys.issubset(d.keys())

    def test_to_dict_includes_config(self, empty_graph: GraphOfThoughts):
        """Config section must contain strategy, max_depth, and thresholds."""
        d = empty_graph.to_dict()
        config = d["config"]
        assert "strategy" in config
        assert "max_depth" in config
        assert "snr_threshold" in config
        assert "ihsan_threshold" in config
        assert config["strategy"] == ReasoningStrategy.BEST_FIRST.value

    def test_to_dict_includes_stats(self, seeded_graph: GraphOfThoughts):
        """Stats section must reflect actual graph operations."""
        d = seeded_graph.to_dict()
        stats = d["stats"]
        assert "nodes_created" in stats
        assert "edges_created" in stats
        assert stats["nodes_created"] >= 4  # root + 2 hyps + 1 evidence
        assert stats["edges_created"] >= 3  # 3 parent-child edges


# ===========================================================================
# 6. TestGraphSigning
# ===========================================================================


class TestGraphSigning:
    """Tests for Ed25519 signing of the graph hash."""

    def test_sign_graph_produces_signature(self, seeded_graph: GraphOfThoughts):
        """Signing with a valid key produces a 128-char hex Ed25519 signature."""
        from core.pci.crypto import generate_keypair

        private_hex, _public_hex = generate_keypair()
        signature = seeded_graph.sign_graph(private_hex)
        assert signature is not None
        # Ed25519 signatures are 64 bytes = 128 hex characters.
        assert len(signature) == 128
        assert re.fullmatch(r"[0-9a-f]{128}", signature) is not None

    def test_sign_graph_deterministic(self, seeded_graph: GraphOfThoughts):
        """The same key and unmodified graph must produce the same signature."""
        from core.pci.crypto import generate_keypair

        private_hex, _public_hex = generate_keypair()
        sig1 = seeded_graph.sign_graph(private_hex)
        sig2 = seeded_graph.sign_graph(private_hex)
        assert sig1 is not None
        assert sig1 == sig2

    def test_sign_graph_bad_key_returns_none(self, seeded_graph: GraphOfThoughts):
        """A malformed private key should return None (not raise)."""
        result = seeded_graph.sign_graph("not_a_valid_hex_key")
        assert result is None


# ===========================================================================
# 7. TestVisualization
# ===========================================================================


class TestVisualization:
    """Tests for ASCII visualization output."""

    def test_ascii_visualization_not_empty(self, seeded_graph: GraphOfThoughts):
        """Visualization of a non-empty graph must produce output."""
        viz = seeded_graph.visualize_ascii()
        assert isinstance(viz, str)
        assert len(viz) > 0
        assert "Graph of Thoughts" in viz

    def test_ascii_visualization_contains_nodes(self, seeded_graph: GraphOfThoughts):
        """Visualization must reference node content from the graph."""
        viz = seeded_graph.visualize_ascii()
        # The root question should appear (possibly truncated).
        assert "What causes X?" in viz or "What causes" in viz
        # Stats footer should show node/edge counts.
        assert "Nodes:" in viz
        assert "Edges:" in viz
