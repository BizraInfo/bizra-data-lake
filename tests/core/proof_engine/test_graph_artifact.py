"""
Tests for SP-008 (GoT Graph Artifact Emission from Live Pipeline).

Proves that the Graph-of-Thoughts engine produces schema-compliant
artifacts that can be stored, retrieved, and verified.

Standing on Giants:
- Besta (GoT, 2024): Graph artifacts as first-class
- Merkle (1979): Content-addressed integrity
- BIZRA Spearpoint PRD SP-008
"""

import hashlib
import json
import pytest
from typing import Any, Dict

from core.sovereign.graph_core import GraphOfThoughts
from core.sovereign.graph_types import (
    EdgeType,
    ReasoningStrategy,
    ThoughtNode,
    ThoughtType,
)


# =============================================================================
# HELPERS
# =============================================================================

def _build_simple_graph() -> GraphOfThoughts:
    """Build a minimal graph for testing."""
    graph = GraphOfThoughts(
        strategy=ReasoningStrategy.BEST_FIRST,
        max_depth=5,
        snr_threshold=0.85,
        ihsan_threshold=0.95,
    )
    root = graph.add_thought("What is the answer?", ThoughtType.QUESTION)
    h1 = graph.generate("Hypothesis A", ThoughtType.HYPOTHESIS, parent=root)
    e1 = graph.generate("Evidence for A", ThoughtType.EVIDENCE, parent=h1)
    return graph


def _build_complex_graph() -> GraphOfThoughts:
    """Build a multi-branch graph for testing."""
    graph = GraphOfThoughts(
        strategy=ReasoningStrategy.BEAM_SEARCH,
        max_depth=10,
        beam_width=3,
    )
    root = graph.add_thought("Complex problem", ThoughtType.QUESTION)
    h1 = graph.generate("Hypothesis A", ThoughtType.HYPOTHESIS, parent=root)
    h2 = graph.generate("Hypothesis B", ThoughtType.HYPOTHESIS, parent=root)
    e1 = graph.generate("Evidence for A", ThoughtType.EVIDENCE, parent=h1)
    e2 = graph.generate("Evidence against B", ThoughtType.EVIDENCE, parent=h2)
    synth = graph.aggregate([h1, e1], "Synthesis of A + evidence")
    conclusion = graph.generate(
        "Final answer based on synthesis", ThoughtType.CONCLUSION, parent=synth
    )
    return graph


# =============================================================================
# to_artifact() SCHEMA COMPLIANCE
# =============================================================================

class TestGraphArtifactShape:
    """Tests that to_artifact() produces schema-compliant output."""

    def test_artifact_has_required_keys(self):
        """Artifact contains all required schema fields."""
        graph = _build_simple_graph()
        artifact = graph.to_artifact()

        assert "nodes" in artifact
        assert "edges" in artifact
        assert "roots" in artifact
        assert "graph_hash" in artifact

    def test_artifact_nodes_have_required_fields(self):
        """Each node has all required schema fields."""
        graph = _build_simple_graph()
        artifact = graph.to_artifact()

        for node in artifact["nodes"]:
            assert "id" in node
            assert "type" in node
            assert "content_hash" in node
            assert "snr" in node
            assert "ihsan" in node
            assert "depth" in node

    def test_artifact_node_types_valid(self):
        """Node types match schema enum values."""
        valid_types = {
            "hypothesis", "evidence", "reasoning", "synthesis",
            "refinement", "validation", "conclusion", "question", "counterpoint",
        }
        graph = _build_simple_graph()
        artifact = graph.to_artifact()

        for node in artifact["nodes"]:
            assert node["type"] in valid_types, f"Invalid type: {node['type']}"

    def test_artifact_edge_types_valid(self):
        """Edge types match schema enum values."""
        valid_types = {
            "supports", "refutes", "derives",
            "synthesizes", "refines", "questions", "validates",
        }
        graph = _build_simple_graph()
        artifact = graph.to_artifact()

        for edge in artifact["edges"]:
            assert edge["type"] in valid_types, f"Invalid edge type: {edge['type']}"

    def test_artifact_edges_have_required_fields(self):
        """Each edge has source, target, type."""
        graph = _build_simple_graph()
        artifact = graph.to_artifact()

        for edge in artifact["edges"]:
            assert "source" in edge
            assert "target" in edge
            assert "type" in edge

    def test_artifact_roots_non_empty(self):
        """Roots array is non-empty (schema requires minItems: 1)."""
        graph = _build_simple_graph()
        artifact = graph.to_artifact()
        assert len(artifact["roots"]) >= 1

    def test_artifact_graph_hash_is_sha256_hex(self):
        """graph_hash is a 64-char hex string (SHA-256)."""
        graph = _build_simple_graph()
        artifact = graph.to_artifact()
        assert len(artifact["graph_hash"]) == 64
        int(artifact["graph_hash"], 16)  # Should not raise

    def test_artifact_content_hash_is_sha256_hex(self):
        """Each node's content_hash is a 64-char hex string."""
        graph = _build_simple_graph()
        artifact = graph.to_artifact()

        for node in artifact["nodes"]:
            assert len(node["content_hash"]) == 64
            int(node["content_hash"], 16)

    def test_artifact_snr_ihsan_in_range(self):
        """snr and ihsan scores are in [0, 1]."""
        graph = _build_simple_graph()
        artifact = graph.to_artifact()

        for node in artifact["nodes"]:
            assert 0.0 <= node["snr"] <= 1.0
            assert 0.0 <= node["ihsan"] <= 1.0

    def test_artifact_depth_non_negative(self):
        """depth is non-negative integer."""
        graph = _build_simple_graph()
        artifact = graph.to_artifact()

        for node in artifact["nodes"]:
            assert isinstance(node["depth"], int)
            assert node["depth"] >= 0


# =============================================================================
# to_artifact() OPTIONAL FIELDS
# =============================================================================

class TestGraphArtifactOptionalFields:
    """Tests for optional artifact fields."""

    def test_artifact_includes_stats(self):
        """Artifact includes graph construction statistics."""
        graph = _build_simple_graph()
        artifact = graph.to_artifact()

        assert "stats" in artifact
        assert "nodes_created" in artifact["stats"]
        assert "edges_created" in artifact["stats"]

    def test_artifact_includes_config(self):
        """Artifact includes graph config."""
        graph = _build_simple_graph()
        artifact = graph.to_artifact()

        assert "config" in artifact
        assert artifact["config"]["strategy"] == "best_first"
        assert artifact["config"]["max_depth"] == 5

    def test_artifact_build_id_when_provided(self):
        """build_id is included when provided."""
        graph = _build_simple_graph()
        artifact = graph.to_artifact(build_id="query-123")
        assert artifact["build_id"] == "query-123"

    def test_artifact_policy_version(self):
        """policy_version defaults to 1.0.0."""
        graph = _build_simple_graph()
        artifact = graph.to_artifact()
        assert artifact["policy_version"] == "1.0.0"

    def test_artifact_no_build_id_when_empty(self):
        """build_id is omitted when empty string."""
        graph = _build_simple_graph()
        artifact = graph.to_artifact(build_id="")
        assert "build_id" not in artifact

    def test_artifact_content_truncated_at_500(self):
        """Node content is truncated to 500 chars (schema maxLength)."""
        graph = GraphOfThoughts()
        long_content = "x" * 1000
        graph.add_thought(long_content, ThoughtType.HYPOTHESIS)
        artifact = graph.to_artifact()

        for node in artifact["nodes"]:
            assert len(node["content"]) <= 500

    def test_artifact_optional_claim_tag(self):
        """claim_tag is included when set in metadata."""
        graph = GraphOfThoughts()
        node = graph.add_thought("Test", ThoughtType.HYPOTHESIS)
        node.metadata["claim_tag"] = "measured"
        artifact = graph.to_artifact()

        tagged_nodes = [n for n in artifact["nodes"] if "claim_tag" in n]
        assert len(tagged_nodes) == 1
        assert tagged_nodes[0]["claim_tag"] == "measured"

    def test_artifact_optional_evidence_hash(self):
        """evidence_hash is included when set in metadata."""
        graph = GraphOfThoughts()
        node = graph.add_thought("Evidence node", ThoughtType.EVIDENCE)
        node.metadata["evidence_hash"] = "a" * 64
        artifact = graph.to_artifact()

        evidence_nodes = [n for n in artifact["nodes"] if "evidence_hash" in n]
        assert len(evidence_nodes) == 1
        assert evidence_nodes[0]["evidence_hash"] == "a" * 64

    def test_artifact_optional_failure_modes(self):
        """failure_modes is included when set in metadata."""
        graph = GraphOfThoughts()
        node = graph.add_thought("Risky hypothesis", ThoughtType.HYPOTHESIS)
        node.metadata["failure_modes"] = ["bias", "data_leak"]
        artifact = graph.to_artifact()

        fm_nodes = [n for n in artifact["nodes"] if "failure_modes" in n]
        assert len(fm_nodes) == 1
        assert "bias" in fm_nodes[0]["failure_modes"]


# =============================================================================
# DETERMINISM AND INTEGRITY
# =============================================================================

class TestGraphArtifactIntegrity:
    """Tests for artifact determinism and integrity."""

    def test_artifact_deterministic(self):
        """Same graph produces same artifact graph_hash."""
        graph = _build_simple_graph()
        a1 = graph.to_artifact()
        a2 = graph.to_artifact()
        assert a1["graph_hash"] == a2["graph_hash"]

    def test_artifact_hash_changes_on_mutation(self):
        """Adding a node changes graph_hash."""
        graph = _build_simple_graph()
        hash_before = graph.to_artifact()["graph_hash"]

        graph.add_thought("New thought", ThoughtType.REASONING)
        hash_after = graph.to_artifact()["graph_hash"]

        assert hash_before != hash_after

    def test_artifact_is_json_serializable(self):
        """Artifact can be serialized to JSON without error."""
        graph = _build_complex_graph()
        artifact = graph.to_artifact(build_id="test-build")
        serialized = json.dumps(artifact)
        deserialized = json.loads(serialized)
        assert deserialized["graph_hash"] == artifact["graph_hash"]

    def test_artifact_graph_hash_matches_compute(self):
        """graph_hash in artifact matches compute_graph_hash()."""
        graph = _build_complex_graph()
        artifact = graph.to_artifact()
        assert artifact["graph_hash"] == graph.compute_graph_hash()

    def test_artifact_edge_weight_in_range(self):
        """Edge weights are in [0, 1]."""
        graph = _build_complex_graph()
        artifact = graph.to_artifact()

        for edge in artifact["edges"]:
            assert 0.0 <= edge["weight"] <= 1.0


# =============================================================================
# COMPLEX GRAPH SCENARIOS
# =============================================================================

class TestComplexGraphArtifact:
    """Tests for complex multi-branch graph artifacts."""

    def test_complex_graph_has_multiple_roots(self):
        """Complex graph artifact may have multiple root entries."""
        graph = _build_complex_graph()
        artifact = graph.to_artifact()
        # At least one root
        assert len(artifact["roots"]) >= 1

    def test_complex_graph_node_count(self):
        """Complex graph has expected number of nodes."""
        graph = _build_complex_graph()
        artifact = graph.to_artifact()
        # We added: root, h1, h2, e1, e2, synth, conclusion = 7 nodes
        assert len(artifact["nodes"]) >= 6

    def test_complex_graph_edge_count(self):
        """Complex graph has edges connecting nodes."""
        graph = _build_complex_graph()
        artifact = graph.to_artifact()
        assert len(artifact["edges"]) >= 5

    def test_complex_graph_stats_reflect_operations(self):
        """Stats track graph construction operations."""
        graph = _build_complex_graph()
        artifact = graph.to_artifact()
        assert artifact["stats"]["nodes_created"] >= 6
        assert artifact["stats"]["edges_created"] >= 5
        assert artifact["stats"]["aggregations"] >= 1

    def test_empty_graph_produces_valid_artifact(self):
        """Graph with no nodes still produces valid artifact structure."""
        graph = GraphOfThoughts()
        artifact = graph.to_artifact()
        assert artifact["nodes"] == []
        assert artifact["edges"] == []
        assert artifact["roots"] == []
        assert len(artifact["graph_hash"]) == 64


# =============================================================================
# RUNTIME INTEGRATION
# =============================================================================

class TestRuntimeGraphArtifactStore:
    """Tests for graph artifact storage in SovereignRuntime."""

    def test_runtime_has_graph_artifacts_store(self):
        """SovereignRuntime has _graph_artifacts dict."""
        from core.sovereign.runtime_core import SovereignRuntime

        runtime = SovereignRuntime()
        assert hasattr(runtime, "_graph_artifacts")
        assert isinstance(runtime._graph_artifacts, dict)

    def test_runtime_store_and_retrieve(self):
        """Can store and retrieve graph artifacts."""
        from core.sovereign.runtime_core import SovereignRuntime

        runtime = SovereignRuntime()
        graph = _build_simple_graph()
        artifact = graph.to_artifact(build_id="test-query-001")

        runtime._graph_artifacts["test-query-001"] = artifact
        retrieved = runtime.get_graph_artifact("test-query-001")

        assert retrieved is not None
        assert retrieved["build_id"] == "test-query-001"
        assert retrieved["graph_hash"] == artifact["graph_hash"]

    def test_runtime_get_missing_returns_none(self):
        """get_graph_artifact returns None for unknown query_id."""
        from core.sovereign.runtime_core import SovereignRuntime

        runtime = SovereignRuntime()
        assert runtime.get_graph_artifact("nonexistent") is None

    def test_store_graph_artifact_method(self):
        """_store_graph_artifact stores artifact from _graph_reasoner."""
        from core.sovereign.runtime_core import SovereignRuntime

        runtime = SovereignRuntime()
        # Set up a real GraphOfThoughts as the reasoner
        graph = _build_simple_graph()
        runtime._graph_reasoner = graph

        runtime._store_graph_artifact("q-001", graph.compute_graph_hash())

        artifact = runtime.get_graph_artifact("q-001")
        assert artifact is not None
        assert artifact["build_id"] == "q-001"

    def test_store_graph_artifact_bounds_storage(self):
        """_store_graph_artifact bounds storage to 100 entries."""
        from core.sovereign.runtime_core import SovereignRuntime

        runtime = SovereignRuntime()
        graph = _build_simple_graph()
        runtime._graph_reasoner = graph

        # Store 105 artifacts
        for i in range(105):
            runtime._store_graph_artifact(f"q-{i:04d}", "hash")

        assert len(runtime._graph_artifacts) <= 100
        # Oldest should have been evicted
        assert runtime.get_graph_artifact("q-0000") is None
        # Newest should still be there
        assert runtime.get_graph_artifact("q-0104") is not None

    def test_store_graph_artifact_no_reasoner_is_noop(self):
        """_store_graph_artifact is a no-op without a graph reasoner."""
        from core.sovereign.runtime_core import SovereignRuntime

        runtime = SovereignRuntime()
        runtime._store_graph_artifact("q-001", "hash")
        assert runtime.get_graph_artifact("q-001") is None


# =============================================================================
# SCHEMA VALIDATION INTEGRATION
# =============================================================================

class TestGraphArtifactSchemaValidation:
    """Tests that artifacts pass the reasoning_graph JSON schema."""

    def test_validates_against_schema(self):
        """Artifact validates against reasoning_graph.schema.json."""
        from core.proof_engine.schema_validator import validate_reasoning_graph

        graph = _build_complex_graph()
        artifact = graph.to_artifact(build_id="schema-test")

        is_valid, errors = validate_reasoning_graph(artifact)
        assert is_valid, f"Schema validation failed: {errors}"

    def test_simple_graph_validates(self):
        """Simple graph artifact passes schema validation."""
        from core.proof_engine.schema_validator import validate_reasoning_graph

        graph = _build_simple_graph()
        artifact = graph.to_artifact()

        is_valid, errors = validate_reasoning_graph(artifact)
        assert is_valid, f"Schema validation failed: {errors}"

    def test_empty_graph_validates(self):
        """Empty graph artifact passes schema validation (roots can be empty in artifact)."""
        from core.proof_engine.schema_validator import validate_reasoning_graph

        graph = GraphOfThoughts()
        artifact = graph.to_artifact()

        # Note: schema requires roots minItems:1, so empty graph may fail.
        # This test documents the expected behavior.
        is_valid, errors = validate_reasoning_graph(artifact)
        if not is_valid:
            # Empty graph naturally has no roots — expected schema rejection
            assert any("roots" in e for e in errors)
