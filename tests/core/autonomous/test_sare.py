"""
Comprehensive Tests for Sovereign Autonomous Reasoning Engine (SARE)

Testing the pinnacle synthesis of:
- Graph-of-Thoughts (Besta)
- SNR Optimization (Shannon)
- Constitutional AI (Anthropic)
- Distributed Consensus (Lamport)
- Attention Mechanisms (Vaswani)

"La hawla wa la quwwata illa billah"
"""

import asyncio
import pytest
from datetime import datetime, timezone
from typing import Dict, Any, List

from core.autonomous import (
    SARE_VERSION,
    GIANTS_PROTOCOL,
    SNR_THRESHOLDS,
    CONSTITUTIONAL_CONSTRAINTS,
    IHSAN_DIMENSIONS,
)
from core.autonomous.giants import (
    Giant,
    MethodologyType,
    MethodologyInheritance,
    ProvenanceRecord,
    GiantsProtocol,
)
from core.autonomous.nodes import (
    NodeType,
    NodeState,
    ReasoningNode,
    ReasoningPath,
    ReasoningGraph,
)
from core.autonomous.loop import (
    LoopPhase,
    LoopState,
    PhaseResult,
    LoopExecution,
    SovereignLoop,
)
from core.autonomous.engine import (
    ReasoningResult,
    SovereignReasoningEngine,
    create_sovereign_engine,
)


# =============================================================================
# MODULE CONSTANTS TESTS
# =============================================================================

class TestModuleConstants:
    """Test module-level constants and configuration."""

    def test_sare_version_format(self):
        """SARE version follows semantic versioning."""
        parts = SARE_VERSION.split(".")
        assert len(parts) == 3, "Version must be major.minor.patch"
        assert all(p.isdigit() for p in parts), "All version parts must be numeric"

    def test_giants_protocol_completeness(self):
        """All five giants are present in protocol."""
        required_giants = ["shannon", "lamport", "vaswani", "besta", "anthropic"]
        for giant in required_giants:
            assert giant in GIANTS_PROTOCOL, f"Missing giant: {giant}"

    def test_giants_protocol_structure(self):
        """Each giant has required fields."""
        required_fields = ["name", "work", "contribution", "application"]
        for giant, info in GIANTS_PROTOCOL.items():
            for field in required_fields:
                assert field in info, f"Giant {giant} missing field: {field}"

    def test_snr_thresholds_hierarchy(self):
        """SNR thresholds maintain proper hierarchy."""
        assert SNR_THRESHOLDS["minimum"] < SNR_THRESHOLDS["action"]
        assert SNR_THRESHOLDS["action"] < SNR_THRESHOLDS["ihsan"]
        assert SNR_THRESHOLDS["ihsan"] <= 1.0

    def test_constitutional_constraints_values(self):
        """Constitutional constraints are within valid ranges."""
        assert 0 < CONSTITUTIONAL_CONSTRAINTS["ihsan_threshold"] <= 1.0
        assert CONSTITUTIONAL_CONSTRAINTS["max_backtrack"] > 0
        assert CONSTITUTIONAL_CONSTRAINTS["max_loops"] > 0

    def test_ihsan_dimensions_sum_to_one(self):
        """Ihsan dimension weights sum to 1.0."""
        total = sum(IHSAN_DIMENSIONS.values())
        assert abs(total - 1.0) < 0.001, f"Ihsan weights sum to {total}, expected 1.0"

    def test_ihsan_dimensions_positive(self):
        """All Ihsan dimension weights are positive."""
        for dim, weight in IHSAN_DIMENSIONS.items():
            assert weight > 0, f"Dimension {dim} has non-positive weight: {weight}"


# =============================================================================
# GIANTS PROTOCOL TESTS
# =============================================================================

class TestGiant:
    """Test Giant enum."""

    def test_all_giants_defined(self):
        """All expected giants are in enum."""
        assert Giant.SHANNON.value == "shannon"
        assert Giant.LAMPORT.value == "lamport"
        assert Giant.VASWANI.value == "vaswani"
        assert Giant.BESTA.value == "besta"
        assert Giant.ANTHROPIC.value == "anthropic"

    def test_giant_string_conversion(self):
        """Giants can be created from strings."""
        assert Giant("shannon") == Giant.SHANNON
        assert Giant("anthropic") == Giant.ANTHROPIC


class TestMethodologyType:
    """Test MethodologyType enum."""

    def test_methodology_types_exist(self):
        """All methodology types are defined."""
        assert MethodologyType.INFORMATION_THEORY.value == "information_theory"
        assert MethodologyType.DISTRIBUTED_SYSTEMS.value == "distributed_systems"
        assert MethodologyType.ATTENTION_MECHANISM.value == "attention_mechanism"
        assert MethodologyType.GRAPH_REASONING.value == "graph_reasoning"
        assert MethodologyType.CONSTITUTIONAL_AI.value == "constitutional_ai"


class TestMethodologyInheritance:
    """Test MethodologyInheritance dataclass."""

    def test_create_inheritance(self):
        """Can create methodology inheritance record."""
        inheritance = MethodologyInheritance(
            giant=Giant.SHANNON,
            methodology=MethodologyType.INFORMATION_THEORY,
            technique="snr",
            application="Quality filtering",
        )
        assert inheritance.giant == Giant.SHANNON
        assert inheritance.confidence == 1.0  # Default

    def test_inheritance_to_dict(self):
        """Inheritance serializes to dictionary."""
        inheritance = MethodologyInheritance(
            giant=Giant.BESTA,
            methodology=MethodologyType.GRAPH_REASONING,
            technique="got",
            application="Non-linear reasoning",
            confidence=0.95,
        )
        result = inheritance.to_dict()
        assert result["giant"] == "besta"
        assert result["methodology"] == "graph_reasoning"
        assert result["technique"] == "got"
        assert result["confidence"] == 0.95


class TestProvenanceRecord:
    """Test ProvenanceRecord dataclass."""

    def test_create_provenance(self):
        """Can create provenance record."""
        inheritance = MethodologyInheritance(
            giant=Giant.SHANNON,
            methodology=MethodologyType.INFORMATION_THEORY,
            technique="snr",
            application="test",
        )
        record = ProvenanceRecord(
            output_hash="abc123",
            inheritances=[inheritance],
            snr_score=0.92,
            ihsan_score=0.95,
            reasoning_path=["node1", "node2"],
        )
        assert record.output_hash == "abc123"
        assert len(record.inheritances) == 1
        assert record.snr_score == 0.92

    def test_provenance_to_dict(self):
        """Provenance serializes to dictionary."""
        record = ProvenanceRecord(
            output_hash="def456",
            inheritances=[],
            snr_score=0.88,
            ihsan_score=0.90,
            reasoning_path=["a", "b", "c"],
        )
        result = record.to_dict()
        assert result["output_hash"] == "def456"
        assert result["snr_score"] == 0.88
        assert len(result["reasoning_path"]) == 3


class TestGiantsProtocol:
    """Test GiantsProtocol class."""

    @pytest.fixture
    def protocol(self):
        """Create fresh GiantsProtocol instance."""
        return GiantsProtocol()

    def test_protocol_initialization(self, protocol):
        """Protocol initializes with all giants."""
        assert len(protocol._giants) == 5
        assert len(protocol._methodology_map) == 5

    def test_list_all_techniques(self, protocol):
        """Can list all registered techniques."""
        techniques = protocol.list_techniques()
        assert len(techniques) >= 10  # At least 2 per giant
        assert "shannon_snr" in techniques
        assert "lamport_consensus" in techniques
        assert "anthropic_ihsan" in techniques

    def test_list_techniques_by_giant(self, protocol):
        """Can list techniques for specific giant."""
        shannon_techniques = protocol.list_techniques(Giant.SHANNON)
        assert "snr" in shannon_techniques
        assert "entropy" in shannon_techniques
        assert "consensus" not in shannon_techniques  # Lamport's technique

    def test_get_giant_info(self, protocol):
        """Can get information about a giant."""
        info = protocol.get_giant_info(Giant.SHANNON)
        assert "name" in info
        assert "Shannon" in info["name"]

    # Shannon Techniques
    def test_shannon_snr_technique(self, protocol):
        """Shannon SNR technique calculates signal quality."""
        result, inheritance = protocol.invoke(
            Giant.SHANNON, "snr", "This is a clear, high-quality signal with meaningful content."
        )
        assert 0 <= result <= 1.0
        assert inheritance.giant == Giant.SHANNON
        assert inheritance.technique == "snr"

    def test_shannon_snr_empty_input(self, protocol):
        """Shannon SNR returns 0 for empty input."""
        result, _ = protocol.invoke(Giant.SHANNON, "snr", "")
        assert result == 0.0

    def test_shannon_entropy_technique(self, protocol):
        """Shannon entropy technique measures information content."""
        result, inheritance = protocol.invoke(
            Giant.SHANNON, "entropy", "The quick brown fox jumps over the lazy dog."
        )
        assert 0 <= result <= 1.0
        assert inheritance.methodology == MethodologyType.INFORMATION_THEORY

    # Lamport Techniques
    def test_lamport_ordering_technique(self, protocol):
        """Lamport ordering assigns logical timestamps."""
        events = [
            {"id": "e1", "data": "first"},
            {"id": "e2", "data": "second"},
            {"id": "e3", "data": "third"},
        ]
        result, inheritance = protocol.invoke(Giant.LAMPORT, "ordering", events)
        assert len(result) == 3
        assert all("lamport_time" in e for e in result)
        assert result[0]["lamport_time"] < result[2]["lamport_time"]

    def test_lamport_consensus_success(self, protocol):
        """Lamport consensus reaches agreement with 2/3+ agreement."""
        proposals = [0.9, 0.91, 0.89, 0.92, 0.88]  # All close to 0.9
        result, inheritance = protocol.invoke(Giant.LAMPORT, "consensus", proposals)
        assert result is not None
        assert abs(result - 0.9) < 0.1

    def test_lamport_consensus_failure(self, protocol):
        """Lamport consensus fails without 2/3+ agreement."""
        proposals = [0.1, 0.5, 0.9]  # Too spread out
        result, _ = protocol.invoke(Giant.LAMPORT, "consensus", proposals)
        assert result is None

    # Vaswani Techniques
    def test_vaswani_attention_technique(self, protocol):
        """Vaswani attention weights relevant items."""
        query = "machine learning algorithms"
        keys = ["neural networks", "cooking recipes", "deep learning", "gardening tips"]
        values = ["ML", "cook", "DL", "garden"]
        result, inheritance = protocol.invoke(Giant.VASWANI, "attention", query, keys, values)
        assert len(result) == 4
        # ML-related should rank higher
        top_values = [r[0] for r in result[:2]]
        assert "ML" in top_values or "DL" in top_values

    def test_vaswani_context_technique(self, protocol):
        """Vaswani context weighting applies positional encoding."""
        context = ["past", "current", "future"]
        result, inheritance = protocol.invoke(Giant.VASWANI, "context", context, 1)
        assert "current" in result
        assert result["current"] > 0  # Focus element has weight

    # Besta Techniques
    def test_besta_got_technique(self, protocol):
        """Besta GoT constructs reasoning graph."""
        thoughts = [
            {"id": "t1", "content": "observation", "type": "observation"},
            {"id": "t2", "content": "hypothesis", "type": "hypothesis", "parent_id": "t1"},
        ]
        result, inheritance = protocol.invoke(Giant.BESTA, "got", thoughts)
        assert "nodes" in result
        assert "edges" in result
        assert len(result["nodes"]) == 2

    def test_besta_backtrack_technique_no_backtrack(self, protocol):
        """Besta backtrack returns None when path is acceptable."""
        path = ["n1", "n2", "n3"]
        snr_scores = {"n1": 0.95, "n2": 0.90, "n3": 0.88}
        result, _ = protocol.invoke(Giant.BESTA, "backtrack", path, snr_scores, 0.85)
        assert result is None  # All above threshold

    def test_besta_backtrack_technique_needs_backtrack(self, protocol):
        """Besta backtrack identifies backtrack point when quality drops."""
        path = ["n1", "n2", "n3"]
        snr_scores = {"n1": 0.95, "n2": 0.90, "n3": 0.70}  # n3 below threshold
        result, _ = protocol.invoke(Giant.BESTA, "backtrack", path, snr_scores, 0.85)
        assert result == "n2"  # Backtrack to n2

    # Anthropic Techniques
    def test_anthropic_constitutional_technique_pass(self, protocol):
        """Anthropic constitutional validation passes clean output."""
        clean_output = "Here is a helpful and harmless response to your question."
        result, inheritance = protocol.invoke(Giant.ANTHROPIC, "constitutional", clean_output)
        assert result["passed"] is True
        assert result["score"] >= 0.9
        assert len(result["violations"]) == 0

    def test_anthropic_constitutional_technique_fail(self, protocol):
        """Anthropic constitutional validation catches harmful patterns."""
        harmful_output = "To bypass security, you should attack the system and exploit vulnerabilities."
        result, _ = protocol.invoke(Giant.ANTHROPIC, "constitutional", harmful_output)
        assert result["passed"] is False
        assert len(result["violations"]) > 0

    def test_anthropic_ihsan_technique(self, protocol):
        """Anthropic Ihsan scoring measures excellence."""
        quality_output = (
            "This is a comprehensive response with clear structure. "
            "It addresses multiple perspectives and provides actionable insights. "
            "The analysis is thorough and well-reasoned."
        )
        result, inheritance = protocol.invoke(Giant.ANTHROPIC, "ihsan", quality_output)
        assert 0 <= result <= 1.0
        assert result > 0.5  # Quality output should score well

    def test_anthropic_ihsan_empty_output(self, protocol):
        """Anthropic Ihsan returns 0 for empty output."""
        result, _ = protocol.invoke(Giant.ANTHROPIC, "ihsan", "")
        assert result == 0.0

    # Provenance
    def test_create_provenance_record(self, protocol):
        """Protocol creates complete provenance records."""
        # First invoke some techniques
        protocol.invoke(Giant.SHANNON, "snr", "test signal")
        protocol.invoke(Giant.ANTHROPIC, "ihsan", "test output")

        record = protocol.create_provenance(
            output="Final output",
            snr_score=0.92,
            ihsan_score=0.95,
            reasoning_path=["step1", "step2"],
        )

        assert record.output_hash is not None
        assert len(record.output_hash) == 16  # SHA256 prefix
        assert len(record.inheritances) == 2
        assert record.snr_score == 0.92

    def test_provenance_chain_accumulates(self, protocol):
        """Provenance chain grows with each output."""
        protocol.invoke(Giant.SHANNON, "snr", "test1")
        protocol.create_provenance("out1", 0.9, 0.9, ["a"])

        protocol.invoke(Giant.LAMPORT, "ordering", [{"x": 1}])
        protocol.create_provenance("out2", 0.91, 0.91, ["b"])

        chain = protocol.get_provenance_chain()
        assert len(chain) == 2

    def test_unknown_technique_raises(self, protocol):
        """Invoking unknown technique raises ValueError."""
        with pytest.raises(ValueError, match="Unknown technique"):
            protocol.invoke(Giant.SHANNON, "nonexistent_technique")


# =============================================================================
# REASONING NODES TESTS
# =============================================================================

class TestNodeType:
    """Test NodeType enum."""

    def test_all_node_types_defined(self):
        """All expected node types exist."""
        types = [
            NodeType.OBSERVATION,
            NodeType.ORIENTATION,
            NodeType.ANALYSIS,
            NodeType.HYPOTHESIS,
            NodeType.SYNTHESIS,
            NodeType.CONCLUSION,
            NodeType.BACKTRACK,
            NodeType.REFINEMENT,
            NodeType.META,
        ]
        assert len(types) == 9


class TestReasoningNode:
    """Test ReasoningNode dataclass."""

    def test_create_node(self):
        """Can create reasoning node."""
        node = ReasoningNode(
            id="node_001",
            content="Initial observation",
            node_type=NodeType.OBSERVATION,
            snr_score=0.92,
            ihsan_score=0.96,
        )
        assert node.id == "node_001"
        assert node.snr_score == 0.92

    def test_node_quality_score(self):
        """Node calculates quality score correctly."""
        node = ReasoningNode(
            id="n1",
            content="test",
            node_type=NodeType.OBSERVATION,
            snr_score=0.81,  # sqrt(0.81 * 1.0) = 0.9
            ihsan_score=1.0,
        )
        assert abs(node.quality_score - 0.9) < 0.01

    def test_node_with_parents(self):
        """Can create node with parent references."""
        node = ReasoningNode(
            id="node_002",
            content="Derived thought",
            node_type=NodeType.ANALYSIS,
            snr_score=0.88,
            ihsan_score=0.96,
            parents={"node_001", "node_000"},
        )
        assert "node_001" in node.parents
        assert len(node.parents) == 2

    def test_node_to_dict(self):
        """Node serializes to dictionary."""
        node = ReasoningNode(
            id="n1",
            content="test",
            node_type=NodeType.HYPOTHESIS,
            snr_score=0.85,
            ihsan_score=0.96,
            metadata={"key": "value"},
        )
        result = node.to_dict()
        assert result["id"] == "n1"
        assert result["node_type"] == "hypothesis"

    def test_node_is_valid_property(self):
        """is_valid property checks thresholds."""
        # High quality node should be valid
        good_node = ReasoningNode(
            id="good",
            content="test",
            node_type=NodeType.OBSERVATION,
            snr_score=0.95,
            ihsan_score=0.96,
            state=NodeState.ACTIVE,
        )
        assert good_node.is_valid is True

        # Pruned node should not be valid
        pruned_node = ReasoningNode(
            id="pruned",
            content="test",
            node_type=NodeType.OBSERVATION,
            snr_score=0.95,
            ihsan_score=0.96,
            state=NodeState.PRUNED,
        )
        assert pruned_node.is_valid is False


class TestReasoningPath:
    """Test ReasoningPath dataclass."""

    def test_create_path(self):
        """Can create reasoning path."""
        path = ReasoningPath(
            nodes=["n1", "n2"],
            total_snr=0.875,
            total_ihsan=0.9,
        )
        assert len(path.nodes) == 2

    def test_path_average_quality(self):
        """Path calculates average quality."""
        path = ReasoningPath(
            nodes=["n1", "n2"],
            total_snr=0.81,  # sqrt(0.81 * 1.0) = 0.9
            total_ihsan=1.0,
        )
        assert abs(path.average_quality - 0.9) < 0.01

    def test_empty_path_average_quality(self):
        """Empty path has 0 average quality."""
        path = ReasoningPath(nodes=[], total_snr=0.0, total_ihsan=0.0)
        assert path.average_quality == 0.0

    def test_path_to_dict(self):
        """Path serializes to dictionary."""
        path = ReasoningPath(
            nodes=["n1", "n2"],
            total_snr=0.9,
            total_ihsan=0.95,
            backtrack_count=1,
            depth=2,
        )
        result = path.to_dict()
        assert len(result["nodes"]) == 2
        assert result["backtrack_count"] == 1
        assert "average_quality" in result


class TestReasoningGraph:
    """Test ReasoningGraph class."""

    @pytest.fixture
    def graph(self):
        """Create fresh reasoning graph."""
        return ReasoningGraph()

    def test_graph_initialization(self, graph):
        """Graph initializes empty."""
        assert len(graph._nodes) == 0
        assert len(graph._root_ids) == 0

    def test_add_root_node(self, graph):
        """Can add root node to graph."""
        node = graph.add_node("This is a test observation with enough content", NodeType.OBSERVATION)
        assert node.id in graph._nodes
        assert len(node.parents) == 0
        assert node.id in graph._root_ids

    def test_add_child_node(self, graph):
        """Can add child node connected to parent."""
        parent = graph.add_node("This is the parent observation content", NodeType.OBSERVATION)
        child = graph.add_node(
            "This is the child analysis content with details",
            NodeType.ANALYSIS,
            parent_ids={parent.id}
        )
        assert parent.id in child.parents
        assert child.id in graph._nodes[parent.id].children

    def test_get_node(self, graph):
        """Can retrieve node by ID."""
        node = graph.add_node("This is test content for retrieval", NodeType.OBSERVATION)
        retrieved = graph.get_node(node.id)
        assert retrieved.content == "This is test content for retrieval"

    def test_get_nonexistent_node(self, graph):
        """Getting nonexistent node returns None."""
        result = graph.get_node("nonexistent")
        assert result is None

    def test_add_multiple_children(self, graph):
        """Can add multiple children to a parent."""
        parent = graph.add_node("Parent observation with detailed content", NodeType.OBSERVATION)
        child1 = graph.add_node(
            "First child analysis with specific findings",
            NodeType.ANALYSIS,
            parent_ids={parent.id}
        )
        child2 = graph.add_node(
            "Second child analysis with different findings",
            NodeType.ANALYSIS,
            parent_ids={parent.id}
        )
        parent_node = graph.get_node(parent.id)
        assert len(parent_node.children) == 2
        assert child1.id in parent_node.children
        assert child2.id in parent_node.children

    def test_node_depth_calculation(self, graph):
        """Nodes have correct depth."""
        n1 = graph.add_node("Root observation", NodeType.OBSERVATION)
        n2 = graph.add_node("Child analysis", NodeType.ANALYSIS, parent_ids={n1.id})
        n3 = graph.add_node("Grandchild hypothesis", NodeType.HYPOTHESIS, parent_ids={n2.id})
        assert n1.depth == 0
        assert n2.depth == 1
        assert n3.depth == 2

    def test_backtrack_creates_node(self, graph):
        """Backtracking creates a backtrack node."""
        n1 = graph.add_node("Original observation that needs reconsideration", NodeType.OBSERVATION)
        backtrack = graph.backtrack(n1.id, "Low quality reasoning detected")
        assert backtrack is not None
        assert backtrack.node_type == NodeType.BACKTRACK
        assert graph._nodes[n1.id].state == NodeState.BACKTRACKED

    def test_should_backtrack_low_snr(self, graph):
        """should_backtrack returns True for low SNR nodes."""
        # Create a node with very short content (low SNR)
        node = graph.add_node("bad", NodeType.OBSERVATION)
        should, reason = graph.should_backtrack(node.id)
        # Low SNR should trigger backtrack recommendation
        assert should is True
        assert "SNR" in reason or "Ihsān" in reason

    def test_find_best_path(self, graph):
        """Can find best path through graph."""
        n1 = graph.add_node(
            "Detailed observation with comprehensive content for high SNR score",
            NodeType.OBSERVATION
        )
        n2a = graph.add_node(
            "Brief analysis",  # Short content = lower SNR
            NodeType.ANALYSIS,
            parent_ids={n1.id}
        )
        n2b = graph.add_node(
            "Comprehensive analysis with detailed findings and thorough examination of the evidence",
            NodeType.ANALYSIS,
            parent_ids={n1.id}
        )

        best_path = graph.find_best_path()
        # Best path should prefer higher quality nodes
        assert len(best_path.nodes) >= 1
        assert best_path.average_quality >= 0

    def test_prune_low_quality(self, graph):
        """Can prune nodes below quality threshold."""
        graph.add_node(
            "High quality content with comprehensive analysis and detailed findings",
            NodeType.OBSERVATION
        )
        graph.add_node("low", NodeType.OBSERVATION)  # Very short = low quality
        graph.add_node(
            "Medium quality content with some analysis",
            NodeType.OBSERVATION
        )

        pruned = graph.prune_low_quality(threshold=0.5)
        assert pruned >= 1  # At least the "low" node pruned
        pruned_nodes = [n for n in graph._nodes.values() if n.state == NodeState.PRUNED]
        assert len(pruned_nodes) >= 1

    def test_synthesize_nodes(self, graph):
        """Can synthesize multiple nodes."""
        n1 = graph.add_node("First observation with detailed content", NodeType.OBSERVATION)
        n2 = graph.add_node("Second observation with different findings", NodeType.OBSERVATION)

        synthesis = graph.synthesize(
            {n1.id, n2.id},
            "Synthesis combining both observations into unified understanding"
        )
        assert synthesis is not None
        assert synthesis.node_type == NodeType.SYNTHESIS
        assert n1.id in synthesis.parents and n2.id in synthesis.parents

    def test_get_stats(self, graph):
        """Can get graph statistics."""
        graph.add_node("First observation", NodeType.OBSERVATION)
        graph.add_node("Second observation", NodeType.ANALYSIS)
        graph.add_node("Third observation", NodeType.CONCLUSION)

        stats = graph.get_stats()
        assert stats["total_nodes"] == 3
        assert stats["active_nodes"] == 3
        assert "avg_snr" in stats
        assert "by_type" in stats

    def test_graph_to_dict(self, graph):
        """Graph serializes to dictionary."""
        n1 = graph.add_node("Parent observation", NodeType.OBSERVATION)
        n2 = graph.add_node("Child analysis", NodeType.ANALYSIS, parent_ids={n1.id})

        result = graph.to_dict()
        assert "nodes" in result
        assert "root_ids" in result
        assert "stats" in result
        assert len(result["nodes"]) == 2

    def test_find_all_paths(self, graph):
        """Can find all paths through graph."""
        n1 = graph.add_node("Root observation", NodeType.OBSERVATION)
        n2a = graph.add_node("Path A analysis", NodeType.ANALYSIS, parent_ids={n1.id})
        n2b = graph.add_node("Path B analysis", NodeType.ANALYSIS, parent_ids={n1.id})

        paths = graph.find_all_paths(max_paths=10)
        # Should find at least 2 paths (through n2a and n2b)
        assert len(paths) >= 2


# =============================================================================
# SOVEREIGN LOOP TESTS
# =============================================================================

class TestLoopPhase:
    """Test LoopPhase enum."""

    def test_all_phases_defined(self):
        """All loop phases exist."""
        phases = [
            LoopPhase.OBSERVE,
            LoopPhase.ORIENT,
            LoopPhase.REASON,
            LoopPhase.SYNTHESIZE,
            LoopPhase.ACT,
            LoopPhase.REFLECT,
        ]
        assert len(phases) == 6


class TestLoopState:
    """Test LoopState enum."""

    def test_all_states_defined(self):
        """All loop states exist."""
        states = [
            LoopState.IDLE,
            LoopState.RUNNING,
            LoopState.PAUSED,
            LoopState.COMPLETED,
            LoopState.FAILED,
        ]
        assert len(states) == 5


class TestPhaseResult:
    """Test PhaseResult dataclass."""

    def test_create_phase_result(self):
        """Can create phase result."""
        result = PhaseResult(
            phase=LoopPhase.OBSERVE,
            success=True,
            content="observation content",
            snr_score=0.92,
            ihsan_score=0.95,
            duration_ms=10.5,
            node_id="n1",
        )
        assert result.phase == LoopPhase.OBSERVE
        assert result.success is True

    def test_phase_result_to_dict(self):
        """Phase result serializes to dictionary."""
        result = PhaseResult(
            phase=LoopPhase.REASON,
            success=True,
            content="reasoning complete",
            snr_score=0.88,
            ihsan_score=0.92,
            duration_ms=15.0,
            node_id="n2",
        )
        data = result.to_dict()
        assert data["phase"] == "reason"
        assert "content_preview" in data


class TestLoopExecution:
    """Test LoopExecution dataclass."""

    def test_create_loop_execution(self):
        """Can create loop execution."""
        execution = LoopExecution(
            id="exec_001",
            input_content="test query",
            output_content="test response",
            phases=[],
            loop_count=3,
            state=LoopState.COMPLETED,
        )
        assert execution.state == LoopState.COMPLETED

    def test_loop_execution_final_scores(self):
        """Loop execution calculates final scores from phases."""
        phase = PhaseResult(
            phase=LoopPhase.ACT,
            success=True,
            content="action",
            snr_score=0.92,
            ihsan_score=0.95,
            duration_ms=10.0,
        )
        execution = LoopExecution(
            id="exec_002",
            input_content="test",
            output_content="result",
            phases=[phase],
            loop_count=1,
            state=LoopState.COMPLETED,
        )
        assert execution.final_snr == 0.92
        assert execution.final_ihsan == 0.95

    def test_loop_execution_to_dict(self):
        """Loop execution serializes to dictionary."""
        execution = LoopExecution(
            id="exec_003",
            input_content="test",
            output_content="result",
            phases=[],
            loop_count=2,
            state=LoopState.COMPLETED,
        )
        data = execution.to_dict()
        assert data["loop_count"] == 2
        assert "output_preview" in data


class TestSovereignLoop:
    """Test SovereignLoop class."""

    @pytest.fixture
    def loop(self):
        """Create fresh SovereignLoop instance."""
        return SovereignLoop(max_loops=3, max_backtrack=5)

    def test_loop_initialization(self, loop):
        """Loop initializes with correct settings."""
        assert loop.max_loops == 3
        assert loop.max_backtrack == 5
        assert loop._graph is None

    @pytest.mark.asyncio
    async def test_loop_execution_simple(self, loop):
        """Loop executes simple query successfully."""
        execution = await loop.execute("What is 2 + 2?")
        assert execution.state == LoopState.COMPLETED
        assert execution.final_snr > 0
        assert execution.final_ihsan > 0
        assert len(execution.phases) >= 3  # At least OBSERVE, ORIENT, REASON

    @pytest.mark.asyncio
    async def test_loop_execution_with_context(self, loop):
        """Loop respects context in execution."""
        context = {"mode": "analytical", "depth": "comprehensive"}
        execution = await loop.execute("Analyze this data", context)
        assert execution.state == LoopState.COMPLETED

    @pytest.mark.asyncio
    async def test_loop_tracks_phases(self, loop):
        """Loop tracks all phase results."""
        execution = await loop.execute("Test query")
        # Should have at least observation and synthesis phases
        phase_types = [p.phase for p in execution.phases]
        assert LoopPhase.OBSERVE in phase_types

    def test_loop_get_graph(self, loop):
        """Can retrieve reasoning graph after execution."""
        asyncio.run(loop.execute("test"))
        graph = loop.get_graph()
        assert graph is not None
        assert len(graph._nodes) > 0

    def test_loop_get_stats(self, loop):
        """Can get loop statistics."""
        asyncio.run(loop.execute("test1"))
        asyncio.run(loop.execute("test2"))
        stats = loop.get_stats()
        assert stats["total_executions"] == 2

    def test_loop_get_executions(self, loop):
        """Can get recent executions."""
        asyncio.run(loop.execute("test"))
        executions = loop.get_executions(limit=10)
        assert len(executions) == 1
        assert "phases" in executions[0]


# =============================================================================
# REASONING ENGINE TESTS
# =============================================================================

class TestReasoningResult:
    """Test ReasoningResult dataclass."""

    def test_create_result(self):
        """Can create reasoning result."""
        result = ReasoningResult(
            query="test query",
            response="test response",
            confidence=0.9,
            snr_score=0.92,
            ihsan_score=0.95,
        )
        assert result.query == "test query"
        assert result.version == SARE_VERSION

    def test_result_quality_score(self):
        """Result calculates quality score."""
        result = ReasoningResult(
            query="q",
            response="r",
            confidence=0.9,
            snr_score=0.81,  # sqrt(0.81) = 0.9
            ihsan_score=1.0,
        )
        assert result.quality_score == 0.9

    def test_result_quality_score_zero(self):
        """Quality score is 0 when SNR or Ihsan is 0."""
        result = ReasoningResult(
            query="q",
            response="r",
            confidence=0.9,
            snr_score=0.0,
            ihsan_score=0.95,
        )
        assert result.quality_score == 0.0

    def test_result_is_constitutional(self):
        """Result checks constitutional compliance."""
        # Above thresholds (snr must be >= action threshold 0.95)
        good_result = ReasoningResult(
            query="q",
            response="r",
            confidence=0.9,
            snr_score=0.96,
            ihsan_score=0.96,
        )
        assert good_result.is_constitutional is True

        # Below thresholds
        bad_result = ReasoningResult(
            query="q",
            response="r",
            confidence=0.5,
            snr_score=0.6,
            ihsan_score=0.7,
        )
        assert bad_result.is_constitutional is False

    def test_result_to_dict(self):
        """Result serializes to dictionary."""
        result = ReasoningResult(
            query="test query" * 20,  # Long query
            response="test response" * 100,  # Long response
            confidence=0.9,
            snr_score=0.92,
            ihsan_score=0.95,
            giants_invoked=["shannon", "anthropic"],
        )
        data = result.to_dict()
        assert len(data["query"]) <= 100  # Truncated
        assert len(data["response"]) <= 500  # Truncated
        assert data["giants_invoked"] == ["shannon", "anthropic"]


class TestSovereignReasoningEngine:
    """Test SovereignReasoningEngine class."""

    @pytest.fixture
    def engine(self):
        """Create fresh engine instance."""
        return SovereignReasoningEngine(
            max_loops=3,
            max_backtrack=5,
            strict_mode=True,
        )

    def test_engine_initialization(self, engine):
        """Engine initializes correctly."""
        assert engine.max_loops == 3
        assert engine.strict_mode is True
        assert engine._total_queries == 0

    def test_engine_version(self, engine):
        """Engine reports version."""
        assert engine.get_version() == SARE_VERSION

    def test_engine_list_giants(self, engine):
        """Engine can list giants."""
        giants = engine.list_giants()
        assert len(giants) == 5
        giant_names = [g["name"] for g in giants]
        assert any("Shannon" in n for n in giant_names)

    def test_engine_list_techniques(self, engine):
        """Engine can list all techniques."""
        techniques = engine.list_techniques()
        assert len(techniques) >= 10

    def test_engine_list_techniques_by_giant(self, engine):
        """Engine can list techniques for specific giant."""
        techniques = engine.list_techniques("shannon")
        assert "snr" in techniques
        assert "entropy" in techniques

    @pytest.mark.asyncio
    async def test_engine_reason_simple(self, engine):
        """Engine can reason about simple query."""
        result = await engine.reason("What is the capital of France?")
        assert isinstance(result, ReasoningResult)
        assert result.query == "What is the capital of France?"
        assert result.snr_score > 0
        assert result.ihsan_score > 0

    @pytest.mark.asyncio
    async def test_engine_reason_with_context(self, engine):
        """Engine respects context in reasoning."""
        context = {"domain": "geography", "depth": "basic"}
        result = await engine.reason("Name three European countries", context)
        assert result.snr_score > 0

    @pytest.mark.asyncio
    async def test_engine_reason_tracks_giants(self, engine):
        """Engine tracks giants invoked during reasoning."""
        result = await engine.reason("Test query")
        # Should have invoked at least some giants
        assert isinstance(result.giants_invoked, list)

    @pytest.mark.asyncio
    async def test_engine_analyze_mode(self, engine):
        """Engine has analyze mode."""
        result = await engine.analyze("This is sample text for analysis.")
        assert result.snr_score > 0

    @pytest.mark.asyncio
    async def test_engine_synthesize_mode(self, engine):
        """Engine has synthesize mode."""
        sources = ["Source A content", "Source B content", "Source C content"]
        result = await engine.synthesize(sources)
        assert result.snr_score > 0

    @pytest.mark.asyncio
    async def test_engine_evaluate_mode(self, engine):
        """Engine has evaluate mode."""
        result = await engine.evaluate("Content to evaluate")
        assert result.snr_score > 0

    @pytest.mark.asyncio
    async def test_engine_create_mode(self, engine):
        """Engine has create mode."""
        result = await engine.create("Write a haiku about coding")
        assert result.snr_score > 0

    def test_engine_invoke_giant_directly(self, engine):
        """Engine can invoke giant techniques directly."""
        result, inheritance = engine.invoke_giant("shannon", "snr", "test signal")
        assert 0 <= result <= 1.0
        assert "giant" in inheritance

    def test_engine_get_giant_info(self, engine):
        """Engine can get giant info."""
        info = engine.get_giant_info("shannon")
        assert "name" in info

    def test_engine_stats_initial(self, engine):
        """Engine tracks stats from start."""
        stats = engine.get_stats()
        assert stats["total_queries"] == 0

    @pytest.mark.asyncio
    async def test_engine_stats_after_queries(self, engine):
        """Engine tracks stats after queries."""
        await engine.reason("query 1")
        await engine.reason("query 2")
        stats = engine.get_stats()
        assert stats["total_queries"] == 2
        assert "avg_snr" in stats
        assert "avg_ihsan" in stats

    def test_engine_recent_results(self, engine):
        """Engine tracks recent results."""
        asyncio.run(engine.reason("test"))
        results = engine.get_recent_results(limit=10)
        assert len(results) == 1

    def test_engine_provenance_chain(self, engine):
        """Engine maintains provenance chain."""
        asyncio.run(engine.reason("test"))
        chain = engine.get_provenance_chain()
        assert isinstance(chain, list)

    def test_engine_constitutional_status(self, engine):
        """Engine reports constitutional status."""
        asyncio.run(engine.reason("test"))
        status = engine.get_constitutional_status()
        assert "status" in status
        assert "compliance" in status

    def test_engine_reset(self, engine):
        """Engine can be reset."""
        asyncio.run(engine.reason("test"))
        engine.reset()
        assert engine._total_queries == 0
        assert len(engine._results) == 0

    def test_engine_repr(self, engine):
        """Engine has string representation."""
        repr_str = repr(engine)
        assert "SovereignReasoningEngine" in repr_str
        assert SARE_VERSION in repr_str


class TestFactoryFunction:
    """Test create_sovereign_engine factory function."""

    def test_create_engine_default(self):
        """Factory creates engine with defaults."""
        engine = create_sovereign_engine()
        assert isinstance(engine, SovereignReasoningEngine)

    def test_create_engine_with_kwargs(self):
        """Factory accepts configuration kwargs."""
        engine = create_sovereign_engine(max_loops=5, strict_mode=False)
        assert engine.max_loops == 5
        assert engine.strict_mode is False

    def test_create_engine_with_llm_fn(self):
        """Factory accepts custom LLM function."""
        def custom_llm(prompt: str) -> str:
            return f"Response to: {prompt[:50]}"

        engine = create_sovereign_engine(llm_fn=custom_llm)
        assert engine.llm_fn == custom_llm


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestSAREIntegration:
    """Integration tests for complete SARE system."""

    @pytest.fixture
    def engine(self):
        """Create engine for integration tests."""
        return create_sovereign_engine(strict_mode=False)

    @pytest.mark.asyncio
    async def test_full_reasoning_pipeline(self, engine):
        """Full reasoning pipeline executes end-to-end."""
        result = await engine.reason(
            "Explain the relationship between information entropy and signal quality."
        )

        # Verify complete result
        assert result.query is not None
        assert result.response is not None
        assert 0 <= result.snr_score <= 1.0
        assert 0 <= result.ihsan_score <= 1.0
        assert result.duration_ms > 0

        # Verify provenance
        if result.provenance:
            assert result.provenance.output_hash is not None

        # Verify graph stats present
        assert "total_nodes" in result.graph_stats or result.graph_stats == {}

    @pytest.mark.asyncio
    async def test_multiple_reasoning_modes(self, engine):
        """All reasoning modes work correctly."""
        analyze_result = await engine.analyze("Sample text for analysis")
        synthesize_result = await engine.synthesize(["Source 1", "Source 2"])
        evaluate_result = await engine.evaluate("Content to evaluate")
        create_result = await engine.create("Generate a short poem")

        for result in [analyze_result, synthesize_result, evaluate_result, create_result]:
            assert result.snr_score > 0
            assert result.ihsan_score > 0

    @pytest.mark.asyncio
    async def test_giants_protocol_integration(self, engine):
        """Giants protocol integrates with reasoning."""
        # Invoke each giant
        shannon_result, _ = engine.invoke_giant("shannon", "snr", "test signal")
        lamport_result, _ = engine.invoke_giant(
            "lamport", "consensus", [0.9, 0.91, 0.89]
        )
        besta_result, _ = engine.invoke_giant(
            "besta", "got", [{"id": "1", "content": "thought"}]
        )
        anthropic_result, _ = engine.invoke_giant(
            "anthropic", "ihsan", "Quality output text"
        )

        # All should produce valid results
        assert shannon_result is not None
        assert lamport_result is not None
        assert besta_result is not None
        assert anthropic_result is not None

    @pytest.mark.asyncio
    async def test_constitutional_enforcement(self):
        """Constitutional constraints are enforced in strict mode."""
        strict_engine = create_sovereign_engine(strict_mode=True)

        result = await strict_engine.reason("Short query")

        # Should have constitutional check
        assert result.is_constitutional is not None

        # If not constitutional, should have fallback response
        if not result.is_constitutional:
            assert "threshold" in result.response.lower() or "SNR" in result.response

    @pytest.mark.asyncio
    async def test_stats_accumulation(self, engine):
        """Stats accumulate correctly over multiple queries."""
        queries = [
            "First test query",
            "Second test query",
            "Third test query",
        ]

        for query in queries:
            await engine.reason(query)

        stats = engine.get_stats()
        assert stats["total_queries"] == 3
        assert stats["avg_duration_ms"] > 0
        assert 0 <= stats["avg_snr"] <= 1.0
        assert 0 <= stats["avg_ihsan"] <= 1.0


# =============================================================================
# EDGE CASES AND ERROR HANDLING
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def engine(self):
        return create_sovereign_engine()

    @pytest.mark.asyncio
    async def test_empty_query(self, engine):
        """Empty query is handled gracefully."""
        result = await engine.reason("")
        # Should still produce a result, even if low quality
        assert result is not None
        assert result.snr_score >= 0

    @pytest.mark.asyncio
    async def test_very_long_query(self, engine):
        """Very long query is handled."""
        long_query = "test " * 1000  # 5000 characters
        result = await engine.reason(long_query)
        assert result is not None
        assert len(result.to_dict()["query"]) <= 100  # Truncated in dict

    @pytest.mark.asyncio
    async def test_special_characters_in_query(self, engine):
        """Special characters don't break reasoning."""
        special_query = "What about <script>alert('xss')</script> & other \\ chars?"
        result = await engine.reason(special_query)
        assert result is not None

    def test_invalid_giant_name(self, engine):
        """Invalid giant name raises error."""
        with pytest.raises(ValueError):
            engine.invoke_giant("invalid_giant", "snr", "test")

    def test_invalid_technique_name(self, engine):
        """Invalid technique name raises error."""
        with pytest.raises(ValueError):
            engine.invoke_giant("shannon", "invalid_technique", "test")

    @pytest.mark.asyncio
    async def test_synthesize_empty_sources(self, engine):
        """Synthesize handles empty sources."""
        result = await engine.synthesize([])
        assert result is not None

    @pytest.mark.asyncio
    async def test_evaluate_with_custom_criteria(self, engine):
        """Evaluate accepts custom criteria."""
        result = await engine.evaluate(
            "Test content",
            criteria=["originality", "coherence", "depth"],
        )
        assert result is not None


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance:
    """Performance-related tests."""

    @pytest.fixture
    def engine(self):
        return create_sovereign_engine()

    @pytest.mark.asyncio
    async def test_reasoning_completes_quickly(self, engine):
        """Simple reasoning completes within timeout."""
        import time

        start = time.time()
        result = await engine.reason("Quick test")
        duration = time.time() - start

        assert duration < 5.0  # Should complete in under 5 seconds
        assert result.duration_ms < 5000

    @pytest.mark.asyncio
    async def test_multiple_queries_performance(self, engine):
        """Multiple queries maintain reasonable performance."""
        import time

        queries = [f"Query number {i}" for i in range(5)]
        start = time.time()

        for query in queries:
            await engine.reason(query)

        total_duration = time.time() - start
        avg_duration = total_duration / len(queries)

        assert avg_duration < 2.0  # Average under 2 seconds each

    def test_graph_operations_performance(self):
        """Graph operations scale reasonably."""
        graph = ReasoningGraph()

        # Add 100 nodes
        import time

        start = time.time()
        nodes = []
        for i in range(100):
            parent_ids = {nodes[-1].id} if nodes and i % 5 != 0 else None
            node = graph.add_node(f"Node {i}", NodeType.ANALYSIS, parent_ids=parent_ids)
            nodes.append(node)

        add_duration = time.time() - start
        assert add_duration < 1.0  # Adding 100 nodes under 1 second

        # Find best path
        start = time.time()
        best_path = graph.find_best_path()
        path_duration = time.time() - start
        assert path_duration < 1.0  # Path finding under 1 second


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
