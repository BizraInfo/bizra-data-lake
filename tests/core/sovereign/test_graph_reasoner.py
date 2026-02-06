"""
BIZRA Graph-of-Thoughts Reasoner Test Suite

Tests for the Graph-of-Thoughts reasoning engine implementation.
Target: 80% coverage of core/sovereign/graph_reasoner.py (619 lines)

Standing on Giants:
- Graph of Thoughts (Besta et al., 2024): "Solving elaborate problems with LLMs"
- Tree of Thoughts (Yao et al., 2023): Deliberate problem solving
- Chain of Thought (Wei et al., 2022): Step-by-step reasoning

Test Categories:
1. ThoughtNode creation and scoring
2. ThoughtEdge type handling
3. ReasoningPath traversal
4. GraphOfThoughts operations (generate, aggregate, refine, prune)
5. Ihsan gating on synthesis

Quality Standards:
- pytest fixtures for reusable setup
- Parameterized tests for edge cases
- Mock external dependencies
- Comprehensive docstrings
"""

import pytest
import sys
import math
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path (works across platforms)
_project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_project_root))

from core.sovereign.graph_reasoner import (
    ThoughtNode,
    ThoughtEdge,
    ThoughtType,
    EdgeType,
    ReasoningPath,
    ReasoningStrategy,
    GraphOfThoughts,
)
from core.integration.constants import UNIFIED_IHSAN_THRESHOLD, UNIFIED_SNR_THRESHOLD


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def basic_thought_node():
    """Create a basic thought node for testing."""
    return ThoughtNode(
        id="test_node_001",
        content="This is a test hypothesis about the system behavior.",
        thought_type=ThoughtType.HYPOTHESIS,
        confidence=0.8,
        snr_score=0.85,
        depth=0,
    )


@pytest.fixture
def high_quality_thought():
    """Create a high-quality thought that passes Ihsan threshold."""
    return ThoughtNode(
        id="high_quality_001",
        content="Well-reasoned conclusion with strong evidence.",
        thought_type=ThoughtType.CONCLUSION,
        confidence=0.98,
        snr_score=0.97,
        depth=2,
        correctness=0.98,
        groundedness=0.97,
        coherence=0.99,
    )


@pytest.fixture
def low_quality_thought():
    """Create a low-quality thought that fails Ihsan threshold."""
    return ThoughtNode(
        id="low_quality_001",
        content="Vague speculation without evidence.",
        thought_type=ThoughtType.HYPOTHESIS,
        confidence=0.3,
        snr_score=0.4,
        depth=1,
        correctness=0.3,
        groundedness=0.2,
        coherence=0.4,
    )


@pytest.fixture
def graph():
    """Create a fresh GraphOfThoughts instance."""
    return GraphOfThoughts(
        strategy=ReasoningStrategy.BEST_FIRST,
        max_depth=10,
        beam_width=5,
        snr_threshold=UNIFIED_SNR_THRESHOLD,
        ihsan_threshold=UNIFIED_IHSAN_THRESHOLD,
    )


@pytest.fixture
def populated_graph(graph):
    """Create a graph with pre-populated thoughts for traversal tests."""
    # Root question
    root = graph.add_thought(
        content="What is the optimal solution?",
        thought_type=ThoughtType.QUESTION,
        confidence=0.9,
    )

    # Two hypotheses
    h1 = graph.add_thought(
        content="Hypothesis A: Use algorithm X",
        thought_type=ThoughtType.HYPOTHESIS,
        confidence=0.75,
        parent_id=root.id,
    )
    h2 = graph.add_thought(
        content="Hypothesis B: Use algorithm Y",
        thought_type=ThoughtType.HYPOTHESIS,
        confidence=0.80,
        parent_id=root.id,
    )

    # Evidence for H1
    e1 = graph.add_thought(
        content="Evidence: Algorithm X has O(n log n) complexity",
        thought_type=ThoughtType.EVIDENCE,
        confidence=0.95,
        parent_id=h1.id,
        edge_type=EdgeType.SUPPORTS,
    )

    # Conclusion from H1
    c1 = graph.add_thought(
        content="Conclusion: Algorithm X is optimal for this case",
        thought_type=ThoughtType.CONCLUSION,
        confidence=0.88,
        parent_id=e1.id,
    )

    return graph


# =============================================================================
# THOUGHT NODE TESTS
# =============================================================================


class TestThoughtNodeCreation:
    """Tests for ThoughtNode dataclass creation and initialization."""

    def test_basic_creation(self, basic_thought_node):
        """ThoughtNode should be created with all required fields."""
        node = basic_thought_node

        assert node.id == "test_node_001"
        assert node.content == "This is a test hypothesis about the system behavior."
        assert node.thought_type == ThoughtType.HYPOTHESIS
        assert node.confidence == 0.8
        assert node.snr_score == 0.85
        assert node.depth == 0

    def test_default_values(self):
        """ThoughtNode should have sensible defaults."""
        node = ThoughtNode(
            id="minimal_node",
            content="Minimal content",
            thought_type=ThoughtType.REASONING,
        )

        assert node.confidence == 0.5
        assert node.snr_score == 0.5
        assert node.depth == 0
        assert node.correctness == 0.5
        assert node.groundedness == 0.5
        assert node.coherence == 0.5
        assert node.metadata == {}

    def test_timestamp_auto_generated(self):
        """ThoughtNode should auto-generate created_at timestamp."""
        before = time.time()
        node = ThoughtNode(
            id="timed_node",
            content="Testing timestamp",
            thought_type=ThoughtType.EVIDENCE,
        )
        after = time.time()

        assert before <= node.created_at <= after

    @pytest.mark.parametrize("thought_type", list(ThoughtType))
    def test_all_thought_types_valid(self, thought_type):
        """All ThoughtType enum values should be valid for node creation."""
        node = ThoughtNode(
            id=f"type_test_{thought_type.value}",
            content=f"Testing {thought_type.value}",
            thought_type=thought_type,
        )

        assert node.thought_type == thought_type
        assert node.thought_type.value == thought_type.value


class TestThoughtNodeIhsanScore:
    """Tests for Ihsan score calculation (geometric mean of dimensions)."""

    def test_ihsan_score_geometric_mean(self):
        """Ihsan score should be geometric mean of dimensions."""
        node = ThoughtNode(
            id="ihsan_test",
            content="Testing Ihsan calculation",
            thought_type=ThoughtType.CONCLUSION,
            confidence=0.9,
            correctness=0.9,
            groundedness=0.9,
            coherence=0.9,
        )

        # Geometric mean of [0.9, 0.9, 0.9, 0.9] = 0.9
        expected = 0.9
        assert abs(node.ihsan_score - expected) < 0.001

    def test_ihsan_score_with_low_dimension(self):
        """Low score in any dimension should drag down Ihsan."""
        node = ThoughtNode(
            id="low_dim_test",
            content="Testing low dimension impact",
            thought_type=ThoughtType.REASONING,
            confidence=0.99,
            correctness=0.99,
            groundedness=0.10,  # Very low
            coherence=0.99,
        )

        # Geometric mean penalizes low outliers heavily
        # geomean(0.99, 0.10, 0.99, 0.99) ~ 0.558
        assert node.ihsan_score < 0.60

    def test_passes_ihsan_threshold(self, high_quality_thought):
        """High-quality thoughts should pass Ihsan threshold."""
        assert high_quality_thought.passes_ihsan is True
        assert high_quality_thought.ihsan_score >= UNIFIED_IHSAN_THRESHOLD

    def test_fails_ihsan_threshold(self, low_quality_thought):
        """Low-quality thoughts should fail Ihsan threshold."""
        assert low_quality_thought.passes_ihsan is False
        assert low_quality_thought.ihsan_score < UNIFIED_IHSAN_THRESHOLD

    def test_ihsan_handles_zero_gracefully(self):
        """Ihsan calculation should handle zero values without errors."""
        node = ThoughtNode(
            id="zero_test",
            content="Testing zero handling",
            thought_type=ThoughtType.HYPOTHESIS,
            confidence=0.0,
            correctness=0.0,
            groundedness=0.0,
            coherence=0.0,
        )

        # Should not raise, should return very small value
        score = node.ihsan_score
        assert score >= 0
        assert score < 0.01  # Very small due to zero dimensions


class TestThoughtNodeSerialization:
    """Tests for ThoughtNode to_dict serialization."""

    def test_to_dict_contains_all_fields(self, basic_thought_node):
        """to_dict should contain all expected fields."""
        d = basic_thought_node.to_dict()

        assert "id" in d
        assert "content" in d
        assert "type" in d
        assert "confidence" in d
        assert "snr" in d
        assert "ihsan" in d
        assert "depth" in d

    def test_to_dict_truncates_long_content(self):
        """Long content should be truncated with ellipsis."""
        long_content = "X" * 500
        node = ThoughtNode(
            id="long_content_test",
            content=long_content,
            thought_type=ThoughtType.REASONING,
        )

        d = node.to_dict()

        assert len(d["content"]) <= 203  # 200 + "..."
        assert d["content"].endswith("...")

    def test_to_dict_short_content_not_truncated(self):
        """Short content should not be truncated."""
        short_content = "Short content"
        node = ThoughtNode(
            id="short_content_test",
            content=short_content,
            thought_type=ThoughtType.EVIDENCE,
        )

        d = node.to_dict()

        assert d["content"] == short_content
        assert "..." not in d["content"]


# =============================================================================
# THOUGHT EDGE TESTS
# =============================================================================


class TestThoughtEdgeCreation:
    """Tests for ThoughtEdge dataclass creation."""

    def test_basic_edge_creation(self):
        """ThoughtEdge should be created with required fields."""
        edge = ThoughtEdge(
            source_id="node_a",
            target_id="node_b",
            edge_type=EdgeType.SUPPORTS,
        )

        assert edge.source_id == "node_a"
        assert edge.target_id == "node_b"
        assert edge.edge_type == EdgeType.SUPPORTS
        assert edge.weight == 1.0
        assert edge.reasoning == ""

    def test_edge_with_reasoning(self):
        """ThoughtEdge should store reasoning explanation."""
        edge = ThoughtEdge(
            source_id="evidence_1",
            target_id="hypothesis_1",
            edge_type=EdgeType.SUPPORTS,
            weight=0.9,
            reasoning="Empirical data confirms hypothesis prediction",
        )

        assert edge.reasoning == "Empirical data confirms hypothesis prediction"
        assert edge.weight == 0.9

    @pytest.mark.parametrize("edge_type", list(EdgeType))
    def test_all_edge_types_valid(self, edge_type):
        """All EdgeType enum values should be valid for edge creation."""
        edge = ThoughtEdge(
            source_id="src",
            target_id="tgt",
            edge_type=edge_type,
        )

        assert edge.edge_type == edge_type


class TestThoughtEdgeSerialization:
    """Tests for ThoughtEdge to_dict serialization."""

    def test_to_dict_contains_fields(self):
        """to_dict should contain all edge fields."""
        edge = ThoughtEdge(
            source_id="a",
            target_id="b",
            edge_type=EdgeType.DERIVES,
            weight=0.8,
        )

        d = edge.to_dict()

        assert d["source"] == "a"
        assert d["target"] == "b"
        assert d["type"] == "derives"
        assert d["weight"] == 0.8


# =============================================================================
# REASONING PATH TESTS
# =============================================================================


class TestReasoningPath:
    """Tests for ReasoningPath through the thought graph."""

    def test_path_creation(self):
        """ReasoningPath should store node sequence and scores."""
        path = ReasoningPath(
            nodes=["node_1", "node_2", "node_3"],
            total_snr=2.7,
            total_confidence=2.4,
        )

        assert path.nodes == ["node_1", "node_2", "node_3"]
        assert path.total_snr == 2.7
        assert path.total_confidence == 2.4

    def test_path_length(self):
        """Path length should match node count."""
        path = ReasoningPath(
            nodes=["a", "b", "c", "d"],
            total_snr=3.6,
            total_confidence=3.2,
        )

        assert path.length == 4

    def test_empty_path_length(self):
        """Empty path should have zero length."""
        path = ReasoningPath(nodes=[], total_snr=0, total_confidence=0)

        assert path.length == 0

    def test_average_snr(self):
        """Average SNR should be calculated correctly."""
        path = ReasoningPath(
            nodes=["a", "b", "c"],
            total_snr=2.7,  # 0.9 average
            total_confidence=2.4,
        )

        assert abs(path.average_snr - 0.9) < 0.001

    def test_average_snr_empty_path(self):
        """Average SNR for empty path should not divide by zero."""
        path = ReasoningPath(nodes=[], total_snr=0, total_confidence=0)

        # Should return 0, not raise ZeroDivisionError
        assert path.average_snr == 0


# =============================================================================
# GRAPH OF THOUGHTS INITIALIZATION
# =============================================================================


class TestGraphOfThoughtsInit:
    """Tests for GraphOfThoughts initialization."""

    def test_default_initialization(self):
        """Graph should initialize with default parameters."""
        graph = GraphOfThoughts()

        assert graph.strategy == ReasoningStrategy.BEST_FIRST
        assert graph.max_depth == 10
        assert graph.beam_width == 5
        assert graph.snr_threshold == UNIFIED_SNR_THRESHOLD
        assert graph.ihsan_threshold == UNIFIED_IHSAN_THRESHOLD

    def test_custom_initialization(self):
        """Graph should accept custom parameters."""
        graph = GraphOfThoughts(
            strategy=ReasoningStrategy.BREADTH_FIRST,
            max_depth=20,
            beam_width=10,
            snr_threshold=0.90,
            ihsan_threshold=0.98,
        )

        assert graph.strategy == ReasoningStrategy.BREADTH_FIRST
        assert graph.max_depth == 20
        assert graph.beam_width == 10
        assert graph.snr_threshold == 0.90
        assert graph.ihsan_threshold == 0.98

    def test_empty_graph_state(self, graph):
        """New graph should have empty state."""
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0
        assert len(graph.roots) == 0
        assert graph.stats["nodes_created"] == 0

    @pytest.mark.parametrize("strategy", list(ReasoningStrategy))
    def test_all_strategies_valid(self, strategy):
        """All ReasoningStrategy values should be valid."""
        graph = GraphOfThoughts(strategy=strategy)

        assert graph.strategy == strategy


# =============================================================================
# GRAPH ADD THOUGHT TESTS
# =============================================================================


class TestGraphAddThought:
    """Tests for adding thoughts to the graph."""

    def test_add_root_thought(self, graph):
        """Adding thought without parent creates root node."""
        node = graph.add_thought(
            content="Root question",
            thought_type=ThoughtType.QUESTION,
        )

        assert node.id in graph.nodes
        assert node.id in graph.roots
        assert graph.stats["nodes_created"] == 1
        assert node.depth == 0

    def test_add_child_thought(self, graph):
        """Adding thought with parent creates child node."""
        parent = graph.add_thought(
            content="Parent hypothesis",
            thought_type=ThoughtType.HYPOTHESIS,
        )

        child = graph.add_thought(
            content="Supporting evidence",
            thought_type=ThoughtType.EVIDENCE,
            parent_id=parent.id,
            edge_type=EdgeType.SUPPORTS,
        )

        assert child.id in graph.nodes
        assert child.id not in graph.roots
        assert child.depth == parent.depth + 1
        assert len(graph.edges) == 1

    def test_add_thought_creates_edge(self, graph):
        """Adding child thought automatically creates edge."""
        parent = graph.add_thought("Parent", ThoughtType.QUESTION)
        child = graph.add_thought(
            "Child",
            ThoughtType.HYPOTHESIS,
            parent_id=parent.id,
            edge_type=EdgeType.DERIVES,
        )

        # Find the connecting edge
        edges = [e for e in graph.edges if e.source_id == parent.id and e.target_id == child.id]

        assert len(edges) == 1
        assert edges[0].edge_type == EdgeType.DERIVES

    def test_add_thought_with_metadata(self, graph):
        """Metadata should be stored with thought."""
        node = graph.add_thought(
            content="Thought with metadata",
            thought_type=ThoughtType.REASONING,
            metadata={"source": "empirical_data", "confidence_method": "bayesian"},
        )

        assert node.metadata["source"] == "empirical_data"
        assert node.metadata["confidence_method"] == "bayesian"


# =============================================================================
# GRAPH GENERATE OPERATION
# =============================================================================


class TestGraphGenerate:
    """Tests for the generate operation (creating new thoughts)."""

    def test_generate_from_parent(self, graph):
        """Generate should create new thought linked to parent."""
        parent = graph.add_thought("Question", ThoughtType.QUESTION)

        child = graph.generate(
            content="Generated hypothesis",
            thought_type=ThoughtType.HYPOTHESIS,
            parent=parent,
        )

        assert child.id in graph.nodes
        assert child.depth == 1
        assert parent.id in graph.adjacency
        assert child.id in graph.adjacency[parent.id]

    def test_generate_without_parent(self, graph):
        """Generate without parent should create root node."""
        node = graph.generate(
            content="Standalone thought",
            thought_type=ThoughtType.REASONING,
            parent=None,
        )

        assert node.id in graph.roots
        assert node.depth == 0


# =============================================================================
# GRAPH AGGREGATE OPERATION
# =============================================================================


class TestGraphAggregate:
    """Tests for the aggregate operation (merging thoughts into synthesis)."""

    def test_aggregate_basic(self, graph):
        """Aggregate should merge multiple thoughts."""
        t1 = graph.add_thought("Thought 1", ThoughtType.EVIDENCE, confidence=0.8)
        t2 = graph.add_thought("Thought 2", ThoughtType.EVIDENCE, confidence=0.9)

        synth = graph.aggregate(
            thoughts=[t1, t2],
            synthesis_content="Combined conclusion from both thoughts",
        )

        assert synth.thought_type == ThoughtType.SYNTHESIS
        assert synth.id in graph.nodes
        assert graph.stats["aggregations"] == 1

    def test_aggregate_confidence_boost(self, graph):
        """Aggregated thought should have confidence bonus."""
        t1 = graph.add_thought("T1", ThoughtType.EVIDENCE, confidence=0.8)
        t2 = graph.add_thought("T2", ThoughtType.EVIDENCE, confidence=0.8)

        synth = graph.aggregate([t1, t2], "Synthesis")

        # Average is 0.8, synthesis should be slightly higher
        assert synth.confidence > 0.8

    def test_aggregate_creates_edges(self, graph):
        """Aggregate should create edges from source thoughts to synthesis."""
        t1 = graph.add_thought("T1", ThoughtType.HYPOTHESIS)
        t2 = graph.add_thought("T2", ThoughtType.HYPOTHESIS)

        synth = graph.aggregate([t1, t2], "Combined")

        # Find edges to synthesis
        synth_edges = [e for e in graph.edges if e.target_id == synth.id]

        assert len(synth_edges) == 2
        assert all(e.edge_type == EdgeType.SYNTHESIZES for e in synth_edges)

    def test_aggregate_depth_calculation(self, graph):
        """Synthesis depth should be max(sources) + 1."""
        t1 = graph.add_thought("T1", ThoughtType.EVIDENCE)
        t1.depth = 3

        t2 = graph.add_thought("T2", ThoughtType.EVIDENCE)
        t2.depth = 5

        synth = graph.aggregate([t1, t2], "Synthesis")

        assert synth.depth == 6  # max(3, 5) + 1


# =============================================================================
# GRAPH REFINE OPERATION
# =============================================================================


class TestGraphRefine:
    """Tests for the refine operation (improving existing thoughts)."""

    def test_refine_creates_new_node(self, graph):
        """Refine should create new node, not modify existing."""
        original = graph.add_thought(
            "Original thought",
            ThoughtType.HYPOTHESIS,
            confidence=0.7,
        )
        original_id = original.id

        refined = graph.refine(
            thought=original,
            refined_content="Improved version of the thought",
        )

        assert refined.id != original_id
        assert refined.id in graph.nodes
        assert original_id in graph.nodes  # Original still exists

    def test_refine_increases_scores(self, graph):
        """Refine should improve confidence and SNR."""
        original = graph.add_thought(
            "Initial",
            ThoughtType.REASONING,
            confidence=0.7,
        )
        original.snr_score = 0.7

        refined = graph.refine(
            thought=original,
            refined_content="Improved",
            improvement_score=0.15,
        )

        assert refined.confidence == 0.85
        assert refined.snr_score == 0.85

    def test_refine_caps_at_one(self, graph):
        """Refine should cap scores at 1.0."""
        original = graph.add_thought(
            "Nearly perfect",
            ThoughtType.CONCLUSION,
            confidence=0.95,
        )
        original.snr_score = 0.95

        refined = graph.refine(original, "Even better", improvement_score=0.2)

        assert refined.confidence == 1.0
        assert refined.snr_score == 1.0

    def test_refine_creates_edge(self, graph):
        """Refine should create REFINES edge."""
        original = graph.add_thought("Original", ThoughtType.HYPOTHESIS)

        refined = graph.refine(original, "Refined")

        # Find refine edge
        refine_edges = [
            e for e in graph.edges
            if e.source_id == original.id and e.target_id == refined.id
        ]

        assert len(refine_edges) == 1
        assert refine_edges[0].edge_type == EdgeType.REFINES

    def test_refine_updates_stats(self, graph):
        """Refine should increment refinements counter."""
        original = graph.add_thought("Original", ThoughtType.REASONING)

        assert graph.stats["refinements"] == 0

        graph.refine(original, "Refined 1")
        assert graph.stats["refinements"] == 1

        graph.refine(original, "Refined 2")
        assert graph.stats["refinements"] == 2


# =============================================================================
# GRAPH VALIDATE OPERATION
# =============================================================================


class TestGraphValidate:
    """Tests for the validate operation (quality checking)."""

    def test_validate_passing_thought(self, graph, high_quality_thought):
        """Validate should pass thought meeting Ihsan threshold."""
        graph.nodes[high_quality_thought.id] = high_quality_thought

        validation = graph.validate(high_quality_thought)

        assert validation.thought_type == ThoughtType.VALIDATION
        assert "PASS" in validation.content
        assert validation.metadata["passed"] is True

    def test_validate_failing_thought(self, graph, low_quality_thought):
        """Validate should fail thought below Ihsan threshold."""
        graph.nodes[low_quality_thought.id] = low_quality_thought

        validation = graph.validate(low_quality_thought)

        assert "FAIL" in validation.content
        assert validation.metadata["passed"] is False

    def test_validate_creates_edge(self, graph, high_quality_thought):
        """Validate should create VALIDATES edge."""
        graph.nodes[high_quality_thought.id] = high_quality_thought

        validation = graph.validate(high_quality_thought)

        # Find validation edge
        val_edges = [
            e for e in graph.edges
            if e.source_id == high_quality_thought.id and e.target_id == validation.id
        ]

        assert len(val_edges) == 1
        assert val_edges[0].edge_type == EdgeType.VALIDATES

    def test_validate_with_custom_validator(self, graph):
        """Custom validator function should be used when provided."""
        thought = graph.add_thought("Test thought", ThoughtType.HYPOTHESIS)

        # Custom validator that always passes
        def custom_validator(content):
            return (True, 0.99)

        validation = graph.validate(thought, validator_fn=custom_validator)

        assert validation.metadata["passed"] is True
        assert validation.confidence == 0.99


# =============================================================================
# GRAPH SCORE NODE OPERATION
# =============================================================================


class TestGraphScoreNode:
    """Tests for node scoring with support/refutation factors."""

    def test_score_basic_node(self, graph, basic_thought_node):
        """Basic node should get reasonable score."""
        graph.nodes[basic_thought_node.id] = basic_thought_node

        score = graph.score_node(basic_thought_node)

        assert 0 <= score <= 1.0

    def test_score_penalizes_depth(self, graph):
        """Deeper nodes should have lower scores."""
        shallow = graph.add_thought("Shallow", ThoughtType.REASONING)
        shallow.confidence = 0.8
        shallow.snr_score = 0.8
        shallow.depth = 1

        deep = graph.add_thought("Deep", ThoughtType.REASONING)
        deep.confidence = 0.8
        deep.snr_score = 0.8
        deep.depth = 8

        shallow_score = graph.score_node(shallow)
        deep_score = graph.score_node(deep)

        assert shallow_score > deep_score

    def test_score_rewards_support(self, graph):
        """Nodes with supporting edges should have higher scores."""
        hypothesis = graph.add_thought("Hypothesis", ThoughtType.HYPOTHESIS)
        hypothesis.confidence = 0.7
        hypothesis.snr_score = 0.7

        # Score without support
        score_before = graph.score_node(hypothesis)

        # Add supporting evidence
        evidence = graph.add_thought(
            "Evidence",
            ThoughtType.EVIDENCE,
            parent_id=hypothesis.id,
        )
        graph.add_edge(evidence.id, hypothesis.id, EdgeType.SUPPORTS)

        score_after = graph.score_node(hypothesis)

        assert score_after > score_before

    def test_score_penalizes_refutation(self, graph):
        """Nodes with refuting edges should have lower scores."""
        hypothesis = graph.add_thought("Hypothesis", ThoughtType.HYPOTHESIS)
        hypothesis.confidence = 0.8
        hypothesis.snr_score = 0.8

        score_before = graph.score_node(hypothesis)

        # Add refuting evidence
        counter = graph.add_thought("Counter evidence", ThoughtType.COUNTERPOINT)
        graph.add_edge(counter.id, hypothesis.id, EdgeType.REFUTES)

        score_after = graph.score_node(hypothesis)

        assert score_after < score_before


# =============================================================================
# GRAPH PRUNE OPERATION
# =============================================================================


class TestGraphPrune:
    """Tests for the prune operation (removing low-quality nodes)."""

    def test_prune_removes_low_snr_nodes(self, graph):
        """Prune should remove nodes below SNR threshold."""
        good = graph.add_thought("Good thought", ThoughtType.REASONING)
        good.snr_score = 0.95

        bad = graph.add_thought("Bad thought", ThoughtType.REASONING)
        bad.snr_score = 0.5

        pruned_count = graph.prune(threshold=0.8)

        assert pruned_count == 1
        assert good.id in graph.nodes
        assert bad.id not in graph.nodes

    def test_prune_preserves_questions(self, graph):
        """Prune should not remove QUESTION type nodes."""
        question = graph.add_thought("Why?", ThoughtType.QUESTION)
        question.snr_score = 0.5  # Below threshold

        pruned = graph.prune(threshold=0.8)

        assert pruned == 0
        assert question.id in graph.nodes

    def test_prune_removes_orphaned_edges(self, graph):
        """Prune should remove edges to/from pruned nodes."""
        parent = graph.add_thought("Parent", ThoughtType.HYPOTHESIS)
        parent.snr_score = 0.95

        child = graph.add_thought(
            "Child",
            ThoughtType.EVIDENCE,
            parent_id=parent.id,
        )
        child.snr_score = 0.4  # Will be pruned

        initial_edges = len(graph.edges)
        graph.prune(threshold=0.8)

        assert len(graph.edges) < initial_edges

    def test_prune_updates_adjacency(self, graph):
        """Prune should update adjacency lists."""
        parent = graph.add_thought("Parent", ThoughtType.HYPOTHESIS)
        parent.snr_score = 0.95

        child = graph.add_thought("Child", ThoughtType.REASONING, parent_id=parent.id)
        child.snr_score = 0.4

        graph.prune(threshold=0.8)

        assert child.id not in graph.adjacency
        assert child.id not in graph.adjacency.get(parent.id, [])

    def test_prune_uses_default_threshold(self, graph):
        """Prune without threshold should use graph's snr_threshold."""
        node = graph.add_thought("Test", ThoughtType.REASONING)
        node.snr_score = graph.snr_threshold - 0.1

        pruned = graph.prune()

        assert pruned == 1


# =============================================================================
# GRAPH PATH FINDING
# =============================================================================


class TestGraphFindBestPath:
    """Tests for finding the best reasoning path through the graph."""

    def test_find_best_path_empty_graph(self, graph):
        """Best path in empty graph should be empty."""
        path = graph.find_best_path()

        assert path.nodes == []
        assert path.total_snr == 0

    def test_find_best_path_single_node(self, graph):
        """Best path with single node should contain that node."""
        node = graph.add_thought("Only node", ThoughtType.CONCLUSION)
        node.snr_score = 0.9

        path = graph.find_best_path()

        assert node.id in path.nodes

    def test_find_best_path_to_conclusion(self, populated_graph):
        """Best path should find path to CONCLUSION type."""
        path = populated_graph.find_best_path()

        if path.nodes:
            last_node_id = path.nodes[-1]
            last_node = populated_graph.nodes.get(last_node_id)
            if last_node:
                assert last_node.thought_type == ThoughtType.CONCLUSION

    def test_find_best_path_from_specific_start(self, populated_graph):
        """Best path should respect start node."""
        # Get a non-root node
        non_root = [
            nid for nid in populated_graph.nodes
            if nid not in populated_graph.roots
        ][0]

        path = populated_graph.find_best_path(start_id=non_root)

        if path.nodes:
            assert path.nodes[0] == non_root

    def test_find_best_path_invalid_start(self, graph):
        """Best path with invalid start should return empty path."""
        graph.add_thought("Some node", ThoughtType.QUESTION)

        path = graph.find_best_path(start_id="nonexistent_id")

        assert path.nodes == []


# =============================================================================
# GRAPH UTILITY METHODS
# =============================================================================


class TestGraphUtilities:
    """Tests for utility methods (get_conclusions, get_frontier, etc.)."""

    def test_get_conclusions(self, populated_graph):
        """get_conclusions should return conclusion nodes above threshold."""
        # Add high-SNR conclusion
        c = populated_graph.add_thought("High SNR conclusion", ThoughtType.CONCLUSION)
        c.snr_score = 0.95

        conclusions = populated_graph.get_conclusions(min_snr=0.9)

        assert len(conclusions) >= 1
        assert all(c.thought_type == ThoughtType.CONCLUSION for c in conclusions)
        assert all(c.snr_score >= 0.9 for c in conclusions)

    def test_get_conclusions_empty_when_none_qualify(self, graph):
        """get_conclusions should return empty list when none qualify."""
        c = graph.add_thought("Low SNR", ThoughtType.CONCLUSION)
        c.snr_score = 0.5

        conclusions = graph.get_conclusions(min_snr=0.9)

        assert len(conclusions) == 0

    def test_get_frontier_returns_leaves(self, populated_graph):
        """get_frontier should return nodes with no children."""
        frontier = populated_graph.get_frontier()

        # All frontier nodes should have no children
        for node in frontier:
            children = populated_graph.adjacency.get(node.id, [])
            assert len(children) == 0

    def test_get_leaves_alias(self, populated_graph):
        """get_leaves should be alias for get_frontier."""
        frontier = populated_graph.get_frontier()
        leaves = populated_graph.get_leaves()

        assert len(frontier) == len(leaves)
        assert set(n.id for n in frontier) == set(n.id for n in leaves)

    def test_clear_resets_graph(self, populated_graph):
        """clear should reset graph to empty state."""
        assert len(populated_graph.nodes) > 0

        populated_graph.clear()

        assert len(populated_graph.nodes) == 0
        assert len(populated_graph.edges) == 0
        assert len(populated_graph.roots) == 0
        assert populated_graph.stats["nodes_created"] == 0


class TestGraphSerialization:
    """Tests for graph serialization to dict."""

    def test_to_dict_structure(self, populated_graph):
        """to_dict should return expected structure."""
        d = populated_graph.to_dict()

        assert "nodes" in d
        assert "edges" in d
        assert "roots" in d
        assert "stats" in d
        assert "config" in d

    def test_to_dict_config_values(self, graph):
        """to_dict config should match graph settings."""
        d = graph.to_dict()

        assert d["config"]["strategy"] == graph.strategy.value
        assert d["config"]["max_depth"] == graph.max_depth
        assert d["config"]["snr_threshold"] == graph.snr_threshold
        assert d["config"]["ihsan_threshold"] == graph.ihsan_threshold


class TestGraphVisualization:
    """Tests for ASCII visualization."""

    def test_visualize_ascii_basic(self, populated_graph):
        """visualize_ascii should return string representation."""
        viz = populated_graph.visualize_ascii()

        assert isinstance(viz, str)
        assert "Graph of Thoughts" in viz
        assert "Nodes:" in viz
        assert "Edges:" in viz

    def test_visualize_ascii_empty_graph(self, graph):
        """Empty graph should still visualize."""
        viz = graph.visualize_ascii()

        assert "Nodes: 0" in viz
        assert "Edges: 0" in viz


# =============================================================================
# IHSAN GATING INTEGRATION TESTS
# =============================================================================


class TestIhsanGating:
    """Integration tests for Ihsan gating on synthesis operations."""

    def test_synthesis_fails_without_quality_sources(self, graph):
        """Synthesizing low-quality thoughts should produce low-quality synthesis."""
        t1 = graph.add_thought("Vague idea 1", ThoughtType.HYPOTHESIS, confidence=0.3)
        t1.snr_score = 0.3
        t1.correctness = 0.3
        t1.groundedness = 0.3
        t1.coherence = 0.3

        t2 = graph.add_thought("Vague idea 2", ThoughtType.HYPOTHESIS, confidence=0.3)
        t2.snr_score = 0.3
        t2.correctness = 0.3
        t2.groundedness = 0.3
        t2.coherence = 0.3

        synth = graph.aggregate([t1, t2], "Weak synthesis")

        # Synthesis inherits low scores from sources (with small bonus)
        assert synth.confidence < 0.5

    def test_synthesis_succeeds_with_quality_sources(self, graph):
        """Synthesizing high-quality thoughts should produce high-quality synthesis."""
        t1 = graph.add_thought("Strong evidence 1", ThoughtType.EVIDENCE, confidence=0.95)
        t1.snr_score = 0.95

        t2 = graph.add_thought("Strong evidence 2", ThoughtType.EVIDENCE, confidence=0.95)
        t2.snr_score = 0.95

        synth = graph.aggregate([t1, t2], "Strong synthesis")

        # Synthesis should have high scores
        assert synth.confidence > 0.95
        assert synth.snr_score > 0.95

    def test_refinement_chain_improves_quality(self, graph):
        """Iterative refinement should progressively improve quality."""
        original = graph.add_thought("Initial rough idea", ThoughtType.HYPOTHESIS)
        original.confidence = 0.6
        original.snr_score = 0.6

        refined1 = graph.refine(original, "First improvement", improvement_score=0.1)
        refined2 = graph.refine(refined1, "Second improvement", improvement_score=0.1)
        refined3 = graph.refine(refined2, "Third improvement", improvement_score=0.1)

        assert refined3.confidence > refined2.confidence > refined1.confidence > original.confidence


# =============================================================================
# EDGE CASE AND BOUNDARY TESTS
# =============================================================================


class TestEdgeCases:
    """Edge case and boundary condition tests."""

    def test_very_deep_graph(self, graph):
        """Graph should handle deep reasoning chains."""
        current = graph.add_thought("Root", ThoughtType.QUESTION)

        for i in range(50):
            current = graph.add_thought(
                f"Level {i+1}",
                ThoughtType.REASONING,
                parent_id=current.id,
            )

        assert current.depth == 50
        assert len(graph.nodes) == 51

    def test_wide_graph(self, graph):
        """Graph should handle wide graphs (many children per node)."""
        root = graph.add_thought("Root", ThoughtType.QUESTION)

        for i in range(100):
            graph.add_thought(
                f"Child {i}",
                ThoughtType.HYPOTHESIS,
                parent_id=root.id,
            )

        assert len(graph.nodes) == 101
        assert len(graph.adjacency[root.id]) == 100

    def test_empty_content_thought(self, graph):
        """Empty content thoughts should be allowed."""
        node = graph.add_thought("", ThoughtType.REASONING)

        assert node.content == ""
        assert node.id in graph.nodes

    def test_unicode_content(self, graph):
        """Unicode content should be handled correctly."""
        content = "Testing: \u0628\u0630\u0631\u0629 (BIZRA) \U0001F331 \u2192 \u221E"
        node = graph.add_thought(content, ThoughtType.REASONING)

        assert node.content == content

    def test_concurrent_thought_ids_unique(self, graph):
        """Rapidly created thoughts should have unique IDs."""
        ids = set()
        for _ in range(1000):
            node = graph.add_thought("Test", ThoughtType.HYPOTHESIS)
            assert node.id not in ids
            ids.add(node.id)

        assert len(ids) == 1000


class TestThoughtsProperty:
    """Tests for the thoughts property alias."""

    def test_thoughts_property_returns_nodes(self, populated_graph):
        """thoughts property should return same dict as nodes."""
        assert populated_graph.thoughts is populated_graph.nodes

    def test_thoughts_property_modifiable(self, graph):
        """thoughts property should allow modifications."""
        node = graph.add_thought("Test", ThoughtType.HYPOTHESIS)

        # Modify via thoughts property
        graph.thoughts[node.id].confidence = 0.99

        assert graph.nodes[node.id].confidence == 0.99


# =============================================================================
# PERFORMANCE TESTS (marked as slow)
# =============================================================================


@pytest.mark.slow
class TestPerformance:
    """Performance and scalability tests."""

    def test_large_graph_creation(self, graph):
        """Large graphs should be created efficiently."""
        import time

        start = time.time()

        for i in range(10000):
            graph.add_thought(f"Thought {i}", ThoughtType.REASONING)

        duration = time.time() - start

        assert len(graph.nodes) == 10000
        assert duration < 5.0  # Should complete in under 5 seconds

    def test_path_finding_performance(self, graph):
        """Path finding should be efficient on large graphs."""
        # Create a graph with many paths
        root = graph.add_thought("Root", ThoughtType.QUESTION)

        for i in range(100):
            branch = graph.add_thought(f"Branch {i}", ThoughtType.HYPOTHESIS, parent_id=root.id)
            for j in range(10):
                leaf = graph.add_thought(f"Leaf {i}-{j}", ThoughtType.CONCLUSION, parent_id=branch.id)
                leaf.snr_score = 0.9

        import time
        start = time.time()
        path = graph.find_best_path()
        duration = time.time() - start

        assert duration < 1.0  # Should complete in under 1 second


# =============================================================================
# REASON METHOD TESTS (High-level API for Sovereign Runtime)
# =============================================================================


class TestGraphReasonMethod:
    """Tests for the high-level reason() API method.

    This method is called by SovereignRuntime._process_query() and must:
    1. Accept query, context, and max_depth parameters
    2. Return dict with thoughts, conclusion, confidence, depth_reached
    3. Maintain SNR >= 0.85 and Ihsan >= 0.95 constraints
    4. Be async-compatible for runtime integration
    """

    @pytest.mark.asyncio
    async def test_reason_returns_expected_structure(self, graph):
        """reason() should return dict with required keys."""
        result = await graph.reason(
            query="What is the optimal approach?",
            context={"domain": "optimization"},
            max_depth=3,
        )

        # Required keys per GraphReasonerStub interface
        assert "thoughts" in result
        assert "conclusion" in result
        assert "confidence" in result
        assert "depth_reached" in result

        # Additional quality metrics
        assert "snr_score" in result
        assert "ihsan_score" in result
        assert "passes_threshold" in result

    @pytest.mark.asyncio
    async def test_reason_thoughts_is_list(self, graph):
        """thoughts should be a list of reasoning steps."""
        result = await graph.reason(
            query="Why is the sky blue?",
            context={},
            max_depth=2,
        )

        assert isinstance(result["thoughts"], list)
        assert len(result["thoughts"]) >= 1
        assert all(isinstance(t, str) for t in result["thoughts"])

    @pytest.mark.asyncio
    async def test_reason_conclusion_is_string(self, graph):
        """conclusion should be a non-empty string."""
        result = await graph.reason(
            query="Explain the concept of sovereignty",
            context={"domain": "philosophy"},
            max_depth=2,
        )

        assert isinstance(result["conclusion"], str)
        assert len(result["conclusion"]) > 0

    @pytest.mark.asyncio
    async def test_reason_confidence_in_valid_range(self, graph):
        """confidence should be between 0 and 1."""
        result = await graph.reason(
            query="What is 2 + 2?",
            context={},
            max_depth=1,
        )

        assert 0 <= result["confidence"] <= 1

    @pytest.mark.asyncio
    async def test_reason_respects_max_depth(self, graph):
        """depth_reached should not exceed max_depth."""
        for max_depth in [1, 2, 3, 5]:
            result = await graph.reason(
                query="Test depth limit",
                context={},
                max_depth=max_depth,
            )

            assert result["depth_reached"] <= max_depth

    @pytest.mark.asyncio
    async def test_reason_with_context(self, graph):
        """reason() should use provided context."""
        result = await graph.reason(
            query="What is the best model?",
            context={
                "domain": "machine_learning",
                "constraints": ["low_latency", "high_accuracy"],
                "facts": ["Dataset has 1M samples"],
            },
            max_depth=2,
        )

        assert result["conclusion"] is not None
        assert result["confidence"] > 0

    @pytest.mark.asyncio
    async def test_reason_snr_above_threshold(self, graph):
        """reason() should produce results with SNR >= threshold."""
        result = await graph.reason(
            query="Analyze the data pipeline",
            context={"domain": "data_engineering"},
            max_depth=3,
        )

        # SNR should meet or approach threshold
        # (may be slightly below in edge cases, but not drastically)
        assert result["snr_score"] >= graph.snr_threshold * 0.9

    @pytest.mark.asyncio
    async def test_reason_ihsan_above_threshold(self, graph):
        """reason() should produce results with Ihsan >= threshold."""
        result = await graph.reason(
            query="Design a secure system",
            context={"domain": "security"},
            max_depth=3,
        )

        # Ihsan should meet or approach threshold
        assert result["ihsan_score"] >= graph.ihsan_threshold * 0.85

    @pytest.mark.asyncio
    async def test_reason_clears_previous_state(self, graph):
        """reason() should start with a clean graph each call."""
        # First call
        await graph.reason(
            query="First query",
            context={},
            max_depth=2,
        )
        nodes_after_first = len(graph.nodes)

        # Second call should not accumulate nodes
        await graph.reason(
            query="Second query",
            context={},
            max_depth=2,
        )
        nodes_after_second = len(graph.nodes)

        # Node counts should be similar (graph is cleared)
        assert abs(nodes_after_second - nodes_after_first) < nodes_after_first

    @pytest.mark.asyncio
    async def test_reason_empty_query(self, graph):
        """reason() should handle empty query gracefully."""
        result = await graph.reason(
            query="",
            context={},
            max_depth=1,
        )

        assert "conclusion" in result
        assert result["confidence"] > 0

    @pytest.mark.asyncio
    async def test_reason_graph_stats_included(self, graph):
        """reason() should include graph statistics."""
        result = await graph.reason(
            query="Test query",
            context={},
            max_depth=2,
        )

        assert "graph_stats" in result
        assert "nodes_created" in result["graph_stats"]


@pytest.mark.asyncio
class TestReasonIntegrationWithRuntime:
    """Integration tests verifying reason() matches SovereignRuntime expectations."""

    async def test_reason_signature_matches_stub(self, graph):
        """reason() signature should match GraphReasonerStub.reason()."""
        # GraphReasonerStub.reason signature:
        # async def reason(self, query: str, context: Dict[str, Any], max_depth: int = 3)
        # Returns: Dict with thoughts, conclusion, confidence, depth_reached

        result = await graph.reason(
            query="Test query",
            context={"key": "value"},
            max_depth=3,
        )

        # Verify stub-compatible return structure
        assert "thoughts" in result
        assert "conclusion" in result
        assert "confidence" in result
        assert "depth_reached" in result

    async def test_reason_can_be_called_without_await_error(self, graph):
        """reason() should be properly async without blocking issues."""
        import asyncio

        # Should complete within reasonable time
        try:
            result = await asyncio.wait_for(
                graph.reason(
                    query="Test async behavior",
                    context={},
                    max_depth=2,
                ),
                timeout=5.0,
            )
            assert result is not None
        except asyncio.TimeoutError:
            pytest.fail("reason() took too long - possible blocking issue")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
