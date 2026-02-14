"""
Tests for the Graph-of-Thoughts Bridge — Multi-Path Reasoning Engine.

Covers: ThoughtNode, ThoughtGraph (all 6 GoT operations),
GoTBridge, visualization, and async reasoning pipeline.

Standing on Giants: Besta (2024), Wei (2022 CoT), Yao (2023 ToT)
"""

import pytest

from core.sovereign.runtime_engines.got_bridge import (
    GoTBridge,
    GoTResult,
    ThoughtEdge,
    ThoughtGraph,
    ThoughtNode,
    ThoughtStatus,
    ThoughtType,
    get_got_bridge,
    think,
    MAX_DEPTH,
    MAX_BRANCHES,
    PRUNE_THRESHOLD,
)


# ═══════════════════════════════════════════════════════════════════════════════
# ThoughtNode Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestThoughtNode:
    """Test ThoughtNode data structure."""

    def test_default_node(self):
        node = ThoughtNode()
        assert node.content == ""
        assert node.thought_type == ThoughtType.GENERATE
        assert node.score == 0.5
        assert node.depth == 0
        assert node.status == ThoughtStatus.ACTIVE
        assert len(node.id) == 8

    def test_combined_score(self):
        node = ThoughtNode(confidence=0.8, coherence=0.6, relevance=0.4)
        expected = 0.8 * 0.4 + 0.6 * 0.3 + 0.4 * 0.3
        assert abs(node.combined_score() - expected) < 1e-9

    def test_combined_score_perfect(self):
        node = ThoughtNode(confidence=1.0, coherence=1.0, relevance=1.0)
        assert abs(node.combined_score() - 1.0) < 1e-9

    def test_combined_score_zero(self):
        node = ThoughtNode(confidence=0.0, coherence=0.0, relevance=0.0)
        assert node.combined_score() == 0.0

    def test_heap_ordering(self):
        """Higher-scoring nodes should come first in heap comparison."""
        high = ThoughtNode(score=0.9)
        low = ThoughtNode(score=0.1)
        assert high < low  # __lt__ inverted for max-heap

    def test_unique_ids(self):
        """Each node gets a unique ID."""
        nodes = [ThoughtNode() for _ in range(100)]
        ids = {n.id for n in nodes}
        assert len(ids) == 100


# ═══════════════════════════════════════════════════════════════════════════════
# ThoughtEdge Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestThoughtEdge:
    """Test ThoughtEdge data structure."""

    def test_default_edge(self):
        edge = ThoughtEdge(source_id="a", target_id="b")
        assert edge.edge_type == "derives"
        assert edge.weight == 1.0

    def test_custom_edge(self):
        edge = ThoughtEdge(source_id="a", target_id="b", edge_type="aggregates", weight=0.5)
        assert edge.edge_type == "aggregates"
        assert edge.weight == 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# ThoughtGraph Core Operations Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestThoughtGraphOperations:
    """Test all 6 GoT operations: Generate, Aggregate, Refine, Validate, Prune, Backtrack."""

    @pytest.fixture
    def graph(self):
        return ThoughtGraph()

    @pytest.fixture
    def rooted_graph(self, graph):
        graph.create_root("Solve the problem")
        return graph

    # --- CREATE ROOT ---

    def test_create_root(self, graph):
        root = graph.create_root("Test goal")
        assert root.thought_type == ThoughtType.ROOT
        assert root.depth == 0
        assert root.score == 1.0
        assert root.id in graph._nodes
        assert graph._root is root

    def test_root_in_frontier(self, graph):
        graph.create_root("Test")
        assert len(graph._frontier) == 1

    # --- GENERATE ---

    def test_generate_creates_children(self, rooted_graph):
        root = rooted_graph._root
        children = rooted_graph.generate(root, "Test goal")
        assert len(children) > 0
        assert all(c.depth == 1 for c in children)
        assert all(c.thought_type == ThoughtType.GENERATE for c in children)

    def test_generate_creates_edges(self, rooted_graph):
        root = rooted_graph._root
        children = rooted_graph.generate(root, "Test goal")
        assert len(rooted_graph._edges) == len(children)
        assert all(e.source_id == root.id for e in rooted_graph._edges)

    def test_generate_respects_max_branches(self):
        graph = ThoughtGraph(max_branches=2)
        root = graph.create_root("Test")
        children = graph.generate(root, "Test")
        assert len(children) <= 2

    def test_generate_respects_max_depth(self):
        graph = ThoughtGraph(max_depth=1)
        root = graph.create_root("Test")
        children = graph.generate(root, "Test")
        # Children are at depth 1, so they can't generate further
        for child in children:
            grandchildren = graph.generate(child, "Test")
            assert len(grandchildren) == 0

    def test_generate_updates_parent_children(self, rooted_graph):
        root = rooted_graph._root
        children = rooted_graph.generate(root, "Test")
        assert len(root.child_ids) == len(children)

    # --- AGGREGATE ---

    def test_aggregate_two_nodes(self, rooted_graph):
        root = rooted_graph._root
        children = rooted_graph.generate(root, "Test")
        assert len(children) >= 2

        aggregated = rooted_graph.aggregate(children[:2])
        assert aggregated.thought_type == ThoughtType.AGGREGATE
        assert len(aggregated.parent_ids) == 2
        assert "Synthesis:" in aggregated.content

    def test_aggregate_depth(self, rooted_graph):
        root = rooted_graph._root
        children = rooted_graph.generate(root, "Test")
        aggregated = rooted_graph.aggregate(children[:2])
        assert aggregated.depth == max(c.depth for c in children[:2]) + 1

    def test_aggregate_score_bonus(self, rooted_graph):
        root = rooted_graph._root
        children = rooted_graph.generate(root, "Test")
        avg = sum(c.score for c in children[:2]) / 2
        aggregated = rooted_graph.aggregate(children[:2])
        # Score gets 1.1x bonus
        assert abs(aggregated.score - avg * 1.1) < 1e-9

    def test_aggregate_empty_raises(self, graph):
        with pytest.raises(ValueError, match="Cannot aggregate empty"):
            graph.aggregate([])

    # --- REFINE ---

    def test_refine_default_iterations(self, rooted_graph):
        root = rooted_graph._root
        children = rooted_graph.generate(root, "Test")
        refined = rooted_graph.refine(children[0])
        assert "[refined v3]" in refined.content
        assert refined.thought_type == ThoughtType.REFINE

    def test_refine_custom_iterations(self, rooted_graph):
        root = rooted_graph._root
        children = rooted_graph.generate(root, "Test")
        refined = rooted_graph.refine(children[0], iterations=1)
        assert "[refined v1]" in refined.content

    def test_refine_improves_score(self, rooted_graph):
        root = rooted_graph._root
        children = rooted_graph.generate(root, "Test")
        original_score = children[0].score
        refined = rooted_graph.refine(children[0])
        assert refined.score >= original_score

    def test_refine_scores_capped_at_one(self, graph):
        node = ThoughtNode(score=0.99, confidence=0.99, coherence=0.99)
        graph._nodes[node.id] = node
        refined = graph.refine(node, iterations=10)
        assert refined.score <= 1.0
        assert refined.confidence <= 1.0

    # --- VALIDATE ---

    def test_validate_updates_score(self, rooted_graph):
        root = rooted_graph._root
        children = rooted_graph.generate(root, "Test")
        old_score = children[0].score
        new_score = rooted_graph.validate(children[0])
        assert children[0].score == new_score

    def test_validate_custom_scorer(self, graph):
        graph.set_scorer(lambda n: 0.42)
        node = ThoughtNode()
        graph._nodes[node.id] = node
        score = graph.validate(node)
        assert score == 0.42
        assert node.score == 0.42

    def test_default_scorer_depth_penalty(self, graph):
        shallow = ThoughtNode(depth=0, confidence=0.5, coherence=0.5, relevance=0.5)
        deep = ThoughtNode(depth=5, confidence=0.5, coherence=0.5, relevance=0.5)
        shallow_score = graph._default_scorer(shallow)
        deep_score = graph._default_scorer(deep)
        assert shallow_score > deep_score

    # --- PRUNE ---

    def test_prune_removes_low_quality(self, graph):
        graph.create_root("Test")
        low = ThoughtNode(score=0.1)  # Below default 0.3 threshold
        graph._nodes[low.id] = low
        graph._frontier.append(low)
        pruned = graph.prune()
        assert pruned >= 1

    def test_prune_keeps_high_quality(self, rooted_graph):
        initial_frontier_size = len(rooted_graph._frontier)
        pruned = rooted_graph.prune()
        # Root has score 1.0, should not be pruned
        assert len(rooted_graph._frontier) == initial_frontier_size - pruned

    def test_prune_custom_threshold(self):
        graph = ThoughtGraph(prune_threshold=0.8)
        root = graph.create_root("Test")
        children = graph.generate(root, "Test")
        # Children have score 0.5, below 0.8 threshold
        pruned = graph.prune()
        assert pruned > 0

    # --- BACKTRACK ---

    def test_backtrack_to_node(self, rooted_graph):
        root = rooted_graph._root
        children = rooted_graph.generate(root, "Test")
        # Explore children, then backtrack to root
        result = rooted_graph.backtrack(root.id)
        assert result is root
        assert len(rooted_graph._frontier) == 1

    def test_backtrack_unknown_node(self, rooted_graph):
        result = rooted_graph.backtrack("nonexistent")
        assert result is None

    # --- MARK SOLUTION ---

    def test_mark_solution(self, graph):
        node = ThoughtNode(content="Answer")
        graph._nodes[node.id] = node
        graph.mark_solution(node)
        assert node.status == ThoughtStatus.SOLUTION
        assert node.thought_type == ThoughtType.SOLUTION
        assert node in graph._solutions

    # --- BEST PATH ---

    def test_get_best_path_empty(self, graph):
        assert graph.get_best_path() == []

    def test_get_best_path_with_solution(self, rooted_graph):
        root = rooted_graph._root
        children = rooted_graph.generate(root, "Test")
        rooted_graph.mark_solution(children[0])
        path = rooted_graph.get_best_path()
        assert len(path) >= 2
        assert path[0] is root
        assert path[-1] is children[0]


# ═══════════════════════════════════════════════════════════════════════════════
# Async Reasoning Pipeline Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestReasoningPipeline:
    """Test the full async reasoning pipeline."""

    @pytest.mark.asyncio
    async def test_reason_finds_solution(self):
        graph = ThoughtGraph()
        result = await graph.reason("Test problem", max_iterations=50)
        assert isinstance(result, GoTResult)
        assert result.goal == "Test problem"
        assert result.explored_nodes > 0
        assert result.execution_time_ms > 0

    @pytest.mark.asyncio
    async def test_reason_respects_max_iterations(self):
        graph = ThoughtGraph()
        result = await graph.reason("Test", max_iterations=3)
        assert result.explored_nodes <= 3

    @pytest.mark.asyncio
    async def test_reason_with_custom_scorer(self):
        graph = ThoughtGraph()
        graph.set_scorer(lambda n: 0.95 if n.depth >= 2 else 0.5)
        result = await graph.reason("Test", max_iterations=30)
        assert result.success

    @pytest.mark.asyncio
    async def test_reason_with_custom_generator(self):
        def gen(parent, goal):
            return [f"Custom thought about {goal}"]

        graph = ThoughtGraph()
        graph.set_generator(gen)
        result = await graph.reason("Custom test", max_iterations=20)
        assert result.explored_nodes > 0


# ═══════════════════════════════════════════════════════════════════════════════
# GoTBridge Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestGoTBridge:
    """Test the GoT bridge (Python fallback mode)."""

    @pytest.mark.asyncio
    async def test_bridge_reason(self):
        bridge = GoTBridge(use_rust=False)
        result = await bridge.reason("Bridge test", max_iterations=10)
        assert isinstance(result, GoTResult)
        assert result.goal == "Bridge test"

    @pytest.mark.asyncio
    async def test_bridge_with_scorer(self):
        bridge = GoTBridge(use_rust=False)
        result = await bridge.reason(
            "Test", max_iterations=10, scorer=lambda n: 0.5
        )
        assert isinstance(result, GoTResult)

    def test_bridge_visualize_empty(self):
        bridge = GoTBridge(use_rust=False)
        viz = bridge.visualize_last_graph()
        # Python graph is initialized but empty
        assert isinstance(viz, str)

    def test_get_got_bridge_singleton(self):
        bridge1 = get_got_bridge()
        bridge2 = get_got_bridge()
        assert bridge1 is bridge2

    @pytest.mark.asyncio
    async def test_think_convenience(self):
        result = await think("Quick test", max_iterations=5)
        assert isinstance(result, GoTResult)


# ═══════════════════════════════════════════════════════════════════════════════
# Visualization Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestVisualization:
    """Test graph visualization."""

    def test_empty_graph_visualization(self):
        graph = ThoughtGraph()
        viz = graph.visualize()
        assert viz == "Empty graph"

    def test_rooted_graph_visualization(self):
        graph = ThoughtGraph()
        graph.create_root("Test goal")
        viz = graph.visualize()
        assert "Graph of Thoughts" in viz
        assert "Test goal" in viz

    def test_visualization_with_children(self):
        graph = ThoughtGraph()
        root = graph.create_root("Goal")
        graph.generate(root, "Goal")
        viz = graph.visualize()
        assert "Approach" in viz  # Default generator uses "Approach" prefix

    def test_visualization_shows_status_icons(self):
        graph = ThoughtGraph()
        root = graph.create_root("Goal")
        children = graph.generate(root, "Goal")
        graph.mark_solution(children[0])
        viz = graph.visualize()
        assert "○" in viz or "★" in viz  # Active or Solution icons


# ═══════════════════════════════════════════════════════════════════════════════
# Constants Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestConstants:
    """Test GoT configuration constants."""

    def test_max_depth_reasonable(self):
        assert 1 <= MAX_DEPTH <= 100

    def test_max_branches_reasonable(self):
        assert 1 <= MAX_BRANCHES <= 50

    def test_prune_threshold_valid(self):
        assert 0.0 <= PRUNE_THRESHOLD <= 1.0
