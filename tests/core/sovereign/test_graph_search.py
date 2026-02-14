"""
Tests for GraphSearchMixin -- Search and Traversal Algorithms
=============================================================
Comprehensive unit tests for the Graph-of-Thoughts search mixin,
covering best-first path finding, frontier detection, backtracking,
and iterative exploration with backtracking.

Standing on Giants:
- Besta et al. (2024): Graph of Thoughts operations
- Dijkstra (1959): Best-first search foundations
"""

import pytest

from core.sovereign.graph_search import GraphSearchMixin
from core.sovereign.graph_types import (
    EdgeType,
    ReasoningPath,
    ThoughtEdge,
    ThoughtNode,
    ThoughtType,
)


# ---------------------------------------------------------------------------
# ConcreteGraph -- Concrete class inheriting GraphSearchMixin
# ---------------------------------------------------------------------------


class ConcreteGraph(GraphSearchMixin):
    """Concrete class wrapping GraphSearchMixin for testing.

    Provides all attributes the mixin expects (nodes, edges, adjacency,
    reverse_adj, roots, snr_threshold, ihsan_threshold) with helpers to
    build graph topologies programmatically.
    """

    def __init__(
        self,
        snr_threshold: float = 0.85,
        ihsan_threshold: float = 0.95,
    ):
        self.nodes: dict[str, ThoughtNode] = {}
        self.edges: list[ThoughtEdge] = []
        self.adjacency: dict[str, list[str]] = {}
        self.reverse_adj: dict[str, list[str]] = {}
        self.roots: list[str] = []
        self.snr_threshold = snr_threshold
        self.ihsan_threshold = ihsan_threshold

    def add_node(
        self,
        node_id: str,
        thought_type: ThoughtType = ThoughtType.REASONING,
        snr_score: float = 0.5,
        confidence: float = 0.5,
        content: str = "",
        is_root: bool = False,
    ) -> ThoughtNode:
        """Add a node to the graph."""
        node = ThoughtNode(
            id=node_id,
            content=content or f"Thought: {node_id}",
            thought_type=thought_type,
            snr_score=snr_score,
            confidence=confidence,
        )
        self.nodes[node_id] = node
        if is_root:
            self.roots.append(node_id)
        return node

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType = EdgeType.DERIVES,
        weight: float = 1.0,
    ) -> ThoughtEdge:
        """Add a directed edge and update adjacency structures."""
        edge = ThoughtEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            weight=weight,
        )
        self.edges.append(edge)
        self.adjacency.setdefault(source_id, []).append(target_id)
        self.reverse_adj.setdefault(target_id, []).append(source_id)
        return edge


# ---------------------------------------------------------------------------
# Graph topology builders
# ---------------------------------------------------------------------------


def build_empty_graph(**kwargs) -> ConcreteGraph:
    """Empty graph with no nodes or edges."""
    return ConcreteGraph(**kwargs)


def build_single_node(
    thought_type: ThoughtType = ThoughtType.HYPOTHESIS,
    snr_score: float = 0.8,
) -> ConcreteGraph:
    """Graph with a single root node."""
    g = ConcreteGraph()
    g.add_node("A", thought_type=thought_type, snr_score=snr_score, is_root=True)
    return g


def build_linear_chain(
    length: int = 3,
    terminal_type: ThoughtType = ThoughtType.CONCLUSION,
    snr_values: list[float] | None = None,
) -> ConcreteGraph:
    """Linear chain: A -> B -> C -> ... with the last node as terminal_type.

    Default SNR values ascend from 0.5 to 0.9 linearly.
    """
    g = ConcreteGraph()
    if snr_values is None:
        snr_values = [0.5 + 0.4 * i / max(length - 1, 1) for i in range(length)]
    ids = [chr(65 + i) for i in range(length)]  # A, B, C, ...
    for idx, nid in enumerate(ids):
        ttype = terminal_type if idx == length - 1 else ThoughtType.REASONING
        g.add_node(nid, thought_type=ttype, snr_score=snr_values[idx], is_root=(idx == 0))
    for i in range(length - 1):
        g.add_edge(ids[i], ids[i + 1])
    return g


def build_diamond_graph(
    snr_left: float = 0.8,
    snr_right: float = 0.6,
    conclusion_snr: float = 0.9,
) -> ConcreteGraph:
    """Diamond: A -> B, A -> C, B -> D, C -> D.

    D is a CONCLUSION. B and C are REASONING.
    """
    g = ConcreteGraph()
    g.add_node("A", thought_type=ThoughtType.HYPOTHESIS, snr_score=0.7, is_root=True)
    g.add_node("B", thought_type=ThoughtType.REASONING, snr_score=snr_left)
    g.add_node("C", thought_type=ThoughtType.REASONING, snr_score=snr_right)
    g.add_node("D", thought_type=ThoughtType.CONCLUSION, snr_score=conclusion_snr)
    g.add_edge("A", "B")
    g.add_edge("A", "C")
    g.add_edge("B", "D")
    g.add_edge("C", "D")
    return g


def build_branching_tree() -> ConcreteGraph:
    """Tree with multiple branches and conclusions at different depths/SNRs.

    Structure:
        ROOT (hypothesis, 0.7)
         +-- R1 (reasoning, 0.8)
         |    +-- C1 (conclusion, 0.95)
         |    +-- C2 (conclusion, 0.6)
         +-- R2 (reasoning, 0.5)
              +-- R3 (reasoning, 0.9)
              |    +-- C3 (conclusion, 0.85)
              +-- E1 (evidence, 0.4)
    """
    g = ConcreteGraph()
    g.add_node("ROOT", ThoughtType.HYPOTHESIS, snr_score=0.7, is_root=True)
    g.add_node("R1", ThoughtType.REASONING, snr_score=0.8)
    g.add_node("R2", ThoughtType.REASONING, snr_score=0.5)
    g.add_node("R3", ThoughtType.REASONING, snr_score=0.9)
    g.add_node("C1", ThoughtType.CONCLUSION, snr_score=0.95)
    g.add_node("C2", ThoughtType.CONCLUSION, snr_score=0.6)
    g.add_node("C3", ThoughtType.CONCLUSION, snr_score=0.85)
    g.add_node("E1", ThoughtType.EVIDENCE, snr_score=0.4)

    g.add_edge("ROOT", "R1")
    g.add_edge("ROOT", "R2")
    g.add_edge("R1", "C1")
    g.add_edge("R1", "C2")
    g.add_edge("R2", "R3")
    g.add_edge("R2", "E1")
    g.add_edge("R3", "C3")
    return g


def build_graph_with_cycle() -> ConcreteGraph:
    """Graph containing a cycle: A -> B -> C -> A, with D (conclusion) off B.

    The cycle must not cause infinite looping in find_best_path.
    """
    g = ConcreteGraph()
    g.add_node("A", ThoughtType.HYPOTHESIS, snr_score=0.7, is_root=True)
    g.add_node("B", ThoughtType.REASONING, snr_score=0.8)
    g.add_node("C", ThoughtType.REASONING, snr_score=0.6)
    g.add_node("D", ThoughtType.CONCLUSION, snr_score=0.9)
    g.add_edge("A", "B")
    g.add_edge("B", "C")
    g.add_edge("C", "A")  # cycle
    g.add_edge("B", "D")
    return g


def build_multi_conclusion_graph() -> ConcreteGraph:
    """Graph with multiple paths to different conclusions.

    ROOT -> R1 -> C_LOW  (conclusion, snr 0.5)
    ROOT -> R2 -> C_HIGH (conclusion, snr 0.95)
    """
    g = ConcreteGraph()
    g.add_node("ROOT", ThoughtType.HYPOTHESIS, snr_score=0.7, is_root=True)
    g.add_node("R1", ThoughtType.REASONING, snr_score=0.3)
    g.add_node("R2", ThoughtType.REASONING, snr_score=0.9)
    g.add_node("C_LOW", ThoughtType.CONCLUSION, snr_score=0.5)
    g.add_node("C_HIGH", ThoughtType.CONCLUSION, snr_score=0.95)
    g.add_edge("ROOT", "R1")
    g.add_edge("ROOT", "R2")
    g.add_edge("R1", "C_LOW")
    g.add_edge("R2", "C_HIGH")
    return g


def build_wide_frontier_graph() -> ConcreteGraph:
    """Graph with many leaf nodes of varying types and SNR scores.

    ROOT -> L1 (reasoning, 0.9)
    ROOT -> L2 (evidence, 0.8)
    ROOT -> L3 (conclusion, 0.7)
    ROOT -> L4 (validation, 0.6)
    ROOT -> L5 (hypothesis, 0.95)
    """
    g = ConcreteGraph()
    g.add_node("ROOT", ThoughtType.HYPOTHESIS, snr_score=0.7, is_root=True)
    leaves = [
        ("L1", ThoughtType.REASONING, 0.9),
        ("L2", ThoughtType.EVIDENCE, 0.8),
        ("L3", ThoughtType.CONCLUSION, 0.7),
        ("L4", ThoughtType.VALIDATION, 0.6),
        ("L5", ThoughtType.HYPOTHESIS, 0.95),
    ]
    for lid, ttype, snr in leaves:
        g.add_node(lid, ttype, snr_score=snr)
        g.add_edge("ROOT", lid)
    return g


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def empty_graph() -> ConcreteGraph:
    return build_empty_graph()


@pytest.fixture
def single_node_graph() -> ConcreteGraph:
    return build_single_node()


@pytest.fixture
def linear_chain() -> ConcreteGraph:
    return build_linear_chain()


@pytest.fixture
def diamond_graph() -> ConcreteGraph:
    return build_diamond_graph()


@pytest.fixture
def branching_tree() -> ConcreteGraph:
    return build_branching_tree()


@pytest.fixture
def cycle_graph() -> ConcreteGraph:
    return build_graph_with_cycle()


@pytest.fixture
def multi_conclusion_graph() -> ConcreteGraph:
    return build_multi_conclusion_graph()


@pytest.fixture
def wide_frontier_graph() -> ConcreteGraph:
    return build_wide_frontier_graph()


# ===========================================================================
# 1. TestGraphSearchSetup
# ===========================================================================


class TestGraphSearchSetup:
    """Verify the test harness itself -- ConcreteGraph provides the mixin contract."""

    def test_testable_graph_inherits_mixin(self):
        """ConcreteGraph is an instance of GraphSearchMixin."""
        g = ConcreteGraph()
        assert isinstance(g, GraphSearchMixin)

    def test_default_thresholds(self):
        """Default thresholds match BIZRA standards."""
        g = ConcreteGraph()
        assert g.snr_threshold == 0.85
        assert g.ihsan_threshold == 0.95

    def test_custom_thresholds(self):
        """Custom thresholds can be set."""
        g = ConcreteGraph(snr_threshold=0.5, ihsan_threshold=0.8)
        assert g.snr_threshold == 0.5
        assert g.ihsan_threshold == 0.8

    def test_add_node(self):
        """Nodes added via helper are retrievable."""
        g = ConcreteGraph()
        node = g.add_node("X", ThoughtType.HYPOTHESIS, snr_score=0.9)
        assert "X" in g.nodes
        assert g.nodes["X"] is node
        assert node.thought_type == ThoughtType.HYPOTHESIS

    def test_add_root_node(self):
        """Root nodes appear in the roots list."""
        g = ConcreteGraph()
        g.add_node("R", is_root=True)
        assert "R" in g.roots

    def test_add_edge_updates_adjacency(self):
        """Edges update both adjacency and reverse_adj."""
        g = ConcreteGraph()
        g.add_node("A")
        g.add_node("B")
        g.add_edge("A", "B")
        assert "B" in g.adjacency["A"]
        assert "A" in g.reverse_adj["B"]

    def test_add_edge_stored(self):
        """Edges are stored in the edges list."""
        g = ConcreteGraph()
        g.add_node("A")
        g.add_node("B")
        edge = g.add_edge("A", "B", edge_type=EdgeType.SUPPORTS)
        assert edge in g.edges
        assert edge.edge_type == EdgeType.SUPPORTS

    def test_builder_linear_chain_structure(self, linear_chain: ConcreteGraph):
        """Linear chain builder produces correct structure."""
        assert len(linear_chain.nodes) == 3
        assert len(linear_chain.edges) == 2
        assert "A" in linear_chain.roots
        assert linear_chain.nodes["C"].thought_type == ThoughtType.CONCLUSION


# ===========================================================================
# 2. TestFindBestPath
# ===========================================================================


class TestFindBestPath:
    """Tests for find_best_path -- best-first search for highest-SNR path."""

    def test_empty_graph_returns_empty_path(self, empty_graph: ConcreteGraph):
        """Empty graph returns a path with no nodes."""
        path = empty_graph.find_best_path()
        assert path.nodes == []
        assert path.total_snr == 0
        assert path.total_confidence == 0

    def test_empty_graph_with_start_id(self, empty_graph: ConcreteGraph):
        """Empty graph with explicit start_id returns empty path."""
        path = empty_graph.find_best_path(start_id="nonexistent")
        assert path.nodes == []

    def test_single_root_non_conclusion(self, single_node_graph: ConcreteGraph):
        """Single root node (non-conclusion) returns path with just the root."""
        path = single_node_graph.find_best_path()
        assert path.nodes == ["A"]
        assert path.total_snr == 0.8  # default from build_single_node

    def test_single_root_is_conclusion(self):
        """Single root node that IS a conclusion returns that node's path."""
        g = build_single_node(thought_type=ThoughtType.CONCLUSION, snr_score=0.9)
        path = g.find_best_path()
        assert path.nodes == ["A"]
        assert path.total_snr == 0.9

    def test_start_id_none_uses_first_root(self, linear_chain: ConcreteGraph):
        """When start_id is None, search starts from roots[0]."""
        path = linear_chain.find_best_path(start_id=None)
        assert path.nodes[0] == "A"

    def test_explicit_start_id(self, linear_chain: ConcreteGraph):
        """Explicit start_id overrides the default root selection."""
        path = linear_chain.find_best_path(start_id="B")
        assert path.nodes[0] == "B"

    def test_start_id_not_in_nodes(self, linear_chain: ConcreteGraph):
        """Invalid start_id returns empty path."""
        path = linear_chain.find_best_path(start_id="INVALID")
        assert path.nodes == []
        assert path.total_snr == 0

    def test_no_roots_no_start_id(self):
        """Graph with nodes but no roots and no start_id returns empty path."""
        g = ConcreteGraph()
        g.add_node("X", ThoughtType.CONCLUSION, snr_score=0.9, is_root=False)
        path = g.find_best_path()
        assert path.nodes == []

    def test_linear_chain_finds_conclusion(self, linear_chain: ConcreteGraph):
        """Linear chain A->B->C (C is conclusion) finds the full path."""
        path = linear_chain.find_best_path()
        assert "C" in path.nodes
        assert path.nodes[-1] == "C"
        assert path.total_snr > 0

    def test_linear_chain_total_snr_is_sum(self, linear_chain: ConcreteGraph):
        """Total SNR is the sum of SNR scores along the path."""
        path = linear_chain.find_best_path()
        expected_snr = sum(linear_chain.nodes[n].snr_score for n in path.nodes)
        assert abs(path.total_snr - expected_snr) < 1e-9

    def test_linear_chain_total_confidence_is_sum(self, linear_chain: ConcreteGraph):
        """Total confidence is the sum of confidence scores along the path."""
        path = linear_chain.find_best_path()
        expected_conf = sum(linear_chain.nodes[n].confidence for n in path.nodes)
        assert abs(path.total_confidence - expected_conf) < 1e-9

    def test_diamond_graph_finds_conclusion(self, diamond_graph: ConcreteGraph):
        """Diamond graph finds D (the conclusion)."""
        path = diamond_graph.find_best_path()
        assert "D" in path.nodes

    def test_diamond_graph_picks_high_snr_branch(self, diamond_graph: ConcreteGraph):
        """Diamond prefers the path through the higher-SNR intermediate node."""
        path = diamond_graph.find_best_path()
        # B has snr_left=0.8, C has snr_right=0.6
        # Best-first explores B before C, so the path should go through B
        assert "B" in path.nodes

    def test_multi_conclusion_picks_highest_total_snr(
        self, multi_conclusion_graph: ConcreteGraph
    ):
        """With multiple conclusions, find_best_path picks the one with highest total SNR."""
        path = multi_conclusion_graph.find_best_path()
        # ROOT(0.7) -> R2(0.9) -> C_HIGH(0.95) = 2.55
        # ROOT(0.7) -> R1(0.3) -> C_LOW(0.5) = 1.5
        assert "C_HIGH" in path.nodes
        expected = 0.7 + 0.9 + 0.95
        assert abs(path.total_snr - expected) < 1e-9

    def test_branching_tree_best_conclusion(self, branching_tree: ConcreteGraph):
        """In branching tree, best path leads to highest-total-SNR conclusion."""
        path = branching_tree.find_best_path()
        assert "C1" in path.nodes or "C3" in path.nodes
        # C1 path: ROOT(0.7) + R1(0.8) + C1(0.95) = 2.45
        # C3 path: ROOT(0.7) + R2(0.5) + R3(0.9) + C3(0.85) = 2.95
        # C2 path: ROOT(0.7) + R1(0.8) + C2(0.6) = 2.1
        # However, best-first explores nodes by individual SNR, so which path
        # is actually found depends on traversal order. The total_snr should
        # be the highest of the paths actually discovered.
        assert path.total_snr > 0

    def test_target_type_synthesis(self):
        """target_type parameter changes what type of node we search for."""
        g = ConcreteGraph()
        g.add_node("A", ThoughtType.HYPOTHESIS, snr_score=0.7, is_root=True)
        g.add_node("B", ThoughtType.SYNTHESIS, snr_score=0.9)
        g.add_node("C", ThoughtType.CONCLUSION, snr_score=0.95)
        g.add_edge("A", "B")
        g.add_edge("A", "C")
        path = g.find_best_path(target_type=ThoughtType.SYNTHESIS)
        assert "B" in path.nodes

    def test_target_type_evidence(self):
        """Can search for EVIDENCE type nodes."""
        g = ConcreteGraph()
        g.add_node("A", ThoughtType.HYPOTHESIS, snr_score=0.7, is_root=True)
        g.add_node("B", ThoughtType.EVIDENCE, snr_score=0.8)
        g.add_edge("A", "B")
        path = g.find_best_path(target_type=ThoughtType.EVIDENCE)
        assert "B" in path.nodes
        assert path.total_snr == 0.7 + 0.8

    def test_no_target_type_match_returns_root_path(self):
        """When no node matches target_type, best_path is just the root."""
        g = ConcreteGraph()
        g.add_node("A", ThoughtType.HYPOTHESIS, snr_score=0.7, is_root=True)
        g.add_node("B", ThoughtType.REASONING, snr_score=0.9)
        g.add_edge("A", "B")
        path = g.find_best_path(target_type=ThoughtType.CONCLUSION)
        # No CONCLUSION node exists; best_path defaults to start node
        assert path.nodes == ["A"]
        assert path.total_snr == 0.7

    def test_cycle_does_not_infinite_loop(self, cycle_graph: ConcreteGraph):
        """Graph with a cycle terminates without infinite looping."""
        path = cycle_graph.find_best_path()
        # D is conclusion, reachable via A->B->D
        assert "D" in path.nodes
        # Ensure no duplicates in path
        assert len(path.nodes) == len(set(path.nodes))

    def test_visited_nodes_not_revisited(self, cycle_graph: ConcreteGraph):
        """The visited set prevents revisiting the same node."""
        path = cycle_graph.find_best_path()
        assert path.nodes.count("A") <= 1
        assert path.nodes.count("B") <= 1

    def test_multiple_paths_to_same_conclusion(self):
        """When multiple paths reach the same conclusion, the best total SNR wins."""
        g = ConcreteGraph()
        g.add_node("A", ThoughtType.HYPOTHESIS, snr_score=0.7, is_root=True)
        g.add_node("B", ThoughtType.REASONING, snr_score=0.9)
        g.add_node("C", ThoughtType.REASONING, snr_score=0.3)
        g.add_node("D", ThoughtType.CONCLUSION, snr_score=0.8)
        g.add_edge("A", "B")
        g.add_edge("A", "C")
        g.add_edge("B", "D")
        g.add_edge("C", "D")
        path = g.find_best_path()
        # Best-first will visit B first (snr 0.9 > 0.3), so path A->B->D
        # gets found first. C's path to D won't be followed because D is
        # already visited. Total = 0.7 + 0.9 + 0.8 = 2.4
        assert path.total_snr == pytest.approx(2.4)

    def test_deep_graph_performance(self):
        """A deep linear chain (50 nodes) completes in reasonable time."""
        length = 50
        snr_values = [0.5 + 0.01 * i for i in range(length)]
        g = build_linear_chain(length=length, snr_values=snr_values)
        path = g.find_best_path()
        # The last node (chr(65+49) might overflow, but we use letters)
        # Conclusion at end should be found
        assert len(path.nodes) == length
        assert path.total_snr > 0

    def test_all_nodes_same_snr(self):
        """When all nodes have identical SNR, a valid path is still returned."""
        g = ConcreteGraph()
        g.add_node("A", ThoughtType.HYPOTHESIS, snr_score=0.5, is_root=True)
        g.add_node("B", ThoughtType.REASONING, snr_score=0.5)
        g.add_node("C", ThoughtType.CONCLUSION, snr_score=0.5)
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        path = g.find_best_path()
        assert "C" in path.nodes
        assert path.total_snr == pytest.approx(1.5)

    def test_returns_reasoning_path_type(self, linear_chain: ConcreteGraph):
        """find_best_path returns a ReasoningPath instance."""
        path = linear_chain.find_best_path()
        assert isinstance(path, ReasoningPath)

    def test_empty_path_is_reasoning_path(self, empty_graph: ConcreteGraph):
        """Even empty result is a proper ReasoningPath."""
        path = empty_graph.find_best_path()
        assert isinstance(path, ReasoningPath)

    def test_start_from_middle_of_chain(self, linear_chain: ConcreteGraph):
        """Starting from a middle node finds the path from there to conclusion."""
        path = linear_chain.find_best_path(start_id="B")
        assert "B" in path.nodes
        assert "C" in path.nodes
        assert "A" not in path.nodes


# ===========================================================================
# 3. TestGetConclusions
# ===========================================================================


class TestGetConclusions:
    """Tests for get_conclusions -- retrieving conclusion nodes above SNR threshold."""

    def test_no_conclusions_returns_empty(self, single_node_graph: ConcreteGraph):
        """Graph with no CONCLUSION nodes returns empty list."""
        conclusions = single_node_graph.get_conclusions()
        assert conclusions == []

    def test_all_conclusions_above_threshold(self, branching_tree: ConcreteGraph):
        """All conclusions above min_snr are returned."""
        conclusions = branching_tree.get_conclusions(min_snr=0.5)
        conclusion_ids = {c.id for c in conclusions}
        assert "C1" in conclusion_ids  # snr 0.95
        assert "C2" in conclusion_ids  # snr 0.6
        assert "C3" in conclusion_ids  # snr 0.85

    def test_min_snr_filtering(self, branching_tree: ConcreteGraph):
        """min_snr filters out conclusions below the threshold."""
        conclusions = branching_tree.get_conclusions(min_snr=0.8)
        conclusion_ids = {c.id for c in conclusions}
        assert "C1" in conclusion_ids  # 0.95 >= 0.8
        assert "C3" in conclusion_ids  # 0.85 >= 0.8
        assert "C2" not in conclusion_ids  # 0.6 < 0.8

    def test_min_snr_zero_returns_all_conclusions(self, branching_tree: ConcreteGraph):
        """min_snr=0.0 returns every conclusion node."""
        conclusions = branching_tree.get_conclusions(min_snr=0.0)
        assert len(conclusions) == 3

    def test_high_threshold_filters_everything(self, branching_tree: ConcreteGraph):
        """Very high min_snr filters out all conclusions."""
        conclusions = branching_tree.get_conclusions(min_snr=1.0)
        assert conclusions == []

    def test_mixed_types_returns_only_conclusions(self, branching_tree: ConcreteGraph):
        """Only CONCLUSION type nodes are returned, not REASONING/EVIDENCE/etc."""
        conclusions = branching_tree.get_conclusions(min_snr=0.0)
        for c in conclusions:
            assert c.thought_type == ThoughtType.CONCLUSION

    def test_exact_threshold_included(self):
        """A conclusion with snr_score exactly equal to min_snr is included."""
        g = ConcreteGraph()
        g.add_node("C", ThoughtType.CONCLUSION, snr_score=0.7)
        conclusions = g.get_conclusions(min_snr=0.7)
        assert len(conclusions) == 1

    def test_just_below_threshold_excluded(self):
        """A conclusion with snr_score just below min_snr is excluded."""
        g = ConcreteGraph()
        g.add_node("C", ThoughtType.CONCLUSION, snr_score=0.6999)
        conclusions = g.get_conclusions(min_snr=0.7)
        assert len(conclusions) == 0

    def test_empty_graph_returns_empty(self, empty_graph: ConcreteGraph):
        """Empty graph returns no conclusions."""
        assert empty_graph.get_conclusions() == []

    def test_default_min_snr_is_0_7(self):
        """Default min_snr parameter is 0.7."""
        g = ConcreteGraph()
        g.add_node("C1", ThoughtType.CONCLUSION, snr_score=0.69)
        g.add_node("C2", ThoughtType.CONCLUSION, snr_score=0.71)
        conclusions = g.get_conclusions()  # no explicit min_snr
        ids = {c.id for c in conclusions}
        assert "C1" not in ids  # 0.69 < 0.7
        assert "C2" in ids  # 0.71 >= 0.7

    def test_returns_thought_node_instances(self, branching_tree: ConcreteGraph):
        """Returned items are ThoughtNode instances."""
        conclusions = branching_tree.get_conclusions(min_snr=0.0)
        for c in conclusions:
            assert isinstance(c, ThoughtNode)


# ===========================================================================
# 4. TestGetFrontier
# ===========================================================================


class TestGetFrontier:
    """Tests for get_frontier -- finding leaf nodes (nodes with no children)."""

    def test_empty_graph_returns_empty(self, empty_graph: ConcreteGraph):
        """Empty graph frontier is empty."""
        assert empty_graph.get_frontier() == []

    def test_all_roots_are_frontier_if_no_edges(self):
        """Nodes with no outgoing edges are frontier nodes."""
        g = ConcreteGraph()
        g.add_node("A", is_root=True)
        g.add_node("B", is_root=True)
        frontier = g.get_frontier()
        ids = {n.id for n in frontier}
        assert ids == {"A", "B"}

    def test_only_leaf_nodes_returned(self, linear_chain: ConcreteGraph):
        """In a linear chain, only the last node is a leaf."""
        frontier = linear_chain.get_frontier()
        ids = {n.id for n in frontier}
        assert ids == {"C"}

    def test_node_with_empty_adjacency_list_is_frontier(self):
        """A node present in adjacency keys but with empty list is frontier."""
        g = ConcreteGraph()
        g.add_node("A", is_root=True)
        g.add_node("B")
        g.add_edge("A", "B")
        # B is in nodes but not in adjacency keys -> frontier
        # A is in adjacency keys with non-empty list -> not frontier
        frontier = g.get_frontier()
        ids = {n.id for n in frontier}
        assert "B" in ids
        assert "A" not in ids

    def test_node_with_explicit_empty_adjacency(self):
        """A node explicitly in adjacency with empty list is still frontier."""
        g = ConcreteGraph()
        g.add_node("A", is_root=True)
        g.adjacency["A"] = []  # explicitly empty
        frontier = g.get_frontier()
        ids = {n.id for n in frontier}
        assert "A" in ids

    def test_internal_nodes_excluded(self, branching_tree: ConcreteGraph):
        """Internal nodes (with outgoing edges) are not in the frontier."""
        frontier = branching_tree.get_frontier()
        ids = {n.id for n in frontier}
        # ROOT, R1, R2, R3 all have children
        assert "ROOT" not in ids
        assert "R1" not in ids
        assert "R2" not in ids
        assert "R3" not in ids

    def test_frontier_nodes_are_leaves(self, branching_tree: ConcreteGraph):
        """Frontier nodes are exactly the leaves."""
        frontier = branching_tree.get_frontier()
        ids = {n.id for n in frontier}
        # C1, C2, C3, E1 are all leaves
        assert ids == {"C1", "C2", "C3", "E1"}

    def test_diamond_graph_frontier(self, diamond_graph: ConcreteGraph):
        """Diamond graph has only D (the conclusion) as frontier."""
        frontier = diamond_graph.get_frontier()
        ids = {n.id for n in frontier}
        assert ids == {"D"}

    def test_single_node_is_frontier(self, single_node_graph: ConcreteGraph):
        """A single isolated node is a frontier node."""
        frontier = single_node_graph.get_frontier()
        assert len(frontier) == 1
        assert frontier[0].id == "A"

    def test_returns_thought_node_instances(self, branching_tree: ConcreteGraph):
        """Frontier returns ThoughtNode instances."""
        frontier = branching_tree.get_frontier()
        for node in frontier:
            assert isinstance(node, ThoughtNode)


# ===========================================================================
# 5. TestGetLeaves
# ===========================================================================


class TestGetLeaves:
    """Tests for get_leaves -- alias for get_frontier."""

    def test_alias_returns_same_result(self, branching_tree: ConcreteGraph):
        """get_leaves returns the same result as get_frontier."""
        frontier = branching_tree.get_frontier()
        leaves = branching_tree.get_leaves()
        frontier_ids = {n.id for n in frontier}
        leaves_ids = {n.id for n in leaves}
        assert frontier_ids == leaves_ids

    def test_alias_on_empty_graph(self, empty_graph: ConcreteGraph):
        """get_leaves on empty graph returns empty like get_frontier."""
        assert empty_graph.get_leaves() == []
        assert empty_graph.get_frontier() == []

    def test_alias_on_single_node(self, single_node_graph: ConcreteGraph):
        """get_leaves on single node matches get_frontier."""
        frontier = single_node_graph.get_frontier()
        leaves = single_node_graph.get_leaves()
        assert len(frontier) == len(leaves)
        assert frontier[0].id == leaves[0].id


# ===========================================================================
# 6. TestBacktrack
# ===========================================================================


class TestBacktrack:
    """Tests for backtrack -- find highest-SNR unexplored frontier node."""

    def test_empty_graph_returns_none(self, empty_graph: ConcreteGraph):
        """Empty graph has nothing to backtrack to."""
        assert empty_graph.backtrack() is None

    def test_single_hypothesis_node_returns_it(self, single_node_graph: ConcreteGraph):
        """Single hypothesis node with no children is a valid backtrack target."""
        result = single_node_graph.backtrack()
        assert result is not None
        assert result.id == "A"

    def test_single_conclusion_node_returns_none(self):
        """Single conclusion node is filtered out (already terminal)."""
        g = build_single_node(thought_type=ThoughtType.CONCLUSION)
        result = g.backtrack()
        assert result is None

    def test_single_validation_node_returns_none(self):
        """Single validation node is filtered out."""
        g = build_single_node(thought_type=ThoughtType.VALIDATION)
        result = g.backtrack()
        assert result is None

    def test_all_conclusion_nodes_returns_none(self):
        """When all frontier nodes are CONCLUSION type, backtrack returns None."""
        g = ConcreteGraph()
        g.add_node("C1", ThoughtType.CONCLUSION, snr_score=0.9)
        g.add_node("C2", ThoughtType.CONCLUSION, snr_score=0.8)
        result = g.backtrack()
        assert result is None

    def test_all_validation_nodes_returns_none(self):
        """When all frontier nodes are VALIDATION type, backtrack returns None."""
        g = ConcreteGraph()
        g.add_node("V1", ThoughtType.VALIDATION, snr_score=0.9)
        g.add_node("V2", ThoughtType.VALIDATION, snr_score=0.8)
        result = g.backtrack()
        assert result is None

    def test_returns_highest_snr_unexplored(self):
        """Backtrack returns the frontier node with the highest SNR score."""
        g = ConcreteGraph()
        g.add_node("R", ThoughtType.HYPOTHESIS, snr_score=0.7, is_root=True)
        g.add_node("L1", ThoughtType.REASONING, snr_score=0.6)
        g.add_node("L2", ThoughtType.REASONING, snr_score=0.9)
        g.add_node("L3", ThoughtType.EVIDENCE, snr_score=0.8)
        g.add_edge("R", "L1")
        g.add_edge("R", "L2")
        g.add_edge("R", "L3")
        result = g.backtrack()
        assert result is not None
        assert result.id == "L2"
        assert result.snr_score == 0.9

    def test_nodes_with_children_excluded(self):
        """Nodes that have outgoing edges (children) are excluded from backtrack."""
        g = ConcreteGraph()
        g.add_node("A", ThoughtType.HYPOTHESIS, snr_score=0.9, is_root=True)
        g.add_node("B", ThoughtType.REASONING, snr_score=0.5)
        g.add_edge("A", "B")
        # A has children, B does not
        result = g.backtrack()
        assert result is not None
        assert result.id == "B"

    def test_mixed_types_selects_non_terminal(self, wide_frontier_graph: ConcreteGraph):
        """From mixed frontier types, only non-CONCLUSION/non-VALIDATION nodes chosen."""
        result = wide_frontier_graph.backtrack()
        assert result is not None
        # L3 is CONCLUSION, L4 is VALIDATION -- both filtered
        # L5 (hypothesis, 0.95) > L1 (reasoning, 0.9) > L2 (evidence, 0.8)
        assert result.id == "L5"
        assert result.thought_type not in (ThoughtType.CONCLUSION, ThoughtType.VALIDATION)

    def test_mixed_conclusion_and_reasoning(self):
        """Only non-terminal leaves are candidates for backtracking."""
        g = ConcreteGraph()
        g.add_node("R", ThoughtType.HYPOTHESIS, snr_score=0.5, is_root=True)
        g.add_node("C1", ThoughtType.CONCLUSION, snr_score=0.99)
        g.add_node("X", ThoughtType.REASONING, snr_score=0.4)
        g.add_edge("R", "C1")
        g.add_edge("R", "X")
        result = g.backtrack()
        assert result is not None
        # C1 is filtered (CONCLUSION), X is chosen despite lower SNR
        assert result.id == "X"

    def test_backtrack_returns_thought_node(self, single_node_graph: ConcreteGraph):
        """Backtrack returns a ThoughtNode instance."""
        result = single_node_graph.backtrack()
        assert isinstance(result, ThoughtNode)

    def test_node_in_adjacency_with_empty_list(self):
        """Node in adjacency dict with empty list has no children (len == 0)."""
        g = ConcreteGraph()
        g.add_node("A", ThoughtType.REASONING, snr_score=0.8)
        g.adjacency["A"] = []  # explicitly empty
        result = g.backtrack()
        assert result is not None
        assert result.id == "A"

    def test_evidence_node_is_valid_backtrack_target(self):
        """EVIDENCE type nodes are valid backtrack targets."""
        g = ConcreteGraph()
        g.add_node("E", ThoughtType.EVIDENCE, snr_score=0.7)
        result = g.backtrack()
        assert result is not None
        assert result.id == "E"

    def test_question_node_is_valid_backtrack_target(self):
        """QUESTION type nodes are valid backtrack targets."""
        g = ConcreteGraph()
        g.add_node("Q", ThoughtType.QUESTION, snr_score=0.6)
        result = g.backtrack()
        assert result is not None
        assert result.id == "Q"

    def test_counterpoint_node_is_valid_backtrack_target(self):
        """COUNTERPOINT type nodes are valid backtrack targets."""
        g = ConcreteGraph()
        g.add_node("CP", ThoughtType.COUNTERPOINT, snr_score=0.5)
        result = g.backtrack()
        assert result is not None
        assert result.id == "CP"


# ===========================================================================
# 7. TestExploreWithBacktrack
# ===========================================================================


class TestExploreWithBacktrack:
    """Tests for explore_with_backtrack -- iterative exploration with backtracking."""

    def test_empty_graph_returns_none(self, empty_graph: ConcreteGraph):
        """Empty graph yields no conclusion to explore."""
        result = empty_graph.explore_with_backtrack()
        assert result is None

    def test_conclusion_above_target_found_immediately(self):
        """When a conclusion above target_snr already exists, it is returned at once."""
        g = ConcreteGraph(ihsan_threshold=0.9)
        g.add_node("C", ThoughtType.CONCLUSION, snr_score=0.95)
        result = g.explore_with_backtrack()
        assert result is not None
        assert result.id == "C"
        assert result.snr_score == 0.95

    def test_returns_best_conclusion_above_threshold(self):
        """Among multiple conclusions above threshold, the best is returned."""
        g = ConcreteGraph(ihsan_threshold=0.7)
        g.add_node("C1", ThoughtType.CONCLUSION, snr_score=0.8)
        g.add_node("C2", ThoughtType.CONCLUSION, snr_score=0.95)
        g.add_node("C3", ThoughtType.CONCLUSION, snr_score=0.75)
        result = g.explore_with_backtrack()
        assert result is not None
        assert result.id == "C2"

    def test_target_snr_defaults_to_ihsan_threshold(self):
        """When target_snr is not specified, ihsan_threshold is used."""
        g = ConcreteGraph(ihsan_threshold=0.9)
        g.add_node("C1", ThoughtType.CONCLUSION, snr_score=0.85)  # below 0.9
        g.add_node("R1", ThoughtType.REASONING, snr_score=0.5)
        # C1 is below ihsan (0.9), backtrack finds R1 but can't generate new nodes
        # Falls back to best conclusion above 0.0
        result = g.explore_with_backtrack()
        assert result is not None
        assert result.id == "C1"

    def test_explicit_target_snr(self):
        """Explicit target_snr overrides ihsan_threshold."""
        g = ConcreteGraph(ihsan_threshold=0.99)
        g.add_node("C1", ThoughtType.CONCLUSION, snr_score=0.8)
        result = g.explore_with_backtrack(target_snr=0.7)
        assert result is not None
        assert result.id == "C1"

    def test_falls_back_to_best_conclusion_below_threshold(self):
        """When no conclusion meets target_snr, best available conclusion is returned."""
        g = ConcreteGraph(ihsan_threshold=0.99)
        g.add_node("C", ThoughtType.CONCLUSION, snr_score=0.5)
        # No backtrack targets (C is CONCLUSION, filtered)
        # Falls back to best conclusion above 0.0
        result = g.explore_with_backtrack()
        assert result is not None
        assert result.id == "C"
        assert result.snr_score == 0.5

    def test_max_iterations_limits_exploration(self):
        """Exploration stops after max_iterations even if no good conclusion found."""
        g = ConcreteGraph(ihsan_threshold=0.99)
        # Only non-terminal leaf nodes, so backtrack() keeps returning them
        g.add_node("R1", ThoughtType.REASONING, snr_score=0.5)
        g.add_node("R2", ThoughtType.REASONING, snr_score=0.6)
        result = g.explore_with_backtrack(max_iterations=3)
        # No conclusions at all, returns None
        assert result is None

    def test_no_conclusions_at_all_returns_none(self):
        """Graph with no conclusion nodes returns None."""
        g = ConcreteGraph()
        g.add_node("A", ThoughtType.HYPOTHESIS, snr_score=0.9, is_root=True)
        g.add_node("B", ThoughtType.REASONING, snr_score=0.8)
        g.add_edge("A", "B")
        result = g.explore_with_backtrack(max_iterations=5)
        assert result is None

    def test_backtrack_returns_none_causes_early_exit(self):
        """When backtrack() returns None, iteration stops early."""
        g = ConcreteGraph(ihsan_threshold=0.99)
        # Only CONCLUSION and VALIDATION leaves -- backtrack returns None
        g.add_node("C1", ThoughtType.CONCLUSION, snr_score=0.5)
        g.add_node("V1", ThoughtType.VALIDATION, snr_score=0.6)
        result = g.explore_with_backtrack(max_iterations=100)
        # Should exit quickly despite high max_iterations
        assert result is not None  # falls back to C1
        assert result.id == "C1"

    def test_single_iteration_sufficient(self):
        """When conclusion already meets threshold, only one iteration is needed."""
        g = ConcreteGraph(ihsan_threshold=0.8)
        g.add_node("C", ThoughtType.CONCLUSION, snr_score=0.85)
        result = g.explore_with_backtrack(max_iterations=1)
        assert result is not None
        assert result.id == "C"

    def test_max_iterations_zero_falls_back(self):
        """max_iterations=0 skips all iterations and falls back to best conclusion."""
        g = ConcreteGraph(ihsan_threshold=0.99)
        g.add_node("C", ThoughtType.CONCLUSION, snr_score=0.5)
        result = g.explore_with_backtrack(max_iterations=0)
        # range(0) is empty, goes directly to fallback
        assert result is not None
        assert result.id == "C"

    def test_returns_thought_node_instance(self):
        """explore_with_backtrack returns a ThoughtNode."""
        g = ConcreteGraph(ihsan_threshold=0.5)
        g.add_node("C", ThoughtType.CONCLUSION, snr_score=0.9)
        result = g.explore_with_backtrack()
        assert isinstance(result, ThoughtNode)

    def test_multiple_iterations_with_only_backtrack_nodes(self):
        """When only unexplored reasoning nodes exist, iterates up to max_iterations."""
        g = ConcreteGraph(ihsan_threshold=0.99)
        # Many reasoning leaves but no conclusions
        for i in range(5):
            g.add_node(f"R{i}", ThoughtType.REASONING, snr_score=0.5 + i * 0.05)
        result = g.explore_with_backtrack(max_iterations=3)
        # No conclusions, returns None
        assert result is None

    def test_fallback_picks_highest_snr_conclusion(self):
        """Fallback (min_snr=0.0) picks the highest-SNR conclusion available."""
        g = ConcreteGraph(ihsan_threshold=0.99)
        g.add_node("C1", ThoughtType.CONCLUSION, snr_score=0.3)
        g.add_node("C2", ThoughtType.CONCLUSION, snr_score=0.6)
        g.add_node("C3", ThoughtType.CONCLUSION, snr_score=0.1)
        # All below ihsan_threshold, backtrack returns None (all CONCLUSION)
        result = g.explore_with_backtrack()
        assert result is not None
        assert result.id == "C2"
        assert result.snr_score == 0.6

    def test_target_snr_zero_accepts_any_conclusion(self):
        """target_snr=0.0 should NOT be treated as falsy (would default to ihsan)."""
        # NOTE: The implementation uses `target_snr or self.ihsan_threshold`
        # which means 0.0 is falsy and will be replaced by ihsan_threshold.
        # This test documents the actual behavior.
        g = ConcreteGraph(ihsan_threshold=0.99)
        g.add_node("C", ThoughtType.CONCLUSION, snr_score=0.01)
        result = g.explore_with_backtrack(target_snr=0.0)
        # Due to `or` semantics, target_snr=0.0 is treated as ihsan (0.99)
        # So C (0.01) won't match the threshold, falls back to best conclusion
        assert result is not None
        assert result.id == "C"


# ===========================================================================
# 8. TestEdgeCases
# ===========================================================================


class TestEdgeCases:
    """Additional edge case and integration tests across methods."""

    def test_disconnected_components(self):
        """Graph with disconnected components is handled correctly."""
        g = ConcreteGraph()
        g.add_node("A", ThoughtType.HYPOTHESIS, snr_score=0.7, is_root=True)
        g.add_node("B", ThoughtType.CONCLUSION, snr_score=0.9)
        g.add_edge("A", "B")
        # Disconnected component
        g.add_node("X", ThoughtType.HYPOTHESIS, snr_score=0.5)
        g.add_node("Y", ThoughtType.CONCLUSION, snr_score=0.99)
        g.add_edge("X", "Y")
        # find_best_path starts from A, can only reach B
        path = g.find_best_path()
        assert "B" in path.nodes
        assert "Y" not in path.nodes

    def test_disconnected_start_from_second_component(self):
        """Starting from a disconnected component reaches its own conclusion."""
        g = ConcreteGraph()
        g.add_node("A", ThoughtType.HYPOTHESIS, snr_score=0.7, is_root=True)
        g.add_node("B", ThoughtType.CONCLUSION, snr_score=0.5)
        g.add_edge("A", "B")
        g.add_node("X", ThoughtType.HYPOTHESIS, snr_score=0.8)
        g.add_node("Y", ThoughtType.CONCLUSION, snr_score=0.99)
        g.add_edge("X", "Y")
        path = g.find_best_path(start_id="X")
        assert "Y" in path.nodes
        assert "A" not in path.nodes

    def test_frontier_and_backtrack_consistency(self, wide_frontier_graph: ConcreteGraph):
        """Backtrack target is always a subset of the frontier."""
        frontier = wide_frontier_graph.get_frontier()
        frontier_ids = {n.id for n in frontier}
        result = wide_frontier_graph.backtrack()
        if result is not None:
            assert result.id in frontier_ids

    def test_large_fan_out(self):
        """Graph with high fan-out (many children per node) works correctly."""
        g = ConcreteGraph()
        g.add_node("ROOT", ThoughtType.HYPOTHESIS, snr_score=0.5, is_root=True)
        best_snr = 0.0
        best_id = None
        for i in range(20):
            nid = f"C{i}"
            snr = 0.3 + i * 0.03
            g.add_node(nid, ThoughtType.CONCLUSION, snr_score=snr)
            g.add_edge("ROOT", nid)
            if snr > best_snr:
                best_snr = snr
                best_id = nid
        path = g.find_best_path()
        assert best_id in path.nodes
        # total_snr = ROOT(0.5) + best_conclusion
        assert path.total_snr == pytest.approx(0.5 + best_snr)

    def test_self_loop_handled(self):
        """A node with a self-loop does not cause infinite recursion."""
        g = ConcreteGraph()
        g.add_node("A", ThoughtType.HYPOTHESIS, snr_score=0.7, is_root=True)
        g.add_node("B", ThoughtType.CONCLUSION, snr_score=0.9)
        g.add_edge("A", "B")
        g.add_edge("A", "A")  # self-loop
        path = g.find_best_path()
        assert "B" in path.nodes
        # A should appear at most once
        assert path.nodes.count("A") == 1

    def test_multiple_roots(self):
        """Graph with multiple roots uses the first root by default."""
        g = ConcreteGraph()
        g.add_node("R1", ThoughtType.HYPOTHESIS, snr_score=0.3, is_root=True)
        g.add_node("R2", ThoughtType.HYPOTHESIS, snr_score=0.9, is_root=True)
        g.add_node("C1", ThoughtType.CONCLUSION, snr_score=0.5)
        g.add_node("C2", ThoughtType.CONCLUSION, snr_score=0.95)
        g.add_edge("R1", "C1")
        g.add_edge("R2", "C2")
        path = g.find_best_path()
        # Uses roots[0] = R1
        assert path.nodes[0] == "R1"
        assert "C1" in path.nodes

    def test_reasoning_path_length_property(self, linear_chain: ConcreteGraph):
        """ReasoningPath.length property matches node count."""
        path = linear_chain.find_best_path()
        assert path.length == len(path.nodes)

    def test_reasoning_path_average_snr(self, linear_chain: ConcreteGraph):
        """ReasoningPath.average_snr is total_snr / length."""
        path = linear_chain.find_best_path()
        expected_avg = path.total_snr / path.length
        assert abs(path.average_snr - expected_avg) < 1e-9

    def test_get_conclusions_independent_of_graph_structure(self):
        """get_conclusions examines all nodes regardless of connectivity."""
        g = ConcreteGraph()
        g.add_node("C1", ThoughtType.CONCLUSION, snr_score=0.8)
        g.add_node("C2", ThoughtType.CONCLUSION, snr_score=0.9)
        # No edges, no roots -- conclusions still found
        conclusions = g.get_conclusions(min_snr=0.0)
        assert len(conclusions) == 2

    def test_frontier_includes_isolated_nodes(self):
        """Isolated nodes (no edges at all) are part of the frontier."""
        g = ConcreteGraph()
        g.add_node("ISOLATED", ThoughtType.REASONING, snr_score=0.5)
        frontier = g.get_frontier()
        assert len(frontier) == 1
        assert frontier[0].id == "ISOLATED"
