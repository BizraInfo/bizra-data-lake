"""Tests for core.nexus.sovereign_nexus -- Sovereign Nexus orchestration engine.

Covers:
- NexusPhase, NexusState, ThoughtType, AgentRole enums
- ThoughtNode, ThoughtEdge, ThoughtGraph data structures
- SNRScore and SNRGate validation
- NexusTask, NexusResult, NexusConfig data classes
- ThoughtGraph: add, path-finding, confidence computation
"""

import math

import pytest

from core.nexus.sovereign_nexus import (
    AgentRole,
    NexusConfig,
    NexusPhase,
    NexusResult,
    NexusState,
    NexusTask,
    SNRGate,
    SNRScore,
    ThoughtEdge,
    ThoughtGraph,
    ThoughtNode,
    ThoughtType,
)


# ---------------------------------------------------------------------------
# ENUM TESTS
# ---------------------------------------------------------------------------


class TestEnums:

    def test_nexus_phases_count(self):
        assert len(NexusPhase) == 10  # OODA(4) + PDCA(4) + SYNTHESIZE + VALIDATE

    def test_nexus_states(self):
        expected = {
            "initializing", "ready", "processing", "reasoning",
            "executing", "validating", "complete", "error",
        }
        actual = {s.value for s in NexusState}
        assert actual == expected

    def test_thought_types(self):
        expected = {
            "hypothesis", "observation", "analysis", "synthesis",
            "conclusion", "contradiction", "refinement",
        }
        actual = {t.value for t in ThoughtType}
        assert actual == expected

    def test_agent_roles(self):
        assert len(AgentRole) == 7


# ---------------------------------------------------------------------------
# ThoughtNode TESTS
# ---------------------------------------------------------------------------


class TestThoughtNode:

    def test_default_values(self):
        node = ThoughtNode(content="Test thought")
        assert node.thought_type == ThoughtType.HYPOTHESIS
        assert node.confidence == 0.5
        assert node.depth == 0
        assert node.validated is False
        assert node.snr_score == 0.0

    def test_to_dict_truncates_long_content(self):
        node = ThoughtNode(content="A" * 300)
        d = node.to_dict()
        assert len(d["content"]) < 300
        assert d["content"].endswith("...")

    def test_to_dict_short_content(self):
        node = ThoughtNode(content="Short")
        d = node.to_dict()
        assert d["content"] == "Short"


# ---------------------------------------------------------------------------
# ThoughtGraph TESTS
# ---------------------------------------------------------------------------


class TestThoughtGraph:

    @pytest.fixture
    def graph(self):
        return ThoughtGraph()

    def test_add_thought_sets_root(self, graph):
        node = graph.add_thought("Root thought")
        assert graph.root_id == node.id
        assert graph.total_thoughts == 1

    def test_add_child_thought(self, graph):
        root = graph.add_thought("Root")
        child = graph.add_thought("Child", parent_id=root.id)
        assert child.depth == 1
        assert root.id in child.parent_ids
        assert child.id in root.child_ids
        assert graph.max_depth_reached == 1

    def test_add_multiple_children(self, graph):
        root = graph.add_thought("Root", confidence=0.9)
        c1 = graph.add_thought("Child 1", parent_id=root.id, confidence=0.8)
        c2 = graph.add_thought("Child 2", parent_id=root.id, confidence=0.7)
        assert graph.total_thoughts == 3
        assert len(root.child_ids) == 2

    def test_get_thought(self, graph):
        node = graph.add_thought("Test")
        found = graph.get_thought(node.id)
        assert found is node

    def test_get_thought_not_found(self, graph):
        assert graph.get_thought("nonexistent") is None

    def test_get_children(self, graph):
        root = graph.add_thought("Root")
        c1 = graph.add_thought("C1", parent_id=root.id)
        c2 = graph.add_thought("C2", parent_id=root.id)
        children = graph.get_children(root.id)
        assert len(children) == 2

    def test_get_leaves(self, graph):
        root = graph.add_thought("Root")
        c1 = graph.add_thought("Leaf1", parent_id=root.id)
        c2 = graph.add_thought("Leaf2", parent_id=root.id)
        leaves = graph.get_leaves()
        assert len(leaves) == 2
        assert root not in leaves

    def test_get_leaves_root_only(self, graph):
        root = graph.add_thought("Root")
        leaves = graph.get_leaves()
        assert len(leaves) == 1
        assert leaves[0] is root

    def test_get_best_path_single_node(self, graph):
        graph.add_thought("Only node", confidence=0.9)
        path = graph.get_best_path()
        assert len(path) == 1

    def test_get_best_path_selects_highest_confidence(self, graph):
        root = graph.add_thought("Root", confidence=1.0)
        high = graph.add_thought("High conf", parent_id=root.id, confidence=0.95)
        low = graph.add_thought("Low conf", parent_id=root.id, confidence=0.3)
        path = graph.get_best_path()
        # Best path should go through high confidence child
        assert high in path

    def test_get_best_path_empty_graph(self, graph):
        path = graph.get_best_path()
        assert path == []

    def test_get_conclusions(self, graph):
        root = graph.add_thought("Root")
        c = graph.add_thought(
            "Conclusion",
            parent_id=root.id,
            thought_type=ThoughtType.CONCLUSION,
        )
        c.validated = True
        conclusions = graph.get_conclusions()
        assert len(conclusions) == 1

    def test_get_conclusions_excludes_unvalidated(self, graph):
        graph.add_thought(
            "Unvalidated conclusion",
            thought_type=ThoughtType.CONCLUSION,
        )
        conclusions = graph.get_conclusions()
        assert len(conclusions) == 0

    def test_compute_graph_confidence_empty(self, graph):
        assert graph.compute_graph_confidence() == 0.0

    def test_compute_graph_confidence_single(self, graph):
        graph.add_thought("Root", confidence=0.9)
        conf = graph.compute_graph_confidence()
        assert abs(conf - 0.9) < 0.01

    def test_compute_graph_confidence_chain(self, graph):
        root = graph.add_thought("Root", confidence=0.8)
        child = graph.add_thought("Child", parent_id=root.id, confidence=0.8)
        conf = graph.compute_graph_confidence()
        # Geometric mean of [0.8, 0.8]
        expected = math.pow(0.8 * 0.8, 0.5)
        assert abs(conf - expected) < 0.01

    def test_to_summary(self, graph):
        graph.add_thought("Root", confidence=0.9)
        summary = graph.to_summary()
        assert summary["total_thoughts"] == 1
        assert summary["max_depth"] == 0
        assert "graph_confidence" in summary


# ---------------------------------------------------------------------------
# SNRScore TESTS
# ---------------------------------------------------------------------------


class TestSNRScore:

    def test_default_snr_is_low(self):
        score = SNRScore()
        assert score.snr <= 0.51  # All components at 0, SNR near 0.5

    def test_high_signal_components(self):
        score = SNRScore(
            relevance=0.95, novelty=0.9, groundedness=0.9,
            coherence=0.9, actionability=0.9,
        )
        assert score.signal_power > 0.5

    def test_noise_power(self):
        score = SNRScore(
            inconsistency=0.5, redundancy=0.5,
            ambiguity=0.5, hallucination_risk=0.5,
        )
        assert score.noise_power > 0

    def test_passed_threshold(self):
        score = SNRScore(
            relevance=0.95, novelty=0.9, groundedness=0.95,
            coherence=0.9, actionability=0.9,
            inconsistency=0.01, redundancy=0.01,
            ambiguity=0.01, hallucination_risk=0.01,
        )
        # With high signal and low noise, SNR should be high
        assert score.snr > 0.5

    def test_to_dict(self):
        score = SNRScore(relevance=0.8)
        d = score.to_dict()
        assert "signal_power" in d
        assert "noise_power" in d
        assert "snr" in d
        assert "passed" in d
        assert "components" in d
        assert "noise" in d


# ---------------------------------------------------------------------------
# SNRGate TESTS
# ---------------------------------------------------------------------------


class TestSNRGate:

    @pytest.fixture
    def gate(self):
        return SNRGate(threshold=0.85)

    def test_validate_returns_score(self, gate):
        score = gate.validate("This is a substantive analysis of system architecture.")
        assert isinstance(score, SNRScore)

    def test_validate_with_sources(self, gate):
        score = gate.validate(
            "This is grounded analysis.",
            sources=["paper1", "paper2", "paper3"],
        )
        assert score.groundedness > 0.5

    def test_validate_thought_updates_node(self, gate):
        node = ThoughtNode(content="Implement a robust testing framework for quality assurance.")
        score = gate.validate_thought(node)
        assert node.snr_score == score.snr
        assert node.validated == score.passed

    def test_get_avg_snr_empty(self, gate):
        assert gate.get_avg_snr() == 0.0

    def test_get_avg_snr_after_validations(self, gate):
        gate.validate("First analysis with some depth.")
        gate.validate("Second analysis with more detail.")
        avg = gate.get_avg_snr()
        assert avg > 0.0


# ---------------------------------------------------------------------------
# NexusTask TESTS
# ---------------------------------------------------------------------------


class TestNexusTask:

    def test_default_task(self):
        task = NexusTask(prompt="Test prompt")
        assert task.ihsan_threshold == 0.95
        assert task.snr_threshold == 0.85
        assert task.priority == 5

    def test_to_dict(self):
        task = NexusTask(prompt="A" * 200)
        d = task.to_dict()
        assert d["prompt"].endswith("...")  # Truncated


# ---------------------------------------------------------------------------
# NexusResult TESTS
# ---------------------------------------------------------------------------


class TestNexusResult:

    def test_successful_result(self):
        result = NexusResult(task_id="t1", success=True, response="Done")
        d = result.to_dict()
        assert d["success"] is True
        assert d["error"] is None

    def test_failed_result(self):
        result = NexusResult(task_id="t2", success=False, error="Something failed")
        d = result.to_dict()
        assert d["success"] is False
        assert d["error"] == "Something failed"


# ---------------------------------------------------------------------------
# NexusConfig TESTS
# ---------------------------------------------------------------------------


class TestNexusConfig:

    def test_default_config(self):
        config = NexusConfig()
        assert config.ihsan_threshold == 0.95
        assert config.snr_threshold == 0.85
        assert config.got_max_branches == 5
        assert config.got_max_depth == 4
        assert len(config.default_agents) == 3
