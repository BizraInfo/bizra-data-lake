"""Tests for Autopoiesis → SDPO Bridge and SDPO → Reflex Bridge.

Covers:
- EvolutionTrace construction, properties, serialization
- AutopoiesisSDPOBridge: collect, filter, flush, contrastive pairs
- TrainingObservation and ReflexCandidate dataclasses
- SDPOReflexBridge: observe, eligible candidates, deny-list, compiled set
- End-to-end: evolution trace → training data → reflex candidate

Blueprint Reference: Section 3.1 — P0 Learning Loop Bridges
"""

from datetime import datetime, timedelta, timezone

import pytest

from core.autopoiesis.sdpo_bridge import (
    AutopoiesisSDPOBridge,
    BridgeResult,
    EvolutionTrace,
)
from core.sdpo.reflex_bridge import (
    REFLEX_IHSAN_THRESHOLD,
    REFLEX_MIN_OBSERVATIONS,
    REFLEX_REPRODUCIBILITY_THRESHOLD,
    ReflexCandidate,
    SDPOReflexBridge,
    TrainingObservation,
)

# ═══════════════════════════════════════════════════════════════════════════
# EvolutionTrace
# ═══════════════════════════════════════════════════════════════════════════


class TestEvolutionTrace:

    def _make_trace(self, **overrides) -> EvolutionTrace:
        defaults = dict(
            genome_id="gen_1_elite_0",
            fitness=0.92,
            ihsan_score=0.96,
            snr_score=0.93,
            task_description="Synthesize research",
            task_output="The key insight is...",
            reasoning_steps=["Step 1", "Step 2"],
            quality_feedback="Good synthesis",
            improvement_suggestions=["Add source B"],
        )
        defaults.update(overrides)
        return EvolutionTrace(**defaults)

    def test_construction(self):
        trace = self._make_trace()
        assert trace.genome_id == "gen_1_elite_0"
        assert trace.fitness == 0.92

    def test_passes_ihsan(self):
        assert self._make_trace(ihsan_score=0.96).passes_ihsan is True
        assert self._make_trace(ihsan_score=0.90).passes_ihsan is False

    def test_passes_snr(self):
        assert self._make_trace(snr_score=0.90).passes_snr is True
        assert self._make_trace(snr_score=0.80).passes_snr is False

    def test_to_dict(self):
        d = self._make_trace().to_dict()
        assert "genome_id" in d
        assert "timestamp" in d
        assert d["fitness"] == 0.92


# ═══════════════════════════════════════════════════════════════════════════
# AutopoiesisSDPOBridge
# ═══════════════════════════════════════════════════════════════════════════


class TestAutopoiesisSDPOBridge:

    def _make_trace(self, fitness=0.92, ihsan=0.96, **kw) -> EvolutionTrace:
        return EvolutionTrace(
            genome_id=kw.get("genome_id", "g1"),
            fitness=fitness,
            ihsan_score=ihsan,
            snr_score=0.93,
            task_description=kw.get("task", "Synthesize"),
            task_output="Output text",
            reasoning_steps=["Step 1"],
            quality_feedback="Good",
            improvement_suggestions=["Improve X"],
        )

    def test_collect_accepts_high_fitness(self):
        bridge = AutopoiesisSDPOBridge()
        assert bridge.collect(self._make_trace(fitness=0.90)) is True
        assert bridge.pending_count == 1

    def test_collect_rejects_low_fitness(self):
        bridge = AutopoiesisSDPOBridge(min_fitness=0.80)
        assert bridge.collect(self._make_trace(fitness=0.70)) is False
        assert bridge.pending_count == 0

    def test_collect_rejects_low_ihsan(self):
        bridge = AutopoiesisSDPOBridge()
        assert bridge.collect(self._make_trace(ihsan=0.90)) is False
        assert bridge.pending_count == 0

    def test_collect_rejects_low_snr(self):
        bridge = AutopoiesisSDPOBridge()
        trace = self._make_trace()
        trace.snr_score = 0.70
        assert bridge.collect(trace) is False
        assert bridge.pending_count == 0

    def test_ready_for_training(self):
        bridge = AutopoiesisSDPOBridge(min_samples=2)
        bridge.collect(self._make_trace())
        assert bridge.ready_for_training is False
        bridge.collect(self._make_trace(genome_id="g2"))
        assert bridge.ready_for_training is True

    def test_flush_returns_none_when_insufficient(self):
        bridge = AutopoiesisSDPOBridge(min_samples=5)
        bridge.collect(self._make_trace())
        assert bridge.flush_to_training_data() is None

    def test_flush_produces_training_data(self):
        bridge = AutopoiesisSDPOBridge(min_samples=2)
        bridge.collect(self._make_trace(genome_id="g1", task="Task A"))
        bridge.collect(self._make_trace(genome_id="g2", task="Task B"))
        bridge.collect(self._make_trace(genome_id="g3", task="Task C"))

        data = bridge.flush_to_training_data()
        assert data is not None
        assert "questions" in data
        assert "corrected_attempts" in data
        assert "feedbacks" in data
        assert "failed_attempts" in data
        assert "quality_scores" in data
        assert "snr_scores" in data
        assert len(data["questions"]) == 3
        assert data["snr_scores"] == [0.93, 0.93, 0.93]

    def test_flush_clears_buffer(self):
        bridge = AutopoiesisSDPOBridge(min_samples=1)
        bridge.collect(self._make_trace())
        bridge.flush_to_training_data()
        assert bridge.pending_count == 0

    def test_flush_caps_at_max_batch(self):
        bridge = AutopoiesisSDPOBridge(min_samples=1, max_batch=2)
        for i in range(5):
            bridge.collect(self._make_trace(genome_id=f"g{i}", task=f"T{i}"))
        data = bridge.flush_to_training_data()
        assert len(data["questions"]) == 2

    def test_bridge_history(self):
        bridge = AutopoiesisSDPOBridge(min_samples=1)
        bridge.collect(self._make_trace())
        bridge.flush_to_training_data()
        history = bridge.get_bridge_history()
        assert len(history) == 1
        assert history[0]["traces_converted"] == 1


# ═══════════════════════════════════════════════════════════════════════════
# ReflexCandidate
# ═══════════════════════════════════════════════════════════════════════════


class TestReflexCandidate:

    def test_eligible_all_gates_pass(self):
        c = ReflexCandidate(
            pattern_id="abc",
            pattern_description="Test",
            source_task="Test",
            avg_ihsan=0.99,
            avg_snr=0.97,
            reproducibility=0.95,
            observation_count=10,
            impact_score=0.12,
        )
        assert c.eligible is True

    def test_not_eligible_low_ihsan(self):
        c = ReflexCandidate(
            pattern_id="abc",
            pattern_description="Test",
            source_task="Test",
            avg_ihsan=0.90,
            avg_snr=0.97,
            reproducibility=0.95,
            observation_count=10,
            impact_score=0.12,
        )
        assert c.eligible is False

    def test_not_eligible_low_reproducibility(self):
        c = ReflexCandidate(
            pattern_id="abc",
            pattern_description="Test",
            source_task="Test",
            avg_ihsan=0.99,
            avg_snr=0.97,
            reproducibility=0.80,
            observation_count=10,
            impact_score=0.12,
        )
        assert c.eligible is False

    def test_not_eligible_zero_impact(self):
        """Anti-gaming: impact must be > 0."""
        c = ReflexCandidate(
            pattern_id="abc",
            pattern_description="Test",
            source_task="Test",
            avg_ihsan=0.99,
            avg_snr=0.85,
            reproducibility=0.95,
            observation_count=10,
            impact_score=0.0,
        )
        assert c.eligible is False

    def test_to_dict(self):
        c = ReflexCandidate(
            pattern_id="abc",
            pattern_description="Synth",
            source_task="Synth",
            avg_ihsan=0.99,
            avg_snr=0.97,
            reproducibility=0.95,
            observation_count=10,
            impact_score=0.12,
        )
        d = c.to_dict()
        assert d["eligible"] is True
        assert "created_at" in d


# ═══════════════════════════════════════════════════════════════════════════
# SDPOReflexBridge
# ═══════════════════════════════════════════════════════════════════════════


class TestSDPOReflexBridge:

    def test_observe_returns_pattern_id(self):
        bridge = SDPOReflexBridge()
        pid = bridge.observe("Task A", 0.98, 0.96, 0.1, True)
        assert isinstance(pid, str)
        assert len(pid) == 64  # SHA-256 hex

    def test_same_task_same_pattern(self):
        bridge = SDPOReflexBridge()
        pid1 = bridge.observe("Task A", 0.98, 0.96, 0.1, True)
        pid2 = bridge.observe("Task A", 0.99, 0.97, 0.08, True)
        assert pid1 == pid2
        assert bridge.pattern_count == 1
        assert bridge.total_observations == 2

    def test_different_tasks_different_patterns(self):
        bridge = SDPOReflexBridge()
        bridge.observe("Task A", 0.98, 0.96, 0.1, True)
        bridge.observe("Task B", 0.99, 0.97, 0.08, True)
        assert bridge.pattern_count == 2

    def test_eligible_after_min_observations(self):
        bridge = SDPOReflexBridge(min_observations=5)
        for _ in range(5):
            bridge.observe("Consistent Task", 0.99, 0.97, 0.05, True)
        candidates = bridge.get_eligible_candidates()
        assert len(candidates) == 1
        assert candidates[0].eligible is True

    def test_custom_min_observations_are_honored(self):
        bridge = SDPOReflexBridge(min_observations=3)
        for _ in range(3):
            bridge.observe("Fast closure task", 0.99, 0.97, 0.05, True)
        candidates = bridge.get_eligible_candidates()
        assert len(candidates) == 1
        assert candidates[0].observation_count == 3

    def test_not_eligible_below_min_observations(self):
        bridge = SDPOReflexBridge(min_observations=5)
        for _ in range(3):
            bridge.observe("Task", 0.99, 0.97, 0.05, True)
        assert len(bridge.get_eligible_candidates()) == 0

    def test_not_eligible_low_reproducibility(self):
        bridge = SDPOReflexBridge(min_observations=5)
        # 2 success, 3 failure → 40% reproducibility
        bridge.observe("T", 0.99, 0.97, 0.05, True)
        bridge.observe("T", 0.99, 0.97, 0.05, True)
        bridge.observe("T", 0.99, 0.97, 0.50, False)
        bridge.observe("T", 0.99, 0.97, 0.50, False)
        bridge.observe("T", 0.99, 0.97, 0.50, False)
        assert len(bridge.get_eligible_candidates()) == 0

    def test_mark_compiled_excludes_from_candidates(self):
        bridge = SDPOReflexBridge(min_observations=5)
        pid = None
        for _ in range(5):
            pid = bridge.observe("Task", 0.99, 0.97, 0.05, True)
        assert len(bridge.get_eligible_candidates()) == 1

        bridge.mark_compiled(pid)
        assert len(bridge.get_eligible_candidates()) == 0
        assert bridge.compiled_count == 1

    def test_deny_excludes_from_candidates(self):
        bridge = SDPOReflexBridge(min_observations=2)
        pid = bridge.observe("Task", 0.99, 0.97, 0.05, True)
        bridge.observe("Task", 0.99, 0.97, 0.05, True)

        bridge.deny(pid, reason="Flagged by Crown")
        assert len(bridge.get_eligible_candidates()) == 0
        assert bridge.denied_count == 1

    def test_get_status(self):
        bridge = SDPOReflexBridge(min_observations=1)
        bridge.observe("Task", 0.99, 0.97, 0.05, True)
        status = bridge.get_status()
        assert status["patterns_tracked"] == 1
        assert status["total_observations"] == 1
        assert "eligible_now" in status

    def test_sorted_by_ihsan_descending(self):
        bridge = SDPOReflexBridge(min_observations=1)
        bridge.observe("Low", 0.98, 0.90, 0.1, True)
        bridge.observe("High", 0.99, 0.97, 0.05, True)
        candidates = bridge.get_eligible_candidates()
        if len(candidates) >= 2:
            assert candidates[0].avg_ihsan >= candidates[1].avg_ihsan


# ═══════════════════════════════════════════════════════════════════════════
# END-TO-END: Evolution → Training → Reflex
# ═══════════════════════════════════════════════════════════════════════════


class TestEndToEndLearningLoop:
    """Verify the complete P0 learning loop: Autopoiesis → SDPO → Reflex."""

    def test_full_pipeline(self):
        # Phase 1: Autopoietic evolution produces traces
        evo_bridge = AutopoiesisSDPOBridge(min_samples=2, min_fitness=0.80)

        trace1 = EvolutionTrace(
            genome_id="gen_5_elite_0",
            fitness=0.93,
            ihsan_score=0.97,
            snr_score=0.95,
            task_description="Classify documents",
            task_output="Category A: legal, Category B: financial",
            reasoning_steps=["Extracted features", "Applied classifier"],
            quality_feedback="Accurate classification",
            improvement_suggestions=["Add confidence scores"],
        )
        trace2 = EvolutionTrace(
            genome_id="gen_5_elite_1",
            fitness=0.91,
            ihsan_score=0.96,
            snr_score=0.94,
            task_description="Summarize findings",
            task_output="Key findings: efficiency improved by 15%",
            reasoning_steps=["Gathered data", "Computed metrics"],
            quality_feedback="Clear summary",
            improvement_suggestions=["Include confidence intervals"],
        )

        assert evo_bridge.collect(trace1)
        assert evo_bridge.collect(trace2)

        # Phase 2: Convert to SDPO training data
        training_data = evo_bridge.flush_to_training_data()
        assert training_data is not None
        assert len(training_data["questions"]) == 2
        assert training_data["snr_scores"] == [0.95, 0.94]

        # Phase 3: Feed into reflex bridge (simulating SDPO training outcomes)
        # Need >= REFLEX_MIN_OBSERVATIONS (5) per pattern for eligibility
        reflex_bridge = SDPOReflexBridge(min_observations=5)

        for question in training_data["questions"]:
            for _ in range(5):
                reflex_bridge.observe(question, 0.99, 0.97, 0.05, True)

        # Phase 4: Check for reflex candidates
        candidates = reflex_bridge.get_eligible_candidates()
        assert len(candidates) == 2
        assert all(c.eligible for c in candidates)

        # Phase 5: Compile one reflex
        reflex_bridge.mark_compiled(candidates[0].pattern_id)
        remaining = reflex_bridge.get_eligible_candidates()
        assert len(remaining) == 1
