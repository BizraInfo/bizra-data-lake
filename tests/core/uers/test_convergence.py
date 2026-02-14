"""Tests for core.uers.convergence -- Convergence Loop data classes and enums.

Covers:
- LoopStage and ConvergenceState enums
- Hypothesis data class
- LoopIteration and ConvergenceResult computed properties
- ConvergenceLoop configuration and state
"""

import pytest

from core.uers.convergence import (
    ConvergenceLoop,
    ConvergenceResult,
    ConvergenceState,
    Hypothesis,
    LoopIteration,
    LoopStage,
)
from core.uers.vectors import VectorType


# ---------------------------------------------------------------------------
# ENUM TESTS
# ---------------------------------------------------------------------------


class TestEnums:

    def test_loop_stages(self):
        expected = {"ingestion", "hypothesis", "probing", "evaluation", "update", "termination"}
        actual = {s.value for s in LoopStage}
        assert actual == expected

    def test_convergence_states(self):
        expected = {
            "initializing", "running", "converging",
            "diverging", "singularity", "failed", "stopped",
        }
        actual = {s.value for s in ConvergenceState}
        assert actual == expected


# ---------------------------------------------------------------------------
# DATA CLASS TESTS
# ---------------------------------------------------------------------------


class TestHypothesis:

    def test_instantiation(self):
        h = Hypothesis(
            id="hyp_001",
            description="Test hypothesis",
            source_vector=VectorType.SURFACE,
            target_vector=VectorType.STRUCTURAL,
            confidence=0.8,
            predicted_delta_e=0.1,
        )
        assert h.validated is None
        assert h.actual_delta_e == 0.0

    def test_to_dict(self):
        h = Hypothesis(
            id="hyp_002",
            description="A" * 200,  # Test truncation
            source_vector=VectorType.BEHAVIORAL,
            target_vector=VectorType.HYPOTHETICAL,
            confidence=0.7,
            predicted_delta_e=0.05,
        )
        d = h.to_dict()
        assert d["id"] == "hyp_002"
        assert len(d["description"]) <= 100
        assert d["source"] == "behavioral"
        assert d["target"] == "hypothetical"


class TestLoopIteration:

    def test_was_productive_positive_delta(self):
        it = LoopIteration(
            iteration=1,
            stage=LoopStage.EVALUATION,
            entropy_before=0.8,
            entropy_after=0.6,
            delta_e=0.2,
            hypotheses_tested=3,
            probes_executed=5,
            duration_ms=100.0,
        )
        assert it.was_productive is True

    def test_was_productive_zero_delta(self):
        it = LoopIteration(
            iteration=2,
            stage=LoopStage.EVALUATION,
            entropy_before=0.5,
            entropy_after=0.5,
            delta_e=0.0,
            hypotheses_tested=2,
            probes_executed=3,
            duration_ms=50.0,
        )
        assert it.was_productive is False

    def test_to_dict(self):
        it = LoopIteration(
            iteration=1,
            stage=LoopStage.EVALUATION,
            entropy_before=0.8,
            entropy_after=0.6,
            delta_e=0.2,
            hypotheses_tested=3,
            probes_executed=5,
            duration_ms=100.0,
        )
        d = it.to_dict()
        assert d["iteration"] == 1
        assert d["was_productive"] is True
        assert d["stage"] == "evaluation"


class TestConvergenceResult:

    def test_convergence_rate_no_iterations(self):
        r = ConvergenceResult(
            id="conv_001",
            initial_entropy=1.0,
            final_entropy=1.0,
            total_delta_e=0.0,
            iterations=0,
            state=ConvergenceState.STOPPED,
            singularity_achieved=False,
        )
        assert r.convergence_rate == 0.0

    def test_convergence_rate_with_iterations(self):
        r = ConvergenceResult(
            id="conv_002",
            initial_entropy=1.0,
            final_entropy=0.5,
            total_delta_e=0.5,
            iterations=10,
            state=ConvergenceState.CONVERGING,
            singularity_achieved=False,
        )
        assert abs(r.convergence_rate - 0.05) < 1e-9

    def test_efficiency_no_history(self):
        r = ConvergenceResult(
            id="conv_003",
            initial_entropy=1.0,
            final_entropy=0.8,
            total_delta_e=0.2,
            iterations=5,
            state=ConvergenceState.STOPPED,
            singularity_achieved=False,
        )
        assert r.efficiency == 0.0

    def test_efficiency_with_mixed_history(self):
        productive = LoopIteration(1, LoopStage.EVALUATION, 0.8, 0.7, 0.1, 1, 1, 10.0)
        stagnant = LoopIteration(2, LoopStage.EVALUATION, 0.7, 0.7, 0.0, 1, 1, 10.0)

        r = ConvergenceResult(
            id="conv_004",
            initial_entropy=1.0,
            final_entropy=0.7,
            total_delta_e=0.3,
            iterations=2,
            state=ConvergenceState.CONVERGING,
            singularity_achieved=False,
            iteration_history=[productive, stagnant],
        )
        assert r.efficiency == 0.5  # 1 out of 2 productive


# ---------------------------------------------------------------------------
# ConvergenceLoop TESTS
# ---------------------------------------------------------------------------


class TestConvergenceLoop:

    def test_default_initialization(self):
        loop = ConvergenceLoop()
        assert loop.get_state() == ConvergenceState.INITIALIZING
        assert loop.max_iterations == 100

    def test_custom_parameters(self):
        loop = ConvergenceLoop(
            max_iterations=50,
            singularity_threshold=0.05,
            stagnation_limit=3,
            min_delta_e=0.01,
        )
        assert loop.max_iterations == 50
        assert loop.singularity_threshold == 0.05
        assert loop.stagnation_limit == 3

    def test_get_manifold(self):
        loop = ConvergenceLoop()
        manifold = loop.get_manifold()
        assert manifold is not None

    def test_get_current_entropy(self):
        loop = ConvergenceLoop()
        entropy = loop.get_current_entropy()
        assert entropy == 1.0  # Initial state

    def test_get_stats_empty(self):
        loop = ConvergenceLoop()
        stats = loop.get_stats()
        assert stats["total_runs"] == 0

    def test_get_results_empty(self):
        loop = ConvergenceLoop()
        results = loop.get_results()
        assert results == []

    def test_reset(self):
        loop = ConvergenceLoop()
        loop.reset()
        assert loop.get_state() == ConvergenceState.INITIALIZING
        assert loop._current_iteration == 0

    def test_set_hypothesis_generator(self):
        loop = ConvergenceLoop()
        loop.set_hypothesis_generator(lambda m: [])
        assert loop._hypothesis_generator is not None

    def test_set_probe_executor(self):
        loop = ConvergenceLoop()
        loop.set_probe_executor(lambda p: p)
        assert loop._probe_executor is not None


@pytest.mark.timeout(60)
class TestConvergenceExecution:

    @pytest.mark.asyncio
    async def test_converge_stops_at_max_iterations(self):
        loop = ConvergenceLoop(max_iterations=3)
        result = await loop.converge()
        assert result.iterations <= 3
        assert isinstance(result, ConvergenceResult)

    @pytest.mark.asyncio
    async def test_converge_with_contextual_data(self):
        loop = ConvergenceLoop(max_iterations=2)
        result = await loop.converge(
            contextual_data={
                "text": "Test query for convergence",
                "intent": 0.9,
                "alignment": 0.85,
            }
        )
        assert result.id.startswith("conv_")
        assert result.duration_ms > 0
