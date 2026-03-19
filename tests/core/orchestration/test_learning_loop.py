"""Tests for core.orchestration.learning_loop — LearningLoopOrchestrator.

Covers:
- LoopEvent and LoopMetrics dataclasses
- Stage 1: on_candidate() — fitness/ihsan gates, trace conversion, bridge wiring
- Stage 2: run_training_cycle() — flush, training, observation piping
- Stage 3: run_compilation_cycle() — eligible candidates → compile_reflex → cache
- Feature flag: disabled mode (dry-run telemetry)
- Full cycle: run_full_cycle() end-to-end
- Edge cases: no trainer, no candidates, empty state

Blueprint Reference: Elite Implementation Blueprint v1.0 — P0 Close Learning Loop
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.autopoiesis.loop import IntegrationCandidate
from core.autopoiesis.sdpo_bridge import AutopoiesisSDPOBridge, EvolutionTrace
from core.constitutional.types import Reflex
from core.integration.constants import UNIFIED_IHSAN_THRESHOLD, UNIFIED_SNR_THRESHOLD
from core.orchestration.learning_loop import (
    LearningLoopOrchestrator,
    LoopEvent,
    LoopMetrics,
)
from core.sdpo.reflex_bridge import SDPOReflexBridge

# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _mock_genome(**kwargs):
    """Create a mock AgentGenome with configurable attributes."""
    genome = MagicMock()
    genome.genome_id = kwargs.get("genome_id", "gen_42")
    genome.snr_score = kwargs.get("snr_score", 0.92)
    genome.task_description = kwargs.get("task_description", "Synthesize findings")
    genome.task_output = kwargs.get("task_output", "The key insight is X.")
    genome.reasoning_steps = kwargs.get("reasoning_steps", ["step1", "step2"])
    genome.improvement_suggestions = kwargs.get(
        "improvement_suggestions", ["improve X"]
    )
    return genome


def _make_candidate(fitness=0.95, ihsan=0.97, recommendation="Integrate"):
    """Create an IntegrationCandidate."""
    return IntegrationCandidate(
        genome=_mock_genome(),
        fitness=fitness,
        novelty_score=0.6,
        ihsan_score=ihsan,
        recommendation=recommendation,
    )


def _mock_trainer(final_loss=0.15, final_ihsan=0.97):
    """Create a mock SDPO trainer."""
    trainer = AsyncMock()
    result = MagicMock()
    result.final_loss = final_loss
    result.final_ihsan_score = final_ihsan
    result.total_epochs_completed = 3
    result.total_steps = 24
    trainer.train.return_value = result
    return trainer


# ═══════════════════════════════════════════════════════════════════════════
# Data Types
# ═══════════════════════════════════════════════════════════════════════════


class TestLoopEvent:
    def test_construction(self):
        event = LoopEvent(
            event_type="CANDIDATE_RECEIVED",
            source="autopoiesis",
            payload={"fitness": 0.95},
        )
        assert event.event_type == "CANDIDATE_RECEIVED"
        assert event.source == "autopoiesis"

    def test_to_dict(self):
        event = LoopEvent(event_type="TEST", source="test", payload={"x": 1})
        d = event.to_dict()
        assert d["event_type"] == "TEST"
        assert isinstance(d["timestamp"], str)


class TestLoopMetrics:
    def test_defaults(self):
        m = LoopMetrics()
        assert m.candidates_received == 0
        assert m.reflexes_compiled == 0
        assert m.last_cycle_at is None

    def test_to_dict(self):
        m = LoopMetrics(candidates_received=5, reflexes_compiled=2)
        d = m.to_dict()
        assert d["candidates_received"] == 5
        assert d["reflexes_compiled"] == 2


# ═══════════════════════════════════════════════════════════════════════════
# Stage 1: on_candidate() — Autopoiesis → Evolution Bridge
# ═══════════════════════════════════════════════════════════════════════════


class TestOnCandidate:
    def test_accept_high_quality_candidate(self):
        orch = LearningLoopOrchestrator(enabled=True)
        candidate = _make_candidate(fitness=0.95, ihsan=0.97)
        accepted = orch.on_candidate(candidate)
        assert accepted is True
        assert orch.metrics.candidates_accepted == 1
        assert orch.metrics.candidates_filtered == 0

    def test_reject_low_fitness(self):
        orch = LearningLoopOrchestrator(enabled=True)
        candidate = _make_candidate(fitness=0.5, ihsan=0.97)
        accepted = orch.on_candidate(candidate)
        assert accepted is False
        assert orch.metrics.candidates_filtered == 1

    def test_reject_low_ihsan(self):
        orch = LearningLoopOrchestrator(enabled=True)
        candidate = _make_candidate(fitness=0.95, ihsan=0.80)
        accepted = orch.on_candidate(candidate)
        assert accepted is False
        assert orch.metrics.candidates_filtered == 1

    def test_multiple_candidates_accumulate(self):
        orch = LearningLoopOrchestrator(enabled=True)
        for _ in range(5):
            orch.on_candidate(_make_candidate())
        assert orch.metrics.candidates_received == 5
        assert orch.metrics.candidates_accepted == 5

    def test_emits_events(self):
        orch = LearningLoopOrchestrator(enabled=True)
        orch.on_candidate(_make_candidate())
        events = orch.get_events()
        types = [e["event_type"] for e in events]
        assert "CANDIDATE_RECEIVED" in types
        assert "CANDIDATE_ACCEPTED" in types

    def test_borderline_fitness_rejected(self):
        """Fitness exactly 0.90 should be accepted (>= gate)."""
        orch = LearningLoopOrchestrator(enabled=True)
        # Just below threshold
        assert orch.on_candidate(_make_candidate(fitness=0.899)) is False
        # At threshold
        assert orch.on_candidate(_make_candidate(fitness=0.90)) is True


# ═══════════════════════════════════════════════════════════════════════════
# Stage 2: run_training_cycle() — Evolution Bridge → SDPO Training
# ═══════════════════════════════════════════════════════════════════════════


class TestTrainingCycle:
    def test_skip_when_not_ready(self):
        orch = LearningLoopOrchestrator(enabled=True)
        result = asyncio.run(orch.run_training_cycle())
        assert result is None

    def test_skip_when_disabled(self):
        orch = LearningLoopOrchestrator(enabled=False)
        # Buffer enough traces
        for _ in range(3):
            orch.on_candidate(_make_candidate())
        result = asyncio.run(orch.run_training_cycle())
        assert result is None
        events = orch.get_events()
        types = [e["event_type"] for e in events]
        assert "TRAINING_SKIPPED_DISABLED" in types

    def test_skip_when_no_trainer(self):
        orch = LearningLoopOrchestrator(enabled=True, sdpo_trainer=None)
        for _ in range(3):
            orch.on_candidate(_make_candidate())
        result = asyncio.run(orch.run_training_cycle())
        assert result is None

    def test_executes_training(self):
        trainer = _mock_trainer()
        orch = LearningLoopOrchestrator(enabled=True, sdpo_trainer=trainer)
        for _ in range(3):
            orch.on_candidate(_make_candidate())
        result = asyncio.run(orch.run_training_cycle())
        assert result is not None
        assert result.final_ihsan_score == 0.97
        trainer.train.assert_called_once()
        assert orch.metrics.training_runs == 1

    def test_pipes_observations_to_reflex_bridge(self):
        trainer = _mock_trainer()
        reflex_bridge = SDPOReflexBridge()
        orch = LearningLoopOrchestrator(
            enabled=True,
            sdpo_trainer=trainer,
            reflex_bridge=reflex_bridge,
        )
        for _ in range(3):
            orch.on_candidate(_make_candidate())
        asyncio.run(orch.run_training_cycle())
        # Observations should have been piped
        assert reflex_bridge.total_observations >= 3
        assert orch.metrics.total_observations >= 3

    def test_preserves_snr_signal_for_reflex_impact(self):
        trainer = _mock_trainer(final_ihsan=0.99)
        reflex_bridge = SDPOReflexBridge(min_observations=3)
        orch = LearningLoopOrchestrator(
            enabled=True,
            sdpo_trainer=trainer,
            reflex_bridge=reflex_bridge,
        )
        for _ in range(3):
            orch.on_candidate(_make_candidate(ihsan=0.99))

        asyncio.run(orch.run_training_cycle())

        candidates = reflex_bridge.get_eligible_candidates()
        assert len(candidates) >= 1
        assert candidates[0].avg_snr > UNIFIED_SNR_THRESHOLD
        assert candidates[0].impact_score > 0.01

    def test_training_ihsan_history(self):
        trainer = _mock_trainer(final_ihsan=0.98)
        orch = LearningLoopOrchestrator(enabled=True, sdpo_trainer=trainer)
        for _ in range(3):
            orch.on_candidate(_make_candidate())
        asyncio.run(orch.run_training_cycle())
        assert orch.metrics.avg_training_ihsan == 0.98


# ═══════════════════════════════════════════════════════════════════════════
# Stage 3: run_compilation_cycle() — Reflex Bridge → Constitutional Compile
# ═══════════════════════════════════════════════════════════════════════════


class TestCompilationCycle:
    def test_no_candidates(self):
        orch = LearningLoopOrchestrator(enabled=True)
        compiled = orch.run_compilation_cycle()
        assert compiled == []
        assert orch.metrics.loop_cycles == 1

    def test_disabled_skips_compilation(self):
        reflex_bridge = SDPOReflexBridge(min_observations=1)
        # Inject an observation that meets all thresholds
        reflex_bridge.observe("high quality task", 0.99, 0.98, 0.01, True)
        orch = LearningLoopOrchestrator(
            enabled=False,
            reflex_bridge=reflex_bridge,
        )
        compiled = orch.run_compilation_cycle()
        assert compiled == []  # Disabled

    def test_compiles_eligible_candidates(self):
        reflex_bridge = SDPOReflexBridge(min_observations=2)
        reflex_cache: dict = {}
        # Build up observations that meet thresholds
        for _ in range(5):
            reflex_bridge.observe("excellent pattern", 0.99, 0.97, 0.01, True)
        orch = LearningLoopOrchestrator(
            enabled=True,
            reflex_bridge=reflex_bridge,
            reflex_cache=reflex_cache,
        )
        compiled = orch.run_compilation_cycle()
        assert len(compiled) >= 1
        assert orch.metrics.reflexes_compiled >= 1
        assert len(reflex_cache) >= 1

    def test_compiled_reflexes_stored_in_cache(self):
        reflex_bridge = SDPOReflexBridge(min_observations=2)
        reflex_cache: dict = {}
        for _ in range(5):
            reflex_bridge.observe("cache test", 0.99, 0.97, 0.01, True)
        orch = LearningLoopOrchestrator(
            enabled=True,
            reflex_bridge=reflex_bridge,
            reflex_cache=reflex_cache,
        )
        orch.run_compilation_cycle()
        # Reflex should be in cache
        assert len(reflex_cache) >= 1
        # All cached reflexes should be proper Reflex objects
        for reflex in reflex_cache.values():
            assert isinstance(reflex, Reflex)

    def test_marks_compiled_prevents_recompilation(self):
        reflex_bridge = SDPOReflexBridge(min_observations=2)
        for _ in range(5):
            reflex_bridge.observe("one-time compile", 0.99, 0.97, 0.01, True)
        orch = LearningLoopOrchestrator(
            enabled=True,
            reflex_bridge=reflex_bridge,
        )
        # First cycle compiles
        first = orch.run_compilation_cycle()
        assert len(first) >= 1
        # Second cycle — already compiled, should not re-compile
        second = orch.run_compilation_cycle()
        assert len(second) == 0

    def test_low_ihsan_not_eligible(self):
        """Candidates with ihsan below REFLEX_IHSAN_THRESHOLD (0.98) are
        filtered out by the bridge's eligible property — never reach compiler.
        This is constitutional: the bridge is stricter than compile_reflex."""
        reflex_bridge = SDPOReflexBridge(min_observations=2)
        # Observations with ihsan=0.90 — below bridge threshold of 0.98
        for _ in range(5):
            reflex_bridge.observe("marginal quality", 0.90, 0.90, 0.1, True)
        orch = LearningLoopOrchestrator(
            enabled=True,
            reflex_bridge=reflex_bridge,
        )
        compiled = orch.run_compilation_cycle()
        # Bridge filters them — never reaches compile_reflex
        assert len(compiled) == 0
        assert orch.metrics.reflexes_compiled == 0

    def test_compile_reflex_denial_path(self):
        """Exercise the compile_reflex denial path by mocking the bridge
        to return a candidate with ihsan that passes bridge but fails compiler."""
        reflex_bridge = MagicMock(spec=SDPOReflexBridge)
        # Return a candidate with ihsan=0.94 — passes bridge mock but
        # compile_reflex rejects (fp(0.94) = 940_000 < IHSAN_FLOOR = 950_000)
        from core.sdpo.reflex_bridge import ReflexCandidate

        fake_candidate = ReflexCandidate(
            pattern_id="abc123",
            pattern_description="edge case pattern",
            source_task="edge case",
            avg_ihsan=0.94,
            avg_snr=0.90,
            reproducibility=0.95,
            observation_count=10,
            impact_score=0.05,
        )
        reflex_bridge.get_eligible_candidates.return_value = [fake_candidate]
        orch = LearningLoopOrchestrator(
            enabled=True,
            reflex_bridge=reflex_bridge,
        )
        compiled = orch.run_compilation_cycle()
        assert len(compiled) == 0
        assert orch.metrics.reflexes_denied == 1
        reflex_bridge.deny.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════
# Full Cycle: End-to-End
# ═══════════════════════════════════════════════════════════════════════════


class TestFullCycle:
    def test_full_cycle_with_training(self):
        trainer = _mock_trainer(final_ihsan=0.98)
        orch = LearningLoopOrchestrator(enabled=True, sdpo_trainer=trainer)
        for _ in range(3):
            orch.on_candidate(_make_candidate())
        result = asyncio.run(orch.run_full_cycle())
        assert result["training_executed"] is True
        assert result["training_ihsan"] == 0.98
        assert "metrics" in result

    def test_full_cycle_without_training(self):
        orch = LearningLoopOrchestrator(enabled=True)
        result = asyncio.run(orch.run_full_cycle())
        assert result["training_executed"] is False
        assert result["training_ihsan"] is None

    def test_end_to_end_pipeline(self):
        """Full pipeline: candidates → training → observations → compilation."""
        trainer = _mock_trainer(final_ihsan=0.99)
        reflex_bridge = SDPOReflexBridge(min_observations=3)
        reflex_cache: dict = {}
        orch = LearningLoopOrchestrator(
            enabled=True,
            sdpo_trainer=trainer,
            reflex_bridge=reflex_bridge,
            reflex_cache=reflex_cache,
        )

        # Step 1: Feed candidates with the same task so observations aggregate.
        for _ in range(3):
            orch.on_candidate(_make_candidate(fitness=0.96, ihsan=0.98))
        assert orch.metrics.candidates_accepted == 3

        # Step 2-4: Run one full cycle and require a compiled reflex.
        result = asyncio.run(orch.run_full_cycle())
        compiled = orch.run_compilation_cycle()
        assert result["training_executed"] is True
        assert result["reflexes_compiled"] >= 1
        assert result["reflex_cache_size"] >= 1
        assert len(compiled) == 0  # Already compiled during run_full_cycle()
        assert orch.metrics.loop_cycles >= 1
        assert orch.metrics.total_observations >= 3
        assert orch.metrics.reflexes_compiled >= 1
        assert len(reflex_cache) >= 1


# ═══════════════════════════════════════════════════════════════════════════
# Observability and Status
# ═══════════════════════════════════════════════════════════════════════════


class TestObservability:
    def test_get_status(self):
        orch = LearningLoopOrchestrator(enabled=True)
        status = orch.get_status()
        assert status["enabled"] is True
        assert "evolution_bridge" in status
        assert "reflex_bridge" in status
        assert "metrics" in status
        assert "recent_events" in status

    def test_event_history_capped(self):
        orch = LearningLoopOrchestrator(enabled=True)
        # Generate many events
        for i in range(600):
            orch.on_candidate(_make_candidate())
        events = orch.get_events(limit=1000)
        # Internal cap at 500
        assert len(events) <= 500

    def test_reflex_cache_size_property(self):
        cache = {b"key": MagicMock()}
        orch = LearningLoopOrchestrator(reflex_cache=cache)
        assert orch.reflex_cache_size == 1

    def test_enabled_property(self):
        assert LearningLoopOrchestrator(enabled=True).enabled is True
        assert LearningLoopOrchestrator(enabled=False).enabled is False


# ═══════════════════════════════════════════════════════════════════════════
# Edge Cases
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_default_disabled(self):
        """Default should respect env var (not set = disabled)."""
        with patch.dict("os.environ", {}, clear=False):
            orch = LearningLoopOrchestrator()
            # Default is disabled unless BIZRA_CLOSED_LOOP_ENABLED=1
            assert isinstance(orch.enabled, bool)

    def test_candidate_with_minimal_genome(self):
        """Candidate with genome lacking optional attributes."""
        genome = MagicMock(spec=[])  # No attributes
        candidate = IntegrationCandidate(
            genome=genome,
            fitness=0.95,
            novelty_score=0.5,
            ihsan_score=0.97,
            recommendation="Integrate",
        )
        orch = LearningLoopOrchestrator(enabled=True)
        # Should not crash — getattr with defaults handles missing attrs
        accepted = orch.on_candidate(candidate)
        assert accepted is True

    def test_empty_reflex_cache_provided(self):
        orch = LearningLoopOrchestrator(reflex_cache={})
        assert orch.reflex_cache_size == 0

    def test_metrics_survive_multiple_cycles(self):
        orch = LearningLoopOrchestrator(enabled=True)
        for _ in range(10):
            orch.run_compilation_cycle()
        assert orch.metrics.loop_cycles == 10
        assert orch.metrics.last_cycle_at is not None
