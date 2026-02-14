"""
Tests for Benchmark Dominance Loop — Recursive Optimization Cycle
==================================================================

Standing on the Shoulders of Giants:
- John Boyd (1995): OODA Loop
- W. Edwards Deming (1950): PDCA Cycle
- Eliyahu Goldratt (1984): Theory of Constraints

إحسان — Excellence in all things.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from core.benchmark.dominance_loop import (
    BenchmarkDominanceLoop,
    LoopPhase,
    LoopState,
    CycleOutcome,
    CycleResult,
    DominanceResult,
)
from core.benchmark.leaderboard import Benchmark, SubmissionResult, SubmissionStatus
from core.benchmark.clear_framework import CLEARMetrics, MetricWeight


# ═══════════════════════════════════════════════════════════════════════════════
# Enums & Dataclasses
# ═══════════════════════════════════════════════════════════════════════════════


class TestLoopPhase:

    def test_all_phases(self):
        assert len(LoopPhase) == 6
        assert LoopPhase.EVALUATE.name == "EVALUATE"
        assert LoopPhase.IDLE in LoopPhase

    def test_ordering(self):
        phases = [LoopPhase.EVALUATE, LoopPhase.ABLATE, LoopPhase.ARCHITECT,
                  LoopPhase.SUBMIT, LoopPhase.ANALYZE, LoopPhase.IDLE]
        assert len(set(phases)) == 6


class TestCycleOutcome:

    def test_all_outcomes(self):
        assert CycleOutcome.IMPROVED.name == "IMPROVED"
        assert CycleOutcome.SOTA_ACHIEVED.name == "SOTA_ACHIEVED"
        assert CycleOutcome.FAILED.name == "FAILED"


class TestLoopState:

    def test_defaults(self):
        s = LoopState()
        assert s.current_phase == LoopPhase.IDLE
        assert s.cycles_completed == 0
        assert s.best_score == 0.0
        assert s.max_cycles == 100

    def test_to_dict(self):
        s = LoopState()
        d = s.to_dict()
        assert d["phase"] == "IDLE"
        assert "cycles_completed" in d
        assert "best_score" in d

    def test_custom_values(self):
        s = LoopState(max_cycles=50, patience=3, improvement_threshold=0.05)
        assert s.max_cycles == 50
        assert s.patience == 3


class TestCycleResult:

    def test_summary(self):
        r = CycleResult(
            cycle_id="abc12345",
            outcome=CycleOutcome.IMPROVED,
            start_score=0.40,
            end_score=0.45,
            improvement=0.05,
            improvement_pct=12.5,
            cycle_cost_usd=0.50,
            cycle_duration_seconds=120.0,
            bottleneck_component="retrieval",
            recommended_action="Upgrade retrieval module",
        )
        s = r.summary()
        assert "IMPROVED" in s
        assert "abc12345" in s
        assert "retrieval" in s

    def test_summary_minimal(self):
        r = CycleResult(cycle_id="min", outcome=CycleOutcome.MAINTAINED)
        s = r.summary()
        assert "MAINTAINED" in s


class TestDominanceResult:

    def test_efficiency_ratio_zero_cost(self):
        r = DominanceResult(
            campaign_id="x",
            target_benchmark=Benchmark.SWE_BENCH,
            total_cost_usd=0.0,
            improvement_from_baseline=0.1,
        )
        assert r.efficiency_ratio() == 0.0

    def test_efficiency_ratio(self):
        r = DominanceResult(
            campaign_id="y",
            target_benchmark=Benchmark.HLE,
            total_cost_usd=10.0,
            improvement_from_baseline=0.5,
        )
        assert r.efficiency_ratio() == pytest.approx(0.05)


# ═══════════════════════════════════════════════════════════════════════════════
# BenchmarkDominanceLoop — Construction
# ═══════════════════════════════════════════════════════════════════════════════


class TestBenchmarkDominanceLoopInit:

    def test_init(self):
        loop = BenchmarkDominanceLoop(
            target_benchmark=Benchmark.SWE_BENCH,
            agent_id="test-agent",
        )
        assert loop.target_benchmark == Benchmark.SWE_BENCH
        assert loop.agent_id == "test-agent"
        assert loop.state.current_phase == LoopPhase.IDLE

    def test_init_custom_weights(self):
        w = MetricWeight(cost=0.1, latency=0.1, efficacy=0.5, assurance=0.2, reliability=0.1)
        loop = BenchmarkDominanceLoop(
            target_benchmark=Benchmark.MMLU_PRO,
            clear_weights=w,
        )
        assert loop.clear_framework.weights.efficacy == 0.5

    def test_set_agent_factory(self):
        loop = BenchmarkDominanceLoop(target_benchmark=Benchmark.HLE)
        factory = MagicMock()
        loop.set_agent_factory(factory)
        assert loop._agent_factory is factory

    def test_set_inference_function(self):
        loop = BenchmarkDominanceLoop(target_benchmark=Benchmark.GPQA)
        fn = MagicMock()
        loop.set_inference_function(fn)
        assert loop._inference_fn is fn


# ═══════════════════════════════════════════════════════════════════════════════
# Loop control logic
# ═══════════════════════════════════════════════════════════════════════════════


class TestShouldContinue:

    def test_stop_on_max_cycles(self):
        loop = BenchmarkDominanceLoop(target_benchmark=Benchmark.SWE_BENCH)
        loop.state.max_cycles = 5
        loop.state.cycles_completed = 5
        assert loop._should_continue(target=1.0, budget=100.0) is False

    def test_stop_on_budget(self):
        loop = BenchmarkDominanceLoop(target_benchmark=Benchmark.SWE_BENCH)
        loop.state.total_cost_usd = 100.0
        assert loop._should_continue(target=1.0, budget=100.0) is False

    def test_stop_on_target(self):
        loop = BenchmarkDominanceLoop(target_benchmark=Benchmark.SWE_BENCH)
        loop.state.current_score = 0.95
        assert loop._should_continue(target=0.90, budget=100.0) is False

    def test_stop_on_patience(self):
        loop = BenchmarkDominanceLoop(target_benchmark=Benchmark.SWE_BENCH)
        loop.state.patience = 3
        loop.state.consecutive_regressions = 3
        assert loop._should_continue(target=1.0, budget=100.0) is False

    def test_continues_normally(self):
        loop = BenchmarkDominanceLoop(target_benchmark=Benchmark.SWE_BENCH)
        loop.state.max_cycles = 100
        loop.state.cycles_completed = 0
        assert loop._should_continue(target=1.0, budget=100.0) is True


# ═══════════════════════════════════════════════════════════════════════════════
# Phase transition
# ═══════════════════════════════════════════════════════════════════════════════


class TestPhaseTransition:

    def test_transition(self):
        loop = BenchmarkDominanceLoop(target_benchmark=Benchmark.SWE_BENCH)
        loop._transition_phase(LoopPhase.EVALUATE)
        assert loop.state.current_phase == LoopPhase.EVALUATE
        assert loop.state.phase_start_time > 0

    def test_callback(self):
        loop = BenchmarkDominanceLoop(target_benchmark=Benchmark.SWE_BENCH)
        phases_seen = []
        loop._on_phase_start = lambda p: phases_seen.append(p)
        loop._transition_phase(LoopPhase.ABLATE)
        assert LoopPhase.ABLATE in phases_seen


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark enum
# ═══════════════════════════════════════════════════════════════════════════════


class TestBenchmarkEnum:

    def test_swe_bench(self):
        b = Benchmark.SWE_BENCH
        assert b.key == "swe-bench"
        assert b.sota_2025 == 0.42

    def test_hle(self):
        b = Benchmark.HLE
        assert b.domain == "Abstract reasoning"

    def test_all_have_key(self):
        for b in Benchmark:
            assert b.key
            assert b.benchmark_name
            assert 0 < b.sota_2025 <= 1.0
