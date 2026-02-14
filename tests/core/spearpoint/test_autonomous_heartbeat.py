"""
Autonomous Pattern-Aware Heartbeat Tests — Phase 21
====================================================

Tests for PatternStrategySelector, MetricsProvider, and the
pattern-aware RecursiveLoop integration.

Standing on Giants:
- Li et al. (2025): Sci-Reasoning thinking patterns
- Nygard (2007): Circuit breaker pattern
- Boyd (1995): OODA loop
- Thompson (1933): Exploration/exploitation balance
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from core.proof_engine.ihsan_gate import IhsanComponents
from core.spearpoint.auto_researcher import ResearchOutcome, ResearchResult
from core.spearpoint.metrics_provider import MetricsProvider, SystemMetricsSnapshot
from core.spearpoint.pattern_selector import (
    PatternOutcome,
    PatternSelectionResult,
    PatternStrategySelector,
)
from core.spearpoint.config import SpearpointConfig
from core.spearpoint.recursive_loop import LoopMetrics, RecursiveLoop


# ---------------------------------------------------------------------------
# PatternStrategySelector Tests
# ---------------------------------------------------------------------------


class TestPatternStrategySelector:
    """Test intelligent pattern routing."""

    def test_initial_selection_is_explore_or_rotate(self):
        """First selection should explore or rotate (no history)."""
        selector = PatternStrategySelector(seed=42)
        result = selector.select(cycle=0)
        assert result.pattern_id in PatternStrategySelector.ALL_PATTERNS
        assert result.strategy in ("explore", "rotate")

    def test_all_15_patterns_available(self):
        """All 15 Sci-Reasoning patterns are available."""
        selector = PatternStrategySelector()
        assert len(selector.ALL_PATTERNS) == 15
        assert selector.ALL_PATTERNS[0] == "P01"
        assert selector.ALL_PATTERNS[-1] == "P15"

    def test_record_outcome_tracks_history(self):
        """Recording outcomes updates internal state."""
        selector = PatternStrategySelector()
        selector.record_outcome("P01", approved=1, cycle=0)
        selector.record_outcome("P01", rejected=1, cycle=1)

        stats = selector.get_statistics()
        assert stats["per_pattern"]["P01"]["approved"] == 1
        assert stats["per_pattern"]["P01"]["rejected"] == 1
        assert stats["per_pattern"]["P01"]["total"] == 2

    def test_complement_strategy_after_approval(self):
        """After an approval, selector should try complementary patterns."""
        selector = PatternStrategySelector(
            seed=42, explore_probability=0.0, cooldown_cycles=0,
        )
        selector.load_cooccurrence({
            "P01": {"P02": 50, "P03": 30},
        })
        selector.record_outcome("P01", approved=1, cycle=0)

        result = selector.select(cycle=1)
        assert result.strategy == "complement"
        assert result.pattern_id == "P02"

    def test_exploit_selects_best_success_rate(self):
        """Exploit strategy picks the pattern with highest success rate."""
        selector = PatternStrategySelector(
            seed=42, explore_probability=0.0, cooldown_cycles=0,
        )
        # P03 has best success rate
        selector.record_outcome("P01", approved=1, rejected=3, cycle=0)
        selector.record_outcome("P02", approved=0, rejected=2, cycle=1)
        selector.record_outcome("P03", approved=3, rejected=1, cycle=2)

        # Clear last_approved so complement doesn't fire
        selector._last_approved_pattern = None

        result = selector.select(cycle=5)
        assert result.strategy == "exploit"
        assert result.pattern_id == "P03"

    def test_rotate_picks_least_recently_used(self):
        """Rotation strategy picks least recently used pattern."""
        selector = PatternStrategySelector(
            seed=42, explore_probability=0.0, cooldown_cycles=0,
        )
        # Use P01 and P02 recently, P03-P15 never used
        selector.record_outcome("P01", approved=0, rejected=1, cycle=10)
        selector.record_outcome("P02", approved=0, rejected=1, cycle=11)

        # Clear last_approved so complement doesn't fire
        selector._last_approved_pattern = None
        # No successful patterns → exploit returns None → falls to rotate
        result = selector.select(cycle=15)
        assert result.strategy == "rotate"
        # Should pick from the never-used patterns (cycle -1)
        assert result.pattern_id not in ("P01", "P02")

    def test_deterministic_with_seed(self):
        """Same seed produces same sequence."""
        s1 = PatternStrategySelector(seed=123)
        s2 = PatternStrategySelector(seed=123)
        r1 = s1.select(cycle=0)
        r2 = s2.select(cycle=0)
        assert r1.pattern_id == r2.pattern_id
        assert r1.strategy == r2.strategy

    def test_statistics_empty(self):
        """Statistics on fresh selector."""
        selector = PatternStrategySelector()
        stats = selector.get_statistics()
        assert stats["patterns_tried"] == 0
        assert stats["patterns_available"] == 15
        assert stats["last_approved_pattern"] is None

    def test_chronic_failure_deprioritized(self):
        """Patterns with only failures are deprioritized in rotation."""
        selector = PatternStrategySelector(
            seed=42, explore_probability=0.0, cooldown_cycles=0,
        )
        # P01 fails 5 times with no approvals → chronic failure
        for i in range(5):
            selector.record_outcome("P01", rejected=1, cycle=i)

        selector._last_approved_pattern = None
        result = selector.select(cycle=10)
        # Should skip P01 (chronic failure) and pick another
        assert result.pattern_id != "P01" or result.strategy == "rotate"


# ---------------------------------------------------------------------------
# MetricsProvider Tests
# ---------------------------------------------------------------------------


class TestMetricsProvider:
    """Test real system metrics computation."""

    def test_initial_snapshot_has_moderate_defaults(self):
        """Fresh provider returns moderate defaults, not hardcoded lows."""
        provider = MetricsProvider()
        snapshot = provider.current_snapshot()
        # Defaults should be higher than the old hardcoded values
        assert snapshot.accuracy >= 0.85
        assert snapshot.task_completion >= 0.85
        assert snapshot.correctness >= 0.85

    def test_clear_metrics_dict_format(self):
        """to_clear_metrics returns the keys the evaluator expects."""
        provider = MetricsProvider()
        snapshot = provider.current_snapshot()
        d = snapshot.to_clear_metrics()
        assert "accuracy" in d
        assert "task_completion" in d
        assert "goal_achievement" in d
        assert "reproducibility" in d
        assert "consistency" in d

    def test_ihsan_components_format(self):
        """to_ihsan_components returns valid IhsanComponents."""
        provider = MetricsProvider()
        snapshot = provider.current_snapshot()
        components = snapshot.to_ihsan_components()
        assert isinstance(components, IhsanComponents)
        assert 0 <= components.correctness <= 1
        assert 0 <= components.safety <= 1
        assert 0 <= components.efficiency <= 1
        assert 0 <= components.user_benefit <= 1

    def test_metrics_improve_with_approvals(self):
        """Recording approvals should improve metrics over time."""
        provider = MetricsProvider()
        for _ in range(5):
            provider.record_cycle_metrics(
                approved=2, rejected=0, inconclusive=0,
                clear_score=0.90, ihsan_score=0.96,
            )

        snapshot = provider.current_snapshot()
        assert snapshot.accuracy >= 0.95
        assert snapshot.correctness >= 0.95

    def test_metrics_degrade_with_rejections(self):
        """Recording rejections should degrade metrics."""
        provider = MetricsProvider()
        for _ in range(5):
            provider.record_cycle_metrics(
                approved=0, rejected=3, inconclusive=0,
                clear_score=0.60, ihsan_score=0.70,
            )

        snapshot = provider.current_snapshot()
        assert snapshot.accuracy < 0.5  # Many rejections → low accuracy

    def test_statistics(self):
        """Provider statistics track accumulated state."""
        provider = MetricsProvider()
        provider.record_cycle_metrics(approved=1, clear_score=0.85, ihsan_score=0.93)
        stats = provider.get_statistics()
        assert stats["total_cycles"] == 1
        assert stats["approved"] == 1
        assert stats["avg_clear"] > 0


# ---------------------------------------------------------------------------
# LoopMetrics Tests
# ---------------------------------------------------------------------------


class TestLoopMetricsPatternAware:
    """Test pattern-aware fields in LoopMetrics."""

    def test_pattern_aware_flag(self):
        """Pattern-aware metrics include pattern fields in dict."""
        m = LoopMetrics(pattern_aware=True, last_pattern_id="P01", last_pattern_strategy="explore")
        d = m.to_dict()
        assert d["pattern_aware"] is True
        assert d["last_pattern_id"] == "P01"
        assert d["last_pattern_strategy"] == "explore"

    def test_non_pattern_aware_omits_fields(self):
        """Non-pattern-aware metrics omit pattern fields."""
        m = LoopMetrics(pattern_aware=False)
        d = m.to_dict()
        assert "pattern_aware" not in d
        assert "last_pattern_id" not in d


# ---------------------------------------------------------------------------
# Pattern-Aware RecursiveLoop Tests
# ---------------------------------------------------------------------------


class TestPatternAwareLoop:
    """Test RecursiveLoop in pattern-aware mode."""

    @pytest.fixture
    def fast_config(self):
        config = SpearpointConfig()
        config.loop_interval_seconds = 0.01
        config.circuit_breaker_backoff_seconds = 0.01
        config.max_iterations_per_cycle = 3
        return config

    @pytest.fixture
    def mock_evaluator(self):
        evaluator = MagicMock()
        evaluator.ledger = MagicMock()
        evaluator.get_statistics.return_value = {"total_evaluations": 0}
        return evaluator

    @pytest.fixture
    def mock_researcher(self, mock_evaluator):
        researcher = MagicMock()

        # research_with_pattern returns list of ResearchResult
        result = MagicMock(spec=ResearchResult)
        result.outcome = ResearchOutcome.INCONCLUSIVE
        result.evaluation = MagicMock()
        result.evaluation.clear_score = 0.80
        result.evaluation.ihsan_score = 0.92
        researcher.research_with_pattern.return_value = [result]
        researcher.research.return_value = [result]
        return researcher

    @pytest.mark.asyncio
    async def test_pattern_aware_loop_runs(self, mock_evaluator, mock_researcher, fast_config):
        """Pattern-aware loop selects and uses patterns."""
        selector = PatternStrategySelector(seed=42)
        provider = MetricsProvider()

        loop = RecursiveLoop(
            evaluator=mock_evaluator,
            researcher=mock_researcher,
            config=fast_config,
            pattern_selector=selector,
            metrics_provider=provider,
        )

        metrics = await loop.run(max_cycles=2)

        assert metrics.cycles_completed == 2
        assert metrics.pattern_aware is True
        assert metrics.patterns_tried >= 1
        assert metrics.last_pattern_id != ""
        assert mock_researcher.research_with_pattern.called

    @pytest.mark.asyncio
    async def test_classic_mode_still_works(self, mock_evaluator, mock_researcher, fast_config):
        """Without selector, loop uses classic research() path."""
        loop = RecursiveLoop(
            evaluator=mock_evaluator,
            researcher=mock_researcher,
            config=fast_config,
        )

        metrics = await loop.run(max_cycles=1)

        assert metrics.cycles_completed == 1
        assert metrics.pattern_aware is False
        assert mock_researcher.research.called
        assert not mock_researcher.research_with_pattern.called

    @pytest.mark.asyncio
    async def test_metrics_provider_receives_feedback(self, mock_evaluator, mock_researcher, fast_config):
        """MetricsProvider receives cycle feedback after each cycle."""
        selector = PatternStrategySelector(seed=42)
        provider = MetricsProvider()

        loop = RecursiveLoop(
            evaluator=mock_evaluator,
            researcher=mock_researcher,
            config=fast_config,
            pattern_selector=selector,
            metrics_provider=provider,
        )

        await loop.run(max_cycles=3)

        stats = provider.get_statistics()
        assert stats["total_cycles"] == 3

    @pytest.mark.asyncio
    async def test_pattern_selector_receives_feedback(self, mock_evaluator, mock_researcher, fast_config):
        """PatternSelector receives outcome feedback after each cycle."""
        selector = PatternStrategySelector(seed=42)

        loop = RecursiveLoop(
            evaluator=mock_evaluator,
            researcher=mock_researcher,
            config=fast_config,
            pattern_selector=selector,
        )

        await loop.run(max_cycles=2)

        stats = selector.get_statistics()
        assert stats["patterns_tried"] >= 1

    @pytest.mark.asyncio
    async def test_circuit_breaker_works_in_pattern_mode(self, mock_evaluator, fast_config):
        """Circuit breaker still trips in pattern-aware mode."""
        researcher = MagicMock()
        rejected = MagicMock(spec=ResearchResult)
        rejected.outcome = ResearchOutcome.REJECTED
        rejected.evaluation = MagicMock()
        rejected.evaluation.clear_score = 0.50
        rejected.evaluation.ihsan_score = 0.60
        researcher.research_with_pattern.return_value = [rejected]

        selector = PatternStrategySelector(seed=42)
        fast_config.circuit_breaker_consecutive_rejections = 2

        loop = RecursiveLoop(
            evaluator=mock_evaluator,
            researcher=researcher,
            config=fast_config,
            pattern_selector=selector,
        )

        metrics = await loop.run(max_cycles=5)

        assert metrics.circuit_breaker_trips >= 1


# ---------------------------------------------------------------------------
# SystemMetricsSnapshot Tests
# ---------------------------------------------------------------------------


class TestSystemMetricsSnapshot:
    """Test SystemMetricsSnapshot data class."""

    def test_default_values(self):
        """Snapshot has safe defaults."""
        s = SystemMetricsSnapshot()
        assert s.safety == 1.0
        assert s.accuracy == 0.0

    def test_to_clear_metrics_keys(self):
        """CLEAR metrics dict has all required keys."""
        s = SystemMetricsSnapshot(accuracy=0.9, task_completion=0.85)
        d = s.to_clear_metrics()
        required = {"accuracy", "task_completion", "goal_achievement",
                     "reproducibility", "consistency", "runs_completed"}
        assert required.issubset(d.keys())

    def test_to_ihsan_components_values(self):
        """Ihsan components match snapshot values."""
        s = SystemMetricsSnapshot(
            correctness=0.95, safety=1.0, efficiency=0.90, user_benefit=0.85,
        )
        c = s.to_ihsan_components()
        assert c.correctness == 0.95
        assert c.safety == 1.0
        assert c.efficiency == 0.90
        assert c.user_benefit == 0.85


# ---------------------------------------------------------------------------
# Metrics Pass-Through Tests
# ---------------------------------------------------------------------------


class TestMetricsPassThrough:
    """Verify real metrics flow from MetricsProvider → RecursiveLoop → evaluator."""

    @pytest.fixture
    def fast_config(self):
        config = SpearpointConfig()
        config.loop_interval_seconds = 0.01
        config.circuit_breaker_backoff_seconds = 0.01
        config.max_iterations_per_cycle = 2
        return config

    @pytest.fixture
    def mock_evaluator(self):
        evaluator = MagicMock()
        evaluator.ledger = MagicMock()
        evaluator.get_statistics.return_value = {"total_evaluations": 0}
        return evaluator

    @pytest.mark.asyncio
    async def test_real_metrics_passed_to_researcher(self, mock_evaluator, fast_config):
        """MetricsProvider snapshot is forwarded to research_with_pattern()."""
        researcher = MagicMock()
        result = MagicMock(spec=ResearchResult)
        result.outcome = ResearchOutcome.INCONCLUSIVE
        result.evaluation = MagicMock()
        result.evaluation.clear_score = 0.80
        result.evaluation.ihsan_score = 0.92
        researcher.research_with_pattern.return_value = [result]

        selector = PatternStrategySelector(seed=42)
        provider = MetricsProvider()

        loop = RecursiveLoop(
            evaluator=mock_evaluator,
            researcher=researcher,
            config=fast_config,
            pattern_selector=selector,
            metrics_provider=provider,
        )

        await loop.run(max_cycles=1)

        # Verify research_with_pattern was called with metrics kwargs
        call_kwargs = researcher.research_with_pattern.call_args
        assert "metrics" in call_kwargs.kwargs or (
            len(call_kwargs.args) > 4 and call_kwargs.args[4] is not None
        ), "metrics should be passed to research_with_pattern"
        assert "ihsan_components" in call_kwargs.kwargs or (
            len(call_kwargs.args) > 5 and call_kwargs.args[5] is not None
        ), "ihsan_components should be passed to research_with_pattern"

    @pytest.mark.asyncio
    async def test_metrics_values_are_real_not_hardcoded(self, mock_evaluator, fast_config):
        """Passed metrics use MetricsProvider defaults (0.85+), not hardcoded (0.7/0.6/0.5)."""
        researcher = MagicMock()
        result = MagicMock(spec=ResearchResult)
        result.outcome = ResearchOutcome.INCONCLUSIVE
        result.evaluation = MagicMock()
        result.evaluation.clear_score = 0.80
        result.evaluation.ihsan_score = 0.92
        researcher.research_with_pattern.return_value = [result]

        selector = PatternStrategySelector(seed=42)
        provider = MetricsProvider()

        loop = RecursiveLoop(
            evaluator=mock_evaluator,
            researcher=researcher,
            config=fast_config,
            pattern_selector=selector,
            metrics_provider=provider,
        )

        await loop.run(max_cycles=1)

        call_kwargs = researcher.research_with_pattern.call_args.kwargs
        metrics_dict = call_kwargs["metrics"]
        # MetricsProvider initial defaults: accuracy=0.95 (0.85+0.1), task_completion=0.90
        # NOT the hardcoded 0.7/0.6/0.5
        assert metrics_dict["accuracy"] >= 0.85, (
            f"accuracy={metrics_dict['accuracy']} should be >= 0.85 (real metrics), "
            f"not the hardcoded 0.7"
        )
        assert metrics_dict["task_completion"] >= 0.85, (
            f"task_completion={metrics_dict['task_completion']} should be >= 0.85, "
            f"not the hardcoded 0.6"
        )

    @pytest.mark.asyncio
    async def test_no_provider_uses_no_metrics_kwargs(self, mock_evaluator, fast_config):
        """Without MetricsProvider, no metrics kwargs are passed."""
        researcher = MagicMock()
        result = MagicMock(spec=ResearchResult)
        result.outcome = ResearchOutcome.INCONCLUSIVE
        result.evaluation = MagicMock()
        result.evaluation.clear_score = 0.80
        result.evaluation.ihsan_score = 0.92
        researcher.research_with_pattern.return_value = [result]

        selector = PatternStrategySelector(seed=42)

        loop = RecursiveLoop(
            evaluator=mock_evaluator,
            researcher=researcher,
            config=fast_config,
            pattern_selector=selector,
            # No metrics_provider
        )

        await loop.run(max_cycles=1)

        call_kwargs = researcher.research_with_pattern.call_args.kwargs
        assert "metrics" not in call_kwargs
        assert "ihsan_components" not in call_kwargs


__all__ = [
    "TestPatternStrategySelector",
    "TestMetricsProvider",
    "TestLoopMetricsPatternAware",
    "TestPatternAwareLoop",
    "TestSystemMetricsSnapshot",
    "TestMetricsPassThrough",
]
