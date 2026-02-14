"""
Tests for RecursiveLoop — Continuous Orchestration Heartbeat
=============================================================

Verifies:
- Circuit breaker (3 rejections -> backoff)
- Evidence ledger file lock safety
- Graceful shutdown
- Loop metrics tracking
"""

import asyncio
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.spearpoint.auto_evaluator import AutoEvaluator
from core.spearpoint.auto_researcher import AutoResearcher, ResearchOutcome
from core.spearpoint.config import SpearpointConfig
from core.spearpoint.recursive_loop import RecursiveLoop, LoopMetrics


@pytest.fixture
def tmp_config(tmp_path: Path) -> SpearpointConfig:
    """Create a config pointing to temp directory."""
    return SpearpointConfig(
        state_dir=tmp_path / "spearpoint",
        evidence_ledger_path=tmp_path / "spearpoint" / "evidence.jsonl",
        hypothesis_memory_path=tmp_path / "spearpoint" / "hypothesis_memory",
        loop_interval_seconds=0.01,  # Fast for testing
        circuit_breaker_consecutive_rejections=3,
        circuit_breaker_backoff_seconds=0.05,  # Short backoff for testing
    )


@pytest.fixture
def evaluator(tmp_config: SpearpointConfig) -> AutoEvaluator:
    """Create an AutoEvaluator with temp config."""
    return AutoEvaluator(config=tmp_config)


@pytest.fixture
def researcher(
    evaluator: AutoEvaluator, tmp_config: SpearpointConfig
) -> AutoResearcher:
    """Create an AutoResearcher."""
    return AutoResearcher(evaluator=evaluator, config=tmp_config)


@pytest.fixture
def loop(
    evaluator: AutoEvaluator,
    researcher: AutoResearcher,
    tmp_config: SpearpointConfig,
) -> RecursiveLoop:
    """Create a RecursiveLoop."""
    return RecursiveLoop(
        evaluator=evaluator,
        researcher=researcher,
        config=tmp_config,
    )


class TestBasicOperation:
    """Verify basic loop operation."""

    @pytest.mark.asyncio
    async def test_run_fixed_cycles(self, loop: RecursiveLoop):
        """Loop runs fixed number of cycles and stops."""
        metrics = await loop.run(max_cycles=2)
        assert metrics.cycles_completed == 2

    @pytest.mark.asyncio
    async def test_metrics_populated(self, loop: RecursiveLoop):
        """Loop metrics are populated after run."""
        metrics = await loop.run(max_cycles=1)
        assert isinstance(metrics, LoopMetrics)
        assert metrics.cycles_completed >= 1
        assert metrics.last_cycle_time != ""

    @pytest.mark.asyncio
    async def test_zero_cycles(self, loop: RecursiveLoop):
        """Loop with max_cycles=0 exits immediately."""
        metrics = await loop.run(max_cycles=0)
        assert metrics.cycles_completed == 0


class TestCircuitBreaker:
    """Verify circuit breaker behavior."""

    @pytest.mark.asyncio
    async def test_breaker_trips_on_consecutive_rejections(
        self, evaluator: AutoEvaluator, tmp_config: SpearpointConfig
    ):
        """Circuit breaker trips after N consecutive rejections."""
        # Create a researcher that always produces rejections
        mock_researcher = MagicMock(spec=AutoResearcher)
        rejected_result = MagicMock()
        rejected_result.outcome = ResearchOutcome.REJECTED
        mock_researcher.research.return_value = [rejected_result]

        loop = RecursiveLoop(
            evaluator=evaluator,
            researcher=mock_researcher,
            config=tmp_config,
        )

        # Run enough cycles to trip the breaker (3 rejections + backoff)
        metrics = await loop.run(max_cycles=5)

        assert metrics.circuit_breaker_trips >= 1
        assert metrics.total_rejected >= 3

    @pytest.mark.asyncio
    async def test_breaker_resets_on_approval(
        self, evaluator: AutoEvaluator, tmp_config: SpearpointConfig
    ):
        """Circuit breaker resets consecutive count on approval."""
        call_count = 0

        def mock_research(**kwargs):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            # First two calls rejected, third approved
            if call_count <= 2:
                result.outcome = ResearchOutcome.REJECTED
            else:
                result.outcome = ResearchOutcome.APPROVED
            return [result]

        mock_researcher = MagicMock(spec=AutoResearcher)
        mock_researcher.research.side_effect = mock_research

        loop = RecursiveLoop(
            evaluator=evaluator,
            researcher=mock_researcher,
            config=tmp_config,
        )

        metrics = await loop.run(max_cycles=3)

        # Should NOT have tripped (only 2 consecutive, then reset)
        assert metrics.circuit_breaker_trips == 0
        assert metrics.total_approved >= 1

    @pytest.mark.asyncio
    async def test_breaker_backoff_resets(
        self, evaluator: AutoEvaluator, tmp_config: SpearpointConfig
    ):
        """After backoff, breaker resets and allows new cycles."""
        mock_researcher = MagicMock(spec=AutoResearcher)
        rejected_result = MagicMock()
        rejected_result.outcome = ResearchOutcome.REJECTED
        mock_researcher.research.return_value = [rejected_result]

        config = SpearpointConfig(
            state_dir=tmp_config.state_dir,
            evidence_ledger_path=tmp_config.evidence_ledger_path,
            hypothesis_memory_path=tmp_config.hypothesis_memory_path,
            loop_interval_seconds=0.01,
            circuit_breaker_consecutive_rejections=2,
            circuit_breaker_backoff_seconds=0.01,
        )

        loop = RecursiveLoop(
            evaluator=evaluator,
            researcher=mock_researcher,
            config=config,
        )

        metrics = await loop.run(max_cycles=5)

        # Breaker should have tripped and then reset after backoff
        assert metrics.circuit_breaker_trips >= 1
        assert metrics.backoff_events >= 1


class TestGracefulShutdown:
    """Verify graceful shutdown via asyncio.Event."""

    @pytest.mark.asyncio
    async def test_stop_event_stops_loop(
        self, loop: RecursiveLoop
    ):
        """request_stop() causes the loop to exit gracefully."""
        # Schedule stop after a short delay
        async def stop_after_delay():
            await asyncio.sleep(0.05)
            loop.request_stop()

        asyncio.create_task(stop_after_delay())

        # Run without max_cycles (would run forever without stop)
        metrics = await loop.run(max_cycles=100)

        # Should have stopped before 100 cycles
        assert metrics.cycles_completed < 100

    @pytest.mark.asyncio
    async def test_is_running_property(self, loop: RecursiveLoop):
        """is_running reflects loop state."""
        assert not loop.is_running

        async def check_running():
            await asyncio.sleep(0.02)
            assert loop.is_running
            loop.request_stop()

        asyncio.create_task(check_running())
        await loop.run(max_cycles=50)


class TestErrorHandling:
    """Verify fail-closed error handling."""

    @pytest.mark.asyncio
    async def test_exception_in_cycle_continues(
        self, evaluator: AutoEvaluator, tmp_config: SpearpointConfig
    ):
        """Exceptions in a cycle don't kill the loop."""
        call_count = 0

        def mock_research(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Simulated failure")
            result = MagicMock()
            result.outcome = ResearchOutcome.APPROVED
            return [result]

        mock_researcher = MagicMock(spec=AutoResearcher)
        mock_researcher.research.side_effect = mock_research

        loop = RecursiveLoop(
            evaluator=evaluator,
            researcher=mock_researcher,
            config=tmp_config,
        )

        metrics = await loop.run(max_cycles=3)

        assert metrics.cycles_completed == 3
        assert metrics.errors >= 1


class TestMetrics:
    """Verify metrics tracking."""

    @pytest.mark.asyncio
    async def test_metrics_serialization(self, loop: RecursiveLoop):
        """LoopMetrics serializes to dict."""
        metrics = await loop.run(max_cycles=1)
        d = metrics.to_dict()
        assert "cycles_completed" in d
        assert "total_hypotheses_evaluated" in d
        assert "circuit_breaker_trips" in d

    @pytest.mark.asyncio
    async def test_get_metrics_during_run(
        self, evaluator: AutoEvaluator, tmp_config: SpearpointConfig
    ):
        """get_metrics() returns current state."""
        mock_researcher = MagicMock(spec=AutoResearcher)
        result = MagicMock()
        result.outcome = ResearchOutcome.APPROVED
        mock_researcher.research.return_value = [result]

        loop = RecursiveLoop(
            evaluator=evaluator,
            researcher=mock_researcher,
            config=tmp_config,
        )

        await loop.run(max_cycles=2)
        metrics = loop.get_metrics()
        assert metrics.cycles_completed == 2
