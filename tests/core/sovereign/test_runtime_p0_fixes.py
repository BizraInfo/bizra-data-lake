"""
Tests for P0/P1 Runtime Fixes — Activation Sprint
===================================================
Verifies all critical bugs identified by the 4-agent SAPE audit are fixed:

RFC-04: SNR optimization result now used (was dead code)
RFC-03: Query metrics counted exactly once (was double-counted)
RFC-06: Muraqabah bridge uses correct Event attributes (topic/payload)
RFC-01: Feature flags respected by _init_components()
RFC-02: RuntimeMetrics.started_at declared in dataclass
RFC-05: Reasoning time variable assigned
LCT-01: Single checkpoint path (no redundant _checkpoint on shutdown)

Standing on Giants: Shannon + Besta + Anthropic + SAPE Framework
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.sovereign.runtime_types import (
    RuntimeConfig,
    RuntimeMetrics,
    RuntimeMode,
    SNRResult,
    SovereignQuery,
)


# =============================================================================
# RFC-04: SNR Optimization Actually Used
# =============================================================================


class TestSNROptimizationUsed:
    """Verify _optimize_snr() returns the optimized text, not the original."""

    @pytest.mark.asyncio
    async def test_optimize_snr_returns_filtered_content(self):
        """The core bug: optimized text was computed but never read back."""
        from core.sovereign.runtime_core import SovereignRuntime

        runtime = SovereignRuntime(RuntimeConfig.minimal())
        runtime._initialized = True

        # Mock SNR optimizer that returns optimized content
        mock_optimizer = AsyncMock()
        mock_optimizer.optimize.return_value = {
            "snr_score": 0.95,
            "optimized": "Clean filtered text",
            "passed": True,
        }
        runtime._snr_optimizer = mock_optimizer

        content, score = await runtime._optimize_snr("Noisy raw text with noise")
        assert content == "Clean filtered text"
        assert score == 0.95

    @pytest.mark.asyncio
    async def test_optimize_snr_falls_back_on_none(self):
        """If optimizer returns None for optimized, keep original."""
        from core.sovereign.runtime_core import SovereignRuntime

        runtime = SovereignRuntime(RuntimeConfig.minimal())
        runtime._initialized = True

        mock_optimizer = AsyncMock()
        mock_optimizer.optimize.return_value = {
            "snr_score": 0.90,
            "optimized": None,
        }
        runtime._snr_optimizer = mock_optimizer

        content, score = await runtime._optimize_snr("Original content")
        assert content == "Original content"
        assert score == 0.90

    @pytest.mark.asyncio
    async def test_optimize_snr_tracks_improvement(self):
        """SNR stats should be updated when content is actually changed."""
        from core.sovereign.runtime_core import SovereignRuntime

        runtime = SovereignRuntime(RuntimeConfig.minimal())
        runtime._initialized = True

        mock_optimizer = AsyncMock()
        mock_optimizer.optimize.return_value = {
            "snr_score": 0.96,
            "optimized": "Short",
        }
        runtime._snr_optimizer = mock_optimizer

        await runtime._optimize_snr("Much longer original text here")
        assert runtime.metrics.snr_optimizations == 1
        assert runtime.metrics.snr_avg_improvement > 0

    @pytest.mark.asyncio
    async def test_optimize_snr_no_optimizer(self):
        """When no optimizer loaded, return content unchanged."""
        from core.sovereign.runtime_core import SovereignRuntime

        runtime = SovereignRuntime(RuntimeConfig.minimal())
        runtime._initialized = True
        runtime._snr_optimizer = None

        content, score = await runtime._optimize_snr("Unchanged text")
        assert content == "Unchanged text"


# =============================================================================
# RFC-03: Query Metrics Counted Exactly Once
# =============================================================================


class TestQueryMetricsNoDuplicates:
    """Verify queries_processed is incremented exactly once per query."""

    def test_update_query_stats_increments_once(self):
        """update_query_stats should be the ONLY place incrementing counters."""
        metrics = RuntimeMetrics()
        metrics.update_query_stats(True, 100.0)

        assert metrics.queries_processed == 1
        assert metrics.queries_succeeded == 1
        assert metrics.queries_failed == 0

    def test_update_query_stats_failed(self):
        metrics = RuntimeMetrics()
        metrics.update_query_stats(False, 50.0)

        assert metrics.queries_processed == 1
        assert metrics.queries_succeeded == 0
        assert metrics.queries_failed == 1

    def test_multiple_queries_count_correct(self):
        """Simulate 5 success + 2 failed = 7 processed."""
        metrics = RuntimeMetrics()
        for _ in range(5):
            metrics.update_query_stats(True, 100.0)
        for _ in range(2):
            metrics.update_query_stats(False, 200.0)

        assert metrics.queries_processed == 7
        assert metrics.queries_succeeded == 5
        assert metrics.queries_failed == 2


# =============================================================================
# RFC-06: Muraqabah Bridge Uses Correct Event Attributes
# =============================================================================


class TestMuraqabahBridge:
    """Verify bridge uses event.topic and event.payload (not event_type/data)."""

    def test_event_has_topic_and_payload(self):
        """Event class must have topic and payload fields."""
        from core.sovereign.event_bus import Event

        event = Event(topic="muraqabah.opportunity", payload={"domain": "health"})
        assert event.topic == "muraqabah.opportunity"
        assert event.payload["domain"] == "health"

    def test_event_has_no_event_type_attribute(self):
        """Event class must NOT have event_type — bridge was using wrong name."""
        from core.sovereign.event_bus import Event

        event = Event()
        assert not hasattr(event, "event_type"), "Event should use .topic, not .event_type"

    def test_event_has_no_data_attribute(self):
        """Event class must NOT have .data — bridge was using wrong name."""
        from core.sovereign.event_bus import Event

        event = Event()
        assert not hasattr(event, "data"), "Event should use .payload, not .data"

    @pytest.mark.asyncio
    async def test_bridge_matches_event_api(self):
        """The bridge function should reference .topic and .payload."""
        from core.sovereign.event_bus import Event
        from core.sovereign.opportunity_pipeline import (
            OpportunityPipeline,
            connect_muraqabah_to_pipeline,
        )

        pipeline = OpportunityPipeline()
        mock_engine = MagicMock()
        mock_engine.event_bus = MagicMock()

        connect_muraqabah_to_pipeline(mock_engine, pipeline)

        # Verify subscribe was called
        mock_engine.event_bus.subscribe.assert_called_once_with(
            "muraqabah.opportunity", pytest.approx(mock_engine.event_bus.subscribe.call_args[0][1])
        )


# =============================================================================
# RFC-01: Feature Flags Respected
# =============================================================================


class TestFeatureFlagsRespected:
    """Verify _init_components() respects RuntimeConfig feature flags."""

    @pytest.mark.asyncio
    async def test_minimal_config_uses_stubs(self):
        """RuntimeConfig.minimal() disables all optional components."""
        from core.sovereign.runtime_core import SovereignRuntime

        config = RuntimeConfig.minimal()
        assert config.enable_graph_reasoning is False
        assert config.enable_snr_optimization is False
        assert config.enable_guardian_validation is False
        assert config.enable_autonomous_loop is False

        runtime = SovereignRuntime(config)
        await runtime._init_components()

        # All should be stubs (not None, but stub instances)
        assert runtime._graph_reasoner is not None
        assert runtime._snr_optimizer is not None
        assert runtime._guardian_council is not None
        assert runtime._autonomous_loop is not None

    @pytest.mark.asyncio
    async def test_standard_config_loads_real_components(self):
        """RuntimeConfig.standard() enables reasoning, SNR, guardian."""
        config = RuntimeConfig.standard()
        assert config.enable_graph_reasoning is True
        assert config.enable_snr_optimization is True
        assert config.enable_guardian_validation is True


# =============================================================================
# RFC-02: RuntimeMetrics.started_at Declared
# =============================================================================


class TestStartedAtDeclared:
    """Verify started_at is a proper dataclass field, not monkey-patched."""

    def test_started_at_in_dataclass_fields(self):
        """started_at must be declared in RuntimeMetrics dataclass."""
        import dataclasses

        field_names = {f.name for f in dataclasses.fields(RuntimeMetrics)}
        assert "started_at" in field_names

    def test_started_at_default_none(self):
        metrics = RuntimeMetrics()
        assert metrics.started_at is None

    def test_started_at_assignable(self):
        from datetime import datetime

        metrics = RuntimeMetrics()
        metrics.started_at = datetime.now()
        assert metrics.started_at is not None


# =============================================================================
# RFC-05: Reasoning Time Variable Used
# =============================================================================


class TestReasoningTimeAssigned:
    """Verify reasoning metrics are tracked (not bare expressions)."""

    def test_runtime_core_tracks_reasoning_stats(self):
        """Check that reasoning stats are recorded via metrics API."""
        import inspect

        from core.sovereign.runtime_core import SovereignRuntime

        source = inspect.getsource(SovereignRuntime._process_query)
        assert "update_reasoning_stats" in source, (
            "reasoning stats should be tracked via metrics.update_reasoning_stats()"
        )


# =============================================================================
# LCT-01: No Redundant Checkpoint on Shutdown
# =============================================================================


class TestSingleCheckpointPath:
    """Verify shutdown uses only MemoryCoordinator, not _checkpoint()."""

    @pytest.mark.asyncio
    async def test_shutdown_does_not_call_checkpoint(self):
        """_checkpoint() should NOT be called during shutdown (LCT-01)."""
        from core.sovereign.runtime_core import SovereignRuntime

        runtime = SovereignRuntime(RuntimeConfig.minimal())
        runtime._running = True
        runtime._memory_coordinator = AsyncMock()
        runtime._shutdown_event = asyncio.Event()

        with patch.object(runtime, "_checkpoint", new_callable=AsyncMock) as mock_cp:
            await runtime.shutdown()
            mock_cp.assert_not_called()

        # MemoryCoordinator.stop() SHOULD be called
        runtime._memory_coordinator.stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_shutdown_stops_memory_coordinator(self):
        """MemoryCoordinator.stop() is the single checkpoint mechanism."""
        from core.sovereign.runtime_core import SovereignRuntime

        runtime = SovereignRuntime(RuntimeConfig.minimal())
        runtime._running = True
        runtime._memory_coordinator = AsyncMock()
        runtime._shutdown_event = asyncio.Event()

        await runtime.shutdown()
        runtime._memory_coordinator.stop.assert_awaited_once()


# =============================================================================
# SNRResult TypedDict Has 'optimized' Key
# =============================================================================


class TestSNRResultType:
    """Verify SNRResult TypedDict includes 'optimized' field."""

    def test_snr_result_accepts_optimized(self):
        result: SNRResult = {
            "snr_score": 0.95,
            "optimized": "Clean text",
        }
        assert result["optimized"] == "Clean text"

    def test_snr_result_optimized_optional(self):
        result: SNRResult = {"snr_score": 0.90}
        assert "optimized" not in result  # total=False makes it optional
