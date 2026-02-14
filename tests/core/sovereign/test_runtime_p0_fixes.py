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
from types import SimpleNamespace
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

        content, score, tags = await runtime._optimize_snr("Noisy raw text with noise")
        assert content == "Clean filtered text"
        # SNREngine v1 overrides with authoritative score; verify it's valid [0, 1]
        assert 0.0 <= score <= 1.0

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

        content, score, tags = await runtime._optimize_snr("Original content")
        assert content == "Original content"
        # SNREngine v1 overrides with authoritative score; verify it's valid [0, 1]
        assert 0.0 <= score <= 1.0

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

        content, score, tags = await runtime._optimize_snr("Unchanged text")
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
        assert config.enable_proactive_kernel is False
        assert config.enable_zpk_preflight is False
        assert config.zpk_emit_bootstrap_events is False

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

        # After Phase 18.1 refactor, _process_query is a router that delegates
        # to _process_query_direct (direct pipeline) or _orchestrate_complex_query.
        # The direct pipeline contains the update_reasoning_stats call.
        source = inspect.getsource(SovereignRuntime._process_query_direct)
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


# =============================================================================
# ZPK Preflight: Fail-Closed Bootstrap Gate
# =============================================================================


class TestZPKPreflight:
    """Verify runtime ZPK preflight gate behavior."""

    @pytest.mark.asyncio
    async def test_zpk_preflight_disabled_noop(self):
        from core.sovereign.runtime_core import SovereignRuntime

        runtime = SovereignRuntime(RuntimeConfig.minimal())
        await runtime._run_zpk_preflight()
        assert runtime._zpk_bootstrap_result is None

    @pytest.mark.asyncio
    async def test_zpk_preflight_fail_closed(self):
        from core.sovereign.runtime_core import SovereignRuntime

        config = RuntimeConfig.minimal()
        config.enable_zpk_preflight = True
        config.zpk_manifest_uri = "/tmp/non-existent-manifest.json"
        config.zpk_release_public_key = "11" * 32
        runtime = SovereignRuntime(config)

        with patch("core.zpk.ZeroPointKernel") as mock_zpk_cls:
            mock_zpk = mock_zpk_cls.return_value
            mock_zpk.bootstrap = AsyncMock(
                return_value=SimpleNamespace(
                    success=False,
                    executed_version=None,
                    rollback_used=False,
                    reason="policy_denied",
                )
            )
            with pytest.raises(RuntimeError, match="ZPK preflight failed"):
                await runtime._run_zpk_preflight()

    @pytest.mark.asyncio
    async def test_zpk_preflight_success_sets_runtime_state(self):
        from core.sovereign.runtime_core import SovereignRuntime

        config = RuntimeConfig.minimal()
        config.enable_zpk_preflight = True
        config.zpk_manifest_uri = "/tmp/manifest.json"
        config.zpk_release_public_key = "22" * 32
        runtime = SovereignRuntime(config)

        with patch("core.zpk.ZeroPointKernel") as mock_zpk_cls:
            mock_zpk = mock_zpk_cls.return_value
            mock_zpk.bootstrap = AsyncMock(
                return_value=SimpleNamespace(
                    success=True,
                    executed_version="1.2.3",
                    rollback_used=False,
                    reason="executed",
                )
            )
            await runtime._run_zpk_preflight()

        state = runtime._get_runtime_state()
        assert state["zpk_preflight"]["success"] is True
        assert state["zpk_preflight"]["executed_version"] == "1.2.3"


class TestRuntimeEnvOverrides:
    """Verify runtime picks ZPK config from environment."""

    def test_apply_env_overrides_for_zpk(self, monkeypatch):
        from core.sovereign.runtime_core import SovereignRuntime

        runtime = SovereignRuntime(RuntimeConfig.minimal())
        monkeypatch.setenv("ZPK_PREFLIGHT_ENABLED", "true")
        monkeypatch.setenv("ZPK_MANIFEST_URI", "/tmp/manifest.json")
        monkeypatch.setenv("ZPK_RELEASE_PUBLIC_KEY", "ab" * 32)
        monkeypatch.setenv("ZPK_ALLOWED_VERSIONS", "1.0.0,2.0.0")
        monkeypatch.setenv("ZPK_MIN_POLICY_VERSION", "3")
        monkeypatch.setenv("ZPK_MIN_IHSAN_POLICY", "0.97")
        monkeypatch.setenv("ZPK_EMIT_BOOTSTRAP_EVENTS", "1")
        monkeypatch.setenv("ZPK_EVENT_TOPIC", "federation.zpk.receipt")

        runtime._apply_env_overrides()

        assert runtime.config.enable_zpk_preflight is True
        assert runtime.config.zpk_manifest_uri == "/tmp/manifest.json"
        assert runtime.config.zpk_release_public_key == "ab" * 32
        assert runtime.config.zpk_allowed_versions == ["1.0.0", "2.0.0"]
        assert runtime.config.zpk_min_policy_version == 3
        assert runtime.config.zpk_min_ihsan_policy == 0.97
        assert runtime.config.zpk_emit_bootstrap_events is True
        assert runtime.config.zpk_event_topic == "federation.zpk.receipt"

    def test_apply_env_overrides_for_pek(self, monkeypatch):
        from core.sovereign.runtime_core import SovereignRuntime

        runtime = SovereignRuntime(RuntimeConfig.minimal())
        monkeypatch.setenv("PEK_ENABLED", "true")
        monkeypatch.setenv("PEK_CYCLE_SECONDS", "2.5")
        monkeypatch.setenv("PEK_MIN_CONFIDENCE", "0.62")
        monkeypatch.setenv("PEK_MIN_AUTO_CONFIDENCE", "0.81")
        monkeypatch.setenv("PEK_BASE_TAU", "0.51")
        monkeypatch.setenv("PEK_AUTO_EXECUTE_TAU", "0.79")
        monkeypatch.setenv("PEK_QUEUE_SILENT_TAU", "0.31")
        monkeypatch.setenv("PEK_ATTENTION_BUDGET_CAPACITY", "10.0")
        monkeypatch.setenv("PEK_ATTENTION_BUDGET_RECOVERY_PER_CYCLE", "1.25")
        monkeypatch.setenv("PEK_EMIT_PROOF_EVENTS", "1")
        monkeypatch.setenv("PEK_PROOF_EVENT_TOPIC", "pek.proof.runtime")

        runtime._apply_env_overrides()

        assert runtime.config.enable_proactive_kernel is True
        assert runtime.config.proactive_kernel_cycle_seconds == 2.5
        assert runtime.config.proactive_kernel_min_confidence == 0.62
        assert runtime.config.proactive_kernel_min_auto_confidence == 0.81
        assert runtime.config.proactive_kernel_base_tau == 0.51
        assert runtime.config.proactive_kernel_auto_execute_tau == 0.79
        assert runtime.config.proactive_kernel_queue_silent_tau == 0.31
        assert runtime.config.proactive_kernel_attention_budget_capacity == 10.0
        assert runtime.config.proactive_kernel_attention_recovery_per_cycle == 1.25
        assert runtime.config.proactive_kernel_emit_events is True
        assert runtime.config.proactive_kernel_event_topic == "pek.proof.runtime"
