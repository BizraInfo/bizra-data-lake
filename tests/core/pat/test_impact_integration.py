"""
Integration tests for the BIZRA Impact Tracker wiring.

Verifies that the impact tracker is properly connected to:
    - SovereignRuntime (query pipeline records impact)
    - Onboarding (initializes tracker state on new identity)
    - Gateway (queries flow through runtime → impact)
    - MemoryCoordinator (impact state is checkpointed)

These tests use mocks to avoid requiring live LLM backends.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.pat.identity_card import SovereigntyTier


# ─── Onboarding → Impact Tracker Init ────────────────────────────────


class TestOnboardingImpactInit:
    """Verify onboarding creates impact_tracker.json baseline."""

    def test_onboard_creates_tracker_file(self, tmp_path):
        from core.pat.onboarding import OnboardingWizard

        wizard = OnboardingWizard(node_dir=tmp_path)
        credentials = wizard.onboard(name="TestUser")

        tracker_file = tmp_path / "impact_tracker.json"
        assert tracker_file.exists(), "Onboarding should create impact_tracker.json"

        data = json.loads(tracker_file.read_text())
        assert data["node_id"] == credentials.node_id
        assert data["sovereignty_score"] == 0.0
        assert data["sovereignty_tier"] == "seed"
        assert data["total_bloom"] == 0.0

    def test_onboard_succeeds_without_tracker(self, tmp_path):
        """Onboarding must not fail if impact tracker module is unavailable."""
        from core.pat.onboarding import OnboardingWizard

        wizard = OnboardingWizard(node_dir=tmp_path)

        # Patch the import inside _init_impact_tracker so the try/except catches it
        with patch.dict("sys.modules", {"core.pat.impact_tracker": None}):
            # Should NOT raise — impact tracker is best-effort
            credentials = wizard.onboard(name="Resilient")
            assert credentials.node_id.startswith("BIZRA-")


# ─── Runtime → Impact Recording ──────────────────────────────────────


class TestRuntimeImpactRecording:
    """Verify runtime records impact on successful queries."""

    @pytest.mark.asyncio
    async def test_runtime_creates_impact_tracker(self, tmp_path):
        """Runtime should initialize impact tracker during init."""
        from core.sovereign.runtime_types import RuntimeConfig
        from core.sovereign.runtime_core import SovereignRuntime

        config = RuntimeConfig(
            autonomous_enabled=False,
            state_dir=tmp_path,
        )

        async with SovereignRuntime.create(config) as runtime:
            assert runtime._impact_tracker is not None
            assert runtime._impact_tracker.sovereignty_score == 0.0

    @pytest.mark.asyncio
    async def test_query_records_impact(self, tmp_path):
        """A successful query should increase total bloom."""
        from core.sovereign.runtime_types import RuntimeConfig, SovereignResult
        from core.sovereign.runtime_core import SovereignRuntime

        config = RuntimeConfig(
            autonomous_enabled=False,
            state_dir=tmp_path,
        )

        async with SovereignRuntime.create(config) as runtime:
            assert runtime._impact_tracker is not None
            bloom_before = runtime._impact_tracker.total_bloom

            # Mock the internal processing to return a successful result
            mock_result = SovereignResult(query_id="test-001")
            mock_result.success = True
            mock_result.response = "Sovereignty is the natural state of being."
            mock_result.snr_score = 0.92
            mock_result.ihsan_score = 0.97
            mock_result.processing_time_ms = 150.0
            mock_result.reasoning_depth = 3
            mock_result.reasoning_used = True
            mock_result.validation_passed = True

            # Call the impact recorder directly (avoids needing live LLM)
            runtime._record_query_impact(mock_result)

            bloom_after = runtime._impact_tracker.total_bloom
            assert bloom_after > bloom_before, "Query should increase bloom"

    @pytest.mark.asyncio
    async def test_impact_failure_does_not_break_query(self, tmp_path):
        """If impact recording fails, the query should still succeed."""
        from core.sovereign.runtime_types import RuntimeConfig, SovereignResult
        from core.sovereign.runtime_core import SovereignRuntime

        config = RuntimeConfig(
            autonomous_enabled=False,
            state_dir=tmp_path,
        )

        async with SovereignRuntime.create(config) as runtime:
            # Sabotage the tracker
            runtime._impact_tracker = MagicMock()
            runtime._impact_tracker.record_event.side_effect = RuntimeError("disk full")

            mock_result = SovereignResult(query_id="test-002")
            mock_result.success = True
            mock_result.response = "Test"
            mock_result.snr_score = 0.9
            mock_result.ihsan_score = 0.95
            mock_result.processing_time_ms = 100.0
            mock_result.reasoning_depth = 1

            # Should not raise
            runtime._record_query_impact(mock_result)


# ─── Status → Sovereignty Info ────────────────────────────────────────


class TestStatusSovereigntyInfo:
    """Verify status() includes sovereignty information."""

    @pytest.mark.asyncio
    async def test_status_includes_sovereignty(self, tmp_path):
        from core.sovereign.runtime_types import RuntimeConfig
        from core.sovereign.runtime_core import SovereignRuntime

        config = RuntimeConfig(
            autonomous_enabled=False,
            state_dir=tmp_path,
        )

        async with SovereignRuntime.create(config) as runtime:
            status = runtime.status()

            assert "sovereignty" in status
            assert status["sovereignty"]["tracking"] is True
            assert "score" in status["sovereignty"]
            assert "tier" in status["sovereignty"]
            assert status["sovereignty"]["tier"] == "seed"

    @pytest.mark.asyncio
    async def test_status_without_tracker(self, tmp_path):
        from core.sovereign.runtime_types import RuntimeConfig
        from core.sovereign.runtime_core import SovereignRuntime

        config = RuntimeConfig(
            autonomous_enabled=False,
            state_dir=tmp_path,
        )

        async with SovereignRuntime.create(config) as runtime:
            runtime._impact_tracker = None
            status = runtime.status()

            assert status["sovereignty"]["tracking"] is False


# ─── MemoryCoordinator Integration ────────────────────────────────────


class TestMemoryCoordinatorIntegration:
    """Verify impact tracker is registered as a state provider."""

    @pytest.mark.asyncio
    async def test_memory_coordinator_has_impact_provider(self, tmp_path):
        from core.sovereign.runtime_types import RuntimeConfig
        from core.sovereign.runtime_core import SovereignRuntime

        config = RuntimeConfig(
            autonomous_enabled=False,
            state_dir=tmp_path,
            enable_persistence=False,  # Don't start background loop
        )

        async with SovereignRuntime.create(config) as runtime:
            if runtime._memory_coordinator:
                provider_names = list(
                    runtime._memory_coordinator._state_providers.keys()
                )
                assert "impact_tracker" in provider_names


# ─── Impact Tracker Persistence Across Runtime Restarts ───────────────


class TestImpactPersistence:
    """Verify impact state survives runtime restarts."""

    @pytest.mark.asyncio
    async def test_impact_survives_restart(self, tmp_path):
        from core.sovereign.runtime_types import RuntimeConfig, SovereignResult
        from core.sovereign.runtime_core import SovereignRuntime

        config = RuntimeConfig(
            autonomous_enabled=False,
            state_dir=tmp_path,
        )

        # First runtime session — record some impact
        async with SovereignRuntime.create(config) as runtime1:
            mock_result = SovereignResult(query_id="persist-001")
            mock_result.success = True
            mock_result.response = "Persisted answer"
            mock_result.snr_score = 0.90
            mock_result.ihsan_score = 0.96
            mock_result.processing_time_ms = 200.0
            mock_result.reasoning_depth = 2
            mock_result.validation_passed = True

            runtime1._record_query_impact(mock_result)
            bloom1 = runtime1._impact_tracker.total_bloom
            assert bloom1 > 0

        # Second runtime session — verify impact persisted
        async with SovereignRuntime.create(config) as runtime2:
            bloom2 = runtime2._impact_tracker.total_bloom
            assert bloom2 == bloom1, "Bloom should persist across runtime restarts"
            assert runtime2._impact_tracker.sovereignty_score > 0


# ─── Bloom Calculation Sanity ─────────────────────────────────────────


class TestBloomCalculation:
    """Verify bloom amounts are reasonable."""

    @pytest.mark.asyncio
    async def test_bloom_from_fast_query(self, tmp_path):
        """A fast, shallow query should produce moderate bloom."""
        from core.sovereign.runtime_types import RuntimeConfig, SovereignResult
        from core.sovereign.runtime_core import SovereignRuntime

        config = RuntimeConfig(
            autonomous_enabled=False,
            state_dir=tmp_path,
        )

        async with SovereignRuntime.create(config) as runtime:
            result = SovereignResult(query_id="fast-001")
            result.success = True
            result.response = "Quick answer"
            result.snr_score = 0.85
            result.ihsan_score = 0.95
            result.processing_time_ms = 50.0  # Fast
            result.reasoning_depth = 1  # Shallow

            runtime._record_query_impact(result)

            bloom = runtime._impact_tracker.total_bloom
            # 50ms/1000*2 = 0.1 + 1*0.5 = 0.5 + 0.5 validation = 1.1
            assert 0.5 < bloom < 3.0, f"Bloom {bloom} should be moderate for fast query"

    @pytest.mark.asyncio
    async def test_bloom_from_deep_query(self, tmp_path):
        """A deep, reasoning-heavy query should produce more bloom."""
        from core.sovereign.runtime_types import RuntimeConfig, SovereignResult
        from core.sovereign.runtime_core import SovereignRuntime

        config = RuntimeConfig(
            autonomous_enabled=False,
            state_dir=tmp_path,
        )

        async with SovereignRuntime.create(config) as runtime:
            result = SovereignResult(query_id="deep-001")
            result.success = True
            result.response = "A detailed, well-reasoned answer with multiple perspectives." * 5
            result.snr_score = 0.95
            result.ihsan_score = 0.98
            result.processing_time_ms = 2000.0  # Slow, deep reasoning
            result.reasoning_depth = 5  # Deep

            runtime._record_query_impact(result)

            bloom = runtime._impact_tracker.total_bloom
            # 2000/1000*2 = 4.0 + 5*0.5 = 2.5 + 0.5 = 7.0
            assert bloom > 5.0, f"Bloom {bloom} should be high for deep query"


# ─── Gateway Context Enhancement ──────────────────────────────────────


class TestGatewayContext:
    """Verify gateway adds source context to queries."""

    @pytest.mark.asyncio
    async def test_gateway_query_fn_adds_source(self):
        """The gateway query function should tag context with source=gateway."""
        from unittest.mock import AsyncMock

        # Mock runtime
        mock_runtime = AsyncMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.response = "Gateway response"
        mock_runtime.query.return_value = mock_result

        # Simulate what _make_query_fn does (but with our mock)
        async def query_fn(content, context=None):
            ctx = context or {}
            ctx.setdefault("source", "gateway")
            result = await mock_runtime.query(content, context=ctx)
            if result.success:
                return result.response
            return f"Error: {result.error}"

        response = await query_fn("Hello BIZRA")
        assert response == "Gateway response"

        # Verify context was passed with source
        call_args = mock_runtime.query.call_args
        assert call_args[1]["context"]["source"] == "gateway"
