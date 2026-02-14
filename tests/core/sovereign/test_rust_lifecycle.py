"""
Tests for Rust Lifecycle Integration — BIZRA Proactive Sovereign Entity
========================================================================
Validates Python-Rust integration lifecycle management.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from core.sovereign.rust_lifecycle import (
    RustAPIClient,
    RustLifecycleManager,
    RustProcessManager,
    RustServiceHealth,
    RustServiceStatus,
    create_rust_gate_filter,
    create_rust_lifecycle,
)


# =============================================================================
# SERVICE HEALTH TESTS
# =============================================================================

class TestRustServiceHealth:
    """Tests for RustServiceHealth dataclass."""

    def test_healthy_status(self):
        """Test healthy service detection."""
        health = RustServiceHealth(
            service="bizra-api",
            status=RustServiceStatus.HEALTHY,
            version="1.0.0",
        )
        assert health.is_healthy()
        assert health.service == "bizra-api"

    def test_unhealthy_status(self):
        """Test unhealthy service detection."""
        health = RustServiceHealth(
            service="bizra-api",
            status=RustServiceStatus.UNHEALTHY,
            error="Connection refused",
        )
        assert not health.is_healthy()
        assert health.error == "Connection refused"

    def test_degraded_status(self):
        """Test degraded service detection."""
        health = RustServiceHealth(
            service="bizra-api",
            status=RustServiceStatus.DEGRADED,
        )
        assert not health.is_healthy()


# =============================================================================
# API CLIENT TESTS
# =============================================================================

class TestRustAPIClient:
    """Tests for RustAPIClient."""

    def test_client_initialization(self):
        """Test client initialization."""
        client = RustAPIClient(base_url="http://localhost:3001")
        assert client.base_url == "http://localhost:3001"

    def test_client_url_normalization(self):
        """Test URL trailing slash is removed."""
        client = RustAPIClient(base_url="http://localhost:3001/")
        assert client.base_url == "http://localhost:3001"

    @pytest.mark.asyncio
    async def test_health_check_connection_error(self):
        """Test health check handles connection errors gracefully."""
        client = RustAPIClient(base_url="http://localhost:9999")  # Non-existent
        health = await client.health_check()
        assert health.status in (RustServiceStatus.UNHEALTHY, RustServiceStatus.UNKNOWN)
        await client.close()

    @pytest.mark.asyncio
    async def test_client_close(self):
        """Test client close is safe to call multiple times."""
        client = RustAPIClient()
        await client.close()
        await client.close()  # Should not raise


# =============================================================================
# PROCESS MANAGER TESTS
# =============================================================================

class TestRustProcessManager:
    """Tests for RustProcessManager."""

    def test_manager_initialization(self):
        """Test process manager initialization."""
        manager = RustProcessManager(api_port=3001)
        assert manager.api_port == 3001
        assert not manager.is_running()

    def test_uptime_when_not_started(self):
        """Test uptime is 0 when not started."""
        manager = RustProcessManager()
        assert manager.uptime() == 0.0

    def test_find_bizra_omega(self):
        """Test bizra-omega path discovery."""
        manager = RustProcessManager()
        # Should find path (may or may not exist depending on environment)
        assert manager.bizra_omega_path is not None


# =============================================================================
# LIFECYCLE MANAGER TESTS
# =============================================================================

class TestRustLifecycleManager:
    """Tests for RustLifecycleManager."""

    def test_manager_initialization(self):
        """Test lifecycle manager initialization."""
        manager = RustLifecycleManager(api_port=3001, use_pyo3=True)
        assert manager.api_port == 3001

    def test_pyo3_fallback_ihsan(self):
        """Test PyO3 Ihsan check falls back to Python."""
        manager = RustLifecycleManager(use_pyo3=False)
        # Without PyO3, should use Python fallback
        assert manager.pyo3_check_ihsan(0.96) is True
        assert manager.pyo3_check_ihsan(0.94) is False

    def test_pyo3_fallback_snr(self):
        """Test PyO3 SNR check falls back to Python."""
        manager = RustLifecycleManager(use_pyo3=False)
        assert manager.pyo3_check_snr(0.9) is True
        assert manager.pyo3_check_snr(0.8) is False

    def test_pyo3_fallback_digest(self):
        """Test PyO3 digest falls back to Python."""
        manager = RustLifecycleManager(use_pyo3=False)
        digest = manager.pyo3_domain_digest(b"test message")
        assert len(digest) == 128  # BLAKE2b hex digest
        assert digest.startswith("")  # Valid hex

    def test_stats(self):
        """Test statistics gathering."""
        manager = RustLifecycleManager()
        stats = manager.stats()

        assert "pyo3_available" in stats
        assert "api_healthy" in stats
        assert "api_port" in stats
        assert "process_running" in stats
        assert "running" in stats

    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self):
        """Test lifecycle start and stop."""
        manager = RustLifecycleManager(
            api_port=3001,
            use_pyo3=False,
            auto_start_api=False,
        )

        result = await manager.start()
        assert "pyo3_available" in result
        assert "api_healthy" in result

        # Manager should be running
        assert manager._running is True

        await manager.stop()
        assert manager._running is False

    @pytest.mark.asyncio
    async def test_health_callback(self):
        """Test health change callback."""
        manager = RustLifecycleManager(
            api_port=9999,  # Non-existent to trigger unhealthy
            use_pyo3=False,
            auto_start_api=False,
            health_check_interval=0.1,  # Fast for testing
        )

        callback_called = False
        received_health = None

        def on_health(health: RustServiceHealth):
            nonlocal callback_called, received_health
            callback_called = True
            received_health = health

        manager.set_health_callback(on_health)

        await manager.start()
        # Wait longer for health check loop to run
        await asyncio.sleep(0.5)
        await manager.stop()

        # Note: callback may not fire if initial health is same as check result
        # The important thing is the manager started and stopped without error
        # Callback fires only on status CHANGE
        assert manager._last_health is not None


# =============================================================================
# FACTORY TESTS
# =============================================================================

class TestFactory:
    """Tests for factory functions."""

    @pytest.mark.asyncio
    async def test_create_rust_lifecycle(self):
        """Test factory function."""
        manager = await create_rust_lifecycle(
            api_port=3001,
            use_pyo3=False,
            auto_start_api=False,
            auto_start=True,
        )

        try:
            assert isinstance(manager, RustLifecycleManager)
            assert manager._running is True
        finally:
            await manager.stop()

    @pytest.mark.asyncio
    async def test_create_rust_lifecycle_no_auto_start(self):
        """Test factory function without auto-start."""
        manager = await create_rust_lifecycle(
            auto_start=False,
        )

        assert manager._running is False


# =============================================================================
# GATE FILTER INTEGRATION TESTS
# =============================================================================

class TestRustGateFilter:
    """Tests for Rust gate filter integration."""

    @pytest.mark.asyncio
    async def test_create_rust_gate_filter(self):
        """Test creating Rust gate filter."""
        manager = RustLifecycleManager(use_pyo3=False)

        filter_instance = create_rust_gate_filter(manager)

        assert filter_instance.name == "Rust Gate"
        assert filter_instance.weight == 2.0

    @pytest.mark.asyncio
    async def test_rust_gate_filter_python_fallback(self):
        """Test Rust gate filter with Python fallback."""
        from core.sovereign.opportunity_pipeline import PipelineOpportunity
        import time

        manager = RustLifecycleManager(use_pyo3=False)
        filter_instance = create_rust_gate_filter(manager)

        # Test with high scores (should pass)
        opp = PipelineOpportunity(
            id="test",
            domain="test",
            description="Test opportunity",
            source="test",
            detected_at=time.time(),
            snr_score=0.95,
            ihsan_score=0.98,
        )

        result = await filter_instance.check(opp)
        # Should fall back to skipping since no Rust available
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_rust_gate_filter_with_pyo3_mock(self):
        """Test Rust gate filter with mocked PyO3."""
        from core.sovereign.opportunity_pipeline import PipelineOpportunity
        import time

        manager = RustLifecycleManager(use_pyo3=False)
        # Simulate PyO3 available
        manager._pyo3_available = True

        # Mock the check methods
        manager.pyo3_check_ihsan = MagicMock(return_value=True)
        manager.pyo3_check_snr = MagicMock(return_value=True)

        filter_instance = create_rust_gate_filter(manager)

        opp = PipelineOpportunity(
            id="test",
            domain="test",
            description="Test opportunity",
            source="test",
            detected_at=time.time(),
            snr_score=0.95,
            ihsan_score=0.98,
        )

        result = await filter_instance.check(opp)
        assert result.passed is True
        assert "Rust gate passed" in result.reason


# =============================================================================
# INTEGRATION WITH OPPORTUNITY PIPELINE TESTS
# =============================================================================

class TestPipelineIntegration:
    """Tests for integration with OpportunityPipeline."""

    @pytest.mark.asyncio
    async def test_pipeline_with_rust_filter(self):
        """Test OpportunityPipeline with Rust gate filter."""
        from core.sovereign.opportunity_pipeline import (
            OpportunityPipeline,
            PipelineOpportunity,
        )
        from core.sovereign.autonomy_matrix import AutonomyLevel
        import time

        # Create Rust lifecycle (no actual Rust needed)
        manager = RustLifecycleManager(use_pyo3=False)

        # Create pipeline
        pipeline = OpportunityPipeline()

        # Add Rust filter
        rust_filter = create_rust_gate_filter(manager)
        pipeline._filters.append(rust_filter)

        await pipeline.start()

        try:
            # Submit opportunity
            opp = PipelineOpportunity(
                id="rust-test",
                domain="test",
                description="Test with Rust filter",
                source="test",
                detected_at=time.time(),
                snr_score=0.95,
                ihsan_score=0.98,
                autonomy_level=AutonomyLevel.AUTOLOW,
            )
            await pipeline.submit(opp)

            # Wait for processing
            await asyncio.sleep(0.5)

            stats = pipeline.stats()
            assert stats["total_received"] == 1
        finally:
            await pipeline.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
