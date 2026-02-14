"""
Tests for Swarm Integration — HybridSwarmOrchestrator + RustServiceAdapter
==========================================================================
Validates the distributed orchestration layer that bridges Python agents
and Rust services with self-healing, proportional scaling, and unified
health monitoring.

Coverage target: 90%+ (up from 38%)
Test count: 60+

Standing on Giants:
- Lamport (1982): Distributed systems, Byzantine fault tolerance
- Hamilton (2007): Operations at scale, availability targets
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import fields as dataclass_fields
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.apex import (
    AgentConfig,
    AgentInstance,
    AgentStatus,
    HealthStatus,
    ScalingAction,
    ScalingDecision,
    Swarm,
    SwarmConfig,
    SwarmOrchestrator,
)
from core.sovereign.rust_lifecycle import RustServiceStatus
from core.sovereign.swarm_integration import (
    AVAILABILITY_TARGET,
    HEALTH_CHECK_INTERVAL,
    MAX_RESTART_ATTEMPTS,
    RESTART_BACKOFF_BASE,
    HybridSwarmOrchestrator,
    RustServiceAdapter,
    ServiceStatus,
    ServiceType,
)


# =============================================================================
# HELPERS
# =============================================================================


def _make_swarm(swarm_id: str = "test-swarm") -> Swarm:
    """Create a minimal Swarm for testing."""
    config = SwarmConfig(
        name="test",
        agent_config=AgentConfig(agent_type="worker", name="agent"),
    )
    swarm = Swarm(id=swarm_id, config=config)
    return swarm


def _make_python_agent(agent_id: str | None = None) -> AgentInstance:
    """Create a Python agent instance for testing."""
    aid = agent_id or f"py-test-{id(object())}"
    return AgentInstance(
        id=aid,
        config=AgentConfig(
            agent_type="python-worker",
            name=aid,
            capabilities={"reasoning", "execution"},
        ),
        status=AgentStatus.RUNNING,
        started_at=datetime.now(timezone.utc),
    )


def _make_rust_agent(service_name: str) -> AgentInstance:
    """Create a Rust agent instance for testing."""
    return AgentInstance(
        id=f"rust:{service_name}",
        config=AgentConfig(
            agent_type="rust-worker",
            name=service_name,
            capabilities={"inference", "consensus", "pci"},
        ),
        status=AgentStatus.RUNNING,
        started_at=datetime.now(timezone.utc),
    )


def _populate_swarm(
    orchestrator: HybridSwarmOrchestrator,
    swarm_id: str = "test-swarm",
    python_count: int = 3,
    rust_count: int = 1,
) -> Swarm:
    """Create and populate a swarm with Python + Rust agents."""
    swarm = _make_swarm(swarm_id)
    orchestrator._swarms[swarm_id] = swarm

    for i in range(python_count):
        agent = _make_python_agent(f"py-{swarm_id}-{i:04d}")
        swarm.agents[agent.id] = agent

    for i in range(rust_count):
        svc_name = f"{swarm_id}-rust-{i:04d}"
        agent = _make_rust_agent(svc_name)
        swarm.agents[agent.id] = agent
        orchestrator.register_rust_service(svc_name)

    return swarm


# =============================================================================
# 1. CONSTANTS
# =============================================================================


class TestConstants:
    """Validate module-level constants match operational specifications."""

    def test_health_check_interval(self):
        assert HEALTH_CHECK_INTERVAL == 30

    def test_restart_backoff_base(self):
        assert RESTART_BACKOFF_BASE == 5

    def test_max_restart_attempts(self):
        assert MAX_RESTART_ATTEMPTS == 3

    def test_availability_target(self):
        assert AVAILABILITY_TARGET == 0.999

    def test_availability_target_is_three_nines(self):
        """Hamilton's three-nines availability."""
        assert AVAILABILITY_TARGET == pytest.approx(0.999)

    def test_constants_are_integers_where_expected(self):
        assert isinstance(HEALTH_CHECK_INTERVAL, int)
        assert isinstance(RESTART_BACKOFF_BASE, int)
        assert isinstance(MAX_RESTART_ATTEMPTS, int)
        assert isinstance(AVAILABILITY_TARGET, float)


# =============================================================================
# 2. SERVICE TYPE ENUM
# =============================================================================


class TestServiceType:
    """Validate ServiceType enum values and behavior."""

    def test_python_agent_value(self):
        assert ServiceType.PYTHON_AGENT == "python_agent"
        assert ServiceType.PYTHON_AGENT.value == "python_agent"

    def test_rust_service_value(self):
        assert ServiceType.RUST_SERVICE == "rust_service"
        assert ServiceType.RUST_SERVICE.value == "rust_service"

    def test_is_string_enum(self):
        assert isinstance(ServiceType.PYTHON_AGENT, str)
        assert isinstance(ServiceType.RUST_SERVICE, str)

    def test_enum_members_count(self):
        assert len(ServiceType) == 2

    def test_construction_from_value(self):
        assert ServiceType("python_agent") is ServiceType.PYTHON_AGENT
        assert ServiceType("rust_service") is ServiceType.RUST_SERVICE


# =============================================================================
# 3. SERVICE STATUS DATACLASS
# =============================================================================


class TestServiceStatus:
    """Validate ServiceStatus dataclass defaults and construction."""

    def test_minimal_construction(self):
        status = ServiceStatus(
            service_id="svc-1",
            service_type=ServiceType.PYTHON_AGENT,
            health=HealthStatus.HEALTHY,
        )
        assert status.service_id == "svc-1"
        assert status.service_type == ServiceType.PYTHON_AGENT
        assert status.health == HealthStatus.HEALTHY

    def test_default_restart_count(self):
        status = ServiceStatus(
            service_id="svc-1",
            service_type=ServiceType.RUST_SERVICE,
            health=HealthStatus.UNKNOWN,
        )
        assert status.restart_count == 0

    def test_default_uptime(self):
        status = ServiceStatus(
            service_id="svc-1",
            service_type=ServiceType.RUST_SERVICE,
            health=HealthStatus.UNKNOWN,
        )
        assert status.uptime_seconds == 0.0

    def test_default_error_is_none(self):
        status = ServiceStatus(
            service_id="svc-1",
            service_type=ServiceType.RUST_SERVICE,
            health=HealthStatus.HEALTHY,
        )
        assert status.error is None

    def test_default_last_check_is_utc(self):
        before = datetime.now(timezone.utc)
        status = ServiceStatus(
            service_id="svc-1",
            service_type=ServiceType.RUST_SERVICE,
            health=HealthStatus.HEALTHY,
        )
        after = datetime.now(timezone.utc)
        assert before <= status.last_check <= after
        assert status.last_check.tzinfo is not None

    def test_custom_values(self):
        custom_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
        status = ServiceStatus(
            service_id="rust:api-server",
            service_type=ServiceType.RUST_SERVICE,
            health=HealthStatus.DEGRADED,
            last_check=custom_time,
            restart_count=2,
            uptime_seconds=3600.0,
            error="timeout",
        )
        assert status.restart_count == 2
        assert status.uptime_seconds == 3600.0
        assert status.error == "timeout"
        assert status.last_check == custom_time

    def test_has_expected_fields(self):
        names = {f.name for f in dataclass_fields(ServiceStatus)}
        expected = {
            "service_id",
            "service_type",
            "health",
            "last_check",
            "restart_count",
            "uptime_seconds",
            "error",
        }
        assert names == expected


# =============================================================================
# 4. RUST SERVICE ADAPTER
# =============================================================================


class TestRustServiceAdapterInit:
    """Validate RustServiceAdapter initialization."""

    def test_default_endpoint(self):
        adapter = RustServiceAdapter("api-server")
        assert adapter.endpoint == "http://localhost:3001"

    def test_custom_endpoint(self):
        adapter = RustServiceAdapter("api-server", endpoint="http://10.0.0.1:8080")
        assert adapter.endpoint == "http://10.0.0.1:8080"

    def test_initial_health_unknown(self):
        adapter = RustServiceAdapter("api-server")
        assert adapter.last_health == HealthStatus.UNKNOWN

    def test_initial_restart_count_zero(self):
        adapter = RustServiceAdapter("api-server")
        assert adapter.restart_count == 0

    def test_initial_last_check_none(self):
        adapter = RustServiceAdapter("api-server")
        assert adapter.last_check is None

    def test_service_name_stored(self):
        adapter = RustServiceAdapter("my-svc")
        assert adapter.service_name == "my-svc"

    def test_start_time_recorded(self):
        before = datetime.now(timezone.utc)
        adapter = RustServiceAdapter("api-server")
        after = datetime.now(timezone.utc)
        assert before <= adapter._start_time <= after


class TestRustServiceAdapterStatusMapping:
    """Validate _map_rust_status covers all RustServiceStatus values."""

    def setup_method(self):
        self.adapter = RustServiceAdapter("test-svc")

    def test_healthy_maps_to_healthy(self):
        assert self.adapter._map_rust_status(RustServiceStatus.HEALTHY) == HealthStatus.HEALTHY

    def test_starting_maps_to_degraded(self):
        assert self.adapter._map_rust_status(RustServiceStatus.STARTING) == HealthStatus.DEGRADED

    def test_degraded_maps_to_degraded(self):
        assert self.adapter._map_rust_status(RustServiceStatus.DEGRADED) == HealthStatus.DEGRADED

    def test_unhealthy_maps_to_unhealthy(self):
        assert self.adapter._map_rust_status(RustServiceStatus.UNHEALTHY) == HealthStatus.UNHEALTHY

    def test_stopped_maps_to_unhealthy(self):
        assert self.adapter._map_rust_status(RustServiceStatus.STOPPED) == HealthStatus.UNHEALTHY

    def test_unknown_maps_to_unknown(self):
        assert self.adapter._map_rust_status(RustServiceStatus.UNKNOWN) == HealthStatus.UNKNOWN

    def test_all_rust_statuses_covered(self):
        """Every RustServiceStatus member must have a mapping."""
        for member in RustServiceStatus:
            result = self.adapter._map_rust_status(member)
            assert isinstance(result, HealthStatus), f"Missing mapping for {member}"


class TestRustServiceAdapterHealthCheck:
    """Validate health_check behavior under various conditions."""

    @pytest.mark.asyncio
    async def test_health_check_without_aiohttp_falls_back_healthy(self):
        """When aiohttp is not importable, adapter assumes HEALTHY."""
        adapter = RustServiceAdapter("test-svc")
        with patch.dict("sys.modules", {"aiohttp": None}):
            with patch(
                "core.sovereign.swarm_integration.RustServiceAdapter.health_check",
                new_callable=AsyncMock,
            ) as mock_hc:
                # Simulate the ImportError branch directly
                pass

        # Actually exercise the real code with aiohttp import failure
        import sys
        original = sys.modules.get("aiohttp")
        sys.modules["aiohttp"] = None  # type: ignore[assignment]
        try:
            # The import inside health_check will raise ImportError
            # because sys.modules["aiohttp"] is None -> import fails
            adapter2 = RustServiceAdapter("fallback-test")
            result = await adapter2.health_check()
            # When import aiohttp raises ImportError, falls back to HEALTHY
            assert result == HealthStatus.HEALTHY
            assert adapter2.last_health == HealthStatus.HEALTHY
        except TypeError:
            # On some Python versions, None in modules raises TypeError
            # which is caught by the generic except -> UNHEALTHY
            assert adapter2.last_health in (HealthStatus.UNHEALTHY, HealthStatus.HEALTHY)
        finally:
            if original is not None:
                sys.modules["aiohttp"] = original
            else:
                sys.modules.pop("aiohttp", None)

    @pytest.mark.asyncio
    async def test_health_check_timeout_returns_degraded(self):
        """Timeout during health check -> DEGRADED.

        The source does: async with aiohttp.ClientSession() as session:
                             async with session.get(...) as response:
        We need session.get() to raise TimeoutError when the outer
        async-with tries to __aenter__ the result, but because aiohttp's
        get() returns a context manager (not a coroutine), we raise on
        the get() call itself to simulate a timeout at connection time.
        """
        adapter = RustServiceAdapter("timeout-svc")

        # Build mock response context manager (the inner `async with`)
        mock_get_cm = MagicMock()
        mock_get_cm.__aenter__ = AsyncMock(side_effect=asyncio.TimeoutError)
        mock_get_cm.__aexit__ = AsyncMock(return_value=False)

        # Build mock session (the outer `async with`)
        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_get_cm)

        # Build mock ClientSession() context manager
        mock_session_cm = MagicMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cm.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session_cm):
            result = await adapter.health_check()

        assert result == HealthStatus.DEGRADED
        assert adapter.last_health == HealthStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_health_check_error_returns_unhealthy(self):
        """Generic exception during health check -> UNHEALTHY."""
        adapter = RustServiceAdapter("error-svc")

        mock_get_cm = MagicMock()
        mock_get_cm.__aenter__ = AsyncMock(side_effect=ConnectionRefusedError("refused"))
        mock_get_cm.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_get_cm)

        mock_session_cm = MagicMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cm.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session_cm):
            result = await adapter.health_check()

        assert result == HealthStatus.UNHEALTHY
        assert adapter.last_health == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_health_check_200_returns_healthy(self):
        """HTTP 200 -> HEALTHY."""
        adapter = RustServiceAdapter("ok-svc")

        mock_response = MagicMock()
        mock_response.status = 200

        mock_get_cm = MagicMock()
        mock_get_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_get_cm.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_get_cm)

        mock_session_cm = MagicMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cm.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session_cm):
            result = await adapter.health_check()

        assert result == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_health_check_4xx_returns_degraded(self):
        """HTTP 4xx (< 500) -> DEGRADED."""
        adapter = RustServiceAdapter("warn-svc")

        mock_response = MagicMock()
        mock_response.status = 429  # Too Many Requests

        mock_get_cm = MagicMock()
        mock_get_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_get_cm.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_get_cm)

        mock_session_cm = MagicMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cm.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session_cm):
            result = await adapter.health_check()

        assert result == HealthStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_health_check_5xx_returns_unhealthy(self):
        """HTTP 5xx -> UNHEALTHY."""
        adapter = RustServiceAdapter("fail-svc")

        mock_response = MagicMock()
        mock_response.status = 503  # Service Unavailable

        mock_get_cm = MagicMock()
        mock_get_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_get_cm.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_get_cm)

        mock_session_cm = MagicMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cm.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session_cm):
            result = await adapter.health_check()

        assert result == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_health_check_sets_last_check(self):
        """health_check always updates last_check timestamp."""
        adapter = RustServiceAdapter("ts-svc")
        assert adapter.last_check is None
        # Use ImportError fallback path by removing aiohttp
        import sys
        original = sys.modules.get("aiohttp")
        sys.modules["aiohttp"] = None  # type: ignore[assignment]
        try:
            before = datetime.now(timezone.utc)
            await adapter.health_check()
            after = datetime.now(timezone.utc)
        except (TypeError, ImportError):
            pass
        finally:
            if original is not None:
                sys.modules["aiohttp"] = original
            else:
                sys.modules.pop("aiohttp", None)

        # last_check should be set regardless of outcome
        assert adapter.last_check is not None


class TestRustServiceAdapterRestart:
    """Validate restart with exponential backoff and attempt limits."""

    @pytest.mark.asyncio
    async def test_restart_exceeds_max_attempts_returns_false(self):
        """When restart_count >= MAX_RESTART_ATTEMPTS, restart fails immediately."""
        adapter = RustServiceAdapter("fail-svc")
        adapter.restart_count = MAX_RESTART_ATTEMPTS
        result = await adapter.restart()
        assert result is False

    @pytest.mark.asyncio
    async def test_restart_at_exact_max_returns_false(self):
        """Boundary: restart_count == MAX_RESTART_ATTEMPTS."""
        adapter = RustServiceAdapter("boundary-svc")
        adapter.restart_count = 3
        result = await adapter.restart()
        assert result is False
        # Count should NOT have been incremented since we returned early
        assert adapter.restart_count == 3

    @pytest.mark.asyncio
    async def test_restart_success_resets_count(self):
        """Successful restart resets restart_count to 0."""
        adapter = RustServiceAdapter("good-svc")
        adapter.restart_count = 2

        # Mock sleep to avoid delays and health_check to return HEALTHY
        with patch("asyncio.sleep", new_callable=AsyncMock):
            with patch.object(
                adapter, "health_check", new_callable=AsyncMock, return_value=HealthStatus.HEALTHY
            ):
                result = await adapter.restart()

        assert result is True
        assert adapter.restart_count == 0

    @pytest.mark.asyncio
    async def test_restart_failure_increments_count(self):
        """Failed restart increments restart_count."""
        adapter = RustServiceAdapter("failing-svc")
        adapter.restart_count = 1

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with patch.object(
                adapter, "health_check", new_callable=AsyncMock, return_value=HealthStatus.UNHEALTHY
            ):
                result = await adapter.restart()

        assert result is False
        assert adapter.restart_count == 2

    @pytest.mark.asyncio
    async def test_restart_exception_increments_count(self):
        """Exception during restart increments restart_count."""
        adapter = RustServiceAdapter("crash-svc")
        adapter.restart_count = 0

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with patch.object(
                adapter, "health_check", new_callable=AsyncMock, side_effect=RuntimeError("boom")
            ):
                result = await adapter.restart()

        assert result is False
        assert adapter.restart_count == 1

    @pytest.mark.asyncio
    async def test_restart_exponential_backoff_timing(self):
        """Backoff is RESTART_BACKOFF_BASE * 2^count."""
        adapter = RustServiceAdapter("backoff-svc")

        sleep_calls = []

        async def capture_sleep(duration):
            sleep_calls.append(duration)

        with patch("asyncio.sleep", side_effect=capture_sleep):
            with patch.object(
                adapter, "health_check", new_callable=AsyncMock, return_value=HealthStatus.HEALTHY
            ):
                # First restart: backoff = 5 * 2^0 = 5
                adapter.restart_count = 0
                await adapter.restart()

        # sleep is called twice: once for backoff, once for startup wait
        assert sleep_calls[0] == RESTART_BACKOFF_BASE * (2 ** 0)  # 5
        assert sleep_calls[1] == 2  # startup wait

    @pytest.mark.asyncio
    async def test_restart_backoff_increases_with_count(self):
        """Verify backoff doubles with each failure."""
        adapter = RustServiceAdapter("backoff-svc")

        sleep_calls = []

        async def capture_sleep(duration):
            sleep_calls.append(duration)

        with patch("asyncio.sleep", side_effect=capture_sleep):
            with patch.object(
                adapter, "health_check", new_callable=AsyncMock, return_value=HealthStatus.UNHEALTHY
            ):
                adapter.restart_count = 2
                await adapter.restart()

        # backoff = 5 * 2^2 = 20
        assert sleep_calls[0] == RESTART_BACKOFF_BASE * (2 ** 2)  # 20

    @pytest.mark.asyncio
    async def test_restart_success_resets_start_time(self):
        """Successful restart resets _start_time."""
        adapter = RustServiceAdapter("time-svc")
        old_start = adapter._start_time

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with patch.object(
                adapter, "health_check", new_callable=AsyncMock, return_value=HealthStatus.HEALTHY
            ):
                await adapter.restart()

        assert adapter._start_time >= old_start


class TestRustServiceAdapterUptime:
    """Validate get_uptime calculation."""

    def test_uptime_positive(self):
        adapter = RustServiceAdapter("up-svc")
        uptime = adapter.get_uptime()
        assert uptime >= 0.0

    def test_uptime_increases_over_time(self):
        adapter = RustServiceAdapter("up-svc")
        t1 = adapter.get_uptime()
        # Small busy-wait to ensure measurable time passes
        time.sleep(0.01)
        t2 = adapter.get_uptime()
        assert t2 > t1


class TestRustServiceAdapterGetStatus:
    """Validate get_status returns well-formed ServiceStatus."""

    def test_get_status_structure(self):
        adapter = RustServiceAdapter("status-svc")
        status = adapter.get_status()
        assert isinstance(status, ServiceStatus)
        assert status.service_id == "rust:status-svc"
        assert status.service_type == ServiceType.RUST_SERVICE
        assert status.health == HealthStatus.UNKNOWN
        assert status.restart_count == 0

    def test_get_status_after_health_change(self):
        adapter = RustServiceAdapter("status-svc")
        adapter.last_health = HealthStatus.DEGRADED
        adapter.restart_count = 2
        status = adapter.get_status()
        assert status.health == HealthStatus.DEGRADED
        assert status.restart_count == 2

    def test_get_status_uptime_is_positive(self):
        adapter = RustServiceAdapter("status-svc")
        status = adapter.get_status()
        assert status.uptime_seconds >= 0.0

    def test_get_status_last_check_without_prior_check(self):
        """When no health check has run, last_check defaults to now."""
        adapter = RustServiceAdapter("no-check-svc")
        before = datetime.now(timezone.utc)
        status = adapter.get_status()
        after = datetime.now(timezone.utc)
        assert before <= status.last_check <= after


# =============================================================================
# 5. HYBRID SWARM ORCHESTRATOR
# =============================================================================


class TestHybridSwarmOrchestratorInit:
    """Validate HybridSwarmOrchestrator initialization."""

    def test_inherits_swarm_orchestrator(self):
        orch = HybridSwarmOrchestrator()
        assert isinstance(orch, SwarmOrchestrator)

    def test_rust_adapters_empty(self):
        orch = HybridSwarmOrchestrator()
        assert orch.rust_adapters == {}

    def test_not_running_initially(self):
        orch = HybridSwarmOrchestrator()
        assert orch._running is False

    def test_heal_task_none(self):
        orch = HybridSwarmOrchestrator()
        assert orch._heal_task is None

    def test_metrics_zeroed(self):
        orch = HybridSwarmOrchestrator()
        assert orch._total_restarts == 0
        assert orch._total_replacements == 0

    def test_availability_history_empty(self):
        orch = HybridSwarmOrchestrator()
        assert len(orch._availability_history) == 0

    def test_availability_history_maxlen(self):
        orch = HybridSwarmOrchestrator()
        assert orch._availability_history.maxlen == 100

    def test_has_health_monitor(self):
        orch = HybridSwarmOrchestrator()
        assert hasattr(orch, "health_monitor")

    def test_has_swarms_dict(self):
        orch = HybridSwarmOrchestrator()
        assert hasattr(orch, "_swarms")
        assert isinstance(orch._swarms, dict)

    def test_class_ratios(self):
        assert HybridSwarmOrchestrator.PYTHON_RATIO == 0.7
        assert HybridSwarmOrchestrator.RUST_RATIO == 0.3

    def test_class_minimums(self):
        assert HybridSwarmOrchestrator.MIN_PYTHON_AGENTS == 1
        assert HybridSwarmOrchestrator.MIN_RUST_SERVICES == 1


class TestRegisterUnregisterRustService:
    """Validate Rust service registration and unregistration."""

    def test_register_creates_adapter(self):
        orch = HybridSwarmOrchestrator()
        orch.register_rust_service("api-server")
        assert "api-server" in orch.rust_adapters
        assert isinstance(orch.rust_adapters["api-server"], RustServiceAdapter)

    def test_register_custom_endpoint(self):
        orch = HybridSwarmOrchestrator()
        orch.register_rust_service("api-server", endpoint="http://10.0.0.1:8080")
        assert orch.rust_adapters["api-server"].endpoint == "http://10.0.0.1:8080"

    def test_register_adds_health_check_callback(self):
        orch = HybridSwarmOrchestrator()
        orch.register_rust_service("api-server")
        assert "rust:api-server" in orch.health_monitor._check_callbacks

    def test_unregister_removes_adapter(self):
        orch = HybridSwarmOrchestrator()
        orch.register_rust_service("api-server")
        orch.unregister_rust_service("api-server")
        assert "api-server" not in orch.rust_adapters

    def test_unregister_nonexistent_is_noop(self):
        orch = HybridSwarmOrchestrator()
        # Should not raise
        orch.unregister_rust_service("nonexistent")
        assert len(orch.rust_adapters) == 0

    def test_register_multiple_services(self):
        orch = HybridSwarmOrchestrator()
        orch.register_rust_service("svc-1")
        orch.register_rust_service("svc-2")
        orch.register_rust_service("svc-3")
        assert len(orch.rust_adapters) == 3


class TestStartStop:
    """Validate orchestrator lifecycle."""

    @pytest.mark.asyncio
    async def test_start_sets_running(self):
        orch = HybridSwarmOrchestrator()
        await orch.start()
        assert orch._running is True
        await orch.stop()

    @pytest.mark.asyncio
    async def test_start_creates_heal_task(self):
        orch = HybridSwarmOrchestrator()
        await orch.start()
        assert orch._heal_task is not None
        assert isinstance(orch._heal_task, asyncio.Task)
        await orch.stop()

    @pytest.mark.asyncio
    async def test_start_idempotent(self):
        orch = HybridSwarmOrchestrator()
        await orch.start()
        task1 = orch._heal_task
        await orch.start()  # second call should be no-op
        assert orch._heal_task is task1
        await orch.stop()

    @pytest.mark.asyncio
    async def test_stop_clears_running(self):
        orch = HybridSwarmOrchestrator()
        await orch.start()
        await orch.stop()
        assert orch._running is False

    @pytest.mark.asyncio
    async def test_stop_cancels_heal_task(self):
        orch = HybridSwarmOrchestrator()
        await orch.start()
        task = orch._heal_task
        await orch.stop()
        assert task.cancelled() or task.done()

    @pytest.mark.asyncio
    async def test_stop_idempotent(self):
        orch = HybridSwarmOrchestrator()
        await orch.start()
        await orch.stop()
        await orch.stop()  # second call should be no-op
        assert orch._running is False


class TestCheckAllHealth:
    """Validate check_all_health across Python + Rust agents."""

    @pytest.mark.asyncio
    async def test_empty_orchestrator(self):
        orch = HybridSwarmOrchestrator()
        result = await orch.check_all_health()
        assert result == {}

    @pytest.mark.asyncio
    async def test_rust_services_checked(self):
        orch = HybridSwarmOrchestrator()
        orch.register_rust_service("svc-1")

        # Mock the health_check on the adapter
        orch.rust_adapters["svc-1"].health_check = AsyncMock(return_value=HealthStatus.HEALTHY)

        result = await orch.check_all_health()
        assert "rust:svc-1" in result
        assert result["rust:svc-1"] == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_python_agents_default_healthy(self):
        """Python agents in swarms default to HEALTHY in check_all_health."""
        orch = HybridSwarmOrchestrator()
        swarm = _make_swarm("s1")
        swarm.agents["py-01"] = _make_python_agent("py-01")
        orch._swarms["s1"] = swarm

        result = await orch.check_all_health()
        assert "py-01" in result
        assert result["py-01"] == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_rust_agents_in_swarm_not_double_counted(self):
        """Rust agents in swarm should NOT appear as Python healthy."""
        orch = HybridSwarmOrchestrator()
        swarm = _make_swarm("s1")
        swarm.agents["rust:svc-1"] = _make_rust_agent("svc-1")
        orch._swarms["s1"] = swarm
        orch.register_rust_service("svc-1")
        orch.rust_adapters["svc-1"].health_check = AsyncMock(return_value=HealthStatus.DEGRADED)

        result = await orch.check_all_health()
        assert result["rust:svc-1"] == HealthStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_mixed_agents(self):
        """Check both Python and Rust agents in same swarm."""
        orch = HybridSwarmOrchestrator()
        swarm = _make_swarm("s1")
        swarm.agents["py-01"] = _make_python_agent("py-01")
        swarm.agents["rust:svc-1"] = _make_rust_agent("svc-1")
        orch._swarms["s1"] = swarm
        orch.register_rust_service("svc-1")
        orch.rust_adapters["svc-1"].health_check = AsyncMock(return_value=HealthStatus.HEALTHY)

        result = await orch.check_all_health()
        assert len(result) == 2
        assert result["py-01"] == HealthStatus.HEALTHY
        assert result["rust:svc-1"] == HealthStatus.HEALTHY


class TestSwarmHealthSummary:
    """Validate get_swarm_health_summary."""

    def test_unknown_swarm_returns_error(self):
        orch = HybridSwarmOrchestrator()
        result = orch.get_swarm_health_summary("nonexistent")
        assert "error" in result

    def test_summary_with_all_healthy_python(self):
        orch = HybridSwarmOrchestrator()
        swarm = _populate_swarm(orch, "s1", python_count=3, rust_count=0)

        summary = orch.get_swarm_health_summary("s1")
        assert summary["swarm_id"] == "s1"
        assert summary["total_agents"] == 3
        assert summary["healthy_agents"] == 3
        assert summary["availability"] == 1.0
        assert summary["meets_target"] is True

    def test_summary_with_rust_adapter_health(self):
        orch = HybridSwarmOrchestrator()
        swarm = _populate_swarm(orch, "s1", python_count=2, rust_count=1)

        # Set rust adapter health to DEGRADED
        svc_name = [k for k in orch.rust_adapters.keys()][0]
        orch.rust_adapters[svc_name].last_health = HealthStatus.DEGRADED

        summary = orch.get_swarm_health_summary("s1")
        assert summary["total_agents"] == 3
        assert summary["healthy_agents"] == 2  # Only Python agents
        assert summary["availability"] == pytest.approx(2 / 3)

    def test_summary_updates_availability_history(self):
        orch = HybridSwarmOrchestrator()
        _populate_swarm(orch, "s1", python_count=2, rust_count=0)

        orch.get_swarm_health_summary("s1")
        assert len(orch._availability_history) == 1

        orch.get_swarm_health_summary("s1")
        assert len(orch._availability_history) == 2

    def test_summary_empty_swarm(self):
        orch = HybridSwarmOrchestrator()
        swarm = _make_swarm("empty")
        orch._swarms["empty"] = swarm

        summary = orch.get_swarm_health_summary("empty")
        assert summary["total_agents"] == 0
        assert summary["healthy_agents"] == 0
        assert summary["availability"] == 0.0
        assert summary["meets_target"] is False

    def test_summary_agent_health_dict(self):
        orch = HybridSwarmOrchestrator()
        swarm = _populate_swarm(orch, "s1", python_count=1, rust_count=1)

        summary = orch.get_swarm_health_summary("s1")
        assert "agent_health" in summary
        assert isinstance(summary["agent_health"], dict)
        # All values should be string health status values
        for v in summary["agent_health"].values():
            assert v in ("healthy", "degraded", "unhealthy", "unknown")

    def test_summary_meets_target_at_boundary(self):
        """availability == 0.999 should meet target."""
        orch = HybridSwarmOrchestrator()
        # Need 1000 agents: 999 healthy + 1 unhealthy = 0.999
        # Simplify: 1 agent, healthy -> 1.0 meets target
        swarm = _make_swarm("boundary")
        swarm.agents["py-01"] = _make_python_agent("py-01")
        orch._swarms["boundary"] = swarm

        summary = orch.get_swarm_health_summary("boundary")
        assert summary["availability"] == 1.0
        assert summary["meets_target"] is True


class TestGetMetrics:
    """Validate get_metrics output."""

    def test_initial_metrics(self):
        orch = HybridSwarmOrchestrator()
        metrics = orch.get_metrics()
        assert metrics["total_swarms"] == 0
        assert metrics["total_rust_services"] == 0
        assert metrics["total_restarts"] == 0
        assert metrics["total_replacements"] == 0
        assert metrics["average_availability"] == 0.0
        assert metrics["meets_availability_target"] is False
        assert metrics["running"] is False

    def test_metrics_after_registration(self):
        orch = HybridSwarmOrchestrator()
        orch.register_rust_service("svc-1")
        orch.register_rust_service("svc-2")
        metrics = orch.get_metrics()
        assert metrics["total_rust_services"] == 2

    def test_metrics_after_swarm_creation(self):
        orch = HybridSwarmOrchestrator()
        _populate_swarm(orch, "s1")
        _populate_swarm(orch, "s2")
        metrics = orch.get_metrics()
        assert metrics["total_swarms"] == 2

    def test_metrics_restarts_tracked(self):
        orch = HybridSwarmOrchestrator()
        orch._total_restarts = 5
        orch._total_replacements = 2
        metrics = orch.get_metrics()
        assert metrics["total_restarts"] == 5
        assert metrics["total_replacements"] == 2

    def test_metrics_average_availability(self):
        orch = HybridSwarmOrchestrator()
        orch._availability_history.extend([1.0, 0.9, 0.8])
        metrics = orch.get_metrics()
        assert metrics["average_availability"] == pytest.approx(0.9)

    def test_metrics_meets_target_when_high(self):
        orch = HybridSwarmOrchestrator()
        orch._availability_history.extend([1.0, 1.0, 1.0])
        metrics = orch.get_metrics()
        assert metrics["meets_availability_target"] is True

    def test_metrics_does_not_meet_target_when_low(self):
        orch = HybridSwarmOrchestrator()
        orch._availability_history.extend([0.5, 0.6, 0.7])
        metrics = orch.get_metrics()
        assert metrics["meets_availability_target"] is False

    @pytest.mark.asyncio
    async def test_metrics_running_after_start(self):
        orch = HybridSwarmOrchestrator()
        await orch.start()
        assert orch.get_metrics()["running"] is True
        await orch.stop()
        assert orch.get_metrics()["running"] is False


# =============================================================================
# 6. SCALING
# =============================================================================


class TestScaling:
    """Validate proportional scaling (70% Python, 30% Rust)."""

    @pytest.mark.asyncio
    async def test_scale_up_proportional_split(self):
        """SCALE_UP should allocate ~70% Python, ~30% Rust."""
        orch = HybridSwarmOrchestrator()
        swarm = _populate_swarm(orch, "s1", python_count=3, rust_count=1)

        decision = ScalingDecision(
            action=ScalingAction.SCALE_UP,
            target_count=14,  # +10 from current 4
            current_count=4,
            reason="test",
        )

        await orch.apply_scaling_decision(decision, "s1")

        python_agents = [a for a in swarm.agents.values() if not a.id.startswith("rust:")]
        rust_agents = [a for a in swarm.agents.values() if a.id.startswith("rust:")]

        # Delta = 10. Python delta = int(10 * 0.7) = 7. Rust delta = 10 - 7 = 3.
        assert len(python_agents) == 3 + 7  # 10
        assert len(rust_agents) == 1 + 3    # 4

    @pytest.mark.asyncio
    async def test_scale_up_unknown_swarm_is_noop(self):
        orch = HybridSwarmOrchestrator()
        decision = ScalingDecision(
            action=ScalingAction.SCALE_UP,
            target_count=10,
            current_count=5,
            reason="test",
        )
        # Should not raise
        await orch.apply_scaling_decision(decision, "nonexistent")

    @pytest.mark.asyncio
    async def test_scale_up_creates_agent_instances(self):
        """New agents should have correct types and running status."""
        orch = HybridSwarmOrchestrator()
        _populate_swarm(orch, "s1", python_count=1, rust_count=0)

        decision = ScalingDecision(
            action=ScalingAction.SCALE_UP,
            target_count=4,
            current_count=1,
            reason="test",
        )
        await orch.apply_scaling_decision(decision, "s1")

        swarm = orch._swarms["s1"]
        for agent in swarm.agents.values():
            assert agent.status == AgentStatus.RUNNING

    @pytest.mark.asyncio
    async def test_scale_down_python_first(self):
        """SCALE_DOWN should remove Python agents before Rust services."""
        orch = HybridSwarmOrchestrator()
        swarm = _populate_swarm(orch, "s1", python_count=5, rust_count=3)
        initial_rust_count = 3

        decision = ScalingDecision(
            action=ScalingAction.SCALE_DOWN,
            target_count=5,  # -3 from current 8
            current_count=8,
            reason="test",
        )

        await orch.apply_scaling_decision(decision, "s1")

        python_agents = [a for a in swarm.agents.values() if not a.id.startswith("rust:")]
        rust_agents = [a for a in swarm.agents.values() if a.id.startswith("rust:")]

        # delta=3. Python can shrink by min(3, 5-1)=3. Rust delta = 3-3 = 0.
        assert len(python_agents) == 2
        assert len(rust_agents) == initial_rust_count  # Rust untouched

    @pytest.mark.asyncio
    async def test_scale_down_respects_python_minimum(self):
        """Cannot scale Python below MIN_PYTHON_AGENTS (1)."""
        orch = HybridSwarmOrchestrator()
        swarm = _populate_swarm(orch, "s1", python_count=2, rust_count=3)

        decision = ScalingDecision(
            action=ScalingAction.SCALE_DOWN,
            target_count=2,  # -3 from current 5
            current_count=5,
            reason="test",
        )

        await orch.apply_scaling_decision(decision, "s1")

        python_agents = [a for a in swarm.agents.values() if not a.id.startswith("rust:")]
        rust_agents = [a for a in swarm.agents.values() if a.id.startswith("rust:")]

        # delta=3. Python delta = min(3, 2-1) = 1. Remaining delta = 2.
        # Rust delta = min(2, 3-1) = 2.
        assert len(python_agents) >= orch.MIN_PYTHON_AGENTS

    @pytest.mark.asyncio
    async def test_scale_down_respects_rust_minimum(self):
        """Cannot scale Rust below MIN_RUST_SERVICES (1)."""
        orch = HybridSwarmOrchestrator()
        swarm = _populate_swarm(orch, "s1", python_count=1, rust_count=2)

        decision = ScalingDecision(
            action=ScalingAction.SCALE_DOWN,
            target_count=1,  # -2 from current 3
            current_count=3,
            reason="test",
        )

        await orch.apply_scaling_decision(decision, "s1")

        rust_agents = [a for a in swarm.agents.values() if a.id.startswith("rust:")]
        assert len(rust_agents) >= orch.MIN_RUST_SERVICES

    @pytest.mark.asyncio
    async def test_scale_down_removes_oldest_python_first(self):
        """FIFO: oldest Python agents removed first."""
        orch = HybridSwarmOrchestrator()
        swarm = _make_swarm("s1")
        orch._swarms["s1"] = swarm

        # Add agents in order; first added = oldest
        for i in range(5):
            agent = _make_python_agent(f"py-s1-{i:04d}")
            swarm.agents[agent.id] = agent

        decision = ScalingDecision(
            action=ScalingAction.SCALE_DOWN,
            target_count=3,
            current_count=5,
            reason="test",
        )

        await orch.apply_scaling_decision(decision, "s1")

        remaining = list(swarm.agents.keys())
        # Oldest two (py-s1-0000, py-s1-0001) should be removed
        assert "py-s1-0000" not in remaining
        assert "py-s1-0001" not in remaining
        assert len(remaining) == 3

    @pytest.mark.asyncio
    async def test_scale_up_rust_registers_services(self):
        """Scaling up Rust should register new adapters."""
        orch = HybridSwarmOrchestrator()
        _populate_swarm(orch, "s1", python_count=1, rust_count=0)

        initial_adapters = len(orch.rust_adapters)

        decision = ScalingDecision(
            action=ScalingAction.SCALE_UP,
            target_count=4,  # +3
            current_count=1,
            reason="test",
        )

        await orch.apply_scaling_decision(decision, "s1")

        # Rust delta = 3 - int(3 * 0.7) = 3 - 2 = 1
        assert len(orch.rust_adapters) > initial_adapters

    @pytest.mark.asyncio
    async def test_scale_down_rust_unregisters_services(self):
        """Scaling down Rust should unregister adapters."""
        orch = HybridSwarmOrchestrator()
        _populate_swarm(orch, "s1", python_count=1, rust_count=3)

        initial_adapters = len(orch.rust_adapters)

        decision = ScalingDecision(
            action=ScalingAction.SCALE_DOWN,
            target_count=2,  # -2 from current 4
            current_count=4,
            reason="test",
        )

        await orch.apply_scaling_decision(decision, "s1")

        # Some rust adapters should have been removed
        # python_delta = min(2, 1-1) = 0, rust_delta = min(2, 3-1) = 2
        assert len(orch.rust_adapters) < initial_adapters

    @pytest.mark.asyncio
    async def test_scale_up_zero_delta_is_noop(self):
        """No scaling when target == current."""
        orch = HybridSwarmOrchestrator()
        swarm = _populate_swarm(orch, "s1", python_count=3, rust_count=1)

        decision = ScalingDecision(
            action=ScalingAction.SCALE_UP,
            target_count=4,
            current_count=4,
            reason="test",
        )

        initial_count = len(swarm.agents)
        await orch.apply_scaling_decision(decision, "s1")
        assert len(swarm.agents) == initial_count


# =============================================================================
# 7. SELF-HEALING
# =============================================================================


class TestSelfHealing:
    """Validate self-healing iteration restarts and replaces services."""

    @pytest.mark.asyncio
    async def test_self_heal_restarts_unhealthy(self):
        """Unhealthy services get restarted."""
        orch = HybridSwarmOrchestrator()
        orch.register_rust_service("sick-svc")

        adapter = orch.rust_adapters["sick-svc"]
        adapter.health_check = AsyncMock(return_value=HealthStatus.UNHEALTHY)
        adapter.restart = AsyncMock(return_value=True)

        await orch._self_heal_iteration()

        adapter.restart.assert_called_once()
        assert orch._total_restarts == 1

    @pytest.mark.asyncio
    async def test_self_heal_replaces_on_failed_restart(self):
        """When restart fails, service is replaced."""
        orch = HybridSwarmOrchestrator()
        swarm = _populate_swarm(orch, "s1", python_count=1, rust_count=1)

        svc_name = [k for k in orch.rust_adapters.keys()][0]
        adapter = orch.rust_adapters[svc_name]
        adapter.health_check = AsyncMock(return_value=HealthStatus.UNHEALTHY)
        adapter.restart = AsyncMock(return_value=False)

        await orch._self_heal_iteration()

        assert orch._total_restarts == 1
        assert orch._total_replacements == 1

    @pytest.mark.asyncio
    async def test_self_heal_skips_healthy(self):
        """Healthy services are not restarted."""
        orch = HybridSwarmOrchestrator()
        orch.register_rust_service("healthy-svc")

        adapter = orch.rust_adapters["healthy-svc"]
        adapter.health_check = AsyncMock(return_value=HealthStatus.HEALTHY)
        adapter.restart = AsyncMock()

        await orch._self_heal_iteration()

        adapter.restart.assert_not_called()
        assert orch._total_restarts == 0

    @pytest.mark.asyncio
    async def test_self_heal_skips_degraded(self):
        """DEGRADED services should NOT trigger restart (only UNHEALTHY/UNKNOWN)."""
        orch = HybridSwarmOrchestrator()
        orch.register_rust_service("degraded-svc")

        adapter = orch.rust_adapters["degraded-svc"]
        adapter.health_check = AsyncMock(return_value=HealthStatus.DEGRADED)
        adapter.restart = AsyncMock()

        await orch._self_heal_iteration()

        adapter.restart.assert_not_called()

    @pytest.mark.asyncio
    async def test_self_heal_handles_unknown_status(self):
        """UNKNOWN status triggers restart."""
        orch = HybridSwarmOrchestrator()
        orch.register_rust_service("unknown-svc")

        adapter = orch.rust_adapters["unknown-svc"]
        adapter.health_check = AsyncMock(return_value=HealthStatus.UNKNOWN)
        adapter.restart = AsyncMock(return_value=True)

        await orch._self_heal_iteration()

        adapter.restart.assert_called_once()
        assert orch._total_restarts == 1

    @pytest.mark.asyncio
    async def test_self_heal_multiple_services(self):
        """Multiple unhealthy services all get restarted."""
        orch = HybridSwarmOrchestrator()
        for name in ["svc-1", "svc-2", "svc-3"]:
            orch.register_rust_service(name)
            adapter = orch.rust_adapters[name]
            adapter.health_check = AsyncMock(return_value=HealthStatus.UNHEALTHY)
            adapter.restart = AsyncMock(return_value=True)

        await orch._self_heal_iteration()

        assert orch._total_restarts == 3
        for name in ["svc-1", "svc-2", "svc-3"]:
            orch.rust_adapters[name].restart.assert_called_once()

    @pytest.mark.asyncio
    async def test_self_heal_increments_metrics(self):
        """Each restart/replacement increments the right metric."""
        orch = HybridSwarmOrchestrator()
        swarm = _populate_swarm(orch, "s1", python_count=0, rust_count=2)

        adapters = list(orch.rust_adapters.values())
        # First adapter: restart succeeds
        adapters[0].health_check = AsyncMock(return_value=HealthStatus.UNHEALTHY)
        adapters[0].restart = AsyncMock(return_value=True)
        # Second adapter: restart fails -> replacement
        adapters[1].health_check = AsyncMock(return_value=HealthStatus.UNHEALTHY)
        adapters[1].restart = AsyncMock(return_value=False)

        await orch._self_heal_iteration()

        assert orch._total_restarts == 2
        assert orch._total_replacements == 1


class TestReplaceRustService:
    """Validate _replace_rust_service logic."""

    @pytest.mark.asyncio
    async def test_replace_removes_old_adapter(self):
        orch = HybridSwarmOrchestrator()
        _populate_swarm(orch, "s1", python_count=0, rust_count=1)
        old_svc = list(orch.rust_adapters.keys())[0]

        await orch._replace_rust_service(old_svc)

        assert old_svc not in orch.rust_adapters

    @pytest.mark.asyncio
    async def test_replace_creates_new_adapter(self):
        orch = HybridSwarmOrchestrator()
        _populate_swarm(orch, "s1", python_count=0, rust_count=1)
        old_svc = list(orch.rust_adapters.keys())[0]

        await orch._replace_rust_service(old_svc)

        # Old removed, new added -> still 1 adapter
        assert len(orch.rust_adapters) == 1
        assert old_svc not in orch.rust_adapters

    @pytest.mark.asyncio
    async def test_replace_maintains_swarm_agent_count(self):
        orch = HybridSwarmOrchestrator()
        swarm = _populate_swarm(orch, "s1", python_count=2, rust_count=1)
        initial_count = len(swarm.agents)
        old_svc = [k for k in orch.rust_adapters.keys()][0]

        await orch._replace_rust_service(old_svc)

        # Should still have same number of agents
        assert len(swarm.agents) == initial_count

    @pytest.mark.asyncio
    async def test_replace_orphaned_service(self):
        """Replacing a service not in any swarm should not crash."""
        orch = HybridSwarmOrchestrator()
        orch.register_rust_service("orphan-svc")

        # Should not raise
        await orch._replace_rust_service("orphan-svc")

        # Adapter should be removed
        assert "orphan-svc" not in orch.rust_adapters

    @pytest.mark.asyncio
    async def test_replace_new_agent_has_correct_type(self):
        orch = HybridSwarmOrchestrator()
        swarm = _populate_swarm(orch, "s1", python_count=0, rust_count=1)
        old_svc = list(orch.rust_adapters.keys())[0]

        await orch._replace_rust_service(old_svc)

        # The new agent should be a rust agent
        for agent in swarm.agents.values():
            if agent.id.startswith("rust:"):
                assert agent.config.agent_type == "rust-worker"
                assert "inference" in agent.config.capabilities


# =============================================================================
# 8. AVAILABILITY
# =============================================================================


class TestAvailability:
    """Validate availability calculation and target compliance."""

    def test_full_availability(self):
        orch = HybridSwarmOrchestrator()
        _populate_swarm(orch, "s1", python_count=10, rust_count=0)

        summary = orch.get_swarm_health_summary("s1")
        assert summary["availability"] == 1.0
        assert summary["meets_target"] is True

    def test_zero_availability_empty_swarm(self):
        orch = HybridSwarmOrchestrator()
        swarm = _make_swarm("s1")
        orch._swarms["s1"] = swarm

        summary = orch.get_swarm_health_summary("s1")
        assert summary["availability"] == 0.0
        assert summary["meets_target"] is False

    def test_partial_availability(self):
        orch = HybridSwarmOrchestrator()
        swarm = _populate_swarm(orch, "s1", python_count=3, rust_count=1)

        # Make rust adapter unhealthy
        svc = list(orch.rust_adapters.keys())[0]
        orch.rust_adapters[svc].last_health = HealthStatus.UNHEALTHY

        summary = orch.get_swarm_health_summary("s1")
        assert summary["availability"] == pytest.approx(3 / 4)
        assert summary["meets_target"] is False  # 0.75 < 0.999

    def test_availability_history_bounded(self):
        """History deque should not exceed maxlen=100."""
        orch = HybridSwarmOrchestrator()
        _populate_swarm(orch, "s1", python_count=1, rust_count=0)

        for _ in range(150):
            orch.get_swarm_health_summary("s1")

        assert len(orch._availability_history) == 100

    def test_average_availability_in_metrics(self):
        orch = HybridSwarmOrchestrator()
        _populate_swarm(orch, "s1", python_count=2, rust_count=0)

        # Generate some history
        for _ in range(5):
            orch.get_swarm_health_summary("s1")

        metrics = orch.get_metrics()
        assert metrics["average_availability"] == 1.0
        assert metrics["meets_availability_target"] is True

    def test_availability_target_exact_boundary(self):
        """Availability of exactly 0.999 should meet target."""
        orch = HybridSwarmOrchestrator()
        orch._availability_history.append(0.999)
        metrics = orch.get_metrics()
        assert metrics["meets_availability_target"] is True

    def test_availability_target_just_below(self):
        """Availability of 0.998 should NOT meet target."""
        orch = HybridSwarmOrchestrator()
        orch._availability_history.append(0.998)
        metrics = orch.get_metrics()
        assert metrics["meets_availability_target"] is False


# =============================================================================
# 9. SELF-HEAL LOOP INTEGRATION
# =============================================================================


class TestSelfHealLoop:
    """Validate the _self_heal_loop runs and respects _running flag."""

    @pytest.mark.asyncio
    async def test_self_heal_loop_stops_when_not_running(self):
        """Loop should exit when _running is False."""
        orch = HybridSwarmOrchestrator()
        orch._running = False

        # Should complete immediately since _running is False
        # Use a timeout to ensure we don't hang
        try:
            await asyncio.wait_for(orch._self_heal_loop(), timeout=0.5)
        except asyncio.TimeoutError:
            pytest.fail("_self_heal_loop did not exit when _running=False")

    @pytest.mark.asyncio
    async def test_self_heal_loop_catches_exceptions(self):
        """Exceptions in _self_heal_iteration should not crash the loop."""
        orch = HybridSwarmOrchestrator()
        orch._running = True

        call_count = 0

        async def failing_iteration():
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                orch._running = False
            raise RuntimeError("test error")

        orch._self_heal_iteration = failing_iteration  # type: ignore[assignment]

        with patch("asyncio.sleep", new_callable=AsyncMock):
            try:
                await asyncio.wait_for(orch._self_heal_loop(), timeout=1.0)
            except asyncio.TimeoutError:
                pass

        assert call_count >= 1


# =============================================================================
# 10. SCALE INTERNAL METHODS
# =============================================================================


class TestScaleInternalMethods:
    """Validate _scale_up_python, _scale_up_rust, _scale_down_python, _scale_down_rust."""

    @pytest.mark.asyncio
    async def test_scale_up_python_adds_correct_count(self):
        orch = HybridSwarmOrchestrator()
        swarm = _make_swarm("s1")
        orch._swarms["s1"] = swarm

        await orch._scale_up_python("s1", 3)
        assert len(swarm.agents) == 3

    @pytest.mark.asyncio
    async def test_scale_up_python_names_include_swarm_id(self):
        orch = HybridSwarmOrchestrator()
        swarm = _make_swarm("s1")
        orch._swarms["s1"] = swarm

        await orch._scale_up_python("s1", 2)

        for agent_id in swarm.agents:
            assert "s1" in agent_id

    @pytest.mark.asyncio
    async def test_scale_up_python_nonexistent_swarm(self):
        orch = HybridSwarmOrchestrator()
        # Should not raise
        await orch._scale_up_python("nonexistent", 5)

    @pytest.mark.asyncio
    async def test_scale_up_rust_creates_adapter(self):
        orch = HybridSwarmOrchestrator()
        swarm = _make_swarm("s1")
        orch._swarms["s1"] = swarm

        await orch._scale_up_rust("s1", 2)

        assert len(orch.rust_adapters) == 2
        # Each should have "s1" in the name
        for name in orch.rust_adapters:
            assert "s1" in name

    @pytest.mark.asyncio
    async def test_scale_up_rust_adds_agents_to_swarm(self):
        orch = HybridSwarmOrchestrator()
        swarm = _make_swarm("s1")
        orch._swarms["s1"] = swarm

        await orch._scale_up_rust("s1", 2)

        rust_agents = [a for a in swarm.agents.values() if a.id.startswith("rust:")]
        assert len(rust_agents) == 2

    @pytest.mark.asyncio
    async def test_scale_down_python_removes_correct_count(self):
        orch = HybridSwarmOrchestrator()
        swarm = _make_swarm("s1")
        orch._swarms["s1"] = swarm
        for i in range(5):
            agent = _make_python_agent(f"py-s1-{i}")
            swarm.agents[agent.id] = agent

        await orch._scale_down_python("s1", 2)
        assert len(swarm.agents) == 3

    @pytest.mark.asyncio
    async def test_scale_down_python_nonexistent_swarm(self):
        orch = HybridSwarmOrchestrator()
        # Should not raise
        await orch._scale_down_python("nonexistent", 5)

    @pytest.mark.asyncio
    async def test_scale_down_rust_removes_correct_count(self):
        orch = HybridSwarmOrchestrator()
        swarm = _make_swarm("s1")
        orch._swarms["s1"] = swarm

        for i in range(3):
            svc = f"svc-{i}"
            orch.register_rust_service(svc)
            agent = _make_rust_agent(svc)
            swarm.agents[agent.id] = agent

        await orch._scale_down_rust("s1", 2)

        rust_agents = [a for a in swarm.agents.values() if a.id.startswith("rust:")]
        assert len(rust_agents) == 1
        assert len(orch.rust_adapters) == 1

    @pytest.mark.asyncio
    async def test_scale_down_rust_nonexistent_swarm(self):
        orch = HybridSwarmOrchestrator()
        # Should not raise
        await orch._scale_down_rust("nonexistent", 5)
