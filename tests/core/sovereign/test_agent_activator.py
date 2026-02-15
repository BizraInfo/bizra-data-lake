"""
Agent Activator + Executor Tests — True Spearpoint Verification
================================================================
16 smoke tests verifying the genesis → active execution bridge.

Standing on Giants:
- pytest (Krekel, 2004): The gold standard
- Hypothesis (MacIver, 2015): Property-based thinking
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers — mock genesis objects
# ---------------------------------------------------------------------------


def _make_agent_identity(
    agent_id: str = "PAT-001",
    role: str = "researcher",
    capabilities: list = None,
    giants: list = None,
    public_key: str = "ed25519:mock",
) -> MagicMock:
    """Create a mock AgentIdentity matching genesis_identity.py shape."""
    mock = MagicMock()
    mock.agent_id = agent_id
    mock.role = role
    mock.capabilities = capabilities or ["search.web", "memory.read"]
    mock.giants = giants or ["Shannon", "Bush"]
    mock.public_key = public_key
    return mock


def _make_genesis_state(
    pat_count: int = 7,
    sat_count: int = 5,
) -> MagicMock:
    """Create a mock GenesisState with PAT and SAT teams."""
    roles_pat = [
        "worker", "researcher", "guardian", "synthesizer",
        "validator", "coordinator", "executor",
    ]
    roles_sat = ["worker", "guardian", "synthesizer", "validator", "coordinator"]

    genesis = MagicMock()
    genesis.pat_team = [
        _make_agent_identity(
            agent_id=f"PAT-{i:03d}",
            role=roles_pat[i % len(roles_pat)],
        )
        for i in range(pat_count)
    ]
    genesis.sat_team = [
        _make_agent_identity(
            agent_id=f"SAT-{i:03d}",
            role=roles_sat[i % len(roles_sat)],
        )
        for i in range(sat_count)
    ]
    genesis.node_id = "BIZRA-00000000"
    genesis.node_name = "Node0-Test"
    return genesis


# ---------------------------------------------------------------------------
# 1. Agent Activator Unit Tests
# ---------------------------------------------------------------------------


class TestAgentActivator:
    """Verify AgentActivator creates and manages active agents."""

    def test_01_import_and_construct(self) -> None:
        """AgentActivator is importable and constructible."""
        from core.sovereign.agent_activator import AgentActivator

        activator = AgentActivator()
        assert activator.agent_count == 0
        assert activator.ready_count == 0

    def test_02_activate_from_genesis(self) -> None:
        """Activates all PAT + SAT agents from genesis state."""
        from core.sovereign.agent_activator import AgentActivator

        activator = AgentActivator()
        genesis = _make_genesis_state(pat_count=7, sat_count=5)
        result = activator.activate_from_genesis(genesis)

        assert result.success
        assert result.activated == 12  # 7 PAT + 5 SAT
        assert result.failed == 0
        assert activator.agent_count == 12
        assert activator.ready_count == 12

    def test_03_agent_roles_preserved(self) -> None:
        """Activated agents preserve their genesis roles."""
        from core.sovereign.agent_activator import AgentActivator

        activator = AgentActivator()
        genesis = _make_genesis_state(pat_count=3, sat_count=0)
        activator.activate_from_genesis(genesis)

        agent = activator.get_agent("PAT-000")
        assert agent is not None
        assert agent.role == "worker"
        assert agent.agent_id == "PAT-000"

    def test_04_system_prompts_generated(self) -> None:
        """Each agent gets a role-appropriate system prompt."""
        from core.sovereign.agent_activator import AgentActivator

        activator = AgentActivator()
        genesis = _make_genesis_state(pat_count=7, sat_count=0)
        activator.activate_from_genesis(genesis)

        researcher = activator.get_agent_for_role("researcher")
        assert researcher is not None
        assert "Researcher" in researcher.system_prompt
        assert len(researcher.system_prompt) > 20

        guardian = activator.get_agent_for_role("guardian")
        assert guardian is not None
        assert "Guardian" in guardian.system_prompt

    def test_05_model_purpose_routing(self) -> None:
        """Agents get correct model_purpose for InferenceGateway routing."""
        from core.sovereign.agent_activator import AgentActivator

        activator = AgentActivator()
        genesis = _make_genesis_state(pat_count=7, sat_count=0)
        activator.activate_from_genesis(genesis)

        researcher = activator.get_agent_for_role("researcher")
        assert researcher is not None
        assert researcher.model_purpose == "reasoning"

        executor = activator.get_agent_for_role("executor")
        assert executor is not None
        assert executor.model_purpose == "agentic"

    def test_06_select_agents_for_task(self) -> None:
        """Agent selection routes by keyword matching."""
        from core.sovereign.agent_activator import AgentActivator

        activator = AgentActivator()
        genesis = _make_genesis_state(pat_count=7, sat_count=0)
        activator.activate_from_genesis(genesis)

        # Research task should include researcher
        agents = activator.select_agents_for_task("Research the impact of AI on education")
        roles = [a.role for a in agents]
        assert "researcher" in roles
        assert "coordinator" in roles  # Always included

    def test_07_select_agents_default_fallback(self) -> None:
        """Tasks with no keyword match get default team."""
        from core.sovereign.agent_activator import AgentActivator

        activator = AgentActivator()
        genesis = _make_genesis_state(pat_count=7, sat_count=0)
        activator.activate_from_genesis(genesis)

        agents = activator.select_agents_for_task("Hello world")
        assert len(agents) >= 2  # Coordinator + fallback

    def test_08_deactivate_all(self) -> None:
        """Deactivation marks all agents as deactivated."""
        from core.sovereign.agent_activator import ActivationStatus, AgentActivator

        activator = AgentActivator()
        genesis = _make_genesis_state(pat_count=3, sat_count=2)
        activator.activate_from_genesis(genesis)
        assert activator.ready_count == 5

        count = activator.deactivate_all()
        assert count == 5
        assert activator.ready_count == 0

        deactivated = activator.get_agents_by_status(ActivationStatus.DEACTIVATED)
        assert len(deactivated) == 5

    def test_09_graceful_with_no_genesis(self) -> None:
        """Activator handles empty genesis gracefully."""
        from core.sovereign.agent_activator import AgentActivator

        activator = AgentActivator()
        genesis = MagicMock()
        genesis.pat_team = []
        genesis.sat_team = []

        result = activator.activate_from_genesis(genesis)
        assert not result.success
        assert result.activated == 0

    def test_10_summary_structure(self) -> None:
        """Summary returns expected structure."""
        from core.sovereign.agent_activator import AgentActivator

        activator = AgentActivator()
        genesis = _make_genesis_state(pat_count=7, sat_count=5)
        activator.activate_from_genesis(genesis)

        summary = activator.summary()
        assert summary["total_agents"] == 12
        assert summary["ready"] == 12
        assert summary["activated"] is True
        assert "by_role" in summary
        assert "by_status" in summary


# ---------------------------------------------------------------------------
# 2. Agent Executor Unit Tests
# ---------------------------------------------------------------------------


class TestAgentExecutor:
    """Verify AgentExecutor dispatches tasks to activated agents."""

    def test_11_import_and_construct(self) -> None:
        """AgentExecutor is importable and constructible."""
        from core.sovereign.agent_executor import AgentExecutor

        executor = AgentExecutor()
        assert executor._executions == 0

    @pytest.mark.asyncio
    async def test_12_execute_with_mock_gateway(self) -> None:
        """Executor dispatches to agents and collects results."""
        from core.sovereign.agent_activator import AgentActivator
        from core.sovereign.agent_executor import AgentExecutor

        # Setup activator
        activator = AgentActivator()
        genesis = _make_genesis_state(pat_count=7, sat_count=0)
        activator.activate_from_genesis(genesis)

        # Setup mock gateway
        mock_result = MagicMock()
        mock_result.content = "Analysis complete: sovereignty patterns identified"
        mock_result.usage = MagicMock()
        mock_result.usage.total_tokens = 150
        mock_gw = MagicMock()
        mock_gw.infer = AsyncMock(return_value=mock_result)

        # Create executor
        executor = AgentExecutor(activator=activator, gateway=mock_gw)

        # Create mock opportunity
        opp = MagicMock()
        opp.id = "test-opp-001"
        opp.domain = "research"
        opp.description = "Research the impact of sovereignty architectures"
        opp.context = {}

        result = await executor.execute(opp)

        assert result["success"]
        assert "content" in result
        assert len(result["agents_used"]) > 0
        assert result["opportunity_id"] == "test-opp-001"

    @pytest.mark.asyncio
    async def test_13_execute_without_gateway(self) -> None:
        """Executor handles missing gateway gracefully."""
        from core.sovereign.agent_activator import AgentActivator
        from core.sovereign.agent_executor import AgentExecutor

        activator = AgentActivator()
        genesis = _make_genesis_state(pat_count=3, sat_count=0)
        activator.activate_from_genesis(genesis)

        executor = AgentExecutor(activator=activator, gateway=None)

        opp = MagicMock()
        opp.id = "test-opp-002"
        opp.domain = "test"
        opp.description = "Test without gateway"
        opp.context = {}

        result = await executor.execute(opp)

        # Should still succeed — just with fallback content
        assert result["success"]
        assert "No gateway configured" in result["content"]

    @pytest.mark.asyncio
    async def test_14_execute_without_activator(self) -> None:
        """Executor handles missing activator gracefully."""
        from core.sovereign.agent_executor import AgentExecutor

        executor = AgentExecutor(activator=None, gateway=None)

        opp = MagicMock()
        opp.id = "test-opp-003"
        opp.domain = "test"
        opp.description = "Test without activator"
        opp.context = {}

        result = await executor.execute(opp)

        assert not result["success"]
        assert "No activated agents" in result["error"]

    @pytest.mark.asyncio
    async def test_15_metrics_tracking(self) -> None:
        """Executor tracks execution metrics."""
        from core.sovereign.agent_activator import AgentActivator
        from core.sovereign.agent_executor import AgentExecutor

        activator = AgentActivator()
        genesis = _make_genesis_state(pat_count=3, sat_count=0)
        activator.activate_from_genesis(genesis)

        mock_result = MagicMock()
        mock_result.content = "Done"
        mock_result.usage = MagicMock()
        mock_result.usage.total_tokens = 100
        mock_gw = MagicMock()
        mock_gw.infer = AsyncMock(return_value=mock_result)

        executor = AgentExecutor(activator=activator, gateway=mock_gw)

        opp = MagicMock()
        opp.id = "test-001"
        opp.domain = "test"
        opp.description = "Analyze this"
        opp.context = {}

        await executor.execute(opp)

        m = executor.metrics()
        assert m["executions"] == 1
        assert m["successes"] == 1
        assert m["failures"] == 0


# ---------------------------------------------------------------------------
# 3. Wiring Integration Tests
# ---------------------------------------------------------------------------


class TestRuntimeWiring:
    """Verify runtime_core.py properly wires activator and executor."""

    def test_16_runtime_has_activator_fields(self) -> None:
        """SovereignRuntime has agent_activator and agent_executor slots."""
        from core.sovereign.runtime_core import SovereignRuntime

        rt = SovereignRuntime()
        assert hasattr(rt, "_agent_activator")
        assert hasattr(rt, "_agent_executor")
        assert rt._agent_activator is None
        assert rt._agent_executor is None

    def test_17_init_agent_activation_no_genesis(self) -> None:
        """_init_agent_activation skips when no genesis identity."""
        from core.sovereign.runtime_core import SovereignRuntime

        rt = SovereignRuntime()
        rt._genesis = None
        rt._init_agent_activation()
        assert rt._agent_activator is None

    def test_18_init_agent_activation_with_genesis(self) -> None:
        """_init_agent_activation creates activator when genesis present."""
        from core.sovereign.runtime_core import SovereignRuntime

        rt = SovereignRuntime()
        rt._genesis = _make_genesis_state(pat_count=7, sat_count=5)
        rt._gateway = MagicMock()

        rt._init_agent_activation()

        assert rt._agent_activator is not None
        assert rt._agent_executor is not None

        # Verify agents were activated
        summary = rt._agent_activator.summary()
        assert summary["total_agents"] == 12
        assert summary["ready"] == 12

    def test_19_component_status_includes_activator(self) -> None:
        """Runtime state snapshot includes agent_activator and agent_executor."""
        from core.sovereign.runtime_core import SovereignRuntime

        rt = SovereignRuntime()
        state = rt._get_runtime_state()
        components = state["components"]
        assert "agent_activator" in components
        assert "agent_executor" in components
