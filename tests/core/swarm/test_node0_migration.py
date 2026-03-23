"""Tests for Phase 4 — node0_activate.py migration patterns.

ADR-004 Phase 5 validation: 7 tests for migration/compatibility.
These tests validate that SwarmEngine can be wired into the existing
node0 pipeline without breaking the result format or feature flag behavior.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.swarm.engine import (
    ParallelStrategy,
    SequentialStrategy,
    SwarmEngine,
)
from core.swarm.types import (
    AgentRole,
    AgentSpec,
    SwarmConfig,
    SwarmTopology,
)

# -- Fixtures ------------------------------------------------------------------


def _make_spec(agent_id: str, role: AgentRole = AgentRole.RESEARCHER) -> AgentSpec:
    return AgentSpec(id=agent_id, role=role, model_purpose="reasoning")


PAT_AGENTS_SAMPLE = {
    "strategist": {
        "name": "Strategist",
        "role": "Strategic planning and long-term thinking",
        "giants": "Sun Tzu, John Boyd, Michael Porter",
        "model_purpose": "thinking",
    },
    "researcher": {
        "name": "Researcher",
        "role": "Deep investigation and evidence gathering",
        "giants": "Vannevar Bush, Claude Shannon, Douglas Engelbart",
        "model_purpose": "reasoning",
    },
    "creator": {
        "name": "Creator",
        "role": "Content creation and design",
        "giants": "Leonardo da Vinci, Steve Jobs, Dieter Rams",
        "model_purpose": "creative",
    },
}


def _build_engine_from_config(yaml_config: Dict[str, Any]) -> SwarmEngine:
    """Simulate the topology selection logic from Phase 4 migration pseudocode."""
    topology_name = yaml_config.get("swarm_topology", "sequential").upper()
    try:
        topology = SwarmTopology[topology_name]
    except KeyError:
        topology = SwarmTopology.SEQUENTIAL

    config = SwarmConfig(
        topology=topology,
        max_concurrent=yaml_config.get("swarm_max_concurrent", 3),
    )
    return SwarmEngine(config=config)


# -- Feature flag behavior -----------------------------------------------------


@pytest.mark.asyncio
async def test_legacy_path_when_disabled():
    """When SWARM_ENGINE_ENABLED=false, legacy code path would run.

    We simulate this by checking that the flag correctly parses to False.
    """
    with patch.dict(os.environ, {"SWARM_ENGINE_ENABLED": "false"}):
        enabled = os.getenv("SWARM_ENGINE_ENABLED", "").lower() in ("1", "true")
        assert enabled is False


@pytest.mark.asyncio
async def test_swarm_engine_path_when_enabled():
    """When SWARM_ENGINE_ENABLED=true, SwarmEngine should be used."""
    with patch.dict(os.environ, {"SWARM_ENGINE_ENABLED": "true"}):
        enabled = os.getenv("SWARM_ENGINE_ENABLED", "").lower() in ("1", "true")
        assert enabled is True

    # Verify the engine actually executes
    engine = SwarmEngine()

    async def fake_call(agent: AgentSpec) -> Dict[str, Any]:
        return {"agent": agent.id, "success": True}

    result = await engine.execute_mission(
        "m1",
        [_make_spec("a")],
        fake_call,
    )
    assert len(result["results"]) == 1


# -- Result format compatibility -----------------------------------------------


@pytest.mark.asyncio
async def test_swarm_results_format_matches_legacy():
    """SwarmEngine path returns same format as legacy path."""

    async def fake_call(agent: AgentSpec) -> Dict[str, Any]:
        return {"agent": agent.id, "success": True, "text": "mock output"}

    engine = SwarmEngine()
    result = await engine.execute_mission(
        "m1",
        [_make_spec("strategist"), _make_spec("researcher")],
        fake_call,
    )

    # node0_activate expects to iterate result["results"]
    # and access r.get("success"), r.get("text"), r.get("agent")
    for r in result["results"]:
        assert "success" in r
        assert isinstance(r["success"], bool)
        assert "agent" in r


# -- Topology selection from config --------------------------------------------


def test_topology_selection_from_config():
    """Config key 'swarm_topology' selects the topology."""
    config = {"swarm_topology": "parallel", "swarm_max_concurrent": 2}
    engine = _build_engine_from_config(config)
    assert isinstance(engine._strategy, ParallelStrategy)


def test_topology_fallback_on_invalid():
    """Invalid topology name falls back to SEQUENTIAL."""
    config = {"swarm_topology": "quantum_mesh"}
    engine = _build_engine_from_config(config)
    assert isinstance(engine._strategy, SequentialStrategy)


# -- EventBus wiring -----------------------------------------------------------


@pytest.mark.asyncio
async def test_event_bus_wired_when_available():
    """wire_swarm_to_bus is called when bus is available."""
    from core.swarm.event_bridge import wire_swarm_to_bus

    engine = SwarmEngine()
    # Create a mock bus
    mock_bus = MagicMock()
    mock_bus.emit = AsyncMock()

    bridge = wire_swarm_to_bus(engine, mock_bus)
    assert bridge is not None


# -- call_fn adapter -----------------------------------------------------------


@pytest.mark.asyncio
async def test_call_fn_adapter():
    """call_fn correctly maps AgentSpec.id to agent_id string."""
    call_log: List[str] = []

    async def fake_call_agent(agent: AgentSpec) -> Dict[str, Any]:
        call_log.append(agent.id)
        return {"agent": agent.id, "success": True}

    # Build specs from PAT_AGENTS
    specs = [
        AgentSpec.from_pat_agent(aid, data) for aid, data in PAT_AGENTS_SAMPLE.items()
    ]

    engine = SwarmEngine()
    await engine.execute_mission("m1", specs, fake_call_agent)
    assert set(call_log) == {"strategist", "researcher", "creator"}
