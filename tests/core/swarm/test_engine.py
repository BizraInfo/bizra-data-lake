"""Tests for core.swarm.engine — Phase 2 SwarmEngine.

ADR-004 Phase 5 validation: 11 unit tests for the execution engine.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

import pytest

from core.swarm.engine import (
    SwarmEngine,
)
from core.swarm.types import (
    AgentRole,
    AgentSpec,
    SwarmConfig,
    SwarmEventKind,
    SwarmTopology,
)

# -- Fixtures ------------------------------------------------------------------


def _make_spec(agent_id: str, role: AgentRole = AgentRole.RESEARCHER) -> AgentSpec:
    return AgentSpec(id=agent_id, role=role, model_purpose="reasoning")


async def _fake_call(agent: AgentSpec) -> Dict[str, Any]:
    return {"agent": agent.id, "success": True, "text": "ok"}


class MockModelRouter:
    """Implements ModelRouterProtocol for testing."""

    def __init__(self, *, should_fail: bool = False):
        self.preload_called = False
        self.preload_agent_ids: Optional[List[str]] = None
        self.equalizer_called = False
        self._should_fail = should_fail

    async def preload_mission_fleet(
        self,
        agent_ids: List[str],
        config: Dict[str, Any],
    ) -> Dict[str, bool]:
        if self._should_fail:
            raise RuntimeError("preload failed")
        self.preload_called = True
        self.preload_agent_ids = agent_ids
        return {aid: True for aid in agent_ids}

    async def check_equalizer(
        self,
        ihsan_score: float,
        backlog: int,
        presence: int,
    ) -> Optional[str]:
        self.equalizer_called = True
        return "STEADY"


# -- Sequential strategy -------------------------------------------------------


@pytest.mark.asyncio
async def test_sequential_execution_order():
    """Sequential strategy executes agents in order."""
    call_order: List[str] = []

    async def tracked_call(agent: AgentSpec) -> Dict[str, Any]:
        call_order.append(agent.id)
        return {"agent": agent.id, "success": True}

    engine = SwarmEngine(config=SwarmConfig(topology=SwarmTopology.SEQUENTIAL))
    result = await engine.execute_mission(
        "m1",
        [_make_spec("a"), _make_spec("b"), _make_spec("c")],
        tracked_call,
    )
    assert call_order == ["a", "b", "c"]
    assert len(result["results"]) == 3


# -- Parallel strategy ----------------------------------------------------------


@pytest.mark.asyncio
async def test_parallel_execution_concurrent():
    """Parallel strategy runs agents concurrently."""
    engine = SwarmEngine(
        config=SwarmConfig(topology=SwarmTopology.PARALLEL, max_concurrent=5),
    )
    agents = [_make_spec(f"a{i}") for i in range(5)]

    async def slow_call(agent: AgentSpec) -> Dict[str, Any]:
        await asyncio.sleep(0.05)
        return {"agent": agent.id, "success": True}

    t0 = time.monotonic()
    result = await engine.execute_mission("m1", agents, slow_call)
    elapsed = time.monotonic() - t0

    assert len(result["results"]) == 5
    # 5 agents x 0.05s sequential = 0.25s; parallel should be ~0.05s
    assert elapsed < 0.2


@pytest.mark.asyncio
async def test_parallel_bounded_by_semaphore():
    """Max concurrent is respected."""
    concurrent_count = 0
    max_seen = 0

    async def counting_call(agent: AgentSpec) -> Dict[str, Any]:
        nonlocal concurrent_count, max_seen
        concurrent_count += 1
        max_seen = max(max_seen, concurrent_count)
        await asyncio.sleep(0.03)
        concurrent_count -= 1
        return {"agent": agent.id, "success": True}

    engine = SwarmEngine(
        config=SwarmConfig(topology=SwarmTopology.PARALLEL, max_concurrent=2),
    )
    agents = [_make_spec(f"a{i}") for i in range(6)]
    await engine.execute_mission("m1", agents, counting_call)
    assert max_seen <= 2


# -- Hierarchical mesh strategy ------------------------------------------------


@pytest.mark.asyncio
async def test_hierarchical_mesh_coordinator_first():
    """Hierarchical mesh runs coordinator before workers."""
    call_order: List[str] = []

    async def tracked_call(agent: AgentSpec) -> Dict[str, Any]:
        call_order.append(agent.id)
        return {"agent": agent.id, "success": True}

    coordinator = _make_spec("coord", AgentRole.COORDINATOR)
    workers = [_make_spec("w1"), _make_spec("w2")]

    engine = SwarmEngine(
        config=SwarmConfig(topology=SwarmTopology.HIERARCHICAL_MESH),
    )
    await engine.execute_mission("hm", [coordinator, *workers], tracked_call)

    assert call_order[0] == "coord"
    assert call_order[-1] == "coord"  # Synthesis pass
    assert set(call_order[1:-1]) == {"w1", "w2"}


# -- Event emission ------------------------------------------------------------


@pytest.mark.asyncio
async def test_event_emission():
    """Engine emits events at each phase transition."""
    events: List[SwarmEventKind] = []
    engine = SwarmEngine()
    engine.on_event(lambda evt: events.append(evt.kind))

    await engine.execute_mission("m1", [_make_spec("a")], _fake_call)

    assert SwarmEventKind.SWARM_CREATED in events
    assert SwarmEventKind.PHASE_CHANGED in events
    assert SwarmEventKind.MISSION_COMPLETE in events
    assert SwarmEventKind.AGENT_STARTED in events
    assert SwarmEventKind.AGENT_COMPLETED in events


# -- Pre-load integration ------------------------------------------------------


@pytest.mark.asyncio
async def test_preload_integration():
    """Engine calls AutoModelRouter.preload_mission_fleet."""
    router = MockModelRouter()
    engine = SwarmEngine(
        config=SwarmConfig(preload_models=True),
        model_router=router,
    )
    await engine.execute_mission("m1", [_make_spec("a")], _fake_call, model_config={})
    assert router.preload_called is True
    assert router.preload_agent_ids == ["a"]


@pytest.mark.asyncio
async def test_graceful_degradation_on_preload_failure():
    """Engine continues if pre-load fails."""
    router = MockModelRouter(should_fail=True)
    engine = SwarmEngine(
        config=SwarmConfig(preload_models=True),
        model_router=router,
    )
    result = await engine.execute_mission("m1", [_make_spec("a")], _fake_call)
    assert len(result["results"]) == 1
    assert result["results"][0]["success"] is True


# -- Failure handling ----------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_failure_does_not_halt_swarm():
    """One agent failure doesn't stop the entire swarm."""

    async def failing_call(agent: AgentSpec) -> Dict[str, Any]:
        if agent.id == "bad":
            raise RuntimeError("boom")
        return {"agent": agent.id, "success": True}

    engine = SwarmEngine()
    result = await engine.execute_mission(
        "m1",
        [_make_spec("good1"), _make_spec("bad"), _make_spec("good2")],
        failing_call,
    )
    successes = sum(1 for r in result["results"] if r.get("success"))
    failures = sum(1 for r in result["results"] if not r.get("success"))
    assert successes == 2
    assert failures == 1


@pytest.mark.asyncio
async def test_result_flattening():
    """Exceptions from gather (parallel) become dicts."""

    async def raise_call(agent: AgentSpec) -> Dict[str, Any]:
        raise ValueError("oops")

    engine = SwarmEngine(
        config=SwarmConfig(topology=SwarmTopology.PARALLEL),
    )
    result = await engine.execute_mission("m1", [_make_spec("a")], raise_call)
    assert len(result["results"]) == 1
    assert result["results"][0]["success"] is False


# -- Equalizer integration -----------------------------------------------------


@pytest.mark.asyncio
async def test_equalizer_integration():
    """Equalizer action included in result."""
    router = MockModelRouter()
    engine = SwarmEngine(
        config=SwarmConfig(equalizer_enabled=True),
        model_router=router,
    )
    result = await engine.execute_mission("m1", [_make_spec("a")], _fake_call)
    assert router.equalizer_called is True
    assert result["eq_action"] == "STEADY"
