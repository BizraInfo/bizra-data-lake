"""Tests for core.swarm.event_bridge — Phase 3 EventBus bridge.

ADR-004 Phase 5 validation: 5 unit tests for the event bridge.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List
from unittest.mock import AsyncMock

import pytest

from core.sovereign.event_bus import Event, EventBus, EventPriority
from core.swarm.engine import SwarmEngine
from core.swarm.event_bridge import SwarmEventBridge, wire_swarm_to_bus
from core.swarm.types import (
    AgentRole,
    AgentSpec,
)


def _make_spec(agent_id: str) -> AgentSpec:
    return AgentSpec(id=agent_id, role=AgentRole.RESEARCHER, model_purpose="reasoning")


async def _fake_call(agent: AgentSpec) -> Dict[str, Any]:
    return {"agent": agent.id, "success": True}


@pytest.mark.asyncio
async def test_bridge_publishes_to_bus():
    """SwarmEvent is translated and published to EventBus."""
    bus = EventBus()
    engine = SwarmEngine()
    _bridge = SwarmEventBridge(engine, bus)

    received: List[Event] = []

    async def handler(event: Event) -> None:
        received.append(event)

    bus.subscribe("swarm.agent_started", handler)

    # Start the bus processing loop
    bus._running = True
    process_task = asyncio.create_task(_drain_bus(bus))

    await engine.execute_mission("m1", [_make_spec("a1")], _fake_call)
    await asyncio.sleep(0.1)

    bus._running = False
    process_task.cancel()
    try:
        await process_task
    except asyncio.CancelledError:
        pass

    # Should have received at least one agent_started event
    started_events = [e for e in received if e.topic == "swarm.agent_started"]
    assert len(started_events) >= 1
    assert started_events[0].payload["agent_id"] == "a1"


@pytest.mark.asyncio
async def test_bridge_correlation_id():
    """swarm_id becomes correlation_id on the bus Event."""
    bus = EventBus()
    engine = SwarmEngine()
    _bridge = SwarmEventBridge(engine, bus)

    received: List[Event] = []

    async def handler(event: Event) -> None:
        received.append(event)

    bus.subscribe("swarm.*", handler)

    bus._running = True
    process_task = asyncio.create_task(_drain_bus(bus))

    await engine.execute_mission("mission-42", [_make_spec("a1")], _fake_call)
    await asyncio.sleep(0.1)

    bus._running = False
    process_task.cancel()
    try:
        await process_task
    except asyncio.CancelledError:
        pass

    assert len(received) > 0
    for evt in received:
        assert evt.correlation_id == "mission-42"


@pytest.mark.asyncio
async def test_bridge_priority_mapping():
    """AGENT_FAILED maps to HIGH priority."""
    bus = EventBus()
    engine = SwarmEngine()
    _bridge = SwarmEventBridge(engine, bus)

    received: List[Event] = []

    async def handler(event: Event) -> None:
        received.append(event)

    bus.subscribe("swarm.agent_failed", handler)

    bus._running = True
    process_task = asyncio.create_task(_drain_bus(bus))

    async def failing_call(agent: AgentSpec) -> Dict[str, Any]:
        raise RuntimeError("timeout")

    await engine.execute_mission("m1", [_make_spec("bad")], failing_call)
    await asyncio.sleep(0.1)

    bus._running = False
    process_task.cancel()
    try:
        await process_task
    except asyncio.CancelledError:
        pass

    failed_events = [e for e in received if e.topic == "swarm.agent_failed"]
    assert len(failed_events) >= 1
    assert failed_events[0].priority == EventPriority.HIGH


def test_bridge_degrades_without_bus():
    """wire_swarm_to_bus returns None when bus is None."""
    engine = SwarmEngine()
    bridge = wire_swarm_to_bus(engine, bus=None)
    assert bridge is None


@pytest.mark.asyncio
async def test_bridge_does_not_break_engine():
    """If bus.emit raises, engine continues normally."""
    bus = EventBus()
    # Make emit always fail
    bus.emit = AsyncMock(side_effect=RuntimeError("bus down"))

    engine = SwarmEngine()
    _bridge = SwarmEventBridge(engine, bus)

    result = await engine.execute_mission("m1", [_make_spec("a1")], _fake_call)
    assert len(result["results"]) == 1
    assert result["results"][0]["success"] is True


# -- Helper to drain the EventBus queue ----------------------------------------


async def _drain_bus(bus: EventBus) -> None:
    """Process events from the bus queue until cancelled."""
    while True:
        try:
            _, _, event = await asyncio.wait_for(
                bus._event_queue.get(),
                timeout=0.05,
            )
            await bus._process_event(event)
        except asyncio.TimeoutError:
            await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            break
