"""SwarmEventBridge — Translates SwarmEngine events into EventBus publications.

ADR-004 Phase 3: Bridges the SwarmEngine internal event system with the
sovereign EventBus, enabling any component subscribed to ``swarm.*`` topics
to observe swarm lifecycle without coupling to the engine directly.

Standing on Giants: Gamma (Observer, 1994) . Hohpe (EIP, 2003) . Lamport (distributed state)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from core.sovereign.event_bus import EventBus, EventPriority
from core.swarm.engine import SwarmEngine
from core.swarm.types import SwarmEvent, SwarmEventKind

logger = logging.getLogger("SwarmEventBridge")

# -- Priority mapping ----------------------------------------------------------

_PRIORITY_MAP = {
    SwarmEventKind.AGENT_FAILED: EventPriority.HIGH,
    SwarmEventKind.EQUALIZER_ACTION: EventPriority.HIGH,
    SwarmEventKind.SWARM_CREATED: EventPriority.NORMAL,
    SwarmEventKind.AGENT_STARTED: EventPriority.NORMAL,
    SwarmEventKind.AGENT_COMPLETED: EventPriority.NORMAL,
    SwarmEventKind.MODEL_PRELOADED: EventPriority.LOW,
    SwarmEventKind.PHASE_CHANGED: EventPriority.NORMAL,
    SwarmEventKind.MISSION_COMPLETE: EventPriority.NORMAL,
}


class SwarmEventBridge:
    """Translates SwarmEngine events into EventBus publications.

    Usage::

        bus = EventBus()
        engine = SwarmEngine(config=cfg)
        bridge = SwarmEventBridge(engine, bus)
        # Now any bus.subscribe("swarm.*", handler) receives swarm events
    """

    def __init__(self, engine: SwarmEngine, bus: EventBus):
        self._engine = engine
        self._bus = bus
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        engine.on_event(self._on_swarm_event)

    def _on_swarm_event(self, swarm_event: SwarmEvent) -> None:
        """Sync callback from SwarmEngine — schedule async publish."""
        try:
            if self._loop is None or self._loop.is_closed():
                self._loop = asyncio.get_running_loop()
            self._loop.create_task(self._publish(swarm_event))
        except RuntimeError:
            logger.debug("No event loop for bridge publish")

    async def _publish(self, swarm_event: SwarmEvent) -> None:
        """Convert SwarmEvent to Event and publish to bus."""
        topic = f"swarm.{swarm_event.kind.value}"
        priority = _PRIORITY_MAP.get(swarm_event.kind, EventPriority.NORMAL)

        payload = {
            "swarm_id": swarm_event.swarm_id,
            "timestamp": swarm_event.timestamp,
            **swarm_event.data,
        }
        if swarm_event.agent_id:
            payload["agent_id"] = swarm_event.agent_id

        try:
            await self._bus.emit(
                topic=topic,
                payload=payload,
                priority=priority,
                source="SwarmEngine",
                correlation_id=swarm_event.swarm_id,
            )
        except (asyncio.CancelledError, RuntimeError, OSError) as e:  # SEC-003 — async boundary
            logger.debug("Bridge publish failed: %s", e)


# -- Factory function ----------------------------------------------------------


def wire_swarm_to_bus(
    engine: SwarmEngine,
    bus: Optional[EventBus] = None,
) -> Optional[SwarmEventBridge]:
    """Wire a SwarmEngine to the sovereign EventBus.

    Returns the bridge instance, or None if bus is unavailable.
    """
    if bus is None:
        return None
    return SwarmEventBridge(engine, bus)
