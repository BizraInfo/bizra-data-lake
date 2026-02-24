# Phase 3: Event Bus Bridge — SwarmEngine ↔ EventBus Integration

> ADR-004 | V3 Unified Swarm Coordination Engine
> Standing on Giants: Gamma (Observer pattern, 1994) · Hohpe (Enterprise Integration, 2003) · Lamport (distributed state, 1978)

## 3.1 Problem Statement

`SwarmEngine` (Phase 2) emits `SwarmEvent` objects through its internal listener
list. The existing `EventBus` (`core/sovereign/event_bus.py`) provides topic-based
async pub/sub with priority queues and wildcard subscriptions.

Today these two systems are disconnected:
- SwarmEngine uses `EventListener = Callable[[SwarmEvent], None]` (sync, fire-and-forget)
- EventBus uses `EventHandler = Callable[[Event], Coroutine]` (async, topic-based)

The bridge translates SwarmEvents into EventBus Events, enabling any component
subscribed to `swarm.*` topics to observe swarm lifecycle without coupling to
the engine directly.

## 3.2 Requirements

- Bridge registers itself as a SwarmEngine event listener
- Translates `SwarmEvent` → `Event` with topic `swarm.<event_kind>`
- Preserves `swarm_id` as `correlation_id` for event chain tracking
- Maps `SwarmEventKind` to `EventPriority` (failures → HIGH, rest → NORMAL)
- Bridge must not block the SwarmEngine (fire-and-forget async publish)
- If EventBus is unavailable, bridge degrades silently (no crash)
- Bridge is optional — SwarmEngine works without it

## 3.3 Pseudocode: `core/swarm/event_bridge.py`

```
IMPORT asyncio
IMPORT logging
IMPORT Optional FROM typing

IMPORT SwarmEvent, SwarmEventKind FROM core.swarm.types
IMPORT Event, EventBus, EventPriority FROM core.sovereign.event_bus
IMPORT SwarmEngine FROM core.swarm.engine

logger = logging.getLogger("SwarmEventBridge")

# ── Priority mapping ──

_PRIORITY_MAP = {
    SwarmEventKind.AGENT_FAILED:     EventPriority.HIGH,
    SwarmEventKind.EQUALIZER_ACTION: EventPriority.HIGH,
    SwarmEventKind.SWARM_CREATED:    EventPriority.NORMAL,
    SwarmEventKind.AGENT_STARTED:    EventPriority.NORMAL,
    SwarmEventKind.AGENT_COMPLETED:  EventPriority.NORMAL,
    SwarmEventKind.MODEL_PRELOADED:  EventPriority.LOW,
    SwarmEventKind.PHASE_CHANGED:    EventPriority.NORMAL,
    SwarmEventKind.MISSION_COMPLETE: EventPriority.NORMAL,
}


CLASS SwarmEventBridge:
    """Translates SwarmEngine events into EventBus publications.

    Usage:
        bus = EventBus()
        engine = SwarmEngine(config=cfg)
        bridge = SwarmEventBridge(engine, bus)
        # Now any bus.subscribe("swarm.*", handler) receives swarm events
    """

    FUNCTION __init__(
        self,
        engine: SwarmEngine,
        bus: EventBus,
    ):
        self._engine = engine
        self._bus = bus
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Register ourselves as a SwarmEngine listener
        engine.on_event(self._on_swarm_event)

    FUNCTION _on_swarm_event(self, swarm_event: SwarmEvent):
        """Sync callback from SwarmEngine — schedule async publish."""
        TRY:
            # Get or cache the event loop
            IF self._loop IS None OR self._loop.is_closed():
                self._loop = asyncio.get_running_loop()

            # Schedule publish as fire-and-forget task
            self._loop.create_task(self._publish(swarm_event))
        EXCEPT RuntimeError:
            # No running event loop — degrade silently
            logger.debug("No event loop for bridge publish")

    ASYNC FUNCTION _publish(self, swarm_event: SwarmEvent):
        """Convert SwarmEvent to Event and publish to bus."""
        topic = f"swarm.{swarm_event.kind.value}"
        priority = _PRIORITY_MAP.get(
            swarm_event.kind, EventPriority.NORMAL
        )

        payload = {
            "swarm_id": swarm_event.swarm_id,
            "timestamp": swarm_event.timestamp,
            **({"agent_id": swarm_event.agent_id} IF swarm_event.agent_id ELSE {}),
            **swarm_event.data,
        }

        TRY:
            AWAIT self._bus.emit(
                topic=topic,
                payload=payload,
                priority=priority,
                source="SwarmEngine",
                correlation_id=swarm_event.swarm_id,
            )
        EXCEPT Exception AS e:
            logger.debug("Bridge publish failed: %s", e)


# ── Factory function ──

FUNCTION wire_swarm_to_bus(
    engine: SwarmEngine,
    bus: Optional[EventBus] = None,
) -> Optional[SwarmEventBridge]:
    """Wire a SwarmEngine to the sovereign EventBus.

    Returns the bridge instance, or None if bus is unavailable.
    Callers don't need to hold the reference — the bridge
    registers itself as an engine listener.
    """
    IF bus IS None:
        RETURN None
    RETURN SwarmEventBridge(engine, bus)
```

## 3.4 Topic Taxonomy

All swarm events publish under the `swarm.*` namespace:

| SwarmEventKind | Topic | Priority | Payload Keys |
|---|---|---|---|
| SWARM_CREATED | `swarm.swarm_created` | NORMAL | `swarm_id`, `agent_count` |
| AGENT_STARTED | `swarm.agent_started` | NORMAL | `swarm_id`, `agent_id` |
| AGENT_COMPLETED | `swarm.agent_completed` | NORMAL | `swarm_id`, `agent_id`, result data |
| AGENT_FAILED | `swarm.agent_failed` | HIGH | `swarm_id`, `agent_id`, `error` |
| MODEL_PRELOADED | `swarm.model_preloaded` | LOW | `swarm_id`, fleet status |
| EQUALIZER_ACTION | `swarm.equalizer_action` | HIGH | `swarm_id`, `action` |
| PHASE_CHANGED | `swarm.phase_changed` | NORMAL | `swarm_id`, `from`, `to` |
| MISSION_COMPLETE | `swarm.mission_complete` | NORMAL | `swarm_id`, `results_count` |

## 3.5 TDD Anchors

```
# tests/core/swarm/test_event_bridge.py

TEST test_bridge_publishes_to_bus():
    """SwarmEvent is translated and published to EventBus."""
    bus = EventBus()
    engine = SwarmEngine()
    bridge = SwarmEventBridge(engine, bus)

    received = []
    ASYNC FUNCTION handler(event):
        received.append(event)

    bus.subscribe("swarm.agent_started", handler)

    # Trigger a swarm event
    engine._emit(SwarmEventKind.AGENT_STARTED, agent_id="a1")

    # Allow async publish to complete
    AWAIT asyncio.sleep(0.01)

    ASSERT len(received) == 1
    ASSERT received[0].payload["agent_id"] == "a1"

TEST test_bridge_correlation_id():
    """swarm_id becomes correlation_id on the bus Event."""
    bus = EventBus()
    engine = SwarmEngine()
    bridge = SwarmEventBridge(engine, bus)

    received = []
    bus.subscribe("swarm.*", LAMBDA evt: received.append(evt))

    engine._swarm_id = "mission-42"
    engine._emit(SwarmEventKind.SWARM_CREATED, data={"agent_count": 3})
    AWAIT asyncio.sleep(0.01)

    ASSERT received[0].correlation_id == "mission-42"

TEST test_bridge_priority_mapping():
    """AGENT_FAILED maps to HIGH priority."""
    bus = EventBus()
    engine = SwarmEngine()
    bridge = SwarmEventBridge(engine, bus)

    received = []
    bus.subscribe("swarm.agent_failed", LAMBDA evt: received.append(evt))

    engine._emit(SwarmEventKind.AGENT_FAILED, agent_id="bad",
                 data={"error": "timeout"})
    AWAIT asyncio.sleep(0.01)

    ASSERT received[0].priority == EventPriority.HIGH

TEST test_bridge_degrades_without_bus():
    """wire_swarm_to_bus returns None when bus is None."""
    engine = SwarmEngine()
    bridge = wire_swarm_to_bus(engine, bus=None)
    ASSERT bridge IS None

TEST test_bridge_does_not_break_engine():
    """If bus.emit raises, engine continues normally."""
    failing_bus = EventBus()
    failing_bus.emit = ASYNC FUNCTION(*a, **kw): RAISE RuntimeError("bus down")

    engine = SwarmEngine()
    bridge = SwarmEventBridge(engine, failing_bus)

    # Engine should still execute without error
    result = AWAIT engine.execute_mission("m1", [agent], fake_call)
    ASSERT len(result["results"]) == 1
```

## 3.6 Compatibility Notes

- `EventBus.emit()` already supports `correlation_id` — no changes needed
- Wildcard subscription `swarm.*` catches all swarm events in one handler
- The bridge is purely additive — existing EventBus subscribers are unaffected
- `asyncio.get_running_loop()` is available in Python 3.10+ (our minimum is 3.11)
