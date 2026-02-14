"""
Tests for Sovereign Event Bus — Async Pub/Sub Messaging
========================================================
Comprehensive tests for EventPriority, Event, EventBus, and the
global singleton factory.

Standing on Giants: Observer Pattern + Async Python + Domain Events
"""

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from core.sovereign.event_bus import (
    Event,
    EventBus,
    EventPriority,
    get_event_bus,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bus() -> EventBus:
    """Fresh EventBus instance for each test."""
    return EventBus()


@pytest.fixture
def small_bus() -> EventBus:
    """EventBus with a tiny queue for boundary tests."""
    return EventBus(max_queue_size=3)


@pytest.fixture(autouse=True)
def _reset_global_bus():
    """Reset the global singleton before each test so tests stay isolated."""
    import core.sovereign.event_bus as mod

    original = mod._global_bus
    mod._global_bus = None
    yield
    mod._global_bus = original


# ---------------------------------------------------------------------------
# 1. TestEventPriority
# ---------------------------------------------------------------------------


class TestEventPriority:
    """Validate the four priority levels and their ordering."""

    def test_critical_exists(self):
        assert EventPriority.CRITICAL is not None

    def test_high_exists(self):
        assert EventPriority.HIGH is not None

    def test_normal_exists(self):
        assert EventPriority.NORMAL is not None

    def test_low_exists(self):
        assert EventPriority.LOW is not None

    def test_exactly_four_members(self):
        assert len(EventPriority) == 4

    def test_critical_less_than_high(self):
        assert EventPriority.CRITICAL.value < EventPriority.HIGH.value

    def test_high_less_than_normal(self):
        assert EventPriority.HIGH.value < EventPriority.NORMAL.value

    def test_normal_less_than_low(self):
        assert EventPriority.NORMAL.value < EventPriority.LOW.value

    def test_ascending_order(self):
        """Full ordering: CRITICAL < HIGH < NORMAL < LOW by value."""
        values = [p.value for p in EventPriority]
        assert values == sorted(values)


# ---------------------------------------------------------------------------
# 2. TestEvent
# ---------------------------------------------------------------------------


class TestEvent:
    """Validate Event dataclass defaults and custom construction."""

    def test_default_id_is_eight_chars(self):
        event = Event()
        assert len(event.id) == 8

    def test_default_id_is_hex_substring(self):
        """ID should be the first 8 chars of a UUID4 string (hex + hyphens)."""
        event = Event()
        # uuid4 format is xxxxxxxx-... so first 8 are hex
        assert all(c in "0123456789abcdef" for c in event.id)

    def test_unique_ids(self):
        ids = {Event().id for _ in range(50)}
        assert len(ids) == 50, "Event IDs must be unique across instances"

    def test_default_topic_is_empty(self):
        assert Event().topic == ""

    def test_default_payload_is_empty_dict(self):
        event = Event()
        assert event.payload == {}
        # Ensure each instance gets its own dict (no shared mutable default)
        event.payload["key"] = "value"
        assert Event().payload == {}

    def test_default_priority_is_normal(self):
        assert Event().priority == EventPriority.NORMAL

    def test_default_source_is_empty(self):
        assert Event().source == ""

    def test_default_timestamp_is_utc(self):
        before = datetime.now(timezone.utc)
        event = Event()
        after = datetime.now(timezone.utc)
        assert before <= event.timestamp <= after

    def test_default_correlation_id_is_none(self):
        assert Event().correlation_id is None

    def test_custom_fields(self):
        event = Event(
            topic="sovereign.boot",
            payload={"status": "ready"},
            priority=EventPriority.HIGH,
            source="test-module",
            correlation_id="corr-001",
        )
        assert event.topic == "sovereign.boot"
        assert event.payload == {"status": "ready"}
        assert event.priority == EventPriority.HIGH
        assert event.source == "test-module"
        assert event.correlation_id == "corr-001"

    def test_custom_id_override(self):
        event = Event(id="custom-id")
        assert event.id == "custom-id"


# ---------------------------------------------------------------------------
# 3. TestEventBusSubscribe
# ---------------------------------------------------------------------------


class TestEventBusSubscribe:
    """Tests for subscribe / unsubscribe mechanics."""

    def test_subscribe_exact_topic(self, bus: EventBus):
        handler = AsyncMock()
        bus.subscribe("topic.a", handler)
        assert handler in bus._subscribers["topic.a"]

    def test_subscribe_multiple_handlers(self, bus: EventBus):
        h1, h2 = AsyncMock(), AsyncMock()
        bus.subscribe("topic.a", h1)
        bus.subscribe("topic.a", h2)
        assert h1 in bus._subscribers["topic.a"]
        assert h2 in bus._subscribers["topic.a"]

    def test_subscribe_wildcard(self, bus: EventBus):
        handler = AsyncMock()
        bus.subscribe("sovereign.*", handler)
        # Wildcard stored under prefix "sovereign." (asterisk stripped)
        assert handler in bus._wildcard_subscribers["sovereign."]

    def test_subscribe_wildcard_does_not_pollute_exact(self, bus: EventBus):
        handler = AsyncMock()
        bus.subscribe("sovereign.*", handler)
        assert "sovereign.*" not in bus._subscribers

    def test_unsubscribe_exact(self, bus: EventBus):
        handler = AsyncMock()
        bus.subscribe("topic.a", handler)
        bus.unsubscribe("topic.a", handler)
        assert handler not in bus._subscribers["topic.a"]

    def test_unsubscribe_wildcard(self, bus: EventBus):
        handler = AsyncMock()
        bus.subscribe("sovereign.*", handler)
        bus.unsubscribe("sovereign.*", handler)
        assert handler not in bus._wildcard_subscribers["sovereign."]

    def test_unsubscribe_nonexistent_no_error(self, bus: EventBus):
        """Unsubscribing a handler that was never subscribed must not raise."""
        handler = AsyncMock()
        bus.unsubscribe("nonexistent.topic", handler)  # Should not raise

    def test_unsubscribe_wildcard_nonexistent_no_error(self, bus: EventBus):
        handler = AsyncMock()
        bus.unsubscribe("nonexistent.*", handler)  # Should not raise

    def test_duplicate_subscribe_is_idempotent(self, bus: EventBus):
        handler = AsyncMock()
        bus.subscribe("topic.a", handler)
        bus.subscribe("topic.a", handler)
        assert len(bus._subscribers["topic.a"]) == 1


# ---------------------------------------------------------------------------
# 4. TestEventBusPublish
# ---------------------------------------------------------------------------


class TestEventBusPublish:
    """Tests for the publish() method."""

    @pytest.mark.asyncio
    async def test_publish_increments_event_count(self, bus: EventBus):
        event = Event(topic="test.publish")
        await bus.publish(event)
        assert bus._event_count == 1

    @pytest.mark.asyncio
    async def test_publish_multiple_increments(self, bus: EventBus):
        for _ in range(5):
            await bus.publish(Event(topic="test.multi"))
        assert bus._event_count == 5

    @pytest.mark.asyncio
    async def test_publish_puts_event_on_queue(self, bus: EventBus):
        event = Event(topic="test.queue", priority=EventPriority.HIGH)
        await bus.publish(event)
        assert bus._event_queue.qsize() == 1

        priority_val, event_id, queued_event = await bus._event_queue.get()
        assert priority_val == EventPriority.HIGH.value
        assert event_id == event.id
        assert queued_event is event

    @pytest.mark.asyncio
    async def test_publish_queue_tuple_structure(self, bus: EventBus):
        event = Event(topic="test.struct", priority=EventPriority.CRITICAL)
        await bus.publish(event)
        item = await bus._event_queue.get()
        assert len(item) == 3
        assert item[0] == EventPriority.CRITICAL.value
        assert item[1] == event.id
        assert item[2] is event


# ---------------------------------------------------------------------------
# 5. TestEventBusEmit
# ---------------------------------------------------------------------------


class TestEventBusEmit:
    """Tests for the emit() convenience method."""

    @pytest.mark.asyncio
    async def test_emit_returns_event_id_string(self, bus: EventBus):
        event_id = await bus.emit("test.emit", {"data": 1})
        assert isinstance(event_id, str)
        assert len(event_id) == 8

    @pytest.mark.asyncio
    async def test_emit_increments_event_count(self, bus: EventBus):
        await bus.emit("test.emit", {})
        assert bus._event_count == 1

    @pytest.mark.asyncio
    async def test_emit_creates_event_with_correct_fields(self, bus: EventBus):
        await bus.emit(
            topic="sovereign.test",
            payload={"key": "val"},
            priority=EventPriority.HIGH,
            source="unit-test",
            correlation_id="corr-xyz",
        )
        _, _, event = await bus._event_queue.get()
        assert event.topic == "sovereign.test"
        assert event.payload == {"key": "val"}
        assert event.priority == EventPriority.HIGH
        assert event.source == "unit-test"
        assert event.correlation_id == "corr-xyz"

    @pytest.mark.asyncio
    async def test_emit_default_priority_is_normal(self, bus: EventBus):
        await bus.emit("test.default", {})
        _, _, event = await bus._event_queue.get()
        assert event.priority == EventPriority.NORMAL

    @pytest.mark.asyncio
    async def test_emit_default_source_is_empty(self, bus: EventBus):
        await bus.emit("test.default", {})
        _, _, event = await bus._event_queue.get()
        assert event.source == ""


# ---------------------------------------------------------------------------
# 6. TestGetHandlers
# ---------------------------------------------------------------------------


class TestGetHandlers:
    """Tests for the _get_handlers() routing logic."""

    def test_exact_match(self, bus: EventBus):
        handler = AsyncMock()
        bus.subscribe("topic.exact", handler)
        handlers = bus._get_handlers("topic.exact")
        assert handler in handlers

    def test_no_match_returns_empty(self, bus: EventBus):
        handlers = bus._get_handlers("topic.nonexistent")
        assert handlers == set()

    def test_wildcard_match(self, bus: EventBus):
        handler = AsyncMock()
        bus.subscribe("sovereign.*", handler)
        handlers = bus._get_handlers("sovereign.boot")
        assert handler in handlers

    def test_wildcard_no_match(self, bus: EventBus):
        handler = AsyncMock()
        bus.subscribe("sovereign.*", handler)
        handlers = bus._get_handlers("federation.boot")
        assert handler not in handlers

    def test_combined_exact_and_wildcard(self, bus: EventBus):
        exact_handler = AsyncMock()
        wildcard_handler = AsyncMock()
        bus.subscribe("sovereign.test", exact_handler)
        bus.subscribe("sovereign.*", wildcard_handler)

        handlers = bus._get_handlers("sovereign.test")
        assert exact_handler in handlers
        assert wildcard_handler in handlers
        assert len(handlers) == 2

    def test_multiple_wildcards_matching(self, bus: EventBus):
        h1, h2 = AsyncMock(), AsyncMock()
        bus.subscribe("sovereign.*", h1)
        bus.subscribe("sovereign.t*", h2)
        # "sovereign.test" starts with both "sovereign." and "sovereign.t"
        handlers = bus._get_handlers("sovereign.test")
        assert h1 in handlers
        assert h2 in handlers

    def test_exact_topic_does_not_match_partial(self, bus: EventBus):
        handler = AsyncMock()
        bus.subscribe("sovereign.boot", handler)
        handlers = bus._get_handlers("sovereign.boot.extra")
        assert handler not in handlers


# ---------------------------------------------------------------------------
# 7. TestProcessEvent
# ---------------------------------------------------------------------------


class TestProcessEvent:
    """Tests for _process_event() execution behavior."""

    @pytest.mark.asyncio
    async def test_handler_called_with_event(self, bus: EventBus):
        handler = AsyncMock()
        bus.subscribe("test.process", handler)

        event = Event(topic="test.process", payload={"x": 1})
        await bus._process_event(event)

        handler.assert_awaited_once_with(event)

    @pytest.mark.asyncio
    async def test_multiple_handlers_all_called(self, bus: EventBus):
        h1, h2, h3 = AsyncMock(), AsyncMock(), AsyncMock()
        bus.subscribe("test.multi", h1)
        bus.subscribe("test.multi", h2)
        bus.subscribe("test.multi", h3)

        event = Event(topic="test.multi")
        await bus._process_event(event)

        h1.assert_awaited_once_with(event)
        h2.assert_awaited_once_with(event)
        h3.assert_awaited_once_with(event)

    @pytest.mark.asyncio
    async def test_handler_exception_does_not_crash(self, bus: EventBus, caplog):
        """An exception in one handler must not prevent others from running."""
        failing_handler = AsyncMock(side_effect=ValueError("boom"))
        good_handler = AsyncMock()
        bus.subscribe("test.error", failing_handler)
        bus.subscribe("test.error", good_handler)

        event = Event(topic="test.error")
        # Must not raise
        await bus._process_event(event)

        # Good handler still called
        good_handler.assert_awaited_once_with(event)

    @pytest.mark.asyncio
    async def test_handler_exception_is_logged(self, bus: EventBus, caplog):
        failing_handler = AsyncMock(side_effect=RuntimeError("test-error-msg"))
        bus.subscribe("test.log", failing_handler)

        with caplog.at_level(logging.ERROR, logger="core.sovereign.event_bus"):
            await bus._process_event(Event(topic="test.log"))

        assert any("test-error-msg" in record.message for record in caplog.records)

    @pytest.mark.asyncio
    async def test_no_handlers_no_error(self, bus: EventBus):
        """Processing an event with zero handlers must not raise."""
        event = Event(topic="test.orphan")
        await bus._process_event(event)  # Should not raise

    @pytest.mark.asyncio
    async def test_no_handlers_does_not_increment_processed(self, bus: EventBus):
        """If no handlers exist, _processed_count should stay at 0 (early return)."""
        event = Event(topic="test.orphan")
        await bus._process_event(event)
        assert bus._processed_count == 0

    @pytest.mark.asyncio
    async def test_processed_count_incremented(self, bus: EventBus):
        handler = AsyncMock()
        bus.subscribe("test.count", handler)

        await bus._process_event(Event(topic="test.count"))
        assert bus._processed_count == 1

    @pytest.mark.asyncio
    async def test_processed_count_incremented_multiple(self, bus: EventBus):
        handler = AsyncMock()
        bus.subscribe("test.count", handler)

        for _ in range(4):
            await bus._process_event(Event(topic="test.count"))
        assert bus._processed_count == 4

    @pytest.mark.asyncio
    async def test_wildcard_handler_called_via_process(self, bus: EventBus):
        handler = AsyncMock()
        bus.subscribe("sovereign.*", handler)

        event = Event(topic="sovereign.lifecycle")
        await bus._process_event(event)

        handler.assert_awaited_once_with(event)


# ---------------------------------------------------------------------------
# 8. TestEventBusStartStop
# ---------------------------------------------------------------------------


class TestEventBusStartStop:
    """Tests for start() and stop() lifecycle."""

    @pytest.mark.asyncio
    async def test_start_sets_running(self, bus: EventBus):
        assert bus._running is False

        # Start in background, then stop after a brief delay
        task = asyncio.create_task(bus.start())
        await asyncio.sleep(0.05)
        assert bus._running is True

        bus.stop()
        await asyncio.sleep(1.2)  # Allow loop timeout to expire
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    def test_stop_sets_running_false(self, bus: EventBus):
        bus._running = True
        bus.stop()
        assert bus._running is False

    @pytest.mark.asyncio
    async def test_start_processes_queued_event(self, bus: EventBus):
        handler = AsyncMock()
        bus.subscribe("test.start", handler)

        await bus.publish(Event(topic="test.start", payload={"startup": True}))

        task = asyncio.create_task(bus.start())
        await asyncio.sleep(0.1)
        bus.stop()
        await asyncio.sleep(1.2)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        handler.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_start_stop_idempotent(self, bus: EventBus):
        """Calling stop() multiple times must not raise."""
        bus.stop()
        bus.stop()
        assert bus._running is False


# ---------------------------------------------------------------------------
# 9. TestStats
# ---------------------------------------------------------------------------


class TestStats:
    """Tests for the stats() method."""

    def test_stats_returns_dict(self, bus: EventBus):
        result = bus.stats()
        assert isinstance(result, dict)

    def test_stats_keys(self, bus: EventBus):
        expected_keys = {
            "topics",
            "wildcard_topics",
            "events_published",
            "events_processed",
            "queue_size",
            "running",
        }
        assert set(bus.stats().keys()) == expected_keys

    def test_stats_initial_values(self, bus: EventBus):
        s = bus.stats()
        assert s["topics"] == 0
        assert s["wildcard_topics"] == 0
        assert s["events_published"] == 0
        assert s["events_processed"] == 0
        assert s["queue_size"] == 0
        assert s["running"] is False

    def test_stats_tracks_topics(self, bus: EventBus):
        bus.subscribe("topic.a", AsyncMock())
        bus.subscribe("topic.b", AsyncMock())
        assert bus.stats()["topics"] == 2

    def test_stats_tracks_wildcard_topics(self, bus: EventBus):
        bus.subscribe("sov.*", AsyncMock())
        bus.subscribe("fed.*", AsyncMock())
        assert bus.stats()["wildcard_topics"] == 2

    @pytest.mark.asyncio
    async def test_stats_tracks_published(self, bus: EventBus):
        await bus.publish(Event(topic="test"))
        await bus.publish(Event(topic="test"))
        assert bus.stats()["events_published"] == 2

    @pytest.mark.asyncio
    async def test_stats_tracks_processed(self, bus: EventBus):
        handler = AsyncMock()
        bus.subscribe("test.stat", handler)
        await bus._process_event(Event(topic="test.stat"))
        assert bus.stats()["events_processed"] == 1

    @pytest.mark.asyncio
    async def test_stats_tracks_queue_size(self, bus: EventBus):
        await bus.publish(Event(topic="q1"))
        await bus.publish(Event(topic="q2"))
        assert bus.stats()["queue_size"] == 2

    def test_stats_running_reflects_state(self, bus: EventBus):
        assert bus.stats()["running"] is False
        bus._running = True
        assert bus.stats()["running"] is True


# ---------------------------------------------------------------------------
# 10. TestGetEventBus (Singleton)
# ---------------------------------------------------------------------------


class TestGetEventBus:
    """Tests for the global singleton factory."""

    def test_returns_event_bus_instance(self):
        bus = get_event_bus()
        assert isinstance(bus, EventBus)

    def test_singleton_returns_same_instance(self):
        bus1 = get_event_bus()
        bus2 = get_event_bus()
        assert bus1 is bus2

    def test_singleton_reset_gives_new_instance(self):
        import core.sovereign.event_bus as mod

        bus1 = get_event_bus()
        mod._global_bus = None
        bus2 = get_event_bus()
        assert bus1 is not bus2


# ---------------------------------------------------------------------------
# 11. TestPriorityOrdering
# ---------------------------------------------------------------------------


class TestPriorityOrdering:
    """Verify that higher-priority events are dequeued before lower ones."""

    @pytest.mark.asyncio
    async def test_critical_dequeued_before_low(self, bus: EventBus):
        """CRITICAL events must come out of the queue before LOW events."""
        low_event = Event(topic="low", priority=EventPriority.LOW)
        critical_event = Event(topic="critical", priority=EventPriority.CRITICAL)

        # Publish LOW first, then CRITICAL
        await bus.publish(low_event)
        await bus.publish(critical_event)

        # Dequeue: CRITICAL should come first despite being published second
        first = await bus._event_queue.get()
        second = await bus._event_queue.get()

        assert first[2].priority == EventPriority.CRITICAL
        assert second[2].priority == EventPriority.LOW

    @pytest.mark.asyncio
    async def test_full_priority_ordering(self, bus: EventBus):
        """Events must dequeue in CRITICAL -> HIGH -> NORMAL -> LOW order."""
        events = [
            Event(topic="low", priority=EventPriority.LOW),
            Event(topic="normal", priority=EventPriority.NORMAL),
            Event(topic="critical", priority=EventPriority.CRITICAL),
            Event(topic="high", priority=EventPriority.HIGH),
        ]
        for e in events:
            await bus.publish(e)

        dequeued = []
        while not bus._event_queue.empty():
            _, _, event = await bus._event_queue.get()
            dequeued.append(event.priority)

        expected = [
            EventPriority.CRITICAL,
            EventPriority.HIGH,
            EventPriority.NORMAL,
            EventPriority.LOW,
        ]
        assert dequeued == expected

    @pytest.mark.asyncio
    async def test_same_priority_preserves_insertion_order(self, bus: EventBus):
        """Events with identical priority should dequeue in ID-sorted order
        (PriorityQueue uses the second tuple element as tiebreaker)."""
        e1 = Event(id="aaaa0001", topic="first", priority=EventPriority.NORMAL)
        e2 = Event(id="aaaa0002", topic="second", priority=EventPriority.NORMAL)
        await bus.publish(e1)
        await bus.publish(e2)

        first = await bus._event_queue.get()
        second = await bus._event_queue.get()

        assert first[2].id == "aaaa0001"
        assert second[2].id == "aaaa0002"

    @pytest.mark.asyncio
    async def test_priority_ordering_via_process(self, bus: EventBus):
        """End-to-end: start the bus and verify handlers fire in priority order."""
        call_order: list[str] = []

        async def record_handler(event: Event):
            call_order.append(event.topic)

        bus.subscribe("p.critical", record_handler)
        bus.subscribe("p.low", record_handler)
        bus.subscribe("p.high", record_handler)

        # Publish in reverse priority order
        await bus.emit("p.low", {}, priority=EventPriority.LOW)
        await bus.emit("p.high", {}, priority=EventPriority.HIGH)
        await bus.emit("p.critical", {}, priority=EventPriority.CRITICAL)

        # Start, let it process, then stop
        task = asyncio.create_task(bus.start())
        await asyncio.sleep(0.2)
        bus.stop()
        await asyncio.sleep(1.2)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert call_order == ["p.critical", "p.high", "p.low"]


# ---------------------------------------------------------------------------
# 12. TestEventBusInit
# ---------------------------------------------------------------------------


class TestEventBusInit:
    """Tests for __init__ parameter handling."""

    def test_default_max_queue_size(self):
        bus = EventBus()
        assert bus._event_queue.maxsize == 1000

    def test_custom_max_queue_size(self):
        bus = EventBus(max_queue_size=50)
        assert bus._event_queue.maxsize == 50

    def test_initial_counters_zero(self):
        bus = EventBus()
        assert bus._event_count == 0
        assert bus._processed_count == 0

    def test_initial_running_false(self):
        bus = EventBus()
        assert bus._running is False

    def test_initial_subscribers_empty(self):
        bus = EventBus()
        assert len(bus._subscribers) == 0
        assert len(bus._wildcard_subscribers) == 0
