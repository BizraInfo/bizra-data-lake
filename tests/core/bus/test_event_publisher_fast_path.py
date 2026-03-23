from __future__ import annotations

import pytest

from core.bus.event_publisher import (
    _publish_expects_single_event,
    publish_topic_event,
    try_publish_topic_event_sync,
)
from core.sovereign.event_bus import EventBus


class _SyncTopicBus:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, str]]] = []

    def publish(self, topic: str, payload: dict[str, str]) -> None:
        self.events.append((topic, payload))


def test_publish_expects_single_event_on_bound_async_publish() -> None:
    bus = EventBus()
    assert _publish_expects_single_event(bus.publish) is True


def test_try_publish_topic_event_sync_detects_async_bus() -> None:
    bus = EventBus()
    dispatched_sync = try_publish_topic_event_sync(
        bus,
        "action.receipt",
        {"source": "test"},
    )
    assert dispatched_sync is False
    assert bus.stats()["events_published"] == 0


@pytest.mark.asyncio
async def test_publish_topic_event_publishes_on_async_bus() -> None:
    bus = EventBus()
    await publish_topic_event(bus, "action.receipt", {"source": "test"})
    assert bus.stats()["events_published"] == 1


def test_try_publish_topic_event_sync_dispatches_plain_topic_bus() -> None:
    bus = _SyncTopicBus()
    dispatched_sync = try_publish_topic_event_sync(
        bus,
        "action.receipt",
        {"source": "test"},
    )
    assert dispatched_sync is True
    assert bus.events == [("action.receipt", {"source": "test"})]
