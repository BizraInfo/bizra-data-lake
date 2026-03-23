"""Tests for the event publisher compatibility layer."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.bus.event_publisher import (
    FanoutEventBus,
    _publish_expects_single_event,
    combine_event_buses,
    publish_topic_event,
)


class TestPublishExpectsSingleEvent:
    def test_single_positional_param_returns_true(self) -> None:
        def publish(event: object) -> None:
            pass

        assert _publish_expects_single_event(publish) is True

    def test_two_positional_params_returns_false(self) -> None:
        def publish(topic: str, payload: dict) -> None:
            pass

        assert _publish_expects_single_event(publish) is False

    def test_varargs_returns_false(self) -> None:
        def publish(*args: Any) -> None:
            pass

        assert _publish_expects_single_event(publish) is False

    def test_uninspectable_returns_false(self) -> None:
        assert _publish_expects_single_event(42) is False


class TestPublishTopicEvent:
    @pytest.mark.asyncio
    async def test_none_bus_is_noop(self) -> None:
        await publish_topic_event(None, "test.topic", {"key": "val"})

    @pytest.mark.asyncio
    async def test_emit_bus(self) -> None:
        bus = MagicMock()
        bus.publish = None
        bus.emit = MagicMock(return_value=None)
        await publish_topic_event(bus, "test.topic", {"k": "v"})
        bus.emit.assert_called_once_with("test.topic", {"k": "v"})

    @pytest.mark.asyncio
    async def test_async_emit_bus(self) -> None:
        bus = MagicMock()
        bus.publish = None
        bus.emit = AsyncMock(return_value=None)
        await publish_topic_event(bus, "test.topic", {"k": "v"})
        bus.emit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_two_arg_publish_bus(self) -> None:
        calls = []

        def publish(topic: str, payload: dict) -> None:
            calls.append((topic, payload))

        bus = MagicMock()
        bus.publish = publish
        bus.emit = None
        await publish_topic_event(bus, "action.receipt", {"a": 1})
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_single_arg_publish_uses_event(self) -> None:
        from core.sovereign.event_bus import EventBus

        bus = EventBus()
        await publish_topic_event(bus, "test.topic", {"k": "v"})
        assert bus.stats()["events_published"] == 1

    @pytest.mark.asyncio
    async def test_no_publish_or_emit_raises(self) -> None:
        bus = MagicMock(spec=[])  # no publish or emit
        with pytest.raises(TypeError, match="publish.*emit"):
            await publish_topic_event(bus, "t", {})


class TestFanoutEventBus:
    @pytest.mark.asyncio
    async def test_fanout_reaches_all_buses(self) -> None:
        bus1 = MagicMock()
        bus1.publish = None
        bus1.emit = AsyncMock()
        bus2 = MagicMock()
        bus2.publish = None
        bus2.emit = AsyncMock()

        fanout = FanoutEventBus(bus1, bus2)
        await fanout.emit("topic", {"k": "v"})

        bus1.emit.assert_awaited_once()
        bus2.emit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_fanout_skips_none(self) -> None:
        bus1 = MagicMock()
        bus1.publish = None
        bus1.emit = AsyncMock()

        fanout = FanoutEventBus(bus1, None)
        await fanout.emit("topic", {"k": "v"})
        bus1.emit.assert_awaited_once()


class TestCombineEventBuses:
    def test_no_buses_returns_none(self) -> None:
        assert combine_event_buses() is None
        assert combine_event_buses(None, None) is None

    def test_single_bus_returns_itself(self) -> None:
        bus = MagicMock()
        assert combine_event_buses(bus) is bus
        assert combine_event_buses(None, bus, None) is bus

    def test_multiple_buses_returns_fanout(self) -> None:
        bus1 = MagicMock()
        bus2 = MagicMock()
        result = combine_event_buses(bus1, bus2)
        assert isinstance(result, FanoutEventBus)
        assert len(result._buses) == 2
