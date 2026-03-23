"""
Event Publisher Compatibility Helpers
=====================================

Normalizes event publication across the CQRS subscriber bus
(`publish(topic, payload)` / sync) and the sovereign async bus
(`publish(Event)` or `emit(topic, payload)`).
"""

from __future__ import annotations

import inspect
from functools import lru_cache
from typing import Any


def _publish_callable_key(publish: Any) -> Any:
    """Return a stable cache key for a publish callable."""
    return (
        getattr(publish, "__func__", publish),
        inspect.ismethod(publish) and getattr(publish, "__self__", None) is not None,
    )


@lru_cache(maxsize=128)
def _cached_publish_expects_single_event(publish_key: Any) -> bool:
    """Cached signature inspection for bound/unbound publish callables."""
    publish_callable, bound_method = publish_key
    try:
        signature = inspect.signature(publish_callable)
    except (TypeError, ValueError):
        return False

    params = list(signature.parameters.values())
    has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
    if has_varargs:
        return False

    positional = [
        p
        for p in params
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    effective_positional = len(positional) - (1 if bound_method else 0)
    return effective_positional == 1


def _publish_expects_single_event(publish: Any) -> bool:
    """Return True when a bound publish() likely expects one Event object."""
    return _cached_publish_expects_single_event(_publish_callable_key(publish))


def _resolve_topic_event_dispatch(
    event_bus: Any,
    topic: str,
    payload: dict[str, Any],
) -> tuple[Any, tuple[Any, ...]]:
    """Return the callable and arguments needed to dispatch a topic event."""
    publish = getattr(event_bus, "publish", None)
    if callable(publish):
        if _publish_expects_single_event(publish):
            from core.sovereign.event_bus import Event

            return publish, (Event(topic=topic, payload=payload),)

        publish_topic: Any = topic
        try:
            from core.bus.subscribers import EventType

            if isinstance(topic, str):
                publish_topic = EventType(topic)
        except (ImportError, ValueError):
            publish_topic = topic
        return publish, (publish_topic, payload)

    emit = getattr(event_bus, "emit", None)
    if callable(emit):
        return emit, (topic, payload)

    raise TypeError("Event bus must expose publish() or emit()")


def try_publish_topic_event_sync(
    event_bus: Any,
    topic: str,
    payload: dict[str, Any],
) -> bool:
    """Try to publish synchronously, returning False if async dispatch is required."""
    if event_bus is None:
        return True

    dispatch, args = _resolve_topic_event_dispatch(event_bus, topic, payload)
    if inspect.iscoroutinefunction(dispatch):
        return False

    result = dispatch(*args)
    if inspect.isawaitable(result):
        return False
    return True


async def publish_topic_event(
    event_bus: Any,
    topic: str,
    payload: dict[str, Any],
) -> None:
    """Publish a topic/payload event across heterogeneous bus interfaces."""
    if event_bus is None:
        return

    dispatch, args = _resolve_topic_event_dispatch(event_bus, topic, payload)
    result = dispatch(*args)
    if inspect.isawaitable(result):
        await result


class FanoutEventBus:
    """Emit one topic/payload event into multiple heterogeneous buses."""

    def __init__(self, *buses: Any) -> None:
        self._buses = tuple(bus for bus in buses if bus is not None)

    async def emit(self, topic: str, payload: dict[str, Any]) -> None:
        for bus in self._buses:
            await publish_topic_event(bus, topic, payload)


def combine_event_buses(*buses: Any) -> Any:
    """Return one bus-like object spanning all provided buses."""
    active = tuple(bus for bus in buses if bus is not None)
    if not active:
        return None
    if len(active) == 1:
        return active[0]
    return FanoutEventBus(*active)
