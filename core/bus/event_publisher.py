"""
Event Publisher Compatibility Helpers
=====================================

Normalizes event publication across the CQRS subscriber bus
(`publish(topic, payload)` / sync) and the sovereign async bus
(`publish(Event)` or `emit(topic, payload)`).
"""

from __future__ import annotations

import inspect
from typing import Any


def _publish_expects_single_event(publish: Any) -> bool:
    """Return True when a bound publish() likely expects one Event object."""
    try:
        signature = inspect.signature(publish)
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
    return len(positional) == 1


async def publish_topic_event(
    event_bus: Any,
    topic: str,
    payload: dict[str, Any],
) -> None:
    """Publish a topic/payload event across heterogeneous bus interfaces."""
    if event_bus is None:
        return

    publish = getattr(event_bus, "publish", None)
    if callable(publish):
        if _publish_expects_single_event(publish):
            from core.sovereign.event_bus import Event

            result = publish(Event(topic=topic, payload=payload))
        else:
            result = publish(topic, payload)

        if inspect.isawaitable(result):
            await result
        return

    emit = getattr(event_bus, "emit", None)
    if callable(emit):
        result = emit(topic, payload)
        if inspect.isawaitable(result):
            await result
        return

    raise TypeError("Event bus must expose publish() or emit()")
