"""
Event Bus — Async Pub/Sub Messaging for Sovereign Components
============================================================
Enables decoupled communication between sovereign modules through
typed events and topic-based subscriptions.

Standing on Giants: Observer Pattern + Async Python + Domain Events
"""

import asyncio
import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Callable, Coroutine, Dict, Optional, Set

logger = logging.getLogger(__name__)


class EventPriority(Enum):
    """Event processing priority."""

    CRITICAL = auto()  # Process immediately
    HIGH = auto()  # Process next
    NORMAL = auto()  # Standard queue
    LOW = auto()  # When available


@dataclass
class Event:
    """Base event structure for the sovereign bus."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    topic: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: EventPriority = EventPriority.NORMAL
    source: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    correlation_id: Optional[str] = None  # For tracking event chains


# Type alias for event handlers
EventHandler = Callable[[Event], Coroutine[Any, Any, None]]


class EventBus:
    """
    Async pub/sub event bus for sovereign component communication.

    Features:
    - Topic-based subscriptions
    - Priority-based processing
    - Wildcard subscriptions (e.g., "sovereign.*")
    - Event correlation tracking
    """

    def __init__(self, max_queue_size: int = 1000):
        self._subscribers: Dict[str, Set[EventHandler]] = defaultdict(set)
        self._wildcard_subscribers: Dict[str, Set[EventHandler]] = defaultdict(set)
        self._event_queue: asyncio.PriorityQueue = asyncio.PriorityQueue(
            maxsize=max_queue_size
        )
        self._running = False
        self._event_count = 0
        self._processed_count = 0

    def subscribe(self, topic: str, handler: EventHandler) -> None:
        """Subscribe to a topic. Use '*' for wildcard (e.g., 'sovereign.*')."""
        if "*" in topic:
            prefix = topic.replace("*", "")
            self._wildcard_subscribers[prefix].add(handler)
            logger.debug(f"Wildcard subscription: {topic}")
        else:
            self._subscribers[topic].add(handler)
            logger.debug(f"Subscription: {topic}")

    def unsubscribe(self, topic: str, handler: EventHandler) -> None:
        """Unsubscribe from a topic."""
        if "*" in topic:
            prefix = topic.replace("*", "")
            self._wildcard_subscribers[prefix].discard(handler)
        else:
            self._subscribers[topic].discard(handler)

    async def publish(self, event: Event) -> None:
        """Publish an event to the bus."""
        # Priority queue uses (priority_value, event_id, event)
        priority_val = event.priority.value
        await self._event_queue.put((priority_val, event.id, event))
        self._event_count += 1

    async def emit(
        self,
        topic: str,
        payload: Dict[str, Any],
        priority: EventPriority = EventPriority.NORMAL,
        source: str = "",
        correlation_id: Optional[str] = None,
    ) -> str:
        """Convenience method to create and publish an event."""
        event = Event(
            topic=topic,
            payload=payload,
            priority=priority,
            source=source,
            correlation_id=correlation_id,
        )
        await self.publish(event)
        return event.id

    def _get_handlers(self, topic: str) -> Set[EventHandler]:
        """Get all handlers for a topic including wildcards."""
        handlers = set(self._subscribers.get(topic, set()))

        # Check wildcard subscriptions
        for prefix, wildcard_handlers in self._wildcard_subscribers.items():
            if topic.startswith(prefix):
                handlers.update(wildcard_handlers)

        return handlers

    async def _process_event(self, event: Event) -> None:
        """Process a single event."""
        handlers = self._get_handlers(event.topic)

        if not handlers:
            logger.debug(f"No handlers for topic: {event.topic}")
            return

        # Execute all handlers concurrently
        tasks = [handler(event) for handler in handlers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Log any errors
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Event handler error: {result}")

        self._processed_count += 1

    async def start(self) -> None:
        """Start the event processing loop."""
        self._running = True
        logger.info("Event bus started")

        while self._running:
            try:
                # Wait for next event with timeout
                _, _, event = await asyncio.wait_for(
                    self._event_queue.get(), timeout=1.0
                )
                await self._process_event(event)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Event bus error: {e}")

    def stop(self) -> None:
        """Stop the event processing loop."""
        self._running = False
        logger.info("Event bus stopped")

    def stats(self) -> Dict[str, Any]:
        """Get event bus statistics."""
        return {
            "topics": len(self._subscribers),
            "wildcard_topics": len(self._wildcard_subscribers),
            "events_published": self._event_count,
            "events_processed": self._processed_count,
            "queue_size": self._event_queue.qsize(),
            "running": self._running,
        }


# Global event bus instance
_global_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get or create the global event bus."""
    global _global_bus
    if _global_bus is None:
        _global_bus = EventBus()
    return _global_bus


__all__ = [
    "Event",
    "EventBus",
    "EventHandler",
    "EventPriority",
    "get_event_bus",
]
