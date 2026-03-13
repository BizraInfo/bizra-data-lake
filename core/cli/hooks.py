"""
BIZRA CLI Hooks Manager
========================

Fires pre/post command events to EventBus subscribers and records
performance telemetry for the constitutional heartbeat.

Standing on Giants:
- Deming (1950): Instrument every operation
- Boyd (1976): Observe-Orient-Decide-Act loop at CLI level
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CLIEvent:
    """Structured event emitted by CLI hooks."""

    event_type: str  # "cli.command.start" | "cli.command.end" | "cli.command.error"
    command: str
    args: List[str]
    timestamp: float = field(default_factory=time.time)
    latency_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


class CLIHooksManager:
    """
    Manages CLI lifecycle hooks with optional EventBus integration.

    Wires into CommandRegistry.add_pre_hook / add_post_hook to emit
    structured events without coupling the registry to the bus.
    """

    def __init__(self, event_bus: Optional[Any] = None) -> None:
        self._event_bus = event_bus
        self._history: List[CLIEvent] = []
        self._max_history = 500

    def pre_command(self, command: str, args: List[str]) -> None:
        """Called before command execution."""
        event = CLIEvent(
            event_type="cli.command.start",
            command=command,
            args=list(args),
        )
        self._record(event)
        self._emit(event)

    def post_command(
        self,
        command: str,
        args: List[str],
        result: Any,
        latency_ms: float,
    ) -> None:
        """Called after command execution."""
        success = getattr(result, "success", True)
        error_msg = None
        if not success:
            error_msg = getattr(result, "message", "unknown error")

        event = CLIEvent(
            event_type="cli.command.end" if success else "cli.command.error",
            command=command,
            args=list(args),
            latency_ms=latency_ms,
            success=success,
            error=error_msg,
        )
        self._record(event)
        self._emit(event)

    def _record(self, event: CLIEvent) -> None:
        """Store in local history ring buffer."""
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]

    def _emit(self, event: CLIEvent) -> None:
        """Forward event to EventBus if wired."""
        if self._event_bus is None:
            return
        try:
            payload = {
                "type": event.event_type,
                "command": event.command,
                "args": event.args,
                "timestamp": event.timestamp,
                "latency_ms": event.latency_ms,
                "success": event.success,
                "error": event.error,
            }
            # Support both publish() and emit() interfaces
            if hasattr(self._event_bus, "publish"):
                self._event_bus.publish("cli.event", payload)
            elif hasattr(self._event_bus, "emit"):
                self._event_bus.emit("cli.event", payload)
        except (RuntimeError, AttributeError, TypeError) as exc:
            logger.debug("CLI hooks: EventBus emit failed (non-fatal): %s", exc)

    @property
    def history(self) -> List[CLIEvent]:
        return list(self._history)

    @property
    def total_commands(self) -> int:
        return sum(1 for e in self._history if e.event_type == "cli.command.start")

    @property
    def total_errors(self) -> int:
        return sum(1 for e in self._history if e.event_type == "cli.command.error")
