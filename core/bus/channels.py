"""
Channel Executor Protocol — Action Dispatch Abstraction
═══════════════════════════════════════════════════════

Protocol for channel executors and result type.
ActionBus dispatches actions to channels by name.

Standing on Giants:
- Hewitt (1973): Actor model
- Fowler (2005): CQRS pattern

Phase 68.01 — Sovereign Instantiation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from core.bus.types import ActionEnvelope


@dataclass
class ChannelResult:
    """Result of executing an action through a channel."""

    success: bool
    outcome_hash: str = ""  # blake3 hex of post-state
    ihsan_score: float = 0.0
    artifacts: list[str] = field(default_factory=list)
    reason: str = ""


@runtime_checkable
class ChannelExecutor(Protocol):
    """Any channel that can execute actions."""

    async def execute(self, action: ActionEnvelope) -> ChannelResult: ...
