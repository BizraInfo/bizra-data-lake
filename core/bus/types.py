"""
Bus Types — Frozen Envelopes for CQRS Command Pipeline
═══════════════════════════════════════════════════════

Immutable data structures for the ActionBus.
All envelopes are frozen (hashable, tamper-evident).

Standing on Giants:
- Fowler (2005): Command Query Responsibility Segregation
- Lamport (1978): Logical clocks and ordering

Phase 68.01 — Sovereign Instantiation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ActionStatus(Enum):
    """Lifecycle states for an action in the CQRS pipeline."""

    PROPOSED = "proposed"
    VALIDATING = "validating"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    DENIED = "denied"
    FAILED = "failed"
    CANCELLED = "cancelled"


class GuardianVerdict(Enum):
    """FATE gate verdict for an action."""

    ALLOWED = "allowed"
    DENIED = "denied"
    CONDITIONAL = "conditional"


@dataclass(frozen=True)
class ActionBudget:
    """Resource budget for a single action execution.

    Enforced by the ActionBus — actions exceeding budget are terminated.
    """

    time_ms: int = 10_000  # Max wall-clock time
    s2_tokens_max: int = 50_000  # Max LLM tokens (System-2 budget)
    retry_max: int = 2  # Max retry attempts
    action_limit: int = 100  # Max sub-actions (OmegaLoop bound)


@dataclass(frozen=True)
class ActionEnvelope:
    """Immutable command envelope for the ActionBus CQRS pipeline.

    Every field is frozen — once created, an envelope cannot be modified.
    The action_id is a content-addressed hash (blake3 of canonical form).
    """

    action_id: str  # blake3(canonical_content)
    kind: str  # e.g., "mission.search.web"
    channel: str  # "desktop" | "file" | "browser" | "llm" | "proof"
    payload: dict = field(default_factory=dict)  # action-specific data
    capabilities: tuple[str, ...] = ()  # required TeleScript capabilities
    telescript: dict = field(default_factory=dict)  # action-level restrictions
    budget: ActionBudget = field(default_factory=ActionBudget)
    correlation_id: str = ""  # mission linkage
    actor_id: bytes = b""  # ed25519 public key
    timestamp: int = 0  # unix ms


@dataclass(frozen=True)
class BusActionReceipt:
    """Immutable receipt proving an action was processed.

    Receipts form a merkle chain via prev_receipt_hash.
    Each receipt is content-addressed by receipt_id.
    """

    receipt_id: str  # blake3(canonical_content)
    action_id: str  # links to ActionEnvelope
    status: ActionStatus
    outcome_hash: str  # blake3(outcome)
    ihsan_score: float = 0.0
    prev_receipt_hash: str = "genesis"  # merkle chain
    guardian_verdict: str = "allowed"  # FATE gate result
    duration_ms: float = 0.0
    error_message: str = ""  # only for FAILED/DENIED (sanitized)
