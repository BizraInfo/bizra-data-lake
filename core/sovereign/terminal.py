"""
Terminal v1 Spine — Sovereign Mission Terminal Contract Types
═════════════════════════════════════════════════════════════

Phase 74: Terminal Genesis — Two-Touch Sovereign Runtime.

Defines the foundational types for the BIZRA terminal product:
- TerminalState: 9-state machine for terminal lifecycle
- PermissionEnvelope: mission-scoped approval bundle
- MissionReceipt: enriched receipt with wallet/reflex/memory deltas
- EventRecord: canonical event schema for timeline rendering
- BriefingContext: morning briefing / session continuity

Standing on Giants:
- Harel (1987): Statecharts — hierarchical state machines
- Thompson (1984): Capability-based security envelopes
- Lamport (1978): Event ordering and hash chains
- Kahneman (2002): System-1/System-2 cognitive split
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# ═══════════════════════════════════════════════════════════════════
# Terminal State Machine — 9 states (Harel statechart)
# ═══════════════════════════════════════════════════════════════════


class TerminalState(str, Enum):
    """Terminal lifecycle states.

    Transitions:
        BOOT → READY                          (runtime health acceptable)
        READY → MISSION_DRAFTING               (user intent)
        MISSION_DRAFTING → PERMISSION_REVIEW   (scope inferred)
        PERMISSION_REVIEW → EXECUTING          (scope accepted)
        EXECUTING → AWAITING_ESCALATION        (action exceeds envelope)
        EXECUTING → COMPLETED                  (receipted success)
        EXECUTING → FAILED_RECOVERABLY         (bounded operational failure)
        EXECUTING → BLOCKED_CONSTITUTIONALLY   (invariant violation risk)
        AWAITING_ESCALATION → EXECUTING        (escalation approved)
        AWAITING_ESCALATION → FAILED_RECOVERABLY (escalation denied)
        COMPLETED → READY                      (cycle reset)
        FAILED_RECOVERABLY → READY             (cycle reset)
        BLOCKED_CONSTITUTIONALLY → READY       (cycle reset after review)
    """

    BOOT = "boot"
    READY = "ready"
    MISSION_DRAFTING = "mission_drafting"
    PERMISSION_REVIEW = "permission_review"
    EXECUTING = "executing"
    AWAITING_ESCALATION = "awaiting_escalation"
    COMPLETED = "completed"
    FAILED_RECOVERABLY = "failed_recoverably"
    BLOCKED_CONSTITUTIONALLY = "blocked_constitutionally"


# Valid transitions (from → {to, ...})
TERMINAL_TRANSITIONS: dict[TerminalState, frozenset[TerminalState]] = {
    TerminalState.BOOT: frozenset({TerminalState.READY}),
    TerminalState.READY: frozenset({TerminalState.MISSION_DRAFTING}),
    TerminalState.MISSION_DRAFTING: frozenset({TerminalState.PERMISSION_REVIEW}),
    TerminalState.PERMISSION_REVIEW: frozenset({TerminalState.EXECUTING}),
    TerminalState.EXECUTING: frozenset(
        {
            TerminalState.AWAITING_ESCALATION,
            TerminalState.COMPLETED,
            TerminalState.FAILED_RECOVERABLY,
            TerminalState.BLOCKED_CONSTITUTIONALLY,
        }
    ),
    TerminalState.AWAITING_ESCALATION: frozenset(
        {TerminalState.EXECUTING, TerminalState.FAILED_RECOVERABLY}
    ),
    TerminalState.COMPLETED: frozenset({TerminalState.READY}),
    TerminalState.FAILED_RECOVERABLY: frozenset({TerminalState.READY}),
    TerminalState.BLOCKED_CONSTITUTIONALLY: frozenset({TerminalState.READY}),
}


class ExecutionPath(str, Enum):
    """How the mission was resolved — System-1, System-2, or mixed."""

    SYSTEM_1_CACHE_HIT = "system_1"  # Reflex cache hit
    SYSTEM_2_NOVEL = "system_2"  # Full reasoning pipeline
    MIXED = "mixed"  # Partial cache, partial novel


# ═══════════════════════════════════════════════════════════════════
# Permission Envelope — Mission-Scoped Approval Bundle
# ═══════════════════════════════════════════════════════════════════


@dataclass
class PermissionEnvelope:
    """Mission-scoped approval bundle.

    Two-Touch Law: The user approves this envelope once at mission start.
    No further confirmations unless action exceeds scope.
    """

    # Filesystem scope — glob patterns for allowed paths
    filesystem: list[str] = field(default_factory=lambda: ["workspace/**"])
    # Application scope — allowed application identifiers
    applications: list[str] = field(default_factory=lambda: ["terminal", "editor"])
    # Network scope — allowed domains/endpoints
    network: list[str] = field(default_factory=list)
    # Data sensitivity class
    data_sensitivity: str = "standard"  # standard | sensitive | restricted
    # Spend budget in USD (0 = no external API calls)
    spend_budget_usd: float = 0.0
    # Time budget in seconds
    time_budget_seconds: int = 900  # 15 minutes default
    # Escalation policy
    escalation: str = "ask-on-boundary-cross"  # ask | block | allow
    # Audit verbosity
    audit_verbosity: str = "standard"  # minimal | standard | detailed

    def to_dict(self) -> dict[str, Any]:
        """Serialize for API transport."""
        return {
            "filesystem": self.filesystem,
            "applications": self.applications,
            "network": self.network,
            "data_sensitivity": self.data_sensitivity,
            "spend_budget_usd": self.spend_budget_usd,
            "time_budget_seconds": self.time_budget_seconds,
            "escalation": self.escalation,
            "audit_verbosity": self.audit_verbosity,
        }

    def allows_path(self, path: str) -> bool:
        """Check if a filesystem path is within the approved scope."""
        import fnmatch

        return any(fnmatch.fnmatch(path, pattern) for pattern in self.filesystem)

    def allows_network(self, domain: str) -> bool:
        """Check if a network domain is within the approved scope."""
        if not self.network:
            return False
        return domain in self.network


# ═══════════════════════════════════════════════════════════════════
# Mission Receipt — Enriched Receipt with Deltas
# ═══════════════════════════════════════════════════════════════════


@dataclass
class WalletDelta:
    """Change in wallet balances from a mission."""

    seed: float = 0.0
    bloom: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {"seed": self.seed, "bloom": self.bloom}


@dataclass
class ReflexDelta:
    """Change in reflex state from a mission.

    Contract §8.1: compiled, near_compile, compile_count, threshold.
    """

    compiled: bool = False
    near_compile: bool = False
    compile_count: int = 0
    threshold: int = 3  # Default: 3 excellent executions to compile

    def to_dict(self) -> dict[str, Any]:
        return {
            "compiled": self.compiled,
            "near_compile": self.near_compile,
            "compile_count": self.compile_count,
            "threshold": self.threshold,
        }


@dataclass
class MemoryDelta:
    """Change in memory state from a mission."""

    episodic: int = 0
    semantic: int = 0
    procedural: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "episodic": self.episodic,
            "semantic": self.semantic,
            "procedural": self.procedural,
        }


@dataclass
class ChannelRecord:
    """Per-channel execution record."""

    channel: str
    success: bool
    duration_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "success": self.success,
            "duration_ms": round(self.duration_ms, 1),
        }


@dataclass
class MissionReceipt:
    """Complete mission receipt with all deltas for terminal rendering.

    This is the canonical output contract for Terminal v1.
    Every completed mission produces exactly one of these.
    """

    mission_id: str
    receipt_id: str
    status: str  # COMPLETE | PARTIAL | FAILED | BLOCKED (Contract §8.1)
    synthesis: str
    ihsan_score: float
    snr_score: float
    duration_ms: float
    channels_executed: list[ChannelRecord]
    execution_path: ExecutionPath = ExecutionPath.SYSTEM_2_NOVEL
    wallet_delta: WalletDelta = field(default_factory=WalletDelta)
    reflex_delta: ReflexDelta = field(default_factory=ReflexDelta)
    memory_delta: MemoryDelta = field(default_factory=MemoryDelta)
    hash_chain_ref: str = ""
    action_count: int = 0
    # Contract §9.4: Cache-hit proof fields (populated on S1 path)
    reflex_pattern: str = ""
    reflex_latency_ms: float = 0.0
    comparison_s2_avg_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize for API transport and timeline rendering."""
        return {
            "mission_id": self.mission_id,
            "receipt_id": self.receipt_id,
            "status": self.status,
            "synthesis": self.synthesis,
            "ihsan_score": self.ihsan_score,
            "snr_score": self.snr_score,
            "duration_ms": round(self.duration_ms, 1),
            "channels_executed": [c.to_dict() for c in self.channels_executed],
            "execution_path": self.execution_path.value,
            "wallet_delta": self.wallet_delta.to_dict(),
            "reflex_delta": self.reflex_delta.to_dict(),
            "memory_delta": self.memory_delta.to_dict(),
            "hash_chain_ref": self.hash_chain_ref,
            "action_count": self.action_count,
            "reflex_pattern": self.reflex_pattern,
            "reflex_latency_ms": round(self.reflex_latency_ms, 1),
            "comparison_s2_avg_ms": round(self.comparison_s2_avg_ms, 1),
        }


# ═══════════════════════════════════════════════════════════════════
# Event Record — Canonical Event Schema for Timeline
# ═══════════════════════════════════════════════════════════════════


class EventSeverity(str, Enum):
    """Event severity levels for timeline rendering.

    Contract §7.1: info, notice, warning, critical.
    """

    INFO = "info"
    NOTICE = "notice"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class EventRecord:
    """Canonical event for timeline rendering.

    Every visible change in the terminal derives from an EventRecord.
    Event-native law: no UI state detached from event spine.
    """

    event_id: str
    timestamp: float  # Unix seconds
    category: str  # mission.*, receipt.*, tick.*, reflex.*, memory.*, etc.
    origin: str  # Subsystem that produced the event
    severity: EventSeverity = EventSeverity.INFO
    mission_id: str = ""
    receipt_id: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "category": self.category,
            "origin": self.origin,
            "severity": self.severity.value,
            "mission_id": self.mission_id,
            "receipt_id": self.receipt_id,
            "payload": self.payload,
        }


# ═══════════════════════════════════════════════════════════════════
# Briefing Context — Session Continuity
# ═══════════════════════════════════════════════════════════════════


@dataclass
class BriefingContext:
    """Contextual briefing generated on terminal open.

    Provides session continuity: what happened, what's active,
    what patterns are near compilation, what to do next.
    """

    time_since_last_mission_s: float = 0.0
    active_project: str = ""
    last_mission_summary: str = ""
    near_compile_patterns: list[str] = field(default_factory=list)
    quality_trend: str = "stable"  # improving | stable | declining
    next_action_suggestion: str = ""
    wallet_snapshot: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "time_since_last_mission_s": round(self.time_since_last_mission_s, 1),
            "active_project": self.active_project,
            "last_mission_summary": self.last_mission_summary,
            "near_compile_patterns": self.near_compile_patterns,
            "quality_trend": self.quality_trend,
            "next_action_suggestion": self.next_action_suggestion,
            "wallet_snapshot": self.wallet_snapshot,
        }


# ═══════════════════════════════════════════════════════════════════
# Terminal State Controller
# ═══════════════════════════════════════════════════════════════════


class TerminalStateController:
    """Manages terminal state transitions with Harel statechart discipline.

    Enforces valid transitions only. Tracks execution path type.
    """

    __slots__ = ("_state", "_execution_path", "_mission_id")

    def __init__(self) -> None:
        self._state = TerminalState.BOOT
        self._execution_path = ExecutionPath.SYSTEM_2_NOVEL
        self._mission_id: str = ""

    @property
    def state(self) -> TerminalState:
        return self._state

    @property
    def execution_path(self) -> ExecutionPath:
        return self._execution_path

    @property
    def mission_id(self) -> str:
        return self._mission_id

    def transition(self, target: TerminalState) -> bool:
        """Attempt state transition. Returns True if valid, False otherwise."""
        allowed = TERMINAL_TRANSITIONS.get(self._state, frozenset())
        if target not in allowed:
            return False
        self._state = target
        return True

    def start_mission(
        self,
        mission_id: str,
        execution_path: ExecutionPath = ExecutionPath.SYSTEM_2_NOVEL,
    ) -> bool:
        """Begin a new mission. Sets execution path metadata."""
        if self._state != TerminalState.READY:
            return False
        self._mission_id = mission_id
        self._execution_path = execution_path
        self._state = TerminalState.MISSION_DRAFTING
        return True

    def complete(self) -> bool:
        """Mark mission as completed and return to ready."""
        if self._state != TerminalState.EXECUTING:
            return False
        self._state = TerminalState.COMPLETED
        return True

    def fail(self) -> bool:
        """Mark mission as failed recoverably."""
        if self._state != TerminalState.EXECUTING:
            return False
        self._state = TerminalState.FAILED_RECOVERABLY
        return True

    def reset(self) -> bool:
        """Return to READY from any terminal state."""
        if self._state in (
            TerminalState.COMPLETED,
            TerminalState.FAILED_RECOVERABLY,
            TerminalState.BLOCKED_CONSTITUTIONALLY,
        ):
            self._state = TerminalState.READY
            self._mission_id = ""
            return True
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self._state.value,
            "execution_path": self._execution_path.value,
            "mission_id": self._mission_id,
        }
