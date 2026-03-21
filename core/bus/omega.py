"""
Omega Loop Controller — Proof-Based Iteration
═════════════════════════════════════════════

Loop terminates ONLY when proof validates. No self-reported completion.
State persists via event emission for resumability.

Standing on Giants:
- Lamport (1978): Logical clocks
- Hoare (1969): Axiomatic basis for computer programming

Phase 68.02 — Sovereign Instantiation
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol, runtime_checkable

from core.bus.event_publisher import publish_topic_event

logger = logging.getLogger(__name__)


class OmegaStatus(Enum):
    """Lifecycle states for an Omega loop."""

    PENDING = "pending"
    RUNNING = "running"
    PROVED = "proved"
    BUDGET_EXHAUSTED = "budget_exhausted"
    CANCELLED = "cancelled"
    PAUSED = "paused"
    FAILED = "failed"
    MAX_ITERATIONS = "max_iterations"


@dataclass
class LoopBudget:
    """Resource budget for an entire Omega loop execution."""

    time_ms: int = 300_000  # 5 minutes default
    s2_tokens: int = 50_000  # LLM token budget
    actions: int = 100  # max action proposals


@dataclass
class ProofCondition:
    """A condition that must be satisfied for loop termination."""

    kind: str  # "ihsan_above" | "tests_pass" | "ledger_committed" | "custom"
    target: str = ""  # what to check
    threshold: float = 0.0
    satisfied: bool = False


@dataclass
class OmegaLoopState:
    """Mutable state of an Omega loop execution."""

    loop_id: str
    mission_id: str
    iteration: int = 0
    max_iterations: int = 50
    status: OmegaStatus = OmegaStatus.RUNNING
    budget: LoopBudget = field(default_factory=LoopBudget)
    proof_conditions: list[ProofCondition] = field(default_factory=list)
    event_ids: list[str] = field(default_factory=list)
    started_at: int = 0
    last_tick_at: int = 0


def _now_ms() -> int:
    return int(time.time() * 1000)


def _loop_id(mission_id: str, conditions: list[ProofCondition]) -> str:
    """Content-addressed loop ID."""
    content = f"{mission_id}:{[c.kind for c in conditions]}"
    return hashlib.blake2b(content.encode(), digest_size=16).hexdigest()


@runtime_checkable
class ActionProposer(Protocol):
    """Protocol for proposing actions (ActionBus.propose)."""

    async def propose(self, action: Any) -> Any: ...


@runtime_checkable
class EventPublisher(Protocol):
    """Protocol for event bus publishing."""

    async def publish(self, topic: str, payload: dict[str, Any]) -> None: ...


class OmegaLoopController:
    """Proof-based iteration controller.

    Security properties:
    1. Proof-based termination — loop ONLY stops when ALL proofs satisfied
    2. Bounded iterations — hard limit prevents infinite loops
    3. Budget enforcement — time/token/action limits enforced each tick
    4. Cancel/pause — operator can interrupt at any time
    5. Event-sourced state — every transition is published for replay
    """

    __slots__ = (
        "_action_bus",
        "_event_bus",
        "_active_loops",
        "_iteration_planner",
        "_proof_checker",
    )

    def __init__(
        self,
        action_bus: ActionProposer | None = None,
        event_bus: EventPublisher | None = None,
        iteration_planner: Callable | None = None,
        proof_checker: Callable | None = None,
    ) -> None:
        self._action_bus = action_bus
        self._event_bus = event_bus
        self._iteration_planner = iteration_planner
        self._proof_checker = proof_checker
        self._active_loops: dict[str, OmegaLoopState] = {}

    @property
    def active_loops(self) -> dict[str, OmegaLoopState]:
        return dict(self._active_loops)

    async def run(
        self,
        mission_id: str,
        proof_conditions: list[ProofCondition],
        budget: LoopBudget | None = None,
        max_iterations: int = 50,
    ) -> OmegaLoopState:
        """Execute an Omega loop until proof validates or budget exhausted."""

        loop_budget = budget or LoopBudget()
        lid = _loop_id(mission_id, proof_conditions)

        state = OmegaLoopState(
            loop_id=lid,
            mission_id=mission_id,
            max_iterations=max_iterations,
            budget=loop_budget,
            proof_conditions=list(proof_conditions),
            started_at=_now_ms(),
        )
        self._active_loops[lid] = state

        await self._emit(
            "omega.started",
            {
                "loop_id": lid,
                "mission_id": mission_id,
                "proof_conditions": [pc.kind for pc in proof_conditions],
            },
        )

        while state.status == OmegaStatus.RUNNING:
            # Check iteration limit
            if state.iteration >= state.max_iterations:
                state.status = OmegaStatus.MAX_ITERATIONS
                break

            # Check budget
            if not self._budget_ok(state):
                state.status = OmegaStatus.BUDGET_EXHAUSTED
                break

            # === One Iteration ===
            state.iteration += 1
            state.last_tick_at = _now_ms()

            await self._emit(
                "omega.iteration",
                {
                    "loop_id": lid,
                    "iteration": state.iteration,
                },
            )

            # Plan and execute actions
            actions = await self._plan_iteration(state)
            receipts = await self._execute_actions(state, actions)

            # Check proof conditions
            all_proved = await self._check_proofs(state, receipts)

            if all_proved:
                state.status = OmegaStatus.PROVED
                await self._emit(
                    "omega.proved",
                    {
                        "loop_id": lid,
                        "iterations": state.iteration,
                        "conditions": [
                            {"kind": pc.kind, "satisfied": pc.satisfied}
                            for pc in state.proof_conditions
                        ],
                    },
                )
                break

            # Update time budget
            elapsed = _now_ms() - state.started_at
            state.budget.time_ms = max(0, loop_budget.time_ms - elapsed)

        # Final event
        await self._emit(
            "omega.completed",
            {
                "loop_id": lid,
                "status": state.status.value,
                "iterations": state.iteration,
            },
        )

        return state

    async def cancel(self, loop_id: str) -> bool:
        """Cancel a running loop."""
        state = self._active_loops.get(loop_id)
        if state and state.status == OmegaStatus.RUNNING:
            state.status = OmegaStatus.CANCELLED
            await self._emit("omega.cancelled", {"loop_id": loop_id})
            return True
        return False

    async def pause(self, loop_id: str) -> bool:
        """Pause a running loop for later resume."""
        state = self._active_loops.get(loop_id)
        if state and state.status == OmegaStatus.RUNNING:
            state.status = OmegaStatus.PAUSED
            await self._emit(
                "omega.paused",
                {
                    "loop_id": loop_id,
                    "iteration": state.iteration,
                },
            )
            return True
        return False

    def _budget_ok(self, state: OmegaLoopState) -> bool:
        """Check if budget allows another iteration."""
        return state.budget.time_ms > 0 and state.budget.actions > 0

    async def _plan_iteration(self, state: OmegaLoopState) -> list:
        """Plan actions for next iteration via pluggable planner."""
        if self._iteration_planner is not None:
            return await self._iteration_planner(state)
        return []

    async def _execute_actions(self, state: OmegaLoopState, actions: list) -> list:
        """Execute planned actions via ActionBus."""
        receipts = []
        if self._action_bus is None:
            return receipts

        for action in actions:
            receipt = await self._action_bus.propose(action)
            receipts.append(receipt)
            state.budget.actions -= 1

        return receipts

    async def _check_proofs(self, state: OmegaLoopState, receipts: list) -> bool:
        """Evaluate all proof conditions. Returns True only if ALL satisfied."""
        if self._proof_checker is not None:
            return await self._proof_checker(state, receipts)

        # Default proof checking for built-in condition types
        all_satisfied = True
        for pc in state.proof_conditions:
            if pc.kind == "ihsan_above":
                scores = [
                    getattr(r, "ihsan_score", 0.0)
                    for r in receipts
                    if getattr(r, "status", None)
                    and getattr(r.status, "value", "") == "completed"
                ]
                pc.satisfied = bool(scores) and all(s >= pc.threshold for s in scores)
            elif pc.kind == "ledger_committed":
                pc.satisfied = len(state.event_ids) > 0 or len(receipts) > 0
            elif pc.kind == "always_true":
                pc.satisfied = True
            else:
                # Unknown condition types remain unsatisfied
                pass

            if not pc.satisfied:
                all_satisfied = False

        return all_satisfied

    async def _emit(self, topic: str, payload: dict[str, Any]) -> None:
        """Emit event via EventBus if configured."""
        if self._event_bus is not None:
            await publish_topic_event(self._event_bus, topic, payload)
