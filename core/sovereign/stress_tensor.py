"""Stress Tensor -- Unified with Action Bus Priority (Omega-7 Gem 6).

The stress tensor Sigma_t is the epistemic tension measure from Definition 1.1.
It IS the Action Bus priority score -- stress measures HOW MUCH tension exists,
the HHMM hidden state captures WHAT the tension is about, and the Action Bus
sorts by stress magnitude.

Priority(m) = Sigma_t * Relevance(m, H_t) * Deadline(m)

Properties:
  P1: Sigma_t >= 0 always (stress is non-negative)
  P2: Sigma_t -> 0 when all tasks resolved
  P3: High stress -> high priority for relevant missions
  P4: Irrelevant missions get low priority regardless of stress
  P5: GCD tick processes highest-priority mission first

Standing on Giants: Kahneman (S1/S2, 2011) | Besta (GoT, 2024) | ARC-001 (Action Bus)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class StressTensor:
    """Epistemic tension measure Sigma_t from Definition 1.1.

    Tracks per-task stress contributions and maintains the aggregate value.
    The stress tensor can be decomposed as:
        Sigma_t = ||tau_expected(t) - tau_actual(t)||
    where tau is the trajectory in the RuVector geometric space.

    Properties:
      P1: value >= 0 always (enforced in __post_init__ and _recompute)
      P2: value -> 0 when all tasks resolved
      P3: increases on new unresolved tasks
      P4: decreases on mission completion
    """

    value: float = 0.0
    _task_stresses: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Enforce P1: stress is non-negative."""
        if self.value < 0:
            self.value = 0.0

    def add_task(self, task_id: str, urgency: float = 1.0) -> None:
        """Register a new unresolved task, increasing stress.

        Args:
            task_id: Unique identifier for the task.
            urgency: Stress contribution of this task (default 1.0).
        """
        self._task_stresses[task_id] = urgency
        self._recompute()

    def resolve_task(self, task_id: str) -> None:
        """Mark a task as resolved, decreasing stress.

        Args:
            task_id: The task to resolve. No-op if task_id not found.
        """
        self._task_stresses.pop(task_id, None)
        self._recompute()

    def _recompute(self) -> None:
        """Recompute total stress from per-task breakdown. Enforces P1."""
        self.value = sum(self._task_stresses.values())
        self.value = max(0.0, self.value)  # P1: non-negative

    @property
    def is_calm(self) -> bool:
        """P2: True when no unresolved tasks and stress is zero."""
        return self.value == 0.0 and len(self._task_stresses) == 0

    @property
    def task_count(self) -> int:
        """Number of unresolved tasks contributing to stress."""
        return len(self._task_stresses)


@dataclass(frozen=True)
class MissionPriority:
    """Priority score for Action Bus ordering.

    Priority = stress * relevance * deadline_urgency

    The GCD tick processes the highest-priority mission first.
    Zero stress or zero relevance always produces zero priority (P3, P4).
    """

    mission_id: str
    stress: float
    relevance: float
    deadline_urgency: float = 1.0

    @property
    def score(self) -> float:
        """The unified priority score: stress * relevance * deadline_urgency."""
        return self.stress * self.relevance * self.deadline_urgency

    @staticmethod
    def compute_relevance(
        mission_domain: str,
        hhmm_state: str,
        domain_map: Optional[dict[str, dict[str, float]]] = None,
    ) -> float:
        """Compute how relevant a mission is to the current HHMM state.

        Args:
            mission_domain: The domain of the mission (e.g., "email", "code").
            hhmm_state: The current HHMM macro-state (e.g., "communicating").
            domain_map: Optional mapping of domain -> {hhmm_state -> relevance}.
                If None, returns 0.5 (moderate relevance).

        Returns:
            Relevance score in [0, 1].

        Example domain_map::

            {"email": {"browsing": 0.2, "communicating": 0.9, "coding": 0.1},
             "code":  {"browsing": 0.3, "communicating": 0.1, "coding": 0.9}}
        """
        if domain_map is None:
            return 0.5

        domain_scores = domain_map.get(mission_domain, {})
        return domain_scores.get(hhmm_state, 0.1)

    @staticmethod
    def compute_deadline_urgency(
        deadline_timestamp: Optional[float],
        current_time: Optional[float] = None,
    ) -> float:
        """Compute urgency multiplier based on deadline proximity.

        Uses a logistic curve that ramps up as the deadline approaches:
            urgency = 1 + 10 / (1 + exp(remaining_hours - 1))

        Args:
            deadline_timestamp: Unix timestamp of the deadline, or None.
            current_time: Current time (defaults to time.time()).

        Returns:
            1.0 if no deadline, up to 10.0 if overdue.
        """
        if deadline_timestamp is None:
            return 1.0

        if current_time is None:
            current_time = time.time()

        remaining_seconds = deadline_timestamp - current_time
        remaining_hours = remaining_seconds / 3600.0

        if remaining_hours <= 0:
            return 10.0  # overdue: maximum urgency

        # Logistic curve: ramps up as deadline approaches
        urgency = 1.0 + 10.0 / (1.0 + math.exp(remaining_hours - 1.0))
        return urgency


def sort_by_priority(
    missions: list[dict[str, Any]],
    stress: StressTensor,
    hhmm_state: str,
) -> list[dict[str, Any]]:
    """Sort missions by priority for Action Bus dispatch.

    The GCD tick processes missions in descending priority order.
    Priority = stress * relevance * deadline_urgency.

    Args:
        missions: List of mission dicts, each with at least "id".
            Optional keys: "domain", "deadline_timestamp".
        stress: Current stress tensor.
        hhmm_state: Current HHMM macro-state for relevance computation.

    Returns:
        Missions sorted by priority (highest first).
    """
    scored: list[tuple[float, dict[str, Any]]] = []

    for m in missions:
        relevance = MissionPriority.compute_relevance(
            m.get("domain", "unknown"), hhmm_state
        )
        deadline_urgency = MissionPriority.compute_deadline_urgency(
            m.get("deadline_timestamp")
        )
        mp = MissionPriority(
            mission_id=m["id"],
            stress=stress.value,
            relevance=relevance,
            deadline_urgency=deadline_urgency,
        )
        scored.append((mp.score, m))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in scored]
