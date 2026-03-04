# Step 6: Stress Tensor ↔ Action Bus Priority Unification

## Standing on Giants: Kahneman (S1/S2, 2011) | Besta (GoT, 2024) | ARC-001 (Action Bus)

**Date:** 2026-03-03
**Ω⁷ Gem:** Ω⁷-6 (Stress tensor IS the Action Bus priority score)
**Intent:** Unify the proof chain's stress tensor Σ with the Action Bus priority function

---

## Problem Statement

Definition 1.1 includes Σ_t ∈ ℝ⁺ as the "stress tensor (epistemic tension measure)."
The Action Bus (from ARC-001) defines a priority score that determines mission
execution order. These are THE SAME CONCEPT expressed in different documents.

**The connection:** Stress measures HOW MUCH tension exists. The HHMM hidden state
captures WHAT the tension is about. The Action Bus sorts by stress magnitude.
The priority function is the dot product of stress and relevance.

---

## Mathematical Formalization

### Unified Priority Function

```
Priority(m) = Σ_t · Relevance(m, H_t) · Deadline(m)

Where:
  Σ_t             ∈ ℝ⁺     — stress tensor (epistemic tension)
  Relevance(m,H)  ∈ [0,1]  — how relevant mission m is to HHMM state H
  Deadline(m)     ∈ [1,∞)  — urgency multiplier (1.0 = no deadline)

Stress update rule:
  Σ_{t+1} = Σ_t + ΔΣ_arrival - ΔΣ_resolution

  ΔΣ_arrival    — stress increase from new unresolved tasks
  ΔΣ_resolution — stress decrease from completed missions

Properties:
  P1: Σ_t ≥ 0 always (stress is non-negative)
  P2: Σ_t → 0 when all tasks are resolved
  P3: High stress → high priority for relevant missions
  P4: Irrelevant missions get low priority regardless of stress
  P5: GCD tick processes highest-priority mission first

Trajectory mismatch (from RuVector Manifold):
  The stress tensor can be decomposed as:

  Σ_t = ||τ_expected(t) - τ_actual(t)||

  Where τ is the trajectory in the RuVector geometric space.
  When on track: Σ → 0.
  When deviated (unexpected event, missed deadline): Σ spikes.
```

---

## Pseudocode

### core/sovereign/stress_tensor.py

```pseudocode
"""Stress Tensor — Unified with Action Bus Priority.

The stress tensor is the epistemic tension measure from Definition 1.1.
It IS the Action Bus priority score (Ω⁷-6 unification).

Standing on Giants: Kahneman (tension) | ARC-001 (Action Bus)
"""

FROM __future__ IMPORT annotations
FROM dataclasses IMPORT dataclass, field
FROM typing IMPORT Optional
IMPORT time


@dataclass
CLASS StressTensor:
    """Epistemic tension measure.

    Properties:
      P1: value >= 0 always
      P2: value → 0 when all tasks resolved
      P3: increases on new unresolved tasks
      P4: decreases on mission completion
    """
    value: float = 0.0
    _task_stresses: dict = field(default_factory=dict)  # task_id → stress

    def __post_init__(self):
        IF self.value < 0:
            self.value = 0.0  # P1: non-negative

    FUNCTION add_task(self, task_id: str, urgency: float = 1.0) -> None:
        """New unresolved task increases stress."""
        self._task_stresses[task_id] = urgency
        self._recompute()

    FUNCTION resolve_task(self, task_id: str) -> None:
        """Completed task decreases stress."""
        self._task_stresses.pop(task_id, None)
        self._recompute()

    FUNCTION _recompute(self) -> None:
        """Recompute total stress from task breakdown."""
        self.value = sum(self._task_stresses.values())
        self.value = max(0.0, self.value)  # P1

    @property
    FUNCTION is_calm(self) -> bool:
        """P2: No unresolved tasks."""
        RETURN self.value == 0.0 AND len(self._task_stresses) == 0

    @property
    FUNCTION task_count(self) -> int:
        """Number of unresolved tasks contributing to stress."""
        RETURN len(self._task_stresses)


@dataclass(frozen=True)
CLASS MissionPriority:
    """Priority score for Action Bus ordering.

    Priority = stress × relevance × deadline_urgency

    The GCD tick processes the highest-priority mission first.
    """
    mission_id: str
    stress: float          # Σ_t — current stress level
    relevance: float       # Relevance(m, H_t) ∈ [0, 1]
    deadline_urgency: float = 1.0  # ≥ 1.0, higher = more urgent

    @property
    FUNCTION score(self) -> float:
        """The unified priority score."""
        RETURN self.stress * self.relevance * self.deadline_urgency

    @staticmethod
    FUNCTION compute_relevance(
        mission_domain: str,
        hhmm_state: str,
        domain_map: dict = None,
    ) -> float:
        """Compute how relevant a mission is to the current HHMM state.

        Example domain_map:
          {"email": {"browsing": 0.2, "communicating": 0.9, "coding": 0.1},
           "code":  {"browsing": 0.3, "communicating": 0.1, "coding": 0.9}}
        """
        IF domain_map IS None:
            RETURN 0.5  # default: moderate relevance

        domain_scores = domain_map.get(mission_domain, {})
        RETURN domain_scores.get(hhmm_state, 0.1)  # default: low

    @staticmethod
    FUNCTION compute_deadline_urgency(
        deadline_timestamp: Optional[float],
        current_time: Optional[float] = None,
    ) -> float:
        """Urgency increases as deadline approaches.

        Returns 1.0 (no deadline) to ∞ (overdue).
        Logistic curve: urgency = 1 + 10 / (1 + e^(remaining_hours - 1))
        """
        IF deadline_timestamp IS None:
            RETURN 1.0  # no deadline

        IF current_time IS None:
            current_time = time.time()

        remaining_seconds = deadline_timestamp - current_time
        remaining_hours = remaining_seconds / 3600

        IF remaining_hours <= 0:
            RETURN 10.0  # overdue: maximum urgency

        # Logistic curve: ramps up as deadline approaches
        IMPORT math
        urgency = 1.0 + 10.0 / (1.0 + math.exp(remaining_hours - 1.0))
        RETURN urgency


FUNCTION sort_by_priority(missions: list, stress: StressTensor, hhmm_state: str) -> list:
    """Sort missions by priority for Action Bus dispatch.

    The GCD tick processes missions in this order.
    """
    priorities = []
    FOR m IN missions:
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
        priorities.append((mp.score, m))

    priorities.sort(key=lambda x: x[0], reverse=True)
    RETURN [m FOR _, m IN priorities]
```

---

## TDD Anchors

```pseudocode
# tests/core/sovereign/test_stress_action_bus.py

TEST stress_starts_at_zero:
    """Initial stress is zero (calm state)."""
    s = StressTensor()
    ASSERT s.value == 0.0
    ASSERT s.is_calm

TEST stress_increases_on_new_task:
    """New task increases stress (P3)."""
    s = StressTensor()
    s.add_task("email_backlog", urgency=2.0)
    ASSERT s.value == 2.0
    ASSERT NOT s.is_calm

TEST stress_decreases_on_resolution:
    """Completed task decreases stress (P4)."""
    s = StressTensor()
    s.add_task("email_backlog", urgency=2.0)
    s.add_task("code_review", urgency=1.5)
    s.resolve_task("email_backlog")
    ASSERT s.value == 1.5

TEST stress_never_negative:
    """P1: Stress is always non-negative."""
    s = StressTensor(value=-5.0)
    ASSERT s.value >= 0.0

TEST stress_returns_to_zero:
    """P2: Resolving all tasks returns to calm."""
    s = StressTensor()
    s.add_task("a", 1.0)
    s.add_task("b", 2.0)
    s.resolve_task("a")
    s.resolve_task("b")
    ASSERT s.is_calm

TEST priority_is_product:
    """Priority = stress × relevance × deadline."""
    mp = MissionPriority(
        mission_id="test", stress=5.0, relevance=0.8, deadline_urgency=2.0
    )
    ASSERT abs(mp.score - 8.0) < 1e-6

TEST zero_stress_zero_priority:
    """Zero stress → zero priority (regardless of relevance)."""
    mp = MissionPriority(
        mission_id="test", stress=0.0, relevance=1.0, deadline_urgency=10.0
    )
    ASSERT mp.score == 0.0

TEST irrelevant_mission_low_priority:
    """P4: Irrelevant mission gets low priority regardless of stress."""
    mp = MissionPriority(
        mission_id="test", stress=100.0, relevance=0.0, deadline_urgency=5.0
    )
    ASSERT mp.score == 0.0

TEST deadline_urgency_no_deadline:
    """No deadline → urgency = 1.0."""
    urgency = MissionPriority.compute_deadline_urgency(None)
    ASSERT urgency == 1.0

TEST deadline_urgency_overdue:
    """Overdue mission → maximum urgency."""
    past = time.time() - 3600  # 1 hour ago
    urgency = MissionPriority.compute_deadline_urgency(past)
    ASSERT urgency >= 9.0

TEST sort_by_priority_orders_correctly:
    """Highest priority mission is first after sort."""
    stress = StressTensor()
    stress.add_task("work", 5.0)
    missions = [
        {"id": "low", "domain": "unknown"},
        {"id": "high", "domain": "email", "deadline_timestamp": time.time() + 60},
    ]
    sorted_m = sort_by_priority(missions, stress, "communicating")
    # "high" should be first (email + communicating = high relevance + deadline)
    ASSERT sorted_m[0]["id"] == "high"
```

---

## Acceptance Criteria

1. `StressTensor` enforces non-negative values (P1)
2. Stress returns to zero when all tasks resolved (P2)
3. `MissionPriority.score` = stress × relevance × deadline
4. Zero stress or zero relevance produces zero priority
5. `sort_by_priority()` correctly orders missions for GCD dispatch
6. All 11 TDD anchors GREEN
7. Full test suite GREEN
