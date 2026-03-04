"""TDD anchors for Stress Tensor and Action Bus Priority (Step 6).

11 tests covering:
  StressTensor: non-negativity (P1), zero convergence (P2), add/resolve
  MissionPriority: product formula, zero-stress/zero-relevance, deadline urgency
  sort_by_priority: correct ordering for GCD dispatch
"""

from __future__ import annotations

import time

from core.sovereign.stress_tensor import (
    MissionPriority,
    StressTensor,
    sort_by_priority,
)


# -- StressTensor tests ------------------------------------------------------


class TestStressTensor:
    """Verify stress tensor properties P1-P4."""

    def test_stress_starts_at_zero(self) -> None:
        """Initial stress is zero (calm state)."""
        s = StressTensor()
        assert s.value == 0.0
        assert s.is_calm

    def test_stress_increases_on_new_task(self) -> None:
        """New task increases stress (P3)."""
        s = StressTensor()
        s.add_task("email_backlog", urgency=2.0)
        assert s.value == 2.0
        assert not s.is_calm

    def test_stress_decreases_on_resolution(self) -> None:
        """Completed task decreases stress (P4)."""
        s = StressTensor()
        s.add_task("email_backlog", urgency=2.0)
        s.add_task("code_review", urgency=1.5)
        s.resolve_task("email_backlog")
        assert s.value == 1.5

    def test_stress_never_negative(self) -> None:
        """P1: Stress is always non-negative."""
        s = StressTensor(value=-5.0)
        assert s.value >= 0.0

    def test_stress_returns_to_zero(self) -> None:
        """P2: Resolving all tasks returns to calm."""
        s = StressTensor()
        s.add_task("a", 1.0)
        s.add_task("b", 2.0)
        s.resolve_task("a")
        s.resolve_task("b")
        assert s.is_calm


# -- MissionPriority tests ---------------------------------------------------


class TestMissionPriority:
    """Verify the unified priority function: stress * relevance * deadline."""

    def test_priority_is_product(self) -> None:
        """Priority = stress * relevance * deadline."""
        mp = MissionPriority(
            mission_id="test", stress=5.0, relevance=0.8, deadline_urgency=2.0
        )
        assert abs(mp.score - 8.0) < 1e-6

    def test_zero_stress_zero_priority(self) -> None:
        """Zero stress -> zero priority (regardless of relevance)."""
        mp = MissionPriority(
            mission_id="test", stress=0.0, relevance=1.0, deadline_urgency=10.0
        )
        assert mp.score == 0.0

    def test_irrelevant_mission_low_priority(self) -> None:
        """P4: Irrelevant mission gets low priority regardless of stress."""
        mp = MissionPriority(
            mission_id="test", stress=100.0, relevance=0.0, deadline_urgency=5.0
        )
        assert mp.score == 0.0


# -- Deadline urgency tests --------------------------------------------------


class TestDeadlineUrgency:
    """Verify logistic deadline urgency curve."""

    def test_no_deadline_urgency_is_1(self) -> None:
        """No deadline -> urgency = 1.0."""
        urgency = MissionPriority.compute_deadline_urgency(None)
        assert urgency == 1.0

    def test_overdue_max_urgency(self) -> None:
        """Overdue mission -> maximum urgency (10.0)."""
        past = time.time() - 3600  # 1 hour ago
        urgency = MissionPriority.compute_deadline_urgency(past)
        assert urgency >= 9.0


# -- sort_by_priority tests --------------------------------------------------


class TestSortByPriority:
    """Verify mission ordering for GCD dispatch."""

    def test_sort_orders_correctly(self) -> None:
        """Highest priority mission is first after sort."""
        stress = StressTensor()
        stress.add_task("work", 5.0)

        missions = [
            {"id": "low", "domain": "unknown"},
            {"id": "high", "domain": "email", "deadline_timestamp": time.time() + 60},
        ]

        # Use a domain map where "email" is highly relevant to "communicating"
        # but sort_by_priority uses default relevance (0.5) without domain_map.
        # The "high" mission has a deadline (urgency > 1.0), so it ranks higher.
        sorted_m = sort_by_priority(missions, stress, "communicating")
        assert sorted_m[0]["id"] == "high"
