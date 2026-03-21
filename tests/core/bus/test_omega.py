"""
Omega Loop Controller Tests — Phase 68.02
══════════════════════════════════════════

TDD anchors for proof-based iteration: termination, budget,
cancel/pause, event emission.

Standing on Giants:
- Beck (2002): TDD by Example
- Hoare (1969): Axiomatic basis for programming
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.bus.omega import (
    LoopBudget,
    OmegaLoopController,
    OmegaLoopState,
    OmegaStatus,
    ProofCondition,
)

# ═══════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════


def _mock_event_bus() -> AsyncMock:
    bus = AsyncMock()
    bus.publish = AsyncMock()
    return bus


def _make_receipt(ihsan: float = 0.96, status: str = "completed") -> SimpleNamespace:
    return SimpleNamespace(
        ihsan_score=ihsan,
        status=SimpleNamespace(value=status),
        action_id="a-test",
    )


async def _planner_one_action(state: OmegaLoopState) -> list:
    """Returns one mock action per iteration."""
    return [SimpleNamespace(action_id=f"act-{state.iteration}")]


async def _proposer_success(action) -> SimpleNamespace:
    return _make_receipt(ihsan=0.96, status="completed")


# ═══════════════════════════════════════════════════════════
# Basic loop behavior
# ═══════════════════════════════════════════════════════════


class TestOmegaLoopBasic:
    """Core loop lifecycle."""

    @pytest.mark.asyncio
    async def test_single_iteration_proves(self) -> None:
        """Loop with always_true condition proves in 1 iteration."""
        ctrl = OmegaLoopController(event_bus=_mock_event_bus())
        state = await ctrl.run(
            mission_id="m-1",
            proof_conditions=[ProofCondition(kind="always_true")],
            max_iterations=10,
        )
        assert state.status == OmegaStatus.PROVED
        assert state.iteration == 1

    @pytest.mark.asyncio
    async def test_max_iterations_stops_loop(self) -> None:
        """Loop stops at max_iterations if proofs never satisfy."""
        ctrl = OmegaLoopController(event_bus=_mock_event_bus())
        state = await ctrl.run(
            mission_id="m-2",
            proof_conditions=[ProofCondition(kind="tests_pass", target="foo")],
            max_iterations=3,
        )
        assert state.status == OmegaStatus.MAX_ITERATIONS
        assert state.iteration == 3

    @pytest.mark.asyncio
    async def test_budget_exhausted_stops_loop(self) -> None:
        """Loop stops when action budget runs out."""
        action_bus = AsyncMock()
        action_bus.propose = AsyncMock(side_effect=_proposer_success)

        ctrl = OmegaLoopController(
            action_bus=action_bus,
            event_bus=_mock_event_bus(),
            iteration_planner=_planner_one_action,
        )
        state = await ctrl.run(
            mission_id="m-3",
            proof_conditions=[ProofCondition(kind="tests_pass", target="bar")],
            budget=LoopBudget(actions=2, time_ms=999_999),
            max_iterations=100,
        )
        assert state.status == OmegaStatus.BUDGET_EXHAUSTED
        assert state.iteration == 2

    @pytest.mark.asyncio
    async def test_multiple_iterations_to_prove(self) -> None:
        """Loop runs multiple iterations then proves."""
        call_count = 0

        async def checker(state, receipts):
            nonlocal call_count
            call_count += 1
            return call_count >= 3  # prove on 3rd iteration

        ctrl = OmegaLoopController(
            event_bus=_mock_event_bus(),
            proof_checker=checker,
        )
        state = await ctrl.run(
            mission_id="m-4",
            proof_conditions=[ProofCondition(kind="custom")],
            max_iterations=10,
        )
        assert state.status == OmegaStatus.PROVED
        assert state.iteration == 3


# ═══════════════════════════════════════════════════════════
# Termination conditions
# ═══════════════════════════════════════════════════════════


class TestOmegaTermination:
    """Proof-based termination logic."""

    @pytest.mark.asyncio
    async def test_terminates_only_when_all_proofs_satisfied(self) -> None:
        """Partial proofs don't terminate the loop."""
        ctrl = OmegaLoopController(event_bus=_mock_event_bus())
        state = await ctrl.run(
            mission_id="m-5",
            proof_conditions=[
                ProofCondition(kind="always_true"),
                ProofCondition(kind="tests_pass", target="missing"),
            ],
            max_iterations=3,
        )
        assert state.status == OmegaStatus.MAX_ITERATIONS

    @pytest.mark.asyncio
    async def test_ihsan_below_floor_continues(self) -> None:
        """Ihsan condition with no receipts remains unsatisfied."""
        ctrl = OmegaLoopController(event_bus=_mock_event_bus())
        state = await ctrl.run(
            mission_id="m-6",
            proof_conditions=[ProofCondition(kind="ihsan_above", threshold=0.95)],
            max_iterations=2,
        )
        # No receipts → ihsan condition never satisfied
        assert state.status == OmegaStatus.MAX_ITERATIONS

    @pytest.mark.asyncio
    async def test_ihsan_with_receipts_proves(self) -> None:
        """Ihsan condition satisfied when receipt scores meet threshold."""
        action_bus = AsyncMock()
        action_bus.propose = AsyncMock(
            return_value=_make_receipt(ihsan=0.97, status="completed")
        )
        ctrl = OmegaLoopController(
            action_bus=action_bus,
            event_bus=_mock_event_bus(),
            iteration_planner=_planner_one_action,
        )
        state = await ctrl.run(
            mission_id="m-7",
            proof_conditions=[ProofCondition(kind="ihsan_above", threshold=0.95)],
            max_iterations=5,
        )
        assert state.status == OmegaStatus.PROVED


# ═══════════════════════════════════════════════════════════
# Cancel and pause
# ═══════════════════════════════════════════════════════════


class TestOmegaCancel:
    """Operator-initiated cancel and pause."""

    @pytest.mark.asyncio
    async def test_cancel_stops_running_loop(self) -> None:
        ctrl = OmegaLoopController(event_bus=_mock_event_bus())
        # Start a loop that would run forever
        state = OmegaLoopState(
            loop_id="l-cancel",
            mission_id="m-cancel",
            status=OmegaStatus.RUNNING,
        )
        ctrl._active_loops["l-cancel"] = state
        result = await ctrl.cancel("l-cancel")
        assert result is True
        assert state.status == OmegaStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_pause_preserves_state(self) -> None:
        ctrl = OmegaLoopController(event_bus=_mock_event_bus())
        state = OmegaLoopState(
            loop_id="l-pause",
            mission_id="m-pause",
            iteration=5,
            status=OmegaStatus.RUNNING,
        )
        ctrl._active_loops["l-pause"] = state
        result = await ctrl.pause("l-pause")
        assert result is True
        assert state.status == OmegaStatus.PAUSED
        assert state.iteration == 5

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_returns_false(self) -> None:
        ctrl = OmegaLoopController()
        result = await ctrl.cancel("nonexistent")
        assert result is False


# ═══════════════════════════════════════════════════════════
# Event emission
# ═══════════════════════════════════════════════════════════


class TestOmegaEvents:
    """Event bus integration."""

    @pytest.mark.asyncio
    async def test_started_event_emitted(self) -> None:
        eb = _mock_event_bus()
        ctrl = OmegaLoopController(event_bus=eb)
        await ctrl.run(
            mission_id="m-ev",
            proof_conditions=[ProofCondition(kind="always_true")],
            max_iterations=1,
        )
        topics = [call.args[0] for call in eb.publish.call_args_list]
        assert "omega.started" in topics

    @pytest.mark.asyncio
    async def test_proved_event_includes_conditions(self) -> None:
        eb = _mock_event_bus()
        ctrl = OmegaLoopController(event_bus=eb)
        await ctrl.run(
            mission_id="m-ev2",
            proof_conditions=[ProofCondition(kind="always_true")],
            max_iterations=5,
        )
        topics = [call.args[0] for call in eb.publish.call_args_list]
        assert "omega.proved" in topics
        # Find the proved event payload
        for call in eb.publish.call_args_list:
            if call.args[0] == "omega.proved":
                payload = call.args[1]
                assert "conditions" in payload
                assert payload["conditions"][0]["satisfied"] is True

    @pytest.mark.asyncio
    async def test_completed_event_always_emitted(self) -> None:
        eb = _mock_event_bus()
        ctrl = OmegaLoopController(event_bus=eb)
        await ctrl.run(
            mission_id="m-ev3",
            proof_conditions=[ProofCondition(kind="tests_pass")],
            max_iterations=1,
        )
        topics = [call.args[0] for call in eb.publish.call_args_list]
        assert "omega.completed" in topics

    @pytest.mark.asyncio
    async def test_supports_sovereign_event_bus_shape(self) -> None:
        from core.sovereign.event_bus import EventBus

        eb = EventBus()
        ctrl = OmegaLoopController(event_bus=eb)
        await ctrl.run(
            mission_id="m-ev4",
            proof_conditions=[ProofCondition(kind="always_true")],
            max_iterations=1,
        )

        stats = eb.stats()
        assert stats["events_published"] == 4
        assert stats["queue_size"] == 4
