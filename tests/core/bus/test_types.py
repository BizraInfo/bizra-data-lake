"""
Bus Types Tests — Phase 68.01
═════════════════════════════

TDD anchors for ActionStatus, ActionEnvelope, ActionBudget, BusActionReceipt.

Standing on Giants:
- Beck (2002): TDD by Example
"""

from __future__ import annotations

from core.bus.types import (
    ActionBudget,
    ActionEnvelope,
    ActionStatus,
    BusActionReceipt,
    GuardianVerdict,
)


class TestActionStatus:
    """Lifecycle state enum."""

    def test_status_values(self) -> None:
        assert ActionStatus.PROPOSED.value == "proposed"
        assert ActionStatus.COMPLETED.value == "completed"
        assert ActionStatus.DENIED.value == "denied"
        assert ActionStatus.FAILED.value == "failed"

    def test_status_count(self) -> None:
        assert len(ActionStatus) == 8


class TestGuardianVerdict:
    """FATE gate verdict enum."""

    def test_verdict_values(self) -> None:
        assert GuardianVerdict.ALLOWED.value == "allowed"
        assert GuardianVerdict.DENIED.value == "denied"
        assert GuardianVerdict.CONDITIONAL.value == "conditional"

    def test_verdict_count(self) -> None:
        assert len(GuardianVerdict) == 3


class TestActionBudget:
    """Resource budget frozen dataclass."""

    def test_defaults(self) -> None:
        b = ActionBudget()
        assert b.time_ms == 10_000
        assert b.s2_tokens_max == 50_000
        assert b.retry_max == 2
        assert b.action_limit == 100

    def test_frozen(self) -> None:
        b = ActionBudget()
        try:
            b.time_ms = 999  # type: ignore[misc]
            assert False, "Should raise FrozenInstanceError"
        except AttributeError:
            pass


class TestActionEnvelope:
    """Immutable command envelope."""

    def test_create_envelope(self) -> None:
        e = ActionEnvelope(
            action_id="abc123",
            kind="mission.search.web",
            channel="browser",
        )
        assert e.action_id == "abc123"
        assert e.kind == "mission.search.web"
        assert e.channel == "browser"
        assert e.payload == {}
        assert e.capabilities == ()
        assert e.budget.time_ms == 10_000

    def test_envelope_frozen(self) -> None:
        e = ActionEnvelope(action_id="x", kind="y", channel="z")
        try:
            e.kind = "changed"  # type: ignore[misc]
            assert False, "Should raise FrozenInstanceError"
        except AttributeError:
            pass


class TestBusActionReceipt:
    """Immutable action receipt with merkle chain."""

    def test_create_receipt(self) -> None:
        r = BusActionReceipt(
            receipt_id="r1",
            action_id="a1",
            status=ActionStatus.COMPLETED,
            outcome_hash="h1",
        )
        assert r.status == ActionStatus.COMPLETED
        assert r.prev_receipt_hash == "genesis"
        assert r.ihsan_score == 0.0
        assert r.guardian_verdict == "allowed"

    def test_receipt_frozen(self) -> None:
        r = BusActionReceipt(
            receipt_id="r1",
            action_id="a1",
            status=ActionStatus.FAILED,
            outcome_hash="h1",
            error_message="something",
        )
        assert r.error_message == "something"
        try:
            r.status = ActionStatus.COMPLETED  # type: ignore[misc]
            assert False, "Should raise FrozenInstanceError"
        except AttributeError:
            pass
