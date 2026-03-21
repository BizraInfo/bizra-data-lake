"""
ActionBus Tests — Phase 68.01
═════════════════════════════

TDD anchors for CQRS command pipeline: propose, cancel, idempotency,
receipt chain, event emission, budget enforcement.

Standing on Giants:
- Beck (2002): TDD by Example
- Fowler (2005): CQRS pattern
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.bus.action_bus import _GENESIS_HASH, ActionBus, FATEResult
from core.bus.channels import ChannelResult
from core.bus.telescript import Capability, TeleScriptEngine, TeleScriptPolicy
from core.bus.types import ActionBudget, ActionEnvelope, ActionStatus

# ═══════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════


def _full_policy() -> TeleScriptPolicy:
    return TeleScriptPolicy(
        allow=frozenset(c.value for c in Capability),
        deny=frozenset(),
    )


def _make_action(
    action_id: str = "act-001",
    kind: str = "file.organize",
    channel: str = "file",
    capabilities: tuple[str, ...] = ("file_read",),
    budget_ms: int = 5000,
) -> ActionEnvelope:
    return ActionEnvelope(
        action_id=action_id,
        kind=kind,
        channel=channel,
        capabilities=capabilities,
        budget=ActionBudget(time_ms=budget_ms),
    )


def _mock_channel(success: bool = True, outcome_hash: str = "h1") -> AsyncMock:
    ch = AsyncMock()
    ch.execute = AsyncMock(
        return_value=ChannelResult(success=success, outcome_hash=outcome_hash)
    )
    return ch


def _mock_event_bus() -> AsyncMock:
    bus = AsyncMock()
    bus.publish = AsyncMock()
    return bus


def _bus(
    channels: dict | None = None,
    fate_gate: MagicMock | None = None,
    event_bus: AsyncMock | None = None,
    policy: TeleScriptPolicy | None = None,
) -> ActionBus:
    return ActionBus(
        telescript=TeleScriptEngine(policy or _full_policy()),
        channels=channels or {"file": _mock_channel()},
        fate_gate=fate_gate,
        event_bus=event_bus,
    )


# ═══════════════════════════════════════════════════════════
# Propose lifecycle
# ═══════════════════════════════════════════════════════════


class TestActionBusPropose:
    """Core propose lifecycle."""

    @pytest.mark.asyncio
    async def test_propose_returns_receipt_on_success(self) -> None:
        bus = _bus()
        receipt = await bus.propose(_make_action())
        assert receipt.status == ActionStatus.COMPLETED
        assert receipt.action_id == "act-001"

    @pytest.mark.asyncio
    async def test_propose_emits_intent_event(self) -> None:
        eb = _mock_event_bus()
        bus = _bus(event_bus=eb)
        await bus.propose(_make_action())
        topics = [call.args[0] for call in eb.publish.call_args_list]
        assert "action.intent" in topics

    @pytest.mark.asyncio
    async def test_propose_denied_by_telescript(self) -> None:
        bus = _bus(
            policy=TeleScriptPolicy(
                allow=frozenset(["file_read"]),
                deny=frozenset(["shell_execute"]),
            )
        )
        action = _make_action(capabilities=("shell_execute",))
        receipt = await bus.propose(action)
        assert receipt.status == ActionStatus.DENIED
        assert receipt.guardian_verdict == "denied"

    @pytest.mark.asyncio
    async def test_propose_denied_by_fate_gate(self) -> None:
        fate = MagicMock()
        fate.evaluate = MagicMock(
            return_value=FATEResult(
                denied=True, reason="constitutional_veto", reason_codes=("F-001",)
            )
        )
        eb = _mock_event_bus()
        bus = _bus(fate_gate=fate, event_bus=eb)
        receipt = await bus.propose(_make_action())
        assert receipt.status == ActionStatus.DENIED
        topics = [call.args[0] for call in eb.publish.call_args_list]
        assert "policy.fate.vetoed" in topics

    @pytest.mark.asyncio
    async def test_propose_timeout_returns_failed(self) -> None:
        async def slow_execute(_action):
            await asyncio.sleep(10)
            return ChannelResult(success=True)  # never reached

        slow_channel = AsyncMock()
        slow_channel.execute = slow_execute
        bus = _bus(channels={"file": slow_channel})
        action = _make_action(budget_ms=50)  # 50ms timeout
        receipt = await bus.propose(action)
        assert receipt.status == ActionStatus.FAILED

    @pytest.mark.asyncio
    async def test_propose_channel_not_found(self) -> None:
        bus = _bus(channels={})
        receipt = await bus.propose(_make_action(channel="nonexistent"))
        assert receipt.status == ActionStatus.FAILED
        assert "Unknown channel" in receipt.error_message


# ═══════════════════════════════════════════════════════════
# Idempotency
# ═══════════════════════════════════════════════════════════


class TestIdempotency:
    """Same action_id must not re-execute."""

    @pytest.mark.asyncio
    async def test_duplicate_action_returns_same_receipt(self) -> None:
        bus = _bus()
        action = _make_action()
        r1 = await bus.propose(action)
        r2 = await bus.propose(action)
        assert r1.receipt_id == r2.receipt_id

    @pytest.mark.asyncio
    async def test_duplicate_action_does_not_re_execute(self) -> None:
        ch = _mock_channel()
        bus = _bus(channels={"file": ch})
        action = _make_action()
        await bus.propose(action)
        await bus.propose(action)
        assert ch.execute.call_count == 1


# ═══════════════════════════════════════════════════════════
# Receipt chain
# ═══════════════════════════════════════════════════════════


class TestReceiptChain:
    """Merkle-linked receipt chain."""

    @pytest.mark.asyncio
    async def test_receipt_chain_genesis_has_zero_prev(self) -> None:
        bus = _bus()
        receipt = await bus.propose(_make_action())
        assert receipt.prev_receipt_hash == _GENESIS_HASH

    @pytest.mark.asyncio
    async def test_receipt_chain_links_prev_hash(self) -> None:
        bus = _bus()
        r1 = await bus.propose(_make_action(action_id="a1"))
        r2 = await bus.propose(_make_action(action_id="a2"))
        assert r2.prev_receipt_hash == r1.receipt_id

    @pytest.mark.asyncio
    async def test_receipt_hash_deterministic(self) -> None:
        bus = _bus()
        r1 = await bus.propose(_make_action())
        assert len(r1.receipt_id) == 64  # blake2b hex


# ═══════════════════════════════════════════════════════════
# Cancel
# ═══════════════════════════════════════════════════════════


class TestCancel:
    """Cancel pending actions."""

    @pytest.mark.asyncio
    async def test_cancel_pending_action(self) -> None:
        bus = _bus()
        receipt = await bus.cancel("pending-001")
        assert receipt.status == ActionStatus.CANCELLED
        assert receipt.action_id == "pending-001"

    @pytest.mark.asyncio
    async def test_cancel_already_executed_raises(self) -> None:
        bus = _bus()
        await bus.propose(_make_action(action_id="done-001"))
        with pytest.raises(RuntimeError, match="Cannot cancel"):
            await bus.cancel("done-001")


# ═══════════════════════════════════════════════════════════
# Event emission
# ═══════════════════════════════════════════════════════════


class TestEventEmission:
    """Event bus integration."""

    @pytest.mark.asyncio
    async def test_success_emits_action_receipt(self) -> None:
        eb = _mock_event_bus()
        bus = _bus(event_bus=eb)
        await bus.propose(_make_action())
        topics = [call.args[0] for call in eb.publish.call_args_list]
        assert "action.receipt" in topics

    @pytest.mark.asyncio
    async def test_failure_emits_action_receipt_failed(self) -> None:
        failing = _mock_channel(success=False)
        eb = _mock_event_bus()
        bus = _bus(channels={"file": failing}, event_bus=eb)
        await bus.propose(_make_action())
        topics = [call.args[0] for call in eb.publish.call_args_list]
        assert "action.receipt.failed" in topics

    @pytest.mark.asyncio
    async def test_deny_emits_policy_event(self) -> None:
        eb = _mock_event_bus()
        bus = _bus(
            policy=TeleScriptPolicy(allow=frozenset(), deny=frozenset()),
            event_bus=eb,
        )
        await bus.propose(_make_action())
        topics = [call.args[0] for call in eb.publish.call_args_list]
        assert "policy.telescript.denied" in topics

    @pytest.mark.asyncio
    async def test_supports_sovereign_event_bus_shape(self) -> None:
        from core.sovereign.event_bus import EventBus

        eb = EventBus()
        bus = _bus(event_bus=eb)
        await bus.propose(_make_action())

        stats = eb.stats()
        assert stats["events_published"] == 2
        assert stats["queue_size"] == 2
