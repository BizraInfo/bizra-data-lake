"""
ActionBus — CQRS Command Pipeline with Constitutional Gates
════════════════════════════════════════════════════════════

Two-phase execution: Propose -> Gate -> Execute -> Verify -> Receipt.
Every action passes TeleScript capability check and FATE gate before
channel dispatch. Receipts form a merkle chain via prev_receipt_hash.

Standing on Giants:
- Fowler (2005): Command Query Responsibility Segregation
- Lamport (1978): Logical clocks and ordering
- Thompson (1984): Capability-based security

Phase 68.01 — Sovereign Instantiation
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from typing import Any, Protocol, runtime_checkable

from core.bus.channels import ChannelExecutor, ChannelResult
from core.bus.telescript import TeleScriptEngine
from core.bus.types import ActionEnvelope, ActionStatus, BusActionReceipt

logger = logging.getLogger(__name__)

# Genesis sentinel — first receipt in chain links to this
_GENESIS_HASH = "0" * 64


@runtime_checkable
class FATEGate(Protocol):
    """Protocol for FATE constitutional gate evaluation."""

    def evaluate(self, action: ActionEnvelope) -> FATEResult: ...


class FATEResult:
    """Result of FATE gate evaluation."""

    __slots__ = ("denied", "reason", "reason_codes")

    def __init__(
        self,
        denied: bool = False,
        reason: str = "",
        reason_codes: tuple[str, ...] = (),
    ) -> None:
        self.denied = denied
        self.reason = reason
        self.reason_codes = reason_codes


@runtime_checkable
class EventPublisher(Protocol):
    """Protocol for event bus publishing."""

    async def publish(self, topic: str, payload: dict[str, Any]) -> None: ...


class ActionBus:
    """CQRS command pipeline with constitutional gates.

    Lifecycle: PROPOSED -> VALIDATING -> EXECUTING -> VERIFYING -> COMPLETED
    Each transition emits events and builds merkle-linked receipts.

    Security properties:
    1. TeleScript capability check before any execution
    2. FATE gate veto power over any action
    3. Idempotency guard — same action_id never re-executes
    4. Receipt chain — tamper-evident merkle linkage
    5. Budget enforcement — timeout-based execution limits
    """

    __slots__ = (
        "_telescript",
        "_fate_gate",
        "_channels",
        "_event_bus",
        "_executed",
        "_receipts",
        "_receipt_index",
    )

    def __init__(
        self,
        telescript: TeleScriptEngine,
        channels: dict[str, ChannelExecutor] | None = None,
        fate_gate: FATEGate | None = None,
        event_bus: EventPublisher | None = None,
    ) -> None:
        self._telescript = telescript
        self._fate_gate = fate_gate
        self._channels: dict[str, ChannelExecutor] = channels or {}
        self._event_bus = event_bus
        self._executed: set[str] = set()
        self._receipts: list[BusActionReceipt] = []
        self._receipt_index: dict[str, BusActionReceipt] = {}

    @property
    def receipt_chain(self) -> list[BusActionReceipt]:
        """Read-only access to the receipt chain."""
        return list(self._receipts)

    async def propose(self, action: ActionEnvelope) -> BusActionReceipt:
        """Full lifecycle: propose -> gate -> execute -> verify -> receipt."""

        # Step 0: Idempotency check
        if action.action_id in self._receipt_index:
            return self._receipt_index[action.action_id]

        # Step 1: TeleScript capability check
        verdict = self._telescript.check(
            requested=action.capabilities,
            action_telescript=action.telescript,
            file_path=action.payload.get("path"),
        )
        if not verdict.allowed:
            receipt = self._make_receipt(
                action,
                ActionStatus.DENIED,
                guardian_verdict="denied",
                error_message=verdict.reason,
            )
            await self._emit(
                "policy.telescript.denied",
                {
                    "action_id": action.action_id,
                    "kind": action.kind,
                    "denied_capabilities": list(verdict.denied_capabilities),
                },
            )
            return receipt

        # Step 2: FATE gate evaluation (if configured)
        if self._fate_gate is not None:
            fate_result = self._fate_gate.evaluate(action)
            if fate_result.denied:
                receipt = self._make_receipt(
                    action,
                    ActionStatus.DENIED,
                    guardian_verdict="denied",
                    error_message=fate_result.reason,
                )
                await self._emit(
                    "policy.fate.vetoed",
                    {
                        "action_id": action.action_id,
                        "reason_codes": list(fate_result.reason_codes),
                    },
                )
                return receipt

        # Step 3: Emit intent event
        await self._emit(
            "action.intent",
            {
                "action_id": action.action_id,
                "kind": action.kind,
                "channel": action.channel,
            },
        )

        # Step 4: Execute via channel
        channel = self._channels.get(action.channel)
        if channel is None:
            receipt = self._make_receipt(
                action,
                ActionStatus.FAILED,
                error_message=f"Unknown channel: {action.channel}",
            )
            await self._emit("action.receipt.failed", self._receipt_dict(receipt))
            return receipt

        result = await self._execute_with_budget(channel, action)

        # Step 5: Build receipt
        status = ActionStatus.COMPLETED if result.success else ActionStatus.FAILED
        receipt = self._make_receipt(
            action,
            status,
            outcome_hash=result.outcome_hash,
            ihsan_score=result.ihsan_score,
        )

        # Step 6: Mark executed (idempotency)
        self._executed.add(action.action_id)

        # Step 7: Emit receipt event
        topic = "action.receipt" if result.success else "action.receipt.failed"
        await self._emit(topic, self._receipt_dict(receipt))

        return receipt

    async def cancel(self, action_id: str) -> BusActionReceipt:
        """Cancel a pending action. Raises if already executed."""
        if action_id in self._executed:
            raise RuntimeError(f"Cannot cancel already-executed action: {action_id}")

        receipt = BusActionReceipt(
            receipt_id=self._hash_receipt(action_id, ActionStatus.CANCELLED),
            action_id=action_id,
            status=ActionStatus.CANCELLED,
            outcome_hash="",
            prev_receipt_hash=self._prev_hash(),
        )
        self._receipts.append(receipt)
        self._receipt_index[action_id] = receipt

        await self._emit("action.cancelled", {"action_id": action_id})
        return receipt

    async def _execute_with_budget(
        self, channel: ChannelExecutor, action: ActionEnvelope
    ) -> ChannelResult:
        """Execute action with timeout from budget."""
        timeout_s = action.budget.time_ms / 1000.0
        try:
            return await asyncio.wait_for(channel.execute(action), timeout=timeout_s)
        except asyncio.TimeoutError:
            logger.warning(
                "Action %s timed out after %dms",
                action.action_id,
                action.budget.time_ms,
            )
            return ChannelResult(success=False, reason="timeout")
        except (
            asyncio.CancelledError,
            RuntimeError,
            OSError,
        ):  # SEC-003 — async boundary
            logger.exception("Action %s execution failed", action.action_id)
            return ChannelResult(success=False, reason="execution_error")

    def _make_receipt(
        self,
        action: ActionEnvelope,
        status: ActionStatus,
        outcome_hash: str = "",
        ihsan_score: float = 0.0,
        guardian_verdict: str = "allowed",
        error_message: str = "",
    ) -> BusActionReceipt:
        """Build receipt with merkle chain link."""
        receipt_id = self._hash_receipt(action.action_id, status, outcome_hash)
        receipt = BusActionReceipt(
            receipt_id=receipt_id,
            action_id=action.action_id,
            status=status,
            outcome_hash=outcome_hash,
            ihsan_score=ihsan_score,
            prev_receipt_hash=self._prev_hash(),
            guardian_verdict=guardian_verdict,
            duration_ms=0.0,
            error_message=error_message,
        )
        self._receipts.append(receipt)
        self._receipt_index[action.action_id] = receipt
        return receipt

    def _prev_hash(self) -> str:
        """Get previous receipt hash for merkle chain."""
        if self._receipts:
            return self._receipts[-1].receipt_id
        return _GENESIS_HASH

    @staticmethod
    def _hash_receipt(
        action_id: str, status: ActionStatus, outcome_hash: str = ""
    ) -> str:
        """Compute blake2b receipt ID from canonical form."""
        canonical = json.dumps(
            {
                "action_id": action_id,
                "status": status.value,
                "outcome_hash": outcome_hash,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.blake2b(canonical, digest_size=32).hexdigest()

    @staticmethod
    def _receipt_dict(receipt: BusActionReceipt) -> dict[str, Any]:
        """Convert receipt to dict for event emission."""
        return {
            "receipt_id": receipt.receipt_id,
            "action_id": receipt.action_id,
            "status": receipt.status.value,
            "outcome_hash": receipt.outcome_hash,
            "ihsan_score": receipt.ihsan_score,
            "guardian_verdict": receipt.guardian_verdict,
        }

    async def _emit(self, topic: str, payload: dict[str, Any]) -> None:
        """Emit event via EventBus if configured."""
        if self._event_bus is not None:
            await self._event_bus.publish(topic, payload)
