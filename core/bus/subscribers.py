"""
BIZRA EventBus Subscriber Wiring — All 12 Subscribers
======================================================
Drop into: core/bus/subscribers.py

This module wires the 12 EventBus subscribers identified in the
Ω∞ Peak Synthesis as the brain-body gap. Each subscriber listens
for a specific event type and triggers the appropriate downstream action.

Phase 1 (Learning Loop): Subscribers 1-4
Phase 2 (Safety): Subscribers 5-7
Phase 3 (Economics): Subscribers 8-12

Standing on Giants: Hewitt (actor model), Deming (PDCA), Boyd (OODA)
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol

logger = logging.getLogger("bizra.bus.subscribers")


# ═══════════════════════════════════════════════════════════════════
# EVENT TYPES
# ═══════════════════════════════════════════════════════════════════


class EventType(str, Enum):
    # Core lifecycle
    ACTION_INTENT = "action.intent"
    ACTION_RECEIPT = "action.receipt"
    ACTION_RECEIPT_FAILED = "action.receipt.failed"
    TELESCRIPT_STEP = "telescript.step"
    TELESCRIPT_COMPLETED = "telescript.completed"
    TELESCRIPT_ROLLED_BACK = "telescript.rolled_back"
    SESSION_END = "session.end"
    # Memory
    MEMORY_PROMOTED = "memory.promoted"
    MEMORY_RETRIEVED = "memory.retrieved"
    # Constitutional
    IHSAN_GATE_BREACHED = "ihsan.gate.breached"
    # Agent
    AGENT_REGISTERED = "agent.registered"


@dataclass
class Event:
    """Immutable event record for the EventBus."""

    event_type: EventType
    payload: Dict[str, Any]
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    event_id: str = ""
    prev_hash: str = ""
    event_hash: str = ""

    def __post_init__(self):
        if not self.event_id:
            self.event_id = f"evt_{int(time.time() * 1000)}_{id(self) % 10000:04d}"
        if not self.event_hash:
            self.event_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        content = json.dumps(
            {
                "type": self.event_type,
                "payload": self.payload,
                "timestamp": self.timestamp,
                "prev_hash": self.prev_hash,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.blake2b(
            b"EVENTBUS_DOMAIN:" + content.encode(), digest_size=32
        ).hexdigest()


@dataclass
class SubscriberDeliveryReceipt:
    """Per-subscriber delivery evidence for one published event."""

    event_id: str
    event_hash: str
    event_type: str
    subscriber_name: str
    status: str
    safety_critical: bool
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    error: str = ""
    delivery_hash: str = ""

    def __post_init__(self) -> None:
        if not self.delivery_hash:
            self.delivery_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        content = json.dumps(
            {
                "event_id": self.event_id,
                "event_hash": self.event_hash,
                "event_type": self.event_type,
                "subscriber_name": self.subscriber_name,
                "status": self.status,
                "safety_critical": self.safety_critical,
                "timestamp": self.timestamp,
                "error": self.error,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.blake2b(
            b"EVENTBUS_DELIVERY_DOMAIN:" + content.encode(),
            digest_size=32,
        ).hexdigest()

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════
# SUBSCRIBER PROTOCOL
# ═══════════════════════════════════════════════════════════════════


class Subscriber(Protocol):
    """Protocol for all EventBus subscribers."""

    event_types: List[EventType]

    def handle(self, event: Event) -> None: ...


# ═══════════════════════════════════════════════════════════════════
# EVENTBUS CORE
# ═══════════════════════════════════════════════════════════════════


class EventBus:
    """
    Append-only, BLAKE3 hash-chained event log.
    Single source of truth for all state changes.
    """

    def __init__(
        self,
        *,
        delivery_receipt_path: Optional[Path | str] = None,
        delivery_receipt_sink: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self._subscribers: Dict[EventType, List[Subscriber]] = {}
        self._chain: List[Event] = []
        self._chain_hash = "0" * 64  # Genesis hash
        self._delivery_receipts: List[SubscriberDeliveryReceipt] = []
        self._last_delivery_ack: Optional[Dict[str, Any]] = None
        self._last_dead_letter: Optional[Dict[str, Any]] = None
        self._delivery_ack_count = 0
        self._dead_letter_count = 0
        self._delivery_receipt_path = (
            Path(delivery_receipt_path) if delivery_receipt_path else None
        )
        self._delivery_receipt_sink = delivery_receipt_sink
        self._persisted_delivery_receipts = 0
        self._last_persistence_error = ""
        self._delivery_sink_failures = 0
        self._last_delivery_sink_error = ""

    def subscribe(self, subscriber: Subscriber) -> None:
        for et in subscriber.event_types:
            self._subscribers.setdefault(et, []).append(subscriber)
            logger.info(
                f"Subscriber wired: {subscriber.__class__.__name__} -> {et.value}"
            )

    def publish(self, event_type: EventType, payload: Dict[str, Any]) -> Event:
        event = Event(
            event_type=event_type,
            payload=payload,
            prev_hash=self._chain_hash,
        )
        self._chain.append(event)
        self._chain_hash = event.event_hash

        handlers = self._subscribers.get(event_type, [])
        for subscriber in handlers:
            safety_critical = self._is_fail_closed_subscriber(subscriber)
            try:
                subscriber.handle(event)
            except Exception as e:  # noqa: BLE001 — boundary boundary
                error = f"{type(e).__name__}: {e}"
                self._record_delivery_receipt(
                    event=event,
                    subscriber=subscriber,
                    status="dead_letter",
                    safety_critical=safety_critical,
                    error=error,
                )
                logger.error(f"Subscriber {subscriber.__class__.__name__} failed: {e}")
                # Fail-open for non-safety subscribers, fail-closed for safety
                if safety_critical:
                    raise
            else:
                self._record_delivery_receipt(
                    event=event,
                    subscriber=subscriber,
                    status="ack",
                    safety_critical=safety_critical,
                )

        return event

    @property
    def chain_height(self) -> int:
        return len(self._chain)

    def verify_chain(self) -> bool:
        prev = "0" * 64
        for event in self._chain:
            if event.prev_hash != prev:
                return False
            prev = event.event_hash
        return True

    def delivery_summary(self) -> Dict[str, Any]:
        total = len(self._delivery_receipts)
        dead_letter_rate = (
            round(self._dead_letter_count / total, 4) if total else 0.0
        )
        return {
            "delivery_receipts": total,
            "delivery_acks": self._delivery_ack_count,
            "delivery_dead_letters": self._dead_letter_count,
            "dead_letter_rate": dead_letter_rate,
            "delivery_persistence_enabled": self._delivery_receipt_path is not None,
            "delivery_receipt_path": (
                str(self._delivery_receipt_path) if self._delivery_receipt_path else ""
            ),
            "persisted_delivery_receipts": self._persisted_delivery_receipts,
            "last_persistence_error": self._last_persistence_error,
            "delivery_sink_enabled": self._delivery_receipt_sink is not None,
            "delivery_sink_failures": self._delivery_sink_failures,
            "last_delivery_sink_error": self._last_delivery_sink_error,
            "last_delivery_ack": self._last_delivery_ack,
            "last_dead_letter": self._last_dead_letter,
        }

    def delivery_receipts(
        self, *, event_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        receipts = self._delivery_receipts
        if event_id is not None:
            receipts = [receipt for receipt in receipts if receipt.event_id == event_id]
        return [receipt.as_dict() for receipt in receipts]

    @property
    def delivery_receipt_count(self) -> int:
        return len(self._delivery_receipts)

    @property
    def dead_letter_count(self) -> int:
        return self._dead_letter_count

    def _is_fail_closed_subscriber(self, subscriber: Subscriber) -> bool:
        return isinstance(subscriber, (IhsanGateBreachHandler, FailedActionQuarantine))

    def _record_delivery_receipt(
        self,
        *,
        event: Event,
        subscriber: Subscriber,
        status: str,
        safety_critical: bool,
        error: str = "",
    ) -> None:
        receipt = SubscriberDeliveryReceipt(
            event_id=event.event_id,
            event_hash=event.event_hash,
            event_type=str(event.event_type.value),
            subscriber_name=subscriber.__class__.__name__,
            status=status,
            safety_critical=safety_critical,
            error=error,
        )
        payload = receipt.as_dict()
        self._delivery_receipts.append(receipt)
        if status == "ack":
            self._delivery_ack_count += 1
            self._last_delivery_ack = payload
        else:
            self._dead_letter_count += 1
            self._last_dead_letter = payload
        self._persist_delivery_receipt(payload)
        self._emit_delivery_receipt(payload)

    def _persist_delivery_receipt(self, payload: Dict[str, Any]) -> None:
        if self._delivery_receipt_path is None:
            return
        try:
            self._delivery_receipt_path.parent.mkdir(parents=True, exist_ok=True)
            with self._delivery_receipt_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
        except (OSError, TypeError, ValueError) as exc:
            self._last_persistence_error = f"{type(exc).__name__}: {exc}"
            logger.error(
                "Failed to persist subscriber delivery receipt: %s",
                self._last_persistence_error,
            )
            return
        self._persisted_delivery_receipts += 1
        self._last_persistence_error = ""

    def _emit_delivery_receipt(self, payload: Dict[str, Any]) -> None:
        if self._delivery_receipt_sink is None:
            return
        try:
            self._delivery_receipt_sink(payload)
        except Exception as exc:  # noqa: BLE001 — boundary boundary
            self._delivery_sink_failures += 1
            self._last_delivery_sink_error = f"{type(exc).__name__}: {exc}"
            logger.error(
                "Failed to mirror subscriber delivery receipt: %s",
                self._last_delivery_sink_error,
            )
        else:
            self._last_delivery_sink_error = ""


# ═══════════════════════════════════════════════════════════════════
# PHASE 1: LEARNING LOOP (Subscribers 1-4)
# ═══════════════════════════════════════════════════════════════════


class ActionReceiptMemoryReinforce:
    """
    Subscriber 1: ActionReceipt → Memory Reinforce
    When an action completes successfully, reinforce the memory trace.
    This is how the system learns from its own actions.
    """

    event_types = [EventType.ACTION_RECEIPT]

    def __init__(self, memory_store):
        self.memory = memory_store

    def handle(self, event: Event) -> None:
        payload = event.payload
        ihsan = payload.get("ihsan_composite", 0.0)
        action_type = payload.get("action_type", "unknown")
        result_summary = payload.get("result_summary", "")

        # Reinforce: strengthen the memory trace proportional to Ihsān score
        memory_entry = {
            "action_type": action_type,
            "result": result_summary,
            "ihsan": ihsan,
            "timestamp": event.timestamp,
            "receipt_hash": event.event_hash,
        }

        self.memory.reinforce(
            key=f"action:{action_type}",
            entry=memory_entry,
            strength=ihsan,  # Higher Ihsān = stronger reinforcement
        )
        logger.info(f"[SUB-1] Memory reinforced: {action_type} (Ihsān={ihsan:.3f})")


class ActionIntentTeleScriptBegin:
    """
    Subscriber 2: ActionIntent → TeleScript Begin
    When a user expresses intent, begin the TeleScript workflow.
    This is the entry point of the OODA loop.
    """

    event_types = [EventType.ACTION_INTENT]

    def __init__(self, telescript_engine):
        self.telescript = telescript_engine

    def handle(self, event: Event) -> None:
        intent = event.payload.get("intent", "")
        context = event.payload.get("context", {})
        session_id = event.payload.get("session_id", "")

        execution_id = self.telescript.begin_execution(
            intent=intent,
            context=context,
            session_id=session_id,
            origin_event=event.event_id,
        )
        logger.info(
            f"[SUB-2] TeleScript begun: {execution_id} for intent: {intent[:50]}..."
        )


class TeleScriptStepReceiptAppend:
    """
    Subscriber 3: TeleScriptStep → Receipt Append
    Each step in a TeleScript workflow produces a receipt.
    This creates the granular proof chain.
    """

    event_types = [EventType.TELESCRIPT_STEP]

    def __init__(self, receipt_chain):
        self.receipts = receipt_chain

    def handle(self, event: Event) -> None:
        step = event.payload
        receipt = {
            "step_id": step.get("step_id", ""),
            "step_type": step.get("step_type", ""),
            "input_hash": step.get("input_hash", ""),
            "output_hash": step.get("output_hash", ""),
            "ihsan_components": step.get("ihsan_components", {}),
            "ihsan_composite": step.get("ihsan_composite", 0.0),
            "duration_ms": step.get("duration_ms", 0),
            "timestamp": event.timestamp,
            "chain_hash": event.event_hash,
        }
        self.receipts.append(receipt)
        logger.info(f"[SUB-3] Receipt appended: step {step.get('step_id', '?')}")


class SessionEndGenesisCompile:
    """
    Subscriber 4: SessionEnd → Genesis Mini-Compile
    When a session ends, check if any action patterns qualify
    for reflex precipitation (myelination).

    Rule: 3+ successful executions with Ihsān ≥ 0.90 → precipitate reflex.
    """

    event_types = [EventType.SESSION_END]

    def __init__(self, reflex_cache, memory_store):
        self.cache = reflex_cache
        self.memory = memory_store
        self.precipitation_threshold = 3
        self.ihsan_floor = 0.90

    def handle(self, event: Event) -> None:
        session_id = event.payload.get("session_id", "")
        session_actions = event.payload.get("actions", [])

        # Count successful high-Ihsān executions per action type
        type_counts: Dict[str, List[float]] = {}
        for action in session_actions:
            atype = action.get("action_type", "")
            ihsan = action.get("ihsan_composite", 0.0)
            if ihsan >= self.ihsan_floor:
                type_counts.setdefault(atype, []).append(ihsan)

        # Precipitate reflexes for qualifying patterns
        precipitated = 0
        for atype, scores in type_counts.items():
            total_count = self.memory.get_success_count(atype)
            total_count += len(scores)
            self.memory.set_success_count(atype, total_count)

            if total_count >= self.precipitation_threshold and atype not in self.cache:
                avg_ihsan = sum(scores) / len(scores)
                self.cache.precipitate(
                    action_type=atype,
                    avg_ihsan=avg_ihsan,
                    execution_count=total_count,
                    source_session=session_id,
                )
                precipitated += 1
                logger.info(
                    f"[SUB-4] ⚡ REFLEX PRECIPITATED: {atype} "
                    f"(count={total_count}, avg_ihsan={avg_ihsan:.3f})"
                )

        if precipitated > 0:
            logger.info(f"[SUB-4] Session {session_id}: {precipitated} new reflexes")


# ═══════════════════════════════════════════════════════════════════
# PHASE 2: SAFETY (Subscribers 5-7)
# ═══════════════════════════════════════════════════════════════════


class IhsanGateBreachHandler:
    """
    Subscriber 5: IhsānGateBreached → Session Halt
    If any output falls below the mission floor (0.85), halt the session.
    This is the constitutional enforcement — the Daughter Test at runtime.

    FAIL-CLOSED: This subscriber raises on failure, halting the pipeline.
    """

    event_types = [EventType.IHSAN_GATE_BREACHED]

    MISSION_FLOOR = 0.85

    def __init__(self, session_manager, audit_log):
        self.sessions = session_manager
        self.audit = audit_log

    def handle(self, event: Event) -> None:
        session_id = event.payload.get("session_id", "")
        ihsan_score = event.payload.get("ihsan_composite", 0.0)
        action_type = event.payload.get("action_type", "")
        violation_dims = event.payload.get("violation_dimensions", [])

        # Log the violation
        self.audit.log_violation(
            session_id=session_id,
            score=ihsan_score,
            floor=self.MISSION_FLOOR,
            action_type=action_type,
            dimensions=violation_dims,
            timestamp=event.timestamp,
        )

        # Halt the session
        self.sessions.halt(
            session_id=session_id,
            reason=f"Ihsān gate breached: {ihsan_score:.3f} < {self.MISSION_FLOOR}",
        )

        logger.warning(
            f"[SUB-5] 🛑 SESSION HALTED: {session_id} "
            f"(Ihsān={ihsan_score:.3f}, violated: {violation_dims})"
        )


class FailedActionQuarantine:
    """
    Subscriber 6: ActionReceipt[failed] → Quarantine
    Failed actions are isolated from memory to prevent toxic learning.
    The system must not learn from its mistakes as if they were successes.

    FAIL-CLOSED: Quarantine failures halt the pipeline.
    """

    event_types = [EventType.ACTION_RECEIPT_FAILED]

    def __init__(self, memory_store, quarantine_store):
        self.memory = memory_store
        self.quarantine = quarantine_store

    def handle(self, event: Event) -> None:
        action_type = event.payload.get("action_type", "")
        error = event.payload.get("error", "")
        context = event.payload.get("context", {})

        # Move to quarantine (never to main memory)
        self.quarantine.isolate(
            action_type=action_type,
            error=error,
            context=context,
            event_hash=event.event_hash,
            timestamp=event.timestamp,
        )

        # Decrement success count (prevent false precipitation)
        current = self.memory.get_success_count(action_type)
        if current > 0:
            self.memory.set_success_count(action_type, current - 1)

        logger.warning(f"[SUB-6] ⚠️ QUARANTINED: {action_type} — {error[:80]}")


class TeleScriptRollbackHealing:
    """
    Subscriber 7: TeleScriptRolledBack → Healing
    When a workflow rolls back, trigger self-repair.
    The system learns what went wrong and adjusts.
    """

    event_types = [EventType.TELESCRIPT_ROLLED_BACK]

    def __init__(self, healing_engine, memory_store):
        self.healer = healing_engine
        self.memory = memory_store

    def handle(self, event: Event) -> None:
        execution_id = event.payload.get("execution_id", "")
        rollback_reason = event.payload.get("reason", "")
        failed_step = event.payload.get("failed_step", "")
        event.payload.get("steps_completed", 0)

        # Route to healing engine
        healing_plan = self.healer.diagnose(
            execution_id=execution_id,
            reason=rollback_reason,
            failed_step=failed_step,
        )

        # Store the failure pattern for future avoidance
        self.memory.record_failure_pattern(
            pattern_key=f"rollback:{failed_step}",
            reason=rollback_reason,
            context=event.payload,
        )

        logger.info(
            f"[SUB-7] 🔧 HEALING: {execution_id} "
            f"(failed at step {failed_step}, plan: {healing_plan.strategy})"
        )


# ═══════════════════════════════════════════════════════════════════
# PHASE 3: ECONOMICS (Subscribers 8-12)
# ═══════════════════════════════════════════════════════════════════


class ActionReceiptHHMMPromotion:
    """
    Subscriber 8: ActionReceipt → HHMM Promotion
    Successful receipts promote action patterns from episodic
    to semantic memory (glacial memory formation).
    """

    event_types = [EventType.ACTION_RECEIPT]

    PROMOTION_THRESHOLD = 0.92  # Higher than precipitation (0.90)

    def __init__(self, hhmm_engine, memory_store):
        self.hhmm = hhmm_engine
        self.memory = memory_store

    def handle(self, event: Event) -> None:
        ihsan = event.payload.get("ihsan_composite", 0.0)
        if ihsan < self.PROMOTION_THRESHOLD:
            return  # Only promote high-quality patterns

        action_type = event.payload.get("action_type", "")
        macro_state = self.hhmm.classify(event.payload)

        promoted = self.memory.promote_to_semantic(
            action_type=action_type,
            macro_state=macro_state,
            ihsan=ihsan,
            evidence_hash=event.event_hash,
        )

        if promoted:
            logger.info(
                f"[SUB-8] 📈 PROMOTED to semantic: {action_type} "
                f"(macro={macro_state}, Ihsān={ihsan:.3f})"
            )


class MemoryPromotedPoICredit:
    """
    Subscriber 9: MemoryPromoted → PoI Credit
    When a pattern is promoted to semantic memory, it earns
    Proof-of-Impact credit toward token minting.
    """

    event_types = [EventType.MEMORY_PROMOTED]

    def __init__(self, poi_engine):
        self.poi = poi_engine

    def handle(self, event: Event) -> None:
        action_type = event.payload.get("action_type", "")
        event.payload.get("macro_state", "")
        ihsan = event.payload.get("ihsan", 0.0)

        credit = self.poi.accumulate(
            source="memory_promotion",
            action_type=action_type,
            quality=ihsan,
            evidence_hash=event.event_hash,
        )

        logger.info(
            f"[SUB-9] 💎 PoI credit: +{credit:.4f} "
            f"for {action_type} (total: {self.poi.total_credit:.4f})"
        )


class TeleScriptCompletedPoIAccumulate:
    """
    Subscriber 10: TeleScriptCompleted → PoI Accumulate
    Completed workflows accumulate Proof-of-Impact toward SEED minting.
    This is where verified work becomes economic value.
    """

    event_types = [EventType.TELESCRIPT_COMPLETED]

    MINTING_FLOOR = 0.95  # From constants.py

    def __init__(self, poi_engine, token_minter):
        self.poi = poi_engine
        self.minter = token_minter

    def handle(self, event: Event) -> None:
        execution_id = event.payload.get("execution_id", "")
        ihsan = event.payload.get("ihsan_composite", 0.0)
        steps = event.payload.get("total_steps", 0)
        duration_ms = event.payload.get("duration_ms", 0)

        # Accumulate PoI
        credit = self.poi.accumulate(
            source="telescript_completion",
            execution_id=execution_id,
            quality=ihsan,
            steps=steps,
            evidence_hash=event.event_hash,
        )

        # Mint SEED if above minting floor
        if ihsan >= self.MINTING_FLOOR:
            seed_amount = self.minter.compute_reward(
                ihsan=ihsan,
                steps=steps,
                duration_ms=duration_ms,
            )
            self.minter.mint_seed(
                amount=seed_amount,
                poi_evidence=event.event_hash,
                ihsan=ihsan,
            )
            logger.info(
                f"[SUB-10] 🌱 SEED MINTED: {seed_amount:.4f} "
                f"(Ihsān={ihsan:.3f}, PoI credit={credit:.4f})"
            )
        else:
            logger.info(
                f"[SUB-10] PoI accumulated but below minting floor: "
                f"Ihsān={ihsan:.3f} < {self.MINTING_FLOOR}"
            )


class MemoryRetrievedBudgetReport:
    """
    Subscriber 11: MemoryRetrieved → Budget Report
    Track context budget per retrieval to prevent overflow.
    Each retrieval consumes context window tokens.
    """

    event_types = [EventType.MEMORY_RETRIEVED]

    MAX_CONTEXT_TOKENS = 128_000  # Conservative for most LLMs

    def __init__(self, context_budget):
        self.budget = context_budget

    def handle(self, event: Event) -> None:
        tokens_used = event.payload.get("tokens_retrieved", 0)
        source = event.payload.get("memory_source", "unknown")
        query = event.payload.get("query", "")

        self.budget.record_retrieval(
            tokens=tokens_used,
            source=source,
            query_hash=hashlib.sha256(query.encode()).hexdigest()[:16],
        )

        self.MAX_CONTEXT_TOKENS - self.budget.total_used
        utilization = self.budget.total_used / self.MAX_CONTEXT_TOKENS

        if utilization > 0.85:
            logger.warning(
                f"[SUB-11] ⚠️ Context budget at {utilization:.0%} "
                f"({self.budget.total_used}/{self.MAX_CONTEXT_TOKENS})"
            )
        else:
            logger.debug(
                f"[SUB-11] Context budget: {utilization:.0%} "
                f"(+{tokens_used} from {source})"
            )


class AgentRegisteredSelfModelUpdate:
    """
    Subscriber 12: AgentRegistered → Self-Model Update
    When an agent registers (or re-registers), update the node's
    self-model to reflect current capabilities.
    This is RSI Pillar I: the system knows what it can do.
    """

    event_types = [EventType.AGENT_REGISTERED]

    def __init__(self, self_model, capability_registry):
        self.model = self_model
        self.registry = capability_registry

    def handle(self, event: Event) -> None:
        agent_id = event.payload.get("agent_id", "")
        agent_type = event.payload.get("agent_type", "")  # PAT or SAT
        capabilities = event.payload.get("capabilities", [])
        version = event.payload.get("version", "0.0.0")

        # Update capability registry
        self.registry.register(
            agent_id=agent_id,
            agent_type=agent_type,
            capabilities=capabilities,
            version=version,
        )

        # Update self-model
        self.model.update_capability_map(
            total_agents=self.registry.count(),
            pat_agents=self.registry.count_by_type("PAT"),
            sat_agents=self.registry.count_by_type("SAT"),
            total_capabilities=self.registry.total_capabilities(),
            capability_vector=self.registry.capability_vector(),
        )

        logger.info(
            f"[SUB-12] 🤖 Self-model updated: {agent_type}:{agent_id} "
            f"({len(capabilities)} capabilities, v{version})"
        )


# ═══════════════════════════════════════════════════════════════════
# WIRING: Connect all 12 subscribers to the EventBus
# ═══════════════════════════════════════════════════════════════════


def wire_all_subscribers(
    bus: EventBus,
    *,
    memory_store,
    telescript_engine,
    receipt_chain,
    reflex_cache,
    session_manager,
    audit_log,
    quarantine_store,
    healing_engine,
    hhmm_engine,
    poi_engine,
    token_minter,
    context_budget,
    self_model,
    capability_registry,
) -> List[Subscriber]:
    """
    Wire all 12 EventBus subscribers.

    Call this during node initialization (genesis sequence).
    All dependencies must be initialized before calling this.

    Returns the list of wired subscribers for testing/inspection.
    """
    subscribers = [
        # Phase 1: Learning Loop
        ActionReceiptMemoryReinforce(memory_store),
        ActionIntentTeleScriptBegin(telescript_engine),
        TeleScriptStepReceiptAppend(receipt_chain),
        SessionEndGenesisCompile(reflex_cache, memory_store),
        # Phase 2: Safety
        IhsanGateBreachHandler(session_manager, audit_log),
        FailedActionQuarantine(memory_store, quarantine_store),
        TeleScriptRollbackHealing(healing_engine, memory_store),
        # Phase 3: Economics
        ActionReceiptHHMMPromotion(hhmm_engine, memory_store),
        MemoryPromotedPoICredit(poi_engine),
        TeleScriptCompletedPoIAccumulate(poi_engine, token_minter),
        MemoryRetrievedBudgetReport(context_budget),
        AgentRegisteredSelfModelUpdate(self_model, capability_registry),
    ]

    for sub in subscribers:
        bus.subscribe(sub)

    logger.info(f"═══ ALL 12 SUBSCRIBERS WIRED ═══ (chain height: {bus.chain_height})")
    return subscribers


# ═══════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════


def _run_smoke_tests():
    """Quick smoke test that all subscribers can be instantiated and wired."""

    # Minimal mock objects
    class MockStore:
        def __init__(self):
            self._data = {}
            self._counts = {}

        def reinforce(self, **kw):
            self._data[kw.get("key", "")] = kw

        def get_success_count(self, key):
            return self._counts.get(key, 0)

        def set_success_count(self, key, val):
            self._counts[key] = val

        def promote_to_semantic(self, **kw):
            return True

        def record_failure_pattern(self, **kw):
            pass

    class MockTeleScript:
        def begin_execution(self, **kw):
            return f"ts_{int(time.time())}"

    class MockReceiptChain(list):
        pass

    class MockReflexCache(dict):
        def precipitate(self, **kw):
            self[kw["action_type"]] = kw

    class MockSessionManager:
        def halt(self, **kw):
            pass

    class MockAuditLog:
        def log_violation(self, **kw):
            pass

    class MockQuarantine:
        def isolate(self, **kw):
            pass

    class MockHealing:
        def diagnose(self, **kw):
            class Plan:
                strategy = "retry"

            return Plan()

    class MockHHMM:
        def classify(self, payload):
            return "macro_general"

    class MockPoI:
        total_credit = 0.0

        def accumulate(self, **kw):
            self.total_credit += 0.01
            return 0.01

    class MockMinter:
        def compute_reward(self, **kw):
            return 0.05

        def mint_seed(self, **kw):
            pass

    class MockBudget:
        total_used = 0

        def record_retrieval(self, **kw):
            self.total_used += kw.get("tokens", 0)

    class MockSelfModel:
        def update_capability_map(self, **kw):
            pass

    class MockCapRegistry:
        def register(self, **kw):
            pass

        def count(self):
            return 7

        def count_by_type(self, t):
            return 7 if t == "PAT" else 5

        def total_capabilities(self):
            return 42

        def capability_vector(self):
            return [1.0] * 8

    # Wire everything
    bus = EventBus()
    subs = wire_all_subscribers(
        bus,
        memory_store=MockStore(),
        telescript_engine=MockTeleScript(),
        receipt_chain=MockReceiptChain(),
        reflex_cache=MockReflexCache(),
        session_manager=MockSessionManager(),
        audit_log=MockAuditLog(),
        quarantine_store=MockQuarantine(),
        healing_engine=MockHealing(),
        hhmm_engine=MockHHMM(),
        poi_engine=MockPoI(),
        token_minter=MockMinter(),
        context_budget=MockBudget(),
        self_model=MockSelfModel(),
        capability_registry=MockCapRegistry(),
    )

    assert len(subs) == 12, f"Expected 12 subscribers, got {len(subs)}"

    # Test Phase 1: Learning loop
    bus.publish(EventType.ACTION_INTENT, {"intent": "test task", "session_id": "s1"})
    bus.publish(EventType.TELESCRIPT_STEP, {"step_id": "1", "ihsan_composite": 0.95})
    bus.publish(
        EventType.ACTION_RECEIPT,
        {"action_type": "test", "ihsan_composite": 0.96, "result_summary": "ok"},
    )
    bus.publish(
        EventType.SESSION_END,
        {
            "session_id": "s1",
            "actions": [
                {"action_type": "test", "ihsan_composite": 0.95},
                {"action_type": "test", "ihsan_composite": 0.93},
                {"action_type": "test", "ihsan_composite": 0.91},
            ],
        },
    )

    # Test Phase 2: Safety
    bus.publish(
        EventType.IHSAN_GATE_BREACHED,
        {
            "session_id": "s2",
            "ihsan_composite": 0.72,
            "action_type": "risky",
            "violation_dimensions": ["safety"],
        },
    )
    bus.publish(
        EventType.ACTION_RECEIPT_FAILED,
        {"action_type": "broken", "error": "null pointer"},
    )
    bus.publish(
        EventType.TELESCRIPT_ROLLED_BACK,
        {"execution_id": "ts1", "reason": "timeout", "failed_step": "step3"},
    )

    # Test Phase 3: Economics
    bus.publish(
        EventType.MEMORY_PROMOTED,
        {"action_type": "promoted", "macro_state": "m1", "ihsan": 0.97},
    )
    bus.publish(
        EventType.TELESCRIPT_COMPLETED,
        {
            "execution_id": "ts2",
            "ihsan_composite": 0.96,
            "total_steps": 5,
            "duration_ms": 1200,
        },
    )
    bus.publish(
        EventType.MEMORY_RETRIEVED,
        {"tokens_retrieved": 500, "memory_source": "episodic", "query": "test"},
    )
    bus.publish(
        EventType.AGENT_REGISTERED,
        {
            "agent_id": "atlas",
            "agent_type": "PAT",
            "capabilities": ["plan", "decompose"],
            "version": "1.0.0",
        },
    )

    # Verify chain integrity
    assert bus.verify_chain(), "Chain integrity check failed!"
    assert bus.chain_height == 11, f"Expected 11 events, got {bus.chain_height}"

    print("═══ ALL 12 SUBSCRIBERS: SMOKE TEST PASSED ═══")
    print(f"  Events processed: {bus.chain_height}")
    print("  Chain integrity: VERIFIED")
    print(f"  Subscribers wired: {len(subs)}/12")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _run_smoke_tests()
