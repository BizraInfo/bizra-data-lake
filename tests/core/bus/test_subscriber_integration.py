"""
EventBus 12-Subscriber Integration Test — Phase 82
═══════════════════════════════════════════════════

Proves all 12 subscribers fire in correct sequence when events are
published through the EventBus. Each subscriber receives its event,
triggers its side-effect, and the hash chain stays valid throughout.

This is the nervous system closure test: if these pass, the BIZRA
organism can react to any action lifecycle event end-to-end.

Standing on Giants: Hewitt (actor model), Deming (PDCA), Boyd (OODA)
"""

from __future__ import annotations

import json
import time

import pytest

from core.bus.subscribers import (
    ActionReceiptMemoryReinforce,
    EventBus,
    EventType,
    FailedActionQuarantine,
    IhsanGateBreachHandler,
    wire_all_subscribers,
)

# ═══════════════════════════════════════════════════════════════
# Mock dependencies — minimal stubs that track side effects
# ═══════════════════════════════════════════════════════════════


class TrackingStore:
    """Memory store that records all calls."""

    def __init__(self):
        self.reinforcements: list[dict] = []
        self._counts: dict[str, int] = {}
        self.promotions: list[dict] = []
        self.failure_patterns: list[dict] = []

    def reinforce(self, **kw):
        self.reinforcements.append(kw)

    def get_success_count(self, key):
        return self._counts.get(key, 0)

    def set_success_count(self, key, val):
        self._counts[key] = val

    def promote_to_semantic(self, **kw):
        self.promotions.append(kw)
        return True

    def record_failure_pattern(self, **kw):
        self.failure_patterns.append(kw)


class TrackingTeleScript:
    """TeleScript engine that records begin_execution calls."""

    def __init__(self):
        self.executions: list[dict] = []

    def begin_execution(self, **kw):
        self.executions.append(kw)
        return f"ts_{int(time.time())}"


class TrackingReceiptChain(list):
    """Receipt chain as a list — appended by subscriber."""

    pass


class TrackingReflexCache(dict):
    """Reflex cache that records precipitations."""

    def __init__(self):
        super().__init__()
        self.precipitations: list[dict] = []

    def precipitate(self, **kw):
        self[kw["action_type"]] = kw
        self.precipitations.append(kw)


class TrackingSessionManager:
    """Session manager that records halts."""

    def __init__(self):
        self.halts: list[dict] = []

    def halt(self, **kw):
        self.halts.append(kw)


class TrackingAuditLog:
    """Audit log that records violations."""

    def __init__(self):
        self.violations: list[dict] = []

    def log_violation(self, **kw):
        self.violations.append(kw)


class TrackingQuarantine:
    """Quarantine store that records isolations."""

    def __init__(self):
        self.isolated: list[dict] = []

    def isolate(self, **kw):
        self.isolated.append(kw)


class TrackingHealing:
    """Healing engine that records diagnoses."""

    def __init__(self):
        self.diagnoses: list[dict] = []

    class Plan:
        strategy = "retry"

    def diagnose(self, **kw):
        self.diagnoses.append(kw)
        return self.Plan()


class TrackingHHMM:
    """Hierarchical HMM that records classifications."""

    def __init__(self):
        self.classifications: list = []

    def classify(self, payload):
        self.classifications.append(payload)
        return "macro_general"


class TrackingPoI:
    """PoI engine that tracks credit accumulations."""

    def __init__(self):
        self.total_credit = 0.0
        self.accumulations: list[dict] = []

    def accumulate(self, **kw):
        self.total_credit += 0.01
        self.accumulations.append(kw)
        return 0.01


class TrackingMinter:
    """Token minter that tracks reward computations and mints."""

    def __init__(self):
        self.rewards: list[dict] = []
        self.mints: list[dict] = []

    def compute_reward(self, **kw):
        self.rewards.append(kw)
        return 0.05

    def mint_seed(self, **kw):
        self.mints.append(kw)


class TrackingBudget:
    """Context budget that tracks retrievals."""

    def __init__(self):
        self.total_used = 0
        self.retrievals: list[dict] = []

    def record_retrieval(self, **kw):
        self.total_used += kw.get("tokens", 0)
        self.retrievals.append(kw)


class TrackingSelfModel:
    """Self-model that tracks capability map updates."""

    def __init__(self):
        self.updates: list[dict] = []

    def update_capability_map(self, **kw):
        self.updates.append(kw)


class TrackingCapRegistry:
    """Capability registry that tracks registrations."""

    def __init__(self):
        self.registrations: list[dict] = []

    def register(self, **kw):
        self.registrations.append(kw)

    def count(self):
        return 7 + len(self.registrations)

    def count_by_type(self, t):
        return 7 if t == "PAT" else 5

    def total_capabilities(self):
        return 42

    def capability_vector(self):
        return [1.0] * 8


# ═══════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════


@pytest.fixture
def deps():
    """All 14 dependency stubs with tracking."""
    return {
        "memory_store": TrackingStore(),
        "telescript_engine": TrackingTeleScript(),
        "receipt_chain": TrackingReceiptChain(),
        "reflex_cache": TrackingReflexCache(),
        "session_manager": TrackingSessionManager(),
        "audit_log": TrackingAuditLog(),
        "quarantine_store": TrackingQuarantine(),
        "healing_engine": TrackingHealing(),
        "hhmm_engine": TrackingHHMM(),
        "poi_engine": TrackingPoI(),
        "token_minter": TrackingMinter(),
        "context_budget": TrackingBudget(),
        "self_model": TrackingSelfModel(),
        "capability_registry": TrackingCapRegistry(),
    }


@pytest.fixture
def wired_bus(deps):
    """EventBus with all 12 subscribers wired."""
    bus = EventBus()
    subs = wire_all_subscribers(bus, **deps)
    return bus, subs, deps


# ═══════════════════════════════════════════════════════════════
# Test: All 12 subscribers wire correctly
# ═══════════════════════════════════════════════════════════════


class TestSubscriberWiring:
    """Verify all 12 subscribers are wired to the EventBus."""

    def test_wire_all_returns_12_subscribers(self, wired_bus):
        _, subs, _ = wired_bus
        assert len(subs) == 12

    def test_wire_all_subscriber_types(self, wired_bus):
        _, subs, _ = wired_bus
        types = {type(s).__name__ for s in subs}
        expected = {
            "ActionReceiptMemoryReinforce",
            "ActionIntentTeleScriptBegin",
            "TeleScriptStepReceiptAppend",
            "SessionEndGenesisCompile",
            "IhsanGateBreachHandler",
            "FailedActionQuarantine",
            "TeleScriptRollbackHealing",
            "ActionReceiptHHMMPromotion",
            "MemoryPromotedPoICredit",
            "TeleScriptCompletedPoIAccumulate",
            "MemoryRetrievedBudgetReport",
            "AgentRegisteredSelfModelUpdate",
        }
        assert types == expected


# ═══════════════════════════════════════════════════════════════
# Test: Phase 1 — Learning Loop (Subscribers 1-4)
# ═══════════════════════════════════════════════════════════════


class TestPhase1LearningLoop:
    """Subscribers 1-4: the learning pipeline."""

    def test_action_receipt_reinforces_memory(self, wired_bus):
        bus, _, deps = wired_bus
        bus.publish(
            EventType.ACTION_RECEIPT,
            {"action_type": "test", "ihsan_composite": 0.96, "result_summary": "ok"},
        )
        assert len(deps["memory_store"].reinforcements) == 1
        r = deps["memory_store"].reinforcements[0]
        assert "test" in r["key"]  # key may be prefixed (e.g. "action:test")

    def test_action_intent_begins_telescript(self, wired_bus):
        bus, _, deps = wired_bus
        bus.publish(
            EventType.ACTION_INTENT,
            {"intent": "analyze data", "session_id": "s1"},
        )
        assert len(deps["telescript_engine"].executions) == 1

    def test_telescript_step_appends_receipt(self, wired_bus):
        bus, _, deps = wired_bus
        bus.publish(
            EventType.TELESCRIPT_STEP,
            {"step_id": "1", "ihsan_composite": 0.95},
        )
        assert len(deps["receipt_chain"]) == 1

    def test_session_end_compiles_reflexes(self, wired_bus):
        bus, _, deps = wired_bus
        bus.publish(
            EventType.SESSION_END,
            {
                "session_id": "s1",
                "actions": [
                    {"action_type": "analyze", "ihsan_composite": 0.95},
                    {"action_type": "analyze", "ihsan_composite": 0.94},
                    {"action_type": "analyze", "ihsan_composite": 0.93},
                ],
            },
        )
        assert len(deps["reflex_cache"].precipitations) >= 1


# ═══════════════════════════════════════════════════════════════
# Test: Phase 2 — Safety (Subscribers 5-7)
# ═══════════════════════════════════════════════════════════════


class TestPhase2Safety:
    """Subscribers 5-7: constitutional safety gates."""

    def test_ihsan_breach_halts_session(self, wired_bus):
        bus, _, deps = wired_bus
        bus.publish(
            EventType.IHSAN_GATE_BREACHED,
            {
                "session_id": "s2",
                "ihsan_composite": 0.72,
                "action_type": "risky",
                "violation_dimensions": ["safety"],
            },
        )
        assert len(deps["session_manager"].halts) == 1
        assert len(deps["audit_log"].violations) == 1

    def test_failed_action_quarantined(self, wired_bus):
        bus, _, deps = wired_bus
        bus.publish(
            EventType.ACTION_RECEIPT_FAILED,
            {"action_type": "broken", "error": "null pointer"},
        )
        assert len(deps["quarantine_store"].isolated) == 1

    def test_telescript_rollback_triggers_healing(self, wired_bus):
        bus, _, deps = wired_bus
        bus.publish(
            EventType.TELESCRIPT_ROLLED_BACK,
            {"execution_id": "ts1", "reason": "timeout", "failed_step": "step3"},
        )
        assert len(deps["healing_engine"].diagnoses) == 1


# ═══════════════════════════════════════════════════════════════
# Test: Phase 3 — Economics (Subscribers 8-12)
# ═══════════════════════════════════════════════════════════════


class TestPhase3Economics:
    """Subscribers 8-12: economic reward and budget tracking."""

    def test_action_receipt_promotes_via_hhmm(self, wired_bus):
        bus, _, deps = wired_bus
        bus.publish(
            EventType.ACTION_RECEIPT,
            {
                "action_type": "promoted",
                "ihsan_composite": 0.97,
                "result_summary": "ok",
            },
        )
        assert len(deps["hhmm_engine"].classifications) == 1

    def test_memory_promoted_credits_poi(self, wired_bus):
        bus, _, deps = wired_bus
        bus.publish(
            EventType.MEMORY_PROMOTED,
            {"action_type": "promoted", "macro_state": "m1", "ihsan": 0.97},
        )
        assert len(deps["poi_engine"].accumulations) == 1

    def test_telescript_completed_accumulates_poi_and_mints(self, wired_bus):
        bus, _, deps = wired_bus
        bus.publish(
            EventType.TELESCRIPT_COMPLETED,
            {
                "execution_id": "ts2",
                "ihsan_composite": 0.96,
                "total_steps": 5,
                "duration_ms": 1200,
            },
        )
        assert deps["poi_engine"].total_credit > 0
        assert len(deps["token_minter"].rewards) == 1

    def test_memory_retrieved_reports_budget(self, wired_bus):
        bus, _, deps = wired_bus
        bus.publish(
            EventType.MEMORY_RETRIEVED,
            {"tokens_retrieved": 500, "memory_source": "episodic", "query": "test"},
        )
        assert deps["context_budget"].total_used == 500

    def test_agent_registered_updates_self_model(self, wired_bus):
        bus, _, deps = wired_bus
        bus.publish(
            EventType.AGENT_REGISTERED,
            {
                "agent_id": "atlas",
                "agent_type": "PAT",
                "capabilities": ["plan", "decompose"],
                "version": "1.0.0",
            },
        )
        assert len(deps["self_model"].updates) == 1
        assert len(deps["capability_registry"].registrations) == 1


# ═══════════════════════════════════════════════════════════════
# Test: Full sequence — all 12 fire in order
# ═══════════════════════════════════════════════════════════════


class TestFullSequence:
    """All 12 subscribers fire in sequence, chain stays valid."""

    def test_full_11_event_sequence_chain_valid(self, wired_bus):
        """Publish all 11 event types in lifecycle order; chain must remain valid."""
        bus, subs, deps = wired_bus

        # Phase 1: Learning Loop
        bus.publish(EventType.ACTION_INTENT, {"intent": "task", "session_id": "s1"})
        bus.publish(
            EventType.TELESCRIPT_STEP, {"step_id": "1", "ihsan_composite": 0.95}
        )
        bus.publish(
            EventType.ACTION_RECEIPT,
            {"action_type": "task", "ihsan_composite": 0.96, "result_summary": "ok"},
        )
        bus.publish(
            EventType.SESSION_END,
            {
                "session_id": "s1",
                "actions": [
                    {"action_type": "task", "ihsan_composite": 0.95},
                    {"action_type": "task", "ihsan_composite": 0.93},
                    {"action_type": "task", "ihsan_composite": 0.91},
                ],
            },
        )

        # Phase 2: Safety
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

        # Phase 3: Economics
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
        assert bus.chain_height == 11
        assert bus.verify_chain(), "Hash chain broken after full sequence"

    def test_full_sequence_all_side_effects_triggered(self, wired_bus):
        """Every subscriber must have triggered at least one side effect."""
        bus, _, deps = wired_bus

        # Fire all events
        bus.publish(EventType.ACTION_INTENT, {"intent": "t", "session_id": "s1"})
        bus.publish(
            EventType.TELESCRIPT_STEP, {"step_id": "1", "ihsan_composite": 0.95}
        )
        bus.publish(
            EventType.ACTION_RECEIPT,
            {"action_type": "t", "ihsan_composite": 0.96, "result_summary": "ok"},
        )
        bus.publish(
            EventType.SESSION_END,
            {
                "session_id": "s1",
                "actions": [
                    {"action_type": "t", "ihsan_composite": 0.95},
                    {"action_type": "t", "ihsan_composite": 0.94},
                    {"action_type": "t", "ihsan_composite": 0.93},
                ],
            },
        )
        bus.publish(
            EventType.IHSAN_GATE_BREACHED,
            {
                "session_id": "s2",
                "ihsan_composite": 0.72,
                "action_type": "r",
                "violation_dimensions": ["safety"],
            },
        )
        bus.publish(
            EventType.ACTION_RECEIPT_FAILED,
            {"action_type": "broken", "error": "err"},
        )
        bus.publish(
            EventType.TELESCRIPT_ROLLED_BACK,
            {"execution_id": "ts1", "reason": "timeout", "failed_step": "s3"},
        )
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
                "capabilities": ["plan"],
                "version": "1.0.0",
            },
        )

        # SUB-1: Memory reinforced
        assert len(deps["memory_store"].reinforcements) >= 1
        # SUB-2: TeleScript begun
        assert len(deps["telescript_engine"].executions) >= 1
        # SUB-3: Receipt appended
        assert len(deps["receipt_chain"]) >= 1
        # SUB-4: Reflex precipitated
        assert len(deps["reflex_cache"].precipitations) >= 1
        # SUB-5: Session halted + audit logged
        assert len(deps["session_manager"].halts) >= 1
        assert len(deps["audit_log"].violations) >= 1
        # SUB-6: Failure quarantined
        assert len(deps["quarantine_store"].isolated) >= 1
        # SUB-7: Healing diagnosed
        assert len(deps["healing_engine"].diagnoses) >= 1
        # SUB-8: HHMM classified
        assert len(deps["hhmm_engine"].classifications) >= 1
        # SUB-9: PoI credited (from memory.promoted)
        assert len(deps["poi_engine"].accumulations) >= 1
        # SUB-10: PoI + token minted (from telescript.completed)
        assert len(deps["token_minter"].rewards) >= 1
        # SUB-11: Budget reported
        assert deps["context_budget"].total_used >= 500
        # SUB-12: Self-model updated
        assert len(deps["self_model"].updates) >= 1
        assert len(deps["capability_registry"].registrations) >= 1

    def test_chain_integrity_after_interleaved_events(self, wired_bus):
        """Non-lifecycle-order events still produce valid chain."""
        bus, _, _ = wired_bus

        # Publish out of order — economics before learning
        bus.publish(
            EventType.MEMORY_RETRIEVED,
            {"tokens_retrieved": 100, "memory_source": "semantic", "query": "q"},
        )
        bus.publish(
            EventType.ACTION_INTENT,
            {"intent": "delayed", "session_id": "s3"},
        )
        bus.publish(
            EventType.AGENT_REGISTERED,
            {
                "agent_id": "hermes",
                "agent_type": "SAT",
                "capabilities": ["route"],
                "version": "2.0.0",
            },
        )
        bus.publish(
            EventType.ACTION_RECEIPT,
            {"action_type": "delayed", "ihsan_composite": 0.99, "result_summary": "ok"},
        )

        assert bus.chain_height == 4
        assert bus.verify_chain()


# ═══════════════════════════════════════════════════════════════
# Test: Safety subscriber fail-closed behavior
# ═══════════════════════════════════════════════════════════════


class TestSafetyFailClosed:
    """Safety-critical subscribers must propagate errors (fail-closed)."""

    def test_ihsan_breach_handler_error_propagates(self):
        """If IhsanGateBreachHandler.handle() raises, EventBus re-raises."""

        class FailingSessionManager:
            def halt(self, **kw):
                raise RuntimeError("Session halt failed")

        class NoOpAudit:
            def log_violation(self, **kw):
                pass

        bus = EventBus()
        sub = IhsanGateBreachHandler(FailingSessionManager(), NoOpAudit())
        bus.subscribe(sub)

        with pytest.raises(RuntimeError, match="Session halt failed"):
            bus.publish(
                EventType.IHSAN_GATE_BREACHED,
                {
                    "session_id": "s1",
                    "ihsan_composite": 0.5,
                    "action_type": "test",
                    "violation_dimensions": ["safety"],
                },
            )
        summary = bus.delivery_summary()
        assert summary["delivery_dead_letters"] == 1
        assert summary["last_dead_letter"]["safety_critical"] is True

    def test_quarantine_error_propagates(self):
        """If FailedActionQuarantine.handle() raises, EventBus re-raises."""

        class FailingStore:
            def record_failure_pattern(self, **kw):
                raise RuntimeError("Store unavailable")

        class FailingQuarantine:
            def isolate(self, **kw):
                raise RuntimeError("Quarantine failed")

        bus = EventBus()
        sub = FailedActionQuarantine(FailingStore(), FailingQuarantine())
        bus.subscribe(sub)

        with pytest.raises(RuntimeError):
            bus.publish(
                EventType.ACTION_RECEIPT_FAILED,
                {"action_type": "broken", "error": "err"},
            )
        summary = bus.delivery_summary()
        assert summary["delivery_dead_letters"] == 1
        assert summary["last_dead_letter"]["subscriber_name"] == (
            "FailedActionQuarantine"
        )


# ═══════════════════════════════════════════════════════════════
# Test: Non-safety subscriber fail-open behavior
# ═══════════════════════════════════════════════════════════════


class TestNonSafetyFailOpen:
    """Non-safety subscribers must NOT crash the bus (fail-open)."""

    def test_learning_subscriber_error_does_not_crash_bus(self):
        """If a learning subscriber raises, bus continues."""

        class FailingStore:
            def reinforce(self, **kw):
                raise RuntimeError("Store down")

            def get_success_count(self, key):
                return 0

            def set_success_count(self, key, val):
                pass

        bus = EventBus()
        sub = ActionReceiptMemoryReinforce(FailingStore())
        bus.subscribe(sub)

        # Should NOT raise — fail-open for learning subscribers
        event = bus.publish(
            EventType.ACTION_RECEIPT,
            {"action_type": "test", "ihsan_composite": 0.96, "result_summary": "ok"},
        )
        assert event.event_hash  # Bus continued, event was recorded
        summary = bus.delivery_summary()
        assert summary["delivery_dead_letters"] == 1
        assert summary["delivery_acks"] == 0
        assert summary["last_dead_letter"]["subscriber_name"] == (
            "ActionReceiptMemoryReinforce"
        )
        assert summary["last_dead_letter"]["status"] == "dead_letter"

    def test_delivery_receipts_capture_acks(self, wired_bus):
        """Successful subscribers should emit ack receipts into the bus ledger."""
        bus, _, _ = wired_bus

        event = bus.publish(
            EventType.ACTION_RECEIPT,
            {
                "action_type": "promote-worthy",
                "ihsan_composite": 0.97,
                "result_summary": "ok",
            },
        )

        receipts = bus.delivery_receipts(event_id=event.event_id)
        assert len(receipts) >= 2
        assert all(receipt["status"] == "ack" for receipt in receipts)
        summary = bus.delivery_summary()
        assert summary["delivery_acks"] >= 2
        assert summary["delivery_dead_letters"] == 0
        assert summary["last_delivery_ack"]["event_id"] == event.event_id

    def test_delivery_receipts_persist_when_path_configured(self, tmp_path):
        """Ack receipts should append to durable JSONL storage when enabled."""
        bus = EventBus(delivery_receipt_path=tmp_path / "audit" / "deliveries.jsonl")
        store = TrackingStore()
        bus.subscribe(ActionReceiptMemoryReinforce(store))

        event = bus.publish(
            EventType.ACTION_RECEIPT,
            {
                "action_type": "persist-me",
                "ihsan_composite": 0.96,
                "result_summary": "ok",
            },
        )

        path = tmp_path / "audit" / "deliveries.jsonl"
        assert path.exists()
        persisted = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert len(persisted) == 1
        assert persisted[0]["event_id"] == event.event_id
        assert persisted[0]["status"] == "ack"
        assert bus.delivery_summary()["persisted_delivery_receipts"] == 1

    def test_delivery_receipt_sink_failure_is_recorded(self):
        """Mirroring failures should be visible without breaking local delivery."""

        def _broken_sink(payload):  # type: ignore[no-untyped-def]
            del payload
            raise RuntimeError("mirror sink unavailable")

        bus = EventBus(delivery_receipt_sink=_broken_sink)
        store = TrackingStore()
        bus.subscribe(ActionReceiptMemoryReinforce(store))

        bus.publish(
            EventType.ACTION_RECEIPT,
            {
                "action_type": "mirror-failure",
                "ihsan_composite": 0.95,
                "result_summary": "ok",
            },
        )

        summary = bus.delivery_summary()
        assert summary["delivery_acks"] == 1
        assert summary["delivery_sink_failures"] == 1
        assert (
            "RuntimeError: mirror sink unavailable"
            in summary["last_delivery_sink_error"]
        )
