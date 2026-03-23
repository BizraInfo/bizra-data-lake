"""
Test: Rust Bridge Topic Translation (Post-Synapse Audit)
=========================================================
Verifies that 3 Python→Rust topic naming mismatches are correctly
resolved by the _TOPIC_TRANSLATE map in RustBridgeSubscriber.

Discovery: Python has 11 EventTypes, Rust has 11 topic constants.
8 match exactly. 3 differ in naming convention.
Without translation, events cross the bridge but miss Rust subscribers.

Phase 87b: Topic Parity Fix
"""

import sys

sys.path.insert(0, r"C:\BIZRA-DATA-LAKE")

from core.bus.rust_bridge import RustBridgeSubscriber
from core.bus.subscribers import EventBus, EventType


class MockRustBridge:
    def __init__(self):
        self.emitted = []
        self.emitted_with_receipt = []

    def emit(self, topic, payload, priority):
        self.emitted.append({"topic": topic, "payload": payload, "priority": priority})
        return 1

    def emit_with_receipt(self, topic, payload, receipt_id, ihsan_score, priority):
        self.emitted_with_receipt.append({"topic": topic, "receipt_id": receipt_id})
        return 1


def test_topic_translation():
    """3 mismatched Python topics are translated to Rust equivalents."""
    mock = MockRustBridge()
    adapter = RustBridgeSubscriber(mock, list(EventType))

    bus = EventBus()
    bus.subscribe(adapter)

    # These 3 Python topics have different names in Rust
    bus.publish(EventType.IHSAN_GATE_BREACHED, {"score": 0.42})
    bus.publish(EventType.TELESCRIPT_ROLLED_BACK, {"reason": "timeout"})
    bus.publish(EventType.TELESCRIPT_STEP, {"step": 1})

    assert adapter._forwarded == 3

    all_topics = [e["topic"] for e in mock.emitted]
    assert "ihsan.breach" in all_topics, f"Expected ihsan.breach, got {all_topics}"
    assert (
        "telescript.rolledback" in all_topics
    ), f"Expected telescript.rolledback, got {all_topics}"
    assert (
        "telescript.step.completed" in all_topics
    ), f"Expected telescript.step.completed, got {all_topics}"

    # Python-native names must NOT appear (they'd miss Rust subscribers)
    assert "ihsan.gate.breached" not in all_topics
    assert "telescript.rolled_back" not in all_topics
    assert "telescript.step" not in all_topics
    print("PASS: test_topic_translation")


def test_untranslated_topics_pass_through():
    """8 matching topics pass through without translation."""
    mock = MockRustBridge()
    adapter = RustBridgeSubscriber(mock, list(EventType))

    bus = EventBus()
    bus.subscribe(adapter)

    bus.publish(EventType.ACTION_RECEIPT, {"action_type": "search"})
    bus.publish(EventType.AGENT_REGISTERED, {"agent_id": "ATLAS"})

    topics = [e["topic"] for e in mock.emitted]
    assert "action.receipt" in topics
    assert "agent.registered" in topics
    print("PASS: test_untranslated_topics_pass_through")


def test_breach_gets_critical_priority_after_translation():
    """ihsan.gate.breached → ihsan.breach AND priority=Critical(3)."""
    mock = MockRustBridge()
    adapter = RustBridgeSubscriber(mock, list(EventType))

    bus = EventBus()
    bus.subscribe(adapter)
    bus.publish(EventType.IHSAN_GATE_BREACHED, {"score": 0.3})

    assert len(mock.emitted) == 1
    assert mock.emitted[0]["topic"] == "ihsan.breach"
    assert mock.emitted[0]["priority"] == 3  # Critical
    print("PASS: test_breach_gets_critical_priority_after_translation")


def test_full_event_taxonomy_parity():
    """All 11 Python EventTypes produce valid Rust topics."""
    mock = MockRustBridge()
    adapter = RustBridgeSubscriber(mock, list(EventType))
    bus = EventBus()
    bus.subscribe(adapter)

    for et in EventType:
        bus.publish(et, {"test": True})

    assert adapter._forwarded == len(list(EventType))
    rust_topics = {e["topic"] for e in mock.emitted}
    expected = {
        "action.intent",
        "action.receipt",
        "action.receipt.failed",
        "agent.registered",
        "ihsan.breach",
        "memory.promoted",
        "memory.retrieved",
        "session.end",
        "telescript.completed",
        "telescript.rolledback",
        "telescript.step.completed",
    }
    assert (
        rust_topics == expected
    ), f"Mismatch: {rust_topics.symmetric_difference(expected)}"
    print("PASS: test_full_event_taxonomy_parity")


if __name__ == "__main__":
    tests = [
        test_topic_translation,
        test_untranslated_topics_pass_through,
        test_breach_gets_critical_priority_after_translation,
        test_full_event_taxonomy_parity,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"FAIL: {t.__name__} — {e}")
            failed += 1
    print(f"\n{'='*50}")
    print(f"Topic Parity Tests: {passed}/{len(tests)} passed")
    if failed:
        sys.exit(1)
    else:
        print("ALL PASSED — Full taxonomy parity verified")
