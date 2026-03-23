"""
Test: Rust Bridge Synapse
=========================
Verifies that RustBridgeSubscriber correctly forwards Python EventBus
events to a mock Rust bridge, proving the wiring pattern works before
the PyO3 module is compiled.

Phase 87: PyO3 EventBridge Synapse Tests
"""

import sys

# Ensure project root is on path
sys.path.insert(0, r"C:\BIZRA-DATA-LAKE")

from core.bus.rust_bridge import RustBridgeSubscriber, diagnose_bridge
from core.bus.subscribers import EventBus, EventType


class MockRustBridge:
    """Simulates PyEventBridge for testing without compiled Rust."""

    def __init__(self):
        self.emitted: list = []
        self.emitted_with_receipt: list = []

    def emit(self, topic: str, payload: str, priority: int) -> int:
        self.emitted.append({"topic": topic, "payload": payload, "priority": priority})
        return 1

    def emit_with_receipt(
        self,
        topic: str,
        payload: str,
        receipt_id: str,
        ihsan_score: float,
        priority: int,
    ) -> int:
        self.emitted_with_receipt.append(
            {
                "topic": topic,
                "payload": payload,
                "receipt_id": receipt_id,
                "ihsan": ihsan_score,
                "priority": priority,
            }
        )
        return 1

    def wire_subscribers(self) -> int:
        return 12

    def health(self) -> dict:
        return {"events_emitted": len(self.emitted) + len(self.emitted_with_receipt)}


def test_bridge_forwards_plain_event():
    """Plain events (no ihsan/receipt) use emit()."""
    mock = MockRustBridge()
    adapter = RustBridgeSubscriber(mock, list(EventType))

    bus = EventBus()
    bus.subscribe(adapter)
    bus.publish(EventType.AGENT_REGISTERED, {"agent_id": "ATLAS", "role": "PAT"})

    assert adapter._forwarded == 1
    assert adapter._failed == 0
    assert len(mock.emitted) == 1
    assert mock.emitted[0]["topic"] == "agent.registered"
    assert "ATLAS" in mock.emitted[0]["payload"]
    print("PASS: test_bridge_forwards_plain_event")


def test_bridge_forwards_receipt_event():
    """Events with ihsan_composite + receipt_hash use emit_with_receipt()."""
    mock = MockRustBridge()
    adapter = RustBridgeSubscriber(mock, list(EventType))

    bus = EventBus()
    bus.subscribe(adapter)
    bus.publish(
        EventType.ACTION_RECEIPT,
        {
            "action_type": "search",
            "ihsan_composite": 0.97,
            "receipt_hash": "abc123def456",
            "result_summary": "Found 42 results",
        },
    )

    assert adapter._forwarded == 1
    assert len(mock.emitted_with_receipt) == 1
    r = mock.emitted_with_receipt[0]
    assert r["topic"] == "action.receipt"
    assert r["receipt_id"] == "abc123def456"
    assert r["ihsan"] == 0.97
    assert r["priority"] == 1  # Normal
    print("PASS: test_bridge_forwards_receipt_event")


def test_bridge_safety_events_critical_priority():
    """Breach/failed events get Critical (3) priority."""
    mock = MockRustBridge()
    adapter = RustBridgeSubscriber(mock, list(EventType))

    bus = EventBus()
    bus.subscribe(adapter)
    bus.publish(EventType.IHSAN_GATE_BREACHED, {"score": 0.42, "threshold": 0.95})

    assert adapter._forwarded == 1
    assert mock.emitted[0]["priority"] == 3  # Critical
    print("PASS: test_bridge_safety_events_critical_priority")


def test_bridge_degradation_on_error():
    """If Rust bridge throws, Python continues — degradation not failure."""

    class BrokenBridge:
        def emit(self, *args):
            raise RuntimeError("Rust segfault simulation")

        def emit_with_receipt(self, *args):
            raise RuntimeError("Rust segfault simulation")

    broken = BrokenBridge()
    adapter = RustBridgeSubscriber(broken, list(EventType))

    bus = EventBus()
    bus.subscribe(adapter)

    # This should NOT raise — bridge degrades gracefully
    bus.publish(EventType.AGENT_REGISTERED, {"agent_id": "test"})

    assert adapter._forwarded == 0
    assert adapter._failed == 1
    assert "segfault" in adapter._last_error
    # Python chain is unaffected
    assert bus.chain_height == 1
    assert bus.verify_chain()
    print("PASS: test_bridge_degradation_on_error")


def test_bridge_stats():
    """Stats property reflects forwarded/failed counts."""
    mock = MockRustBridge()
    adapter = RustBridgeSubscriber(mock, list(EventType))

    bus = EventBus()
    bus.subscribe(adapter)

    for i in range(5):
        bus.publish(EventType.TELESCRIPT_STEP, {"step": i})

    s = adapter.stats
    assert s["forwarded"] == 5
    assert s["failed"] == 0
    assert s["bridge_healthy"] is True
    print("PASS: test_bridge_stats")


def test_diagnose_bridge():
    """diagnose_bridge() reports Rust availability without crashing."""
    result = diagnose_bridge()
    assert "rust_available" in result
    assert "error" in result
    # In test env Rust module likely not compiled — that's fine
    if not result["rust_available"]:
        assert result["error"] is not None
    print(f"PASS: test_diagnose_bridge (rust_available={result['rust_available']})")


def test_chain_integrity_preserved():
    """Python hash chain is unbroken even with bridge forwarding."""
    mock = MockRustBridge()
    adapter = RustBridgeSubscriber(mock, list(EventType))

    bus = EventBus()
    bus.subscribe(adapter)

    # Emit a variety of events
    bus.publish(EventType.ACTION_INTENT, {"query": "organize invoices"})
    bus.publish(
        EventType.ACTION_RECEIPT,
        {
            "action_type": "organize",
            "ihsan_composite": 0.96,
            "receipt_hash": "hash_abc",
        },
    )
    bus.publish(EventType.SESSION_END, {"duration_s": 42})

    # Python chain must be intact
    assert bus.chain_height == 3
    assert bus.verify_chain()

    # Rust bridge received all 3 events
    assert adapter._forwarded == 3
    total_rust = len(mock.emitted) + len(mock.emitted_with_receipt)
    assert total_rust == 3
    print("PASS: test_chain_integrity_preserved")


# ═══════════════════════════════════════════════════════════════════
# RUNNER
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    tests = [
        test_bridge_forwards_plain_event,
        test_bridge_forwards_receipt_event,
        test_bridge_safety_events_critical_priority,
        test_bridge_degradation_on_error,
        test_bridge_stats,
        test_diagnose_bridge,
        test_chain_integrity_preserved,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"FAIL: {t.__name__} — {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Rust Bridge Synapse Tests: {passed}/{len(tests)} passed")
    if failed:
        print(f"  {failed} FAILED")
        sys.exit(1)
    else:
        print("  ALL PASSED -- Synapse verified")
        sys.exit(0)
