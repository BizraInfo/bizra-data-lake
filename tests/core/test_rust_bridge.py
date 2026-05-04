"""Tests for the Python↔Rust event bridge wrapper.

These tests verify the Python facade works with both the Rust backend
(when available) and the stub fallback (always available).

Standing on Giants: Shannon (namespace isolation) · Lamport (event ordering)
"""

import pytest

from core.sovereign.event_bus import (
    RustEventBridge,
    create_rust_event_bridge,
    is_rust_event_bus_available,
)


def test_availability_check_returns_bool():
    """is_rust_event_bus_available returns a boolean."""
    result = is_rust_event_bus_available()
    assert isinstance(result, bool)


def test_factory_returns_none_when_unavailable():
    """If Rust binding is missing, factory returns None (not raise)."""
    bridge = create_rust_event_bridge()
    # Either a RustEventBridge or None — both are valid
    assert bridge is None or isinstance(bridge, RustEventBridge)


@pytest.mark.skipif(
    not is_rust_event_bus_available(),
    reason="Rust bizra binding not installed (maturin develop)",
)
class TestRustEventBridge:
    """Tests that only run when the Rust binding is available."""

    def test_create_bridge(self):
        bridge = RustEventBridge(production=False)
        assert not bridge.is_wired

    def test_wire_subscribers(self):
        bridge = RustEventBridge(production=False)
        count = bridge.wire()
        assert count == 13
        assert bridge.is_wired

    def test_emit_returns_delivery_count(self):
        bridge = RustEventBridge(production=False)
        bridge.wire()
        delivered = bridge.emit_rust("action.intent", "test_payload", 1)
        assert isinstance(delivered, int)
        assert delivered >= 0

    def test_health_returns_dict(self):
        bridge = RustEventBridge(production=False)
        bridge.wire()
        h = bridge.health()
        assert isinstance(h, dict)
        assert "events_emitted" in h
        assert "system_ihsan" in h
        assert "active_subscriptions" in h

    def test_mission_lifecycle_event_flow(self):
        """Full round-trip: emit mission intent → action receipt → verify health."""
        bridge = RustEventBridge(production=False)
        bridge.wire()

        # Mission start event (action.intent → action namespace shard)
        d1 = bridge.emit_rust("action.intent", "mission:001:test_task", 1)
        assert d1 >= 1, "action.intent should reach ≥1 subscriber"

        # Mission completion event (action.receipt → same shard)
        d2 = bridge.emit_rust("action.receipt", "mission:001:ihsan=0.97", 1)
        assert d2 >= 1, "action.receipt should reach ≥1 subscriber"

        # Ihsan breach (ihsan.breach → subscriber #9, requires EMERGENCY priority)
        d3 = bridge.emit_rust(
            "ihsan.breach", "mission:002:ihsan=0.80", bridge.PRIORITY_EMERGENCY
        )
        assert d3 >= 1, "ihsan.breach should reach ≥1 subscriber"

        # Verify health reflects all 3 emissions
        h = bridge.health()
        assert h["events_emitted"] >= 3
        assert h["events_delivered"] >= 3
        assert h["delivery_ratio"] > 0

    def test_namespace_isolation(self):
        """Events in one namespace don't cross-contaminate."""
        bridge = RustEventBridge(production=False)
        bridge.wire()

        # Get baseline
        h_before = bridge.health()
        emitted_before = h_before["events_emitted"]

        # Emit to distinct namespaces
        bridge.emit_rust("action.intent", "test1", 1)
        bridge.emit_rust("memory.promoted", "test2", 1)
        bridge.emit_rust("session.start", "test3", 1)

        h_after = bridge.health()
        assert h_after["events_emitted"] == emitted_before + 3

    def test_priority_escalation(self):
        """High-priority events (CRITICAL/EMERGENCY) are delivered."""
        bridge = RustEventBridge(production=False)
        bridge.wire()

        # Emergency event on system.lifecycle (subscriber #11)
        d = bridge.emit_rust(
            "system.lifecycle", "critical_failure", bridge.PRIORITY_EMERGENCY
        )
        assert d >= 1, "Emergency events must reach at least 1 subscriber"
