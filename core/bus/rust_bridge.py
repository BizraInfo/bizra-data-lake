"""
BIZRA Rust Bridge — The Synapse
================================
Connects Python EventBus (cognitive) to Rust PyEventBridge (constitutional).

This is the single most important module in the Python codebase.
Before this file, Python cognition and Rust proof-carrying execution were
two isolated nervous systems — 12 Python subscribers and 12 Rust subscribers
running in parallel, neither aware of the other.

After this file, every Python cognitive event simultaneously flows through
Rust's constitutional verification pipeline. A thought in Python becomes
a cryptographically signed proof fragment in Rust.

The language boundary IS the trust boundary.
PAT (Python) serves the user. SAT (Rust) validates independently.
This bridge is the membrane between them.

Standing on Giants:
  - Hewitt (1973): Actor model — message-passing across boundaries
  - Lamport (1978): Event ordering preserved across the bridge
  - Maturana (1980): Autopoiesis — the loop that makes the system self-sustaining

Phase 87: PyO3 EventBridge Synapse
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from core.bus.subscribers import Event, EventBus

logger = logging.getLogger("bizra.bus.rust_bridge")

# ═══════════════════════════════════════════════════════════════════
# RUST BRIDGE SUBSCRIBER
# ═══════════════════════════════════════════════════════════════════


class RustBridgeSubscriber:
    """
    Forwards every Python EventBus event to Rust's constitutional pipeline.

    This subscriber listens to ALL Python event types and forwards each one
    through the PyEventBridge into bizra-hooks::BizraSystem, where Rust's
    12 constitutional subscribers process them independently.

    The result: every Python cognitive operation (memory reinforcement,
    reflex compilation, action receipts) is simultaneously verified by
    Rust's type-safe constitutional validators.

    Constitutional guarantee: if the Rust bridge is unavailable, Python
    continues operating (degradation, not failure). Events are counted
    but not lost — the Python EventBus chain remains intact.
    """

    def __init__(self, bridge, event_types: list):
        self.bridge = bridge
        self.event_types = event_types
        self._forwarded: int = 0
        self._failed: int = 0
        self._last_error: Optional[str] = None

    # Topic translation: Python EventType values → Rust subscriber topic constants.
    # Three naming mismatches exist between the Python and Rust taxonomies.
    # Without this map, events cross the bridge but miss their Rust handlers.
    _TOPIC_TRANSLATE = {
        "ihsan.gate.breached": "ihsan.breach",
        "telescript.rolled_back": "telescript.rolledback",
        "telescript.step": "telescript.step.completed",
    }

    def handle(self, event: Event) -> None:
        """
        Forward a single Python event to the Rust nervous system.

        If the event carries an ihsan_composite and receipt_hash, we use
        emit_with_receipt() to bind the Rust event to the proof chain.
        Otherwise, we use plain emit() for informational events.

        Priority mapping: Python events default to Normal (1).
        Safety-critical events (ihsan.gate.breached) use Critical (3).

        Topic translation: 3 naming mismatches are resolved here so that
        Python events match Rust subscriber topic filters exactly.
        """
        try:
            # Translate Python topic → Rust topic (identity for 8 matching topics)
            raw_topic = event.event_type.value
            topic = self._TOPIC_TRANSLATE.get(raw_topic, raw_topic)
            payload_str = json.dumps(event.payload, default=str)

            # Determine priority: safety events are Critical
            priority = 1  # Normal
            if "breach" in topic or "failed" in topic:
                priority = 3  # Critical

            # Extract proof-chain binding fields if present
            ihsan = event.payload.get("ihsan_composite", 0.0)
            receipt_id = event.payload.get("receipt_hash", "")

            if ihsan > 0 and receipt_id:
                # Identity-aware handoff: bind to verified proof chain
                self.bridge.emit_with_receipt(
                    topic, payload_str, receipt_id, ihsan, priority
                )
            else:
                self.bridge.emit(topic, payload_str, priority)

            self._forwarded += 1

        except (
            Exception
        ) as e:  # noqa: BLE001 — boundary: never let Rust failure crash Python
            self._failed += 1
            self._last_error = str(e)
            logger.warning(f"Rust bridge forward failed ({self._failed} total): {e}")
            # Constitutional degradation: Python continues, Rust misses this event.
            # The Python EventBus chain is unaffected — only the Rust mirror is incomplete.

    @property
    def stats(self) -> dict:
        """Bridge health statistics."""
        return {
            "forwarded": self._forwarded,
            "failed": self._failed,
            "last_error": self._last_error,
            "bridge_healthy": self._failed == 0
            or (self._forwarded > 0 and self._failed / self._forwarded < 0.01),
        }


# ═══════════════════════════════════════════════════════════════════
# BRIDGE WIRING — The single function that closes the loop
# ═══════════════════════════════════════════════════════════════════


def wire_rust_bridge(
    bus: EventBus,
    production: bool = False,
) -> Optional[RustBridgeSubscriber]:
    """
    Wire the Rust constitutional bridge into an existing Python EventBus.

    This is the single function call that closes the autopoietic loop:
    Python cognition → Rust verification → proof chain → back to Python.

    Args:
        bus: The Python EventBus instance (core.bus.subscribers.EventBus)
        production: If True, use production-mode Rust system (stricter gates)

    Returns:
        The RustBridgeSubscriber if wired successfully, None if Rust unavailable.

    Usage:
        from core.bus.subscribers import EventBus
        from core.bus.rust_bridge import wire_rust_bridge

        bus = EventBus()
        # ... wire Python subscribers ...
        bridge_sub = wire_rust_bridge(bus, production=False)
        if bridge_sub:
            print(f"Rust bridge active: {bridge_sub.stats}")
    """
    try:
        from bizra import PyEventBridge
    except ImportError:
        logger.warning(
            "bizra Rust module not available — running Python-only mode. "
            "Build with: cd bizra-omega && maturin develop -p bizra-python"
        )
        return None

    try:
        # 1. Create the Rust nervous system
        bridge = PyEventBridge(production=production)

        # 2. Wire Rust's 12 constitutional subscribers
        wired_count = bridge.wire_subscribers()
        logger.info(
            f"Rust PyEventBridge initialized: {wired_count} constitutional "
            f"subscribers wired (production={production})"
        )

        # 3. Get all Python event types to forward
        from core.bus.subscribers import EventType

        all_event_types = list(EventType)

        # 4. Create the bridge subscriber
        adapter = RustBridgeSubscriber(bridge, all_event_types)

        # 5. Subscribe to the Python EventBus — this is THE connection
        bus.subscribe(adapter)

        logger.info(
            f"╔══════════════════════════════════════════════╗\n"
            f"║  RUST BRIDGE SYNAPSE ACTIVE                  ║\n"
            f"║  {len(all_event_types)} Python event types → "
            f"Rust constitutional pipeline  ║\n"
            f"║  Language boundary = Trust boundary           ║\n"
            f"╚══════════════════════════════════════════════╝"
        )

        return adapter

    except Exception as e:  # noqa: BLE001
        logger.error(
            f"Failed to wire Rust bridge: {e}. "
            f"System continues in Python-only mode."
        )
        return None


# ═══════════════════════════════════════════════════════════════════
# DIAGNOSTIC — Quick health check callable from anywhere
# ═══════════════════════════════════════════════════════════════════


def diagnose_bridge() -> dict:
    """
    Quick diagnostic: can the Rust bridge be loaded?

    Returns a dict with:
      - rust_available: bool — is the PyO3 module importable?
      - version: str — bizra module version if available
      - ihsan_threshold: float — constitutional constant from Rust
      - error: str | None — error message if unavailable
    """
    try:
        import bizra

        return {
            "rust_available": True,
            "version": getattr(bizra, "__version__", "unknown"),
            "ihsan_threshold": getattr(bizra, "IHSAN_THRESHOLD", 0.0),
            "snr_threshold": getattr(bizra, "SNR_THRESHOLD", 0.0),
            "error": None,
        }
    except ImportError as e:
        return {
            "rust_available": False,
            "version": None,
            "ihsan_threshold": None,
            "snr_threshold": None,
            "error": str(e),
        }
