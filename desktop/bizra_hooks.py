"""
bizra_hooks — Python interface to the Node0 Nervous System.

This module wraps the Rust bizra-hooks shared library via ctypes,
providing a Pythonic API for Phase 46 engines and all future components.

Usage:
    from bizra_hooks import Kernel, ComponentKind, EventKind

    kernel = Kernel.boot()
    search_id = kernel.register("vector_search", ComponentKind.ENGINE)
    kernel.set_health(search_id, "healthy")
    event_id = kernel.publish(EventKind.SEARCH_EXECUTED, search_id, "query: investor deck")
    print(kernel.architecture_json())
    kernel.shutdown()

Standing on Giants: Python ctypes (stdlib) · Rust C-ABI
"""

from __future__ import annotations

import ctypes
import json
import os
import platform
from enum import IntEnum
from pathlib import Path
from typing import Any, Optional

# ═══════════════════════════════════════════════════════════════════════════════
# LIBRARY LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def _find_library() -> Path:
    """Find the bizra_hooks shared library."""
    # Search order:
    # 1. BIZRA_HOOKS_LIB environment variable
    # 2. Same directory as this file
    # 3. ../target/release/
    # 4. System library path

    env_path = os.environ.get("BIZRA_HOOKS_LIB")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p

    system = platform.system()
    if system == "Linux":
        lib_name = "libbizra_hooks.so"
    elif system == "Darwin":
        lib_name = "libbizra_hooks.dylib"
    elif system == "Windows":
        lib_name = "bizra_hooks.dll"
    else:
        lib_name = "libbizra_hooks.so"

    # Same directory as this file.
    here = Path(__file__).parent
    candidates = [
        here / lib_name,
        here / "lib" / lib_name,
        here.parent / "target" / "release" / lib_name,
        here.parent.parent / "bizra-hooks" / "target" / "release" / lib_name,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not find {lib_name}. Set BIZRA_HOOKS_LIB or build with: "
        f"cargo build --release --features ffi"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS — Mirror Rust enum ordinals exactly.
# ═══════════════════════════════════════════════════════════════════════════════

class ComponentKind(IntEnum):
    """Component type for registration."""
    CORE = 0
    AGENT = 1
    ENGINE = 2
    INTEGRATION = 3
    INTERFACE = 4
    OPERATIONS = 5


class HealthStatus(IntEnum):
    """Component health state."""
    HEALTHY = 0
    DEGRADED = 1
    FAILED = 2
    UNINITIALIZED = 3


class EventKind(IntEnum):
    """Event type for publishing. Must match Rust EventKind ordinal exactly."""
    # Conversation lifecycle
    USER_MESSAGE = 0
    AGENT_RESPONSE = 1
    CONVERSATION_START = 2
    CONVERSATION_END = 3

    # Memory lifecycle
    MEMORY_STORE = 4
    MEMORY_RETRIEVE = 5
    MEMORY_SYNTHESIZE = 6
    MEMORY_FORGET = 7

    # Agent lifecycle
    COMPONENT_REGISTERED = 8
    COMPONENT_HEALTH_CHANGED = 9
    TASK_START = 10
    TASK_COMPLETE = 11
    TASK_FAILED = 12

    # Intelligence pipeline
    SEARCH_EXECUTED = 13
    REASONING_COMPLETE = 14
    PREDICTION_UPDATED = 15
    RESONANCE_COMPLETE = 16

    # Desktop / Integration
    MCP_TOOL_CALL = 17
    DESKTOP_ACTION = 18
    API_CALL = 19

    # System
    HOOK_EXECUTED = 20
    CANARY_ROUTED = 21
    ROLLBACK_TRIGGERED = 22
    IHSAN_SCORED = 23

    # Self-modification (RSI)
    MUTATION_PROPOSED = 24
    MUTATION_VERIFIED = 25
    MUTATION_DEPLOYED = 26
    MUTATION_ROLLED_BACK = 27


# ═══════════════════════════════════════════════════════════════════════════════
# KERNEL — The Python interface to Node0's nervous system.
# ═══════════════════════════════════════════════════════════════════════════════

class Kernel:
    """
    Python interface to the BIZRA Node0 Kernel (Rust).

    Wraps the C-ABI exported functions from libbizra_hooks.
    Thread-safe: the Rust side uses Arc<RwLock<..>> internally.
    """

    def __init__(self, lib: ctypes.CDLL):
        self._lib = lib
        self._setup_signatures()
        self._booted = False

    def _setup_signatures(self):
        """Declare C function signatures for type safety."""
        L = self._lib

        # Lifecycle
        L.bizra_init.restype = ctypes.c_int32
        L.bizra_init.argtypes = []

        L.bizra_init_with_capacity.restype = ctypes.c_int32
        L.bizra_init_with_capacity.argtypes = [ctypes.c_uint64]

        L.bizra_shutdown.restype = ctypes.c_int32
        L.bizra_shutdown.argtypes = []

        # Registration
        L.bizra_register.restype = ctypes.c_uint64
        L.bizra_register.argtypes = [ctypes.c_char_p, ctypes.c_uint32]

        L.bizra_set_health.restype = ctypes.c_int32
        L.bizra_set_health.argtypes = [ctypes.c_uint64, ctypes.c_uint32]

        # Publishing
        L.bizra_publish.restype = ctypes.c_uint64
        L.bizra_publish.argtypes = [ctypes.c_uint32, ctypes.c_uint64, ctypes.c_char_p]

        L.bizra_publish_scored.restype = ctypes.c_uint64
        L.bizra_publish_scored.argtypes = [
            ctypes.c_uint32, ctypes.c_uint64, ctypes.c_char_p,
            ctypes.c_double, ctypes.c_double, ctypes.c_uint64,
        ]

        # Queries
        L.bizra_component_count.restype = ctypes.c_uint64
        L.bizra_component_count.argtypes = []

        L.bizra_events_published.restype = ctypes.c_uint64
        L.bizra_events_published.argtypes = []

        L.bizra_has_cycles.restype = ctypes.c_int32
        L.bizra_has_cycles.argtypes = []

        L.bizra_architecture_json.restype = ctypes.c_char_p
        L.bizra_architecture_json.argtypes = []

        L.bizra_free_string.restype = None
        L.bizra_free_string.argtypes = [ctypes.c_char_p]

    @classmethod
    def boot(cls, capacity: Optional[int] = None) -> "Kernel":
        """Boot the kernel. Call once at process startup."""
        lib_path = _find_library()
        lib = ctypes.CDLL(str(lib_path))
        kernel = cls(lib)

        if capacity:
            result = lib.bizra_init_with_capacity(capacity)
        else:
            result = lib.bizra_init()

        if result < 0:
            raise RuntimeError("Failed to initialize BIZRA kernel (lock poisoned)")
        if result == 0:
            pass  # Already initialized — reuse.

        kernel._booted = True
        return kernel

    def shutdown(self):
        """Shutdown the kernel and release all resources."""
        if self._booted:
            self._lib.bizra_shutdown()
            self._booted = False

    def __del__(self):
        if self._booted:
            self.shutdown()

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self, name: str, kind: ComponentKind) -> int:
        """
        Register a component. Returns its ComponentId.

        Usage:
            search_id = kernel.register("vector_search", ComponentKind.ENGINE)
        """
        cid = self._lib.bizra_register(name.encode("utf-8"), int(kind))
        if cid == 0:
            raise ValueError(f"Failed to register '{name}' (duplicate name or kernel not initialized)")
        return cid

    def set_health(self, component_id: int, health: HealthStatus | str) -> None:
        """
        Set component health status.

        Accepts HealthStatus enum or string: "healthy", "degraded", "failed", "uninitialized"
        """
        if isinstance(health, str):
            health = HealthStatus[health.upper()]
        result = self._lib.bizra_set_health(component_id, int(health))
        if result <= 0:
            raise ValueError(f"Failed to set health for component {component_id}")

    # ── Event Publishing ──────────────────────────────────────────────────────

    def publish(
        self,
        kind: EventKind,
        source_id: int,
        text: Optional[str] = None,
    ) -> int:
        """
        Publish an event. Returns EventId.

        Usage:
            eid = kernel.publish(EventKind.SEARCH_EXECUTED, search_id, "query: deck")
        """
        text_bytes = text.encode("utf-8") if text else None
        eid = self._lib.bizra_publish(int(kind), source_id, text_bytes)
        if eid == 0:
            raise RuntimeError(f"Failed to publish event {kind.name}")
        return eid

    def publish_scored(
        self,
        kind: EventKind,
        source_id: int,
        text: Optional[str] = None,
        *,
        snr: float = 0.0,
        confidence: float = 0.0,
        latency_us: int = 0,
    ) -> int:
        """
        Publish an event with إحسان quality score.

        Usage:
            eid = kernel.publish_scored(
                EventKind.RESONANCE_COMPLETE, engine_id,
                "pipeline output",
                snr=0.97, confidence=0.95, latency_us=1500
            )
        """
        text_bytes = text.encode("utf-8") if text else None
        eid = self._lib.bizra_publish_scored(
            int(kind), source_id, text_bytes,
            snr, confidence, latency_us,
        )
        if eid == 0:
            raise RuntimeError(f"Failed to publish scored event {kind.name}")
        return eid

    # ── Queries ───────────────────────────────────────────────────────────────

    @property
    def component_count(self) -> int:
        """Number of registered components (including kernel self)."""
        return self._lib.bizra_component_count()

    @property
    def events_published(self) -> int:
        """Total events published since boot."""
        return self._lib.bizra_events_published()

    @property
    def has_cycles(self) -> bool:
        """True if the architecture dependency graph has cycles (design error)."""
        return self._lib.bizra_has_cycles() == 1

    def architecture(self) -> dict[str, Any]:
        """
        Get the architecture graph as a Python dict.

        Returns: {"nodes": [...], "edges": [...]}
        """
        raw = self._lib.bizra_architecture_json()
        if raw is None:
            return {"nodes": [], "edges": []}
        try:
            return json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {"nodes": [], "edges": []}

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def status(self) -> dict[str, Any]:
        """Quick status summary."""
        return {
            "booted": self._booted,
            "components": self.component_count,
            "events_published": self.events_published,
            "has_cycles": self.has_cycles,
        }

    def __repr__(self) -> str:
        return (
            f"Kernel(components={self.component_count}, "
            f"events={self.events_published}, "
            f"booted={self._booted})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE — For Phase 46 engine integration.
# ═══════════════════════════════════════════════════════════════════════════════

_GLOBAL_KERNEL: Optional[Kernel] = None

def boot(**kwargs) -> Kernel:
    """Boot the global kernel singleton."""
    global _GLOBAL_KERNEL
    if _GLOBAL_KERNEL is None:
        _GLOBAL_KERNEL = Kernel.boot(**kwargs)
    return _GLOBAL_KERNEL

def get_kernel() -> Kernel:
    """Get the global kernel. Raises if not booted."""
    if _GLOBAL_KERNEL is None:
        raise RuntimeError("Kernel not booted. Call bizra_hooks.boot() first.")
    return _GLOBAL_KERNEL


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 46 INTEGRATION — Register existing engines.
# ═══════════════════════════════════════════════════════════════════════════════

def register_phase46_engines(kernel: Kernel) -> dict[str, int]:
    """
    Register all Phase 46 cognitive engines with the kernel.

    Returns a dict mapping engine names to their ComponentIds.
    """
    engines = {
        "vector_search": ComponentKind.ENGINE,
        "got_bridge": ComponentKind.ENGINE,
        "hmm_engine": ComponentKind.ENGINE,
        "cognitive_resonance": ComponentKind.ENGINE,
        "mcp_server": ComponentKind.INTEGRATION,
        "canary_router": ComponentKind.OPERATIONS,
        "rollback_engine": ComponentKind.OPERATIONS,
        "phase46_metrics": ComponentKind.OPERATIONS,
    }

    ids = {}
    for name, kind in engines.items():
        try:
            cid = kernel.register(name, kind)
            kernel.set_health(cid, HealthStatus.HEALTHY)
            ids[name] = cid
        except ValueError:
            pass  # Already registered.

    return ids
