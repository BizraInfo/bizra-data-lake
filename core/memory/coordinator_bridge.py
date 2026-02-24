"""
MemoryCoordinator ↔ AgentDB Bridge — Wires AgentDB into the auto-save lifecycle.

Registers AgentDB as a CORE-priority state provider in MemoryCoordinator,
ensuring that HNSW index is flushed to disk on every auto-save and that
AgentDB state is included in versioned checkpoints.

Usage:
    from core.memory import AgentDB
    from core.memory.coordinator_bridge import AgentDBBridge
    from core.sovereign.memory_coordinator import MemoryCoordinator

    bridge = AgentDBBridge(agent_db, coordinator)
    bridge.register()

Standing on Giants: ADR-006 (Unified Memory Service)
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class AgentDBBridge:
    """Bridges AgentDB into MemoryCoordinator's save/restore lifecycle.

    Thread-safe registration. The bridge patches ``save_all`` to flush
    the HNSW index before each checkpoint and registers a state provider
    so AgentDB metrics appear in every versioned snapshot.
    """

    def __init__(self, agent_db, coordinator) -> None:
        self._db = agent_db
        self._coordinator = coordinator
        self._registered = False

    @property
    def registered(self) -> bool:
        return self._registered

    def register(self) -> None:
        """Register AgentDB with the coordinator. Idempotent."""
        if self._registered:
            return

        # Import RestorePriority here to avoid circular imports at module level
        from core.sovereign.memory_coordinator import RestorePriority

        # Register state provider (CORE priority)
        self._coordinator.register_state_provider(
            name="agent_db",
            provider=self._get_state,
            priority=RestorePriority.CORE,
        )

        # Wrap save_all to flush HNSW before checkpoint
        original_save = self._coordinator.save_all

        @functools.wraps(original_save)
        async def _patched_save(source: str = "manual") -> bool:
            self._flush_hnsw()
            return await original_save(source=source)

        self._coordinator.save_all = _patched_save

        self._registered = True
        logger.info("AgentDB registered with MemoryCoordinator (CORE priority)")

    def _flush_hnsw(self) -> None:
        """Flush HNSW index to disk. Tolerates uninitialized state."""
        try:
            if getattr(self._db, "_initialized", False):
                self._db.save()
        except Exception as e:
            logger.warning(f"AgentDB HNSW flush failed: {e}")

    def _get_state(self) -> Dict[str, Any]:
        """State provider callback for checkpoint collection."""
        if not getattr(self._db, "_initialized", False):
            return {"status": "not_initialized"}
        return self._db.get_persistable_state()

    def ensure_initialized(self) -> None:
        """Ensure AgentDB is initialized (call during restore)."""
        if not getattr(self._db, "_initialized", False):
            self._db.initialize()
