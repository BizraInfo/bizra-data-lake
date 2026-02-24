# Phase 39 — Pseudocode Module 01: MemoryCoordinator ↔ AgentDB Bridge

**FR-01** | Priority: 1 (first to implement) | Risk: Low | New files: 1

---

## Overview

Wire `MemoryCoordinator` to treat `AgentDB` as a first-class subsystem:
register it as a state provider, flush HNSW on auto-save, restore on startup.

---

## Flow Diagram

```
MemoryCoordinator
  │
  ├── save_all()
  │     ├── collect state from providers
  │     ├── AgentDB.save()          ← NEW: flush HNSW to disk
  │     ├── LivingMemory._save()
  │     └── StateCheckpointer.checkpoint()
  │
  └── restore_latest()
        ├── StateCheckpointer.restore()
        ├── AgentDB.initialize()    ← NEW: restore HNSW + SQLite
        └── restore_by_priority()
```

---

## Pseudocode: `core/memory/coordinator_bridge.py`

```
MODULE coordinator_bridge

IMPORT AgentDB from core.memory
IMPORT MemoryCoordinator, RestorePriority from core.sovereign.memory_coordinator

CLASS AgentDBBridge:
    """Bridges AgentDB into MemoryCoordinator's save/restore lifecycle."""

    CONSTRUCTOR(agent_db: AgentDB, coordinator: MemoryCoordinator):
        self._db = agent_db
        self._coordinator = coordinator
        self._registered = False

    METHOD register():
        """Register AgentDB as a state provider + hook into save lifecycle."""

        IF self._registered:
            RETURN

        # Register state provider (CORE priority — after SAFETY, before QUALITY)
        self._coordinator.register_state_provider(
            name="agent_db",
            provider=self._get_state,
            priority=RestorePriority.CORE
        )

        # Monkey-patch save_all to include HNSW flush
        # (Alternative: use an event hook if coordinator supports it)
        original_save = self._coordinator.save_all

        ASYNC METHOD patched_save(source="manual"):
            # Flush HNSW index before checkpoint
            TRY:
                IF self._db._initialized:
                    self._db.save()
            EXCEPT Exception as e:
                LOG.warning(f"AgentDB HNSW flush failed during save: {e}")

            # Proceed with original save
            RETURN AWAIT original_save(source=source)

        self._coordinator.save_all = patched_save
        self._registered = True
        LOG.info("AgentDB registered with MemoryCoordinator")

    METHOD _get_state() -> dict:
        """State provider callback for checkpoint collection."""
        IF NOT self._db._initialized:
            RETURN {"status": "not_initialized"}

        RETURN self._db.get_persistable_state()

    METHOD ensure_initialized():
        """Called during restore to ensure AgentDB is ready."""
        IF NOT self._db._initialized:
            self._db.initialize()
```

---

## Pseudocode: Integration in `core/sovereign/runtime.py` (existing file, add wiring)

```
# In SovereignRuntime.__init__ or initialize():

FROM core.memory import AgentDB, MemoryConfig
FROM core.memory.coordinator_bridge import AgentDBBridge

# Create AgentDB
self._agent_db = AgentDB(MemoryConfig())
self._agent_db.initialize()

# Bridge into coordinator
self._db_bridge = AgentDBBridge(self._agent_db, self._memory_coordinator)
self._db_bridge.register()
```

---

## TDD Anchors

```
TEST test_bridge_registers_provider:
    db = AgentDB(config)
    db.initialize()
    coord = MemoryCoordinator()
    coord.initialize()

    bridge = AgentDBBridge(db, coord)
    bridge.register()

    ASSERT "agent_db" IN coord._state_providers
    ASSERT coord._state_providers["agent_db"][1] == RestorePriority.CORE

TEST test_save_all_flushes_hnsw:
    db = AgentDB(config)
    db.initialize()
    db.store("test content", embedding=[0.1]*768)

    coord = MemoryCoordinator()
    coord.initialize()
    bridge = AgentDBBridge(db, coord)
    bridge.register()

    AWAIT coord.save_all()

    # HNSW index file should exist on disk
    ASSERT config.hnsw_path.exists()

TEST test_save_all_tolerates_uninitialized_db:
    db = AgentDB(config)  # NOT initialized
    coord = MemoryCoordinator()
    coord.initialize()
    bridge = AgentDBBridge(db, coord)
    bridge.register()

    result = AWAIT coord.save_all()
    ASSERT result == True  # Should not crash

TEST test_state_provider_returns_counts:
    db = AgentDB(config)
    db.initialize()
    db.store("alpha")
    db.store("beta")

    coord = MemoryCoordinator()
    coord.initialize()
    bridge = AgentDBBridge(db, coord)
    bridge.register()

    state = bridge._get_state()
    ASSERT state["record_count"] == 2
    ASSERT "sqlite_path" IN state

TEST test_double_register_is_noop:
    bridge = AgentDBBridge(db, coord)
    bridge.register()
    bridge.register()  # Should not error or double-register
    ASSERT coord._state_providers.keys().count("agent_db") == 1
```

---

## Error Handling

| Scenario | Behavior |
|----------|----------|
| AgentDB not initialized at save time | Skip HNSW flush, log warning, continue |
| Disk full during HNSW save | Catch IOError, log error, save_all still returns checkpoint |
| Coordinator not initialized | `register()` works — provider is stored, called later |
| AgentDB crashes during state collection | Provider returns `{"error": "..."}`, checkpoint still saved |
