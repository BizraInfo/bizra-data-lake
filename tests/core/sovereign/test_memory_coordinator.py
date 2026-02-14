"""
Tests for Memory Coordinator — Unified Auto-Save Engine
=======================================================
Verifies that MemoryCoordinator correctly:
- Collects state from registered providers
- Creates versioned checkpoints with checksums
- Restores from saved state
- Auto-save loop runs and persists
- Integrates with LivingMemoryCore
- Includes genesis identity in every save

Standing on Giants: Event Sourcing + Snapshot Pattern + Write-Ahead Logging
"""

import asyncio
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from core.sovereign.memory_coordinator import (
    MemoryCoordinator,
    MemoryCoordinatorConfig,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def state_dir(tmp_path):
    """Create a temporary state directory."""
    return tmp_path


@pytest.fixture
def config(state_dir):
    """Create a test configuration."""
    return MemoryCoordinatorConfig(
        state_dir=state_dir,
        auto_save_interval=1.0,
        max_checkpoints=5,
    )


@pytest.fixture
def coordinator(config):
    """Create an initialized coordinator."""
    mc = MemoryCoordinator(config)
    mc.initialize(node_id="node0_test123", node_name="TestNode")
    return mc


# =============================================================================
# TESTS: Initialization
# =============================================================================


class TestInit:
    def test_default_config(self):
        mc = MemoryCoordinator()
        assert mc.config.auto_save_interval == 120.0
        assert mc.config.max_checkpoints == 20
        assert not mc._running

    def test_custom_config(self, config):
        mc = MemoryCoordinator(config)
        assert mc.config.auto_save_interval == 1.0
        assert mc.config.max_checkpoints == 5

    def test_initialize_creates_dirs(self, config, state_dir):
        mc = MemoryCoordinator(config)
        mc.initialize(node_id="node0_abc", node_name="Test")
        assert (state_dir / "checkpoints").exists()

    def test_initialize_stores_identity(self, coordinator):
        assert coordinator._node_id == "node0_test123"
        assert coordinator._node_name == "TestNode"

    def test_stats_before_save(self, coordinator):
        stats = coordinator.stats()
        assert stats["node_id"] == "node0_test123"
        assert stats["running"] is False
        assert stats["save_count"] == 0
        assert stats["last_save"] is None
        assert stats["providers"] == []


# =============================================================================
# TESTS: State Providers
# =============================================================================


class TestStateProviders:
    def test_register_provider(self, coordinator):
        coordinator.register_state_provider("test", lambda: {"key": "val"})
        assert "test" in coordinator._state_providers

    def test_register_multiple_providers(self, coordinator):
        coordinator.register_state_provider("a", lambda: {"x": 1})
        coordinator.register_state_provider("b", lambda: {"y": 2})
        assert len(coordinator._state_providers) == 2

    def test_provider_appears_in_stats(self, coordinator):
        coordinator.register_state_provider("metrics", lambda: {})
        stats = coordinator.stats()
        assert "metrics" in stats["providers"]


# =============================================================================
# TESTS: Save
# =============================================================================


class TestSave:
    @pytest.mark.asyncio
    async def test_save_empty(self, coordinator):
        """Save with no providers should still succeed."""
        ok = await coordinator.save_all(source="test")
        assert ok is True
        assert coordinator._save_count == 1

    @pytest.mark.asyncio
    async def test_save_with_provider(self, coordinator):
        coordinator.register_state_provider(
            "runtime", lambda: {"queries": 42, "score": 0.95}
        )
        ok = await coordinator.save_all()
        assert ok is True

    @pytest.mark.asyncio
    async def test_save_creates_checkpoint_file(self, coordinator, state_dir):
        await coordinator.save_all()
        cp_files = list((state_dir / "checkpoints").glob("cp-*.json"))
        assert len(cp_files) == 1

    @pytest.mark.asyncio
    async def test_save_increments_count(self, coordinator):
        await coordinator.save_all()
        await coordinator.save_all()
        assert coordinator._save_count == 2

    @pytest.mark.asyncio
    async def test_save_updates_last_save(self, coordinator):
        assert coordinator._last_save is None
        await coordinator.save_all()
        assert coordinator._last_save is not None

    @pytest.mark.asyncio
    async def test_save_includes_node_id(self, coordinator, state_dir):
        await coordinator.save_all()
        cp_file = list((state_dir / "checkpoints").glob("cp-*.json"))[0]
        data = json.loads(cp_file.read_text())
        assert data["state"]["coordinator"]["node_id"] == "node0_test123"

    @pytest.mark.asyncio
    async def test_save_includes_provider_data(self, coordinator, state_dir):
        coordinator.register_state_provider(
            "test_data", lambda: {"answer": 42}
        )
        await coordinator.save_all()
        cp_file = list((state_dir / "checkpoints").glob("cp-*.json"))[0]
        data = json.loads(cp_file.read_text())
        assert data["state"]["test_data"]["answer"] == 42

    @pytest.mark.asyncio
    async def test_save_handles_provider_error(self, coordinator):
        """A failing provider should not crash the save."""
        coordinator.register_state_provider(
            "broken", lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        coordinator.register_state_provider("good", lambda: {"ok": True})
        ok = await coordinator.save_all()
        assert ok is True


# =============================================================================
# TESTS: Restore
# =============================================================================


class TestRestore:
    @pytest.mark.asyncio
    async def test_restore_no_checkpoints(self, coordinator):
        state = await coordinator.restore_latest()
        assert state is None

    @pytest.mark.asyncio
    async def test_restore_after_save(self, coordinator):
        coordinator.register_state_provider(
            "rt", lambda: {"cycles": 100}
        )
        await coordinator.save_all()
        state = await coordinator.restore_latest()
        assert state is not None
        assert state["rt"]["cycles"] == 100

    @pytest.mark.asyncio
    async def test_restore_gets_latest(self, coordinator):
        coordinator.register_state_provider(
            "ver", lambda: {"v": coordinator._save_count}
        )
        await coordinator.save_all()
        await coordinator.save_all()
        state = await coordinator.restore_latest()
        assert state["ver"]["v"] == 1  # provider returns count before increment

    @pytest.mark.asyncio
    async def test_restore_preserves_genesis(self, coordinator):
        await coordinator.save_all()
        state = await coordinator.restore_latest()
        assert state["coordinator"]["node_id"] == "node0_test123"
        assert state["coordinator"]["node_name"] == "TestNode"


# =============================================================================
# TESTS: Living Memory Integration
# =============================================================================


class TestLivingMemory:
    @pytest.mark.asyncio
    async def test_register_living_memory(self, coordinator):
        mock_lm = MagicMock()
        coordinator.register_living_memory(mock_lm)
        assert coordinator._living_memory is mock_lm

    @pytest.mark.asyncio
    async def test_save_calls_living_memory_save(self, coordinator):
        mock_lm = MagicMock()
        mock_lm._save_memories = AsyncMock()
        mock_lm.get_stats = MagicMock(return_value=MagicMock(
            to_dict=lambda: {"total_entries": 5}
        ))
        coordinator.register_living_memory(mock_lm)

        await coordinator.save_all()
        mock_lm._save_memories.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_save_includes_living_memory_stats(self, coordinator, state_dir):
        mock_lm = MagicMock()
        mock_lm._save_memories = AsyncMock()
        mock_lm.get_stats = MagicMock(return_value=MagicMock(
            to_dict=lambda: {"total_entries": 10, "active_entries": 8}
        ))
        coordinator.register_living_memory(mock_lm)

        await coordinator.save_all()
        state = await coordinator.restore_latest()
        assert state["living_memory"]["total_entries"] == 10


# =============================================================================
# TESTS: Auto-Save Loop
# =============================================================================


class TestAutoSave:
    @pytest.mark.asyncio
    async def test_start_auto_save(self, coordinator):
        await coordinator.start_auto_save()
        assert coordinator._running is True
        assert coordinator._auto_save_task is not None
        # Clean up
        await coordinator.stop()

    @pytest.mark.asyncio
    async def test_auto_save_creates_checkpoints(self, config, state_dir):
        config.auto_save_interval = 0.2  # 200ms for test speed
        mc = MemoryCoordinator(config)
        mc.initialize(node_id="node0_fast")
        mc.register_state_provider("tick", lambda: {"t": mc._save_count})

        await mc.start_auto_save()
        await asyncio.sleep(0.7)  # Should trigger ~3 auto-saves
        await mc.stop()

        assert mc._save_count >= 3  # 3 auto-saves + 1 shutdown save

    @pytest.mark.asyncio
    async def test_stop_performs_final_save(self, coordinator):
        await coordinator.start_auto_save()
        await coordinator.stop()
        assert coordinator._save_count >= 1  # shutdown save
        assert coordinator._running is False


# =============================================================================
# TESTS: Checkpoint Rotation
# =============================================================================


class TestRotation:
    @pytest.mark.asyncio
    async def test_respects_max_checkpoints(self, config, state_dir):
        config.max_checkpoints = 3
        mc = MemoryCoordinator(config)
        mc.initialize(node_id="node0_rotate")

        for i in range(6):
            await mc.save_all()

        cp_files = list((state_dir / "checkpoints").glob("cp-*.json"))
        assert len(cp_files) <= 3


# =============================================================================
# TESTS: Stats
# =============================================================================


class TestStats:
    @pytest.mark.asyncio
    async def test_stats_after_save(self, coordinator):
        coordinator.register_state_provider("x", lambda: {})
        await coordinator.save_all()

        stats = coordinator.stats()
        assert stats["save_count"] == 1
        assert stats["last_save"] is not None
        assert "x" in stats["providers"]
        assert stats["checkpointer"]["checkpoint_count"] == 1

    @pytest.mark.asyncio
    async def test_stats_running_state(self, coordinator):
        await coordinator.start_auto_save()
        assert coordinator.stats()["running"] is True
        await coordinator.stop()
        assert coordinator.stats()["running"] is False
