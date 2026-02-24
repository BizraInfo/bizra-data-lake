"""Tests for FR-01: MemoryCoordinator ↔ AgentDB Bridge."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.memory.agent_db import AgentDB
from core.memory.config import MemoryConfig
from core.memory.coordinator_bridge import AgentDBBridge


@pytest.fixture
def tmp_config(tmp_path: Path) -> MemoryConfig:
    cfg = MemoryConfig(data_dir=tmp_path / "agent_db")
    cfg.auto_embed = False
    return cfg


@pytest.fixture
def agent_db(tmp_config: MemoryConfig) -> AgentDB:
    db = AgentDB(tmp_config)
    db.initialize()
    return db


@pytest.fixture
def mock_coordinator():
    """Minimal mock of MemoryCoordinator interface."""
    coord = MagicMock()
    coord._state_providers = {}
    coord.save_all = AsyncMock(return_value=True)

    def _register(name, provider, priority):
        coord._state_providers[name] = (provider, priority)

    coord.register_state_provider = _register
    return coord


class TestBridgeRegistration:
    def test_register_adds_state_provider(self, agent_db, mock_coordinator):
        bridge = AgentDBBridge(agent_db, mock_coordinator)
        bridge.register()
        assert "agent_db" in mock_coordinator._state_providers

    def test_register_is_idempotent(self, agent_db, mock_coordinator):
        bridge = AgentDBBridge(agent_db, mock_coordinator)
        bridge.register()
        bridge.register()
        assert bridge.registered
        # Only one provider registered
        assert len(mock_coordinator._state_providers) == 1

    def test_registered_property(self, agent_db, mock_coordinator):
        bridge = AgentDBBridge(agent_db, mock_coordinator)
        assert not bridge.registered
        bridge.register()
        assert bridge.registered


class TestStateProvider:
    def test_get_state_returns_counts(self, agent_db, mock_coordinator):
        agent_db.store("alpha", importance=0.8)
        agent_db.store("beta", importance=0.7)

        bridge = AgentDBBridge(agent_db, mock_coordinator)
        bridge.register()

        provider_fn = mock_coordinator._state_providers["agent_db"][0]
        state = provider_fn()
        assert state["record_count"] == 2
        assert "sqlite_path" in state

    def test_get_state_uninitialized(self, tmp_config, mock_coordinator):
        db = AgentDB(tmp_config)  # NOT initialized
        bridge = AgentDBBridge(db, mock_coordinator)
        bridge.register()

        provider_fn = mock_coordinator._state_providers["agent_db"][0]
        state = provider_fn()
        assert state["status"] == "not_initialized"


class TestSaveIntegration:
    def test_save_all_flushes_hnsw(self, agent_db, mock_coordinator, tmp_config):
        import numpy as np

        dim = tmp_config.hnsw.dimensions
        agent_db.store("test", embedding=list(np.random.randn(dim).astype(float)))

        bridge = AgentDBBridge(agent_db, mock_coordinator)
        bridge.register()

        asyncio.get_event_loop().run_until_complete(
            mock_coordinator.save_all(source="test")
        )

        # Check that either the hnswlib index or numpy npz was saved
        assert (
            tmp_config.hnsw_path.exists()
            or tmp_config.hnsw_path.with_suffix(".meta.json").exists()
            or tmp_config.hnsw_path.with_suffix(".npz").exists()
        )

    def test_save_all_tolerates_uninitialized_db(self, tmp_config, mock_coordinator):
        db = AgentDB(tmp_config)  # NOT initialized
        bridge = AgentDBBridge(db, mock_coordinator)
        bridge.register()

        # Should not crash
        asyncio.get_event_loop().run_until_complete(
            mock_coordinator.save_all(source="test")
        )


class TestEnsureInitialized:
    def test_ensure_initialized_calls_init(self, tmp_config, mock_coordinator):
        db = AgentDB(tmp_config)
        bridge = AgentDBBridge(db, mock_coordinator)
        assert not db._initialized
        bridge.ensure_initialized()
        assert db._initialized
