"""Tests for FR-04: Health & Metrics API."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.memory.agent_db import AgentDB
from core.memory.config import MemoryConfig
from core.memory.health import (
    AgentDBHealthChecker,
    AgentDBMetrics,
    HealthReport,
    HealthStatus,
)


@pytest.fixture
def tmp_config(tmp_path: Path) -> MemoryConfig:
    cfg = MemoryConfig(data_dir=tmp_path / "agent_db")
    cfg.auto_embed = False
    return cfg


@pytest.fixture
def db(tmp_config: MemoryConfig) -> AgentDB:
    d = AgentDB(tmp_config)
    d.initialize()
    return d


@pytest.mark.xdist_group(name="serial_integration")
class TestHealthReport:
    def test_healthy_db(self, db):
        db.store("test record")
        checker = AgentDBHealthChecker(db)
        report = checker.check()

        assert report.status == HealthStatus.HEALTHY
        assert report.is_healthy()
        assert "sqlite" in report.components
        assert "hnsw" in report.components
        assert "memory" in report.components

    def test_sqlite_details(self, db):
        db.store("content A")
        checker = AgentDBHealthChecker(db)
        report = checker.check()

        sqlite = report.components["sqlite"]
        assert sqlite.status == HealthStatus.HEALTHY
        assert sqlite.details["writable"] is True
        assert sqlite.details["fts5_ok"] is True
        assert sqlite.details["records"]["active"] == 1

    def test_hnsw_details(self, db):
        checker = AgentDBHealthChecker(db)
        report = checker.check()

        hnsw = report.components["hnsw"]
        assert hnsw.details["vector_count"] >= 0
        assert hnsw.details["dimensions"] == 768

    def test_uninitialized_db_is_down(self, tmp_config):
        db = AgentDB(tmp_config)  # NOT initialized
        checker = AgentDBHealthChecker(db)
        report = checker.check()

        assert report.status == HealthStatus.DOWN
        assert report.components["sqlite"].status == HealthStatus.DOWN

    def test_to_dict(self, db):
        checker = AgentDBHealthChecker(db)
        report = checker.check()
        d = report.to_dict()

        assert d["status"] == "healthy"
        assert "sqlite" in d["components"]
        assert "timestamp" in d

    def test_memory_component(self, db):
        checker = AgentDBHealthChecker(db)
        report = checker.check()

        mem = report.components["memory"]
        assert mem.status == HealthStatus.HEALTHY
        assert "estimated_mb" in mem.details
        assert mem.details["estimated_mb"] >= 0


@pytest.mark.xdist_group(name="serial_integration")
class TestAgentDBMetrics:
    def test_metrics_enabled(self, db):
        pytest.importorskip("prometheus_client")
        metrics = AgentDBMetrics(db)
        assert metrics.enabled

    def test_update_does_not_crash(self, db):
        db.store("test")
        metrics = AgentDBMetrics(db)
        metrics.update()  # Should not raise

    def test_observe_search(self, db):
        metrics = AgentDBMetrics(db)
        metrics.observe_search(0.001)  # Should not raise

    def test_observe_store(self, db):
        metrics = AgentDBMetrics(db)
        metrics.observe_store(0.002)  # Should not raise
