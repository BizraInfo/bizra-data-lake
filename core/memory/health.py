"""
Health & Metrics — Structured health checks and optional Prometheus metrics.

AgentDBHealthChecker produces a HealthReport suitable for monitoring
dashboards and /health endpoints. AgentDBMetrics optionally exports
Prometheus gauges and histograms.

Usage:
    from core.memory.health import AgentDBHealthChecker
    checker = AgentDBHealthChecker(db)
    report = checker.check()
    print(report.status)  # "healthy" | "degraded" | "down"

Standing on Giants: ADR-006, Prometheus exposition format (2012)
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict

from .types import RecordState

logger = logging.getLogger(__name__)

try:
    from prometheus_client import CollectorRegistry, Gauge, Histogram

    _HAS_PROM = True
except ImportError:
    _HAS_PROM = False


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"


@dataclass
class ComponentHealth:
    name: str
    status: HealthStatus
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthReport:
    status: HealthStatus = HealthStatus.HEALTHY
    components: Dict[str, ComponentHealth] = field(default_factory=dict)
    timestamp_iso: str = ""

    def is_healthy(self) -> bool:
        return self.status == HealthStatus.HEALTHY

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "timestamp": self.timestamp_iso,
            "components": {
                name: {"status": c.status.value, **c.details}
                for name, c in self.components.items()
            },
        }


class AgentDBHealthChecker:
    """Produces structured health reports for AgentDB."""

    def __init__(self, agent_db) -> None:
        self._db = agent_db

    def check(self) -> HealthReport:
        report = HealthReport(
            timestamp_iso=datetime.now(timezone.utc).isoformat(),
        )

        report.components["sqlite"] = self._check_sqlite()
        report.components["hnsw"] = self._check_hnsw()
        report.components["memory"] = self._check_memory()

        statuses = [c.status for c in report.components.values()]
        if HealthStatus.DOWN in statuses:
            report.status = HealthStatus.DOWN
        elif HealthStatus.DEGRADED in statuses:
            report.status = HealthStatus.DEGRADED

        return report

    def _check_sqlite(self) -> ComponentHealth:
        try:
            if not getattr(self._db, "_initialized", False):
                return ComponentHealth(
                    "sqlite", HealthStatus.DOWN, {"reason": "not_initialized"}
                )

            store = self._db.backend
            path = store.db_path

            if not path.exists():
                return ComponentHealth(
                    "sqlite", HealthStatus.DOWN, {"reason": "file_missing"}
                )

            writable = os.access(str(path), os.W_OK)
            file_size_mb = path.stat().st_size / (1024 * 1024)

            counts: Dict[str, int] = {}
            for state in RecordState:
                counts[state.value] = store.count(state=state)

            fts_ok = True
            try:
                store.keyword_search("__health_check__", top_k=1)
            except Exception:
                fts_ok = False

            status = HealthStatus.HEALTHY
            if not writable or not fts_ok:
                status = HealthStatus.DEGRADED

            return ComponentHealth(
                "sqlite",
                status,
                {
                    "writable": writable,
                    "file_size_mb": round(file_size_mb, 2),
                    "records": counts,
                    "fts5_ok": fts_ok,
                    "path": str(path),
                },
            )
        except Exception as e:
            return ComponentHealth("sqlite", HealthStatus.DOWN, {"error": str(e)})

    def _check_hnsw(self) -> ComponentHealth:
        try:
            hnsw = self._db.hnsw

            if not getattr(hnsw, "_initialized", False):
                return ComponentHealth(
                    "hnsw", HealthStatus.DEGRADED, {"reason": "not_initialized"}
                )

            count = hnsw.count
            capacity = hnsw.capacity
            ratio = count / max(capacity, 1)

            status = HealthStatus.HEALTHY
            if ratio > 0.9:
                status = HealthStatus.DEGRADED

            return ComponentHealth(
                "hnsw",
                status,
                {
                    "vector_count": count,
                    "capacity": capacity,
                    "capacity_ratio": round(ratio, 4),
                    "dimensions": hnsw._config.dimensions,
                    "backend": "hnswlib" if hnsw._use_hnswlib else "numpy",
                },
            )
        except Exception as e:
            return ComponentHealth("hnsw", HealthStatus.DOWN, {"error": str(e)})

    def _check_memory(self) -> ComponentHealth:
        try:
            est_bytes = 0
            hnsw = self._db.hnsw
            est_bytes += sys.getsizeof(hnsw._id_map) + len(hnsw._id_map) * 80
            est_bytes += sys.getsizeof(hnsw._reverse_map) + len(hnsw._reverse_map) * 80
            for vec in hnsw._fallback_vectors.values():
                est_bytes += vec.nbytes
            est_mb = est_bytes / (1024 * 1024)

            status = HealthStatus.HEALTHY
            if est_mb > 1000:
                status = HealthStatus.DEGRADED

            return ComponentHealth("memory", status, {"estimated_mb": round(est_mb, 2)})
        except Exception as e:
            return ComponentHealth(
                "memory", HealthStatus.HEALTHY, {"estimated_mb": -1, "error": str(e)}
            )


class AgentDBMetrics:
    """Prometheus metrics for AgentDB. No-op if prometheus_client unavailable.

    Uses a dedicated CollectorRegistry to avoid duplication in tests
    and multi-instance scenarios.
    """

    def __init__(self, agent_db, registry=None) -> None:
        self._db = agent_db
        self._enabled = _HAS_PROM

        if self._enabled:
            reg = registry or CollectorRegistry(auto_describe=True)
            self._registry = reg
            self._records_gauge = Gauge(
                "agentdb_records_total",
                "Total memory records",
                labelnames=["state"],
                registry=reg,
            )
            self._vectors_gauge = Gauge(
                "agentdb_vectors_total",
                "Indexed vectors in HNSW",
                registry=reg,
            )
            self._search_hist = Histogram(
                "agentdb_search_duration_seconds",
                "Search query duration",
                buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
                registry=reg,
            )
            self._store_hist = Histogram(
                "agentdb_store_duration_seconds",
                "Store operation duration",
                buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
                registry=reg,
            )
            self._capacity_gauge = Gauge(
                "agentdb_hnsw_capacity_ratio",
                "HNSW fill ratio",
                registry=reg,
            )

    @property
    def enabled(self) -> bool:
        return self._enabled

    def update(self) -> None:
        """Refresh all gauges from current state."""
        if not self._enabled:
            return
        try:
            store = self._db.backend
            for state in RecordState:
                self._records_gauge.labels(state=state.value).set(
                    store.count(state=state)
                )
            self._vectors_gauge.set(self._db.hnsw.count)
            cap = self._db.hnsw.capacity
            if cap > 0:
                self._capacity_gauge.set(self._db.hnsw.count / cap)
        except Exception as e:
            logger.warning(f"Metrics update failed: {e}")

    def observe_search(self, duration_seconds: float) -> None:
        if self._enabled:
            self._search_hist.observe(duration_seconds)

    def observe_store(self, duration_seconds: float) -> None:
        if self._enabled:
            self._store_hist.observe(duration_seconds)
