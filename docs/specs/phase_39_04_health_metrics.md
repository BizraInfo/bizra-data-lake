# Phase 39 — Pseudocode Module 04: Health & Metrics API

**FR-04** | Priority: 4 | Risk: Low | New files: 1

---

## Overview

Expose AgentDB internals for monitoring — structured health reports for
programmatic checks, Prometheus metrics for Grafana dashboards.

---

## Flow Diagram

```
AgentDB.health()
  ├── SQLite: writable? file_size? record_count? FTS5 ok?
  ├── HNSW: loaded? vector_count? capacity? ratio?
  ├── Memory: estimated RAM usage
  └── Returns: HealthReport (status: healthy | degraded | down)

Prometheus (optional)
  ├── agentdb_records_total{kind, state}
  ├── agentdb_vectors_total
  ├── agentdb_search_duration_seconds
  ├── agentdb_store_duration_seconds
  └── agentdb_hnsw_capacity_ratio
```

---

## Pseudocode: `core/memory/health.py`

```
MODULE health

IMPORT logging, os, time
FROM dataclasses IMPORT dataclass, field
FROM enum IMPORT Enum
FROM typing IMPORT Dict, Optional

FROM .agent_db IMPORT AgentDB
FROM .config IMPORT MemoryConfig
FROM .types IMPORT MemoryKind, RecordState

LOG = logging.getLogger(__name__)

# Optional Prometheus
TRY:
    FROM prometheus_client IMPORT Gauge, Histogram, Counter
    _HAS_PROM = True
EXCEPT ImportError:
    _HAS_PROM = False


CLASS HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"


@dataclass
CLASS ComponentHealth:
    name: str
    status: HealthStatus
    details: Dict[str, object] = field(default_factory=dict)


@dataclass
CLASS HealthReport:
    status: HealthStatus
    components: Dict[str, ComponentHealth] = field(default_factory=dict)
    timestamp_iso: str = ""

    METHOD is_healthy() -> bool:
        RETURN self.status == HealthStatus.HEALTHY


CLASS AgentDBHealthChecker:
    """Produces structured health reports for AgentDB."""

    CONSTRUCTOR(agent_db: AgentDB):
        self._db = agent_db

    METHOD check() -> HealthReport:
        """Run all health checks, return aggregated report."""
        FROM datetime IMPORT datetime, timezone
        report = HealthReport(
            status=HealthStatus.HEALTHY,
            timestamp_iso=datetime.now(timezone.utc).isoformat()
        )

        # Check SQLite
        sqlite_health = self._check_sqlite()
        report.components["sqlite"] = sqlite_health

        # Check HNSW
        hnsw_health = self._check_hnsw()
        report.components["hnsw"] = hnsw_health

        # Check memory pressure
        mem_health = self._check_memory()
        report.components["memory"] = mem_health

        # Aggregate: worst component status wins
        statuses = [c.status FOR c IN report.components.values()]
        IF HealthStatus.DOWN IN statuses:
            report.status = HealthStatus.DOWN
        ELIF HealthStatus.DEGRADED IN statuses:
            report.status = HealthStatus.DEGRADED

        RETURN report

    METHOD _check_sqlite() -> ComponentHealth:
        TRY:
            IF NOT self._db._initialized:
                RETURN ComponentHealth("sqlite", HealthStatus.DOWN, {"reason": "not_initialized"})

            store = self._db.backend
            path = store.db_path

            # Check file exists and is writable
            IF NOT path.exists():
                RETURN ComponentHealth("sqlite", HealthStatus.DOWN, {"reason": "file_missing"})

            writable = os.access(str(path), os.W_OK)
            file_size_mb = path.stat().st_size / (1024 * 1024)

            # Count records by state
            counts = {}
            FOR state IN RecordState:
                counts[state.value] = store.count(state=state)

            # Test FTS5 by running a trivial query
            fts_ok = True
            TRY:
                store.keyword_search("__health_check__", top_k=1)
            EXCEPT Exception:
                fts_ok = False

            status = HealthStatus.HEALTHY
            IF NOT writable:
                status = HealthStatus.DEGRADED
            IF NOT fts_ok:
                status = HealthStatus.DEGRADED

            RETURN ComponentHealth("sqlite", status, {
                "writable": writable,
                "file_size_mb": round(file_size_mb, 2),
                "records": counts,
                "fts5_ok": fts_ok,
                "path": str(path),
            })

        EXCEPT Exception as e:
            RETURN ComponentHealth("sqlite", HealthStatus.DOWN, {"error": str(e)})

    METHOD _check_hnsw() -> ComponentHealth:
        TRY:
            hnsw = self._db.hnsw

            IF NOT hnsw._initialized:
                RETURN ComponentHealth("hnsw", HealthStatus.DEGRADED, {"reason": "not_initialized"})

            count = hnsw.count
            capacity = hnsw.capacity
            ratio = count / max(capacity, 1)

            status = HealthStatus.HEALTHY
            IF ratio > 0.9:
                status = HealthStatus.DEGRADED  # Near capacity

            RETURN ComponentHealth("hnsw", status, {
                "vector_count": count,
                "capacity": capacity,
                "capacity_ratio": round(ratio, 4),
                "dimensions": hnsw._config.dimensions,
                "backend": "hnswlib" IF hnsw._use_hnswlib ELSE "numpy",
                "index_path": str(self._db._config.hnsw_path),
            })

        EXCEPT Exception as e:
            RETURN ComponentHealth("hnsw", HealthStatus.DOWN, {"error": str(e)})

    METHOD _check_memory() -> ComponentHealth:
        """Estimate RAM usage of in-memory structures."""
        TRY:
            IMPORT sys

            est_bytes = 0

            # HNSW id maps
            hnsw = self._db.hnsw
            est_bytes += sys.getsizeof(hnsw._id_map) + len(hnsw._id_map) * 80
            est_bytes += sys.getsizeof(hnsw._reverse_map) + len(hnsw._reverse_map) * 80

            # Fallback vectors (numpy)
            FOR vec IN hnsw._fallback_vectors.values():
                est_bytes += vec.nbytes

            est_mb = est_bytes / (1024 * 1024)

            status = HealthStatus.HEALTHY
            IF est_mb > 1000:  # > 1GB in maps
                status = HealthStatus.DEGRADED

            RETURN ComponentHealth("memory", status, {
                "estimated_mb": round(est_mb, 2),
            })

        EXCEPT Exception as e:
            RETURN ComponentHealth("memory", HealthStatus.HEALTHY, {
                "estimated_mb": -1,
                "error": str(e)
            })


CLASS AgentDBMetrics:
    """Prometheus metrics for AgentDB (no-op if prometheus_client unavailable)."""

    CONSTRUCTOR(agent_db: AgentDB):
        self._db = agent_db
        self._enabled = _HAS_PROM

        IF self._enabled:
            self._records_gauge = Gauge(
                "agentdb_records_total",
                "Total memory records",
                labelnames=["kind", "state"]
            )
            self._vectors_gauge = Gauge(
                "agentdb_vectors_total",
                "Total indexed vectors in HNSW"
            )
            self._search_hist = Histogram(
                "agentdb_search_duration_seconds",
                "Search query duration",
                buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
            )
            self._store_hist = Histogram(
                "agentdb_store_duration_seconds",
                "Store operation duration",
                buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
            )
            self._capacity_gauge = Gauge(
                "agentdb_hnsw_capacity_ratio",
                "HNSW index fill ratio (count / max_elements)"
            )

    METHOD update():
        """Refresh all gauges from current AgentDB state."""
        IF NOT self._enabled:
            RETURN

        TRY:
            store = self._db.backend

            # Record counts by kind x state
            FOR kind IN MemoryKind:
                FOR state IN RecordState:
                    # This would need a combined query — approximate with total
                    pass

            # Simpler: just total by state
            FOR state IN RecordState:
                count = store.count(state=state)
                self._records_gauge.labels(kind="all", state=state.value).set(count)

            self._vectors_gauge.set(self._db.hnsw.count)

            capacity = self._db.hnsw.capacity
            IF capacity > 0:
                self._capacity_gauge.set(self._db.hnsw.count / capacity)

        EXCEPT Exception as e:
            LOG.warning(f"Metrics update failed: {e}")

    METHOD observe_search(duration_seconds: float):
        IF self._enabled:
            self._search_hist.observe(duration_seconds)

    METHOD observe_store(duration_seconds: float):
        IF self._enabled:
            self._store_hist.observe(duration_seconds)
```

---

## TDD Anchors

```
TEST test_health_report_healthy:
    db = AgentDB(config)
    db.initialize()
    db.store("test record")

    checker = AgentDBHealthChecker(db)
    report = checker.check()

    ASSERT report.status == HealthStatus.HEALTHY
    ASSERT report.is_healthy()
    ASSERT "sqlite" IN report.components
    ASSERT "hnsw" IN report.components
    ASSERT report.components["sqlite"].details["fts5_ok"] == True

TEST test_health_report_uninitialized:
    db = AgentDB(config)  # NOT initialized
    checker = AgentDBHealthChecker(db)
    report = checker.check()

    ASSERT report.status == HealthStatus.DOWN
    ASSERT report.components["sqlite"].status == HealthStatus.DOWN

TEST test_health_hnsw_near_capacity:
    config = MemoryConfig()
    config.hnsw.max_elements = 10  # Tiny capacity
    db = AgentDB(config)
    db.initialize()

    # Fill to 90%+
    FOR i IN range(10):
        db.store(f"record {i}", embedding=[float(i)] * 768)

    checker = AgentDBHealthChecker(db)
    report = checker.check()

    ASSERT report.components["hnsw"].details["capacity_ratio"] >= 0.9
    ASSERT report.components["hnsw"].status == HealthStatus.DEGRADED

TEST test_metrics_no_prometheus:
    # Mock prometheus_client as unavailable
    WITH mock.patch.dict(sys.modules, {"prometheus_client": None}):
        db = AgentDB(config)
        metrics = AgentDBMetrics(db)
        metrics.update()  # Should not crash
        metrics.observe_search(0.001)  # No-op

TEST test_metrics_with_prometheus:
    db = AgentDB(config)
    db.initialize()
    db.store("test")

    metrics = AgentDBMetrics(db)
    metrics.update()

    # Verify gauges are set (prometheus_client exposes .get() on gauges)
    ASSERT metrics._vectors_gauge._value.get() >= 0
```
