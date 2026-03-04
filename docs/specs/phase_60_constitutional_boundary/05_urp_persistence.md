# Step 5: URP Shared Persistence (StorageBackend Abstraction)

## Standing on Giants: Lamport (state machine replication) | Brewer (CAP theorem) | Codd (relational algebra)

## Problem Statement

SAPE audit finding F5 (verified TRUE) identified that all 4 URP services use
process-local in-memory dictionaries as their data store:

```python
# Pattern found in all URP services:
_DB: dict[str, Any] = {}
```

Services affected:
- `urp_registry` — model registry
- `urp_knowledge_graph` — reflex knowledge store
- `urp_consensus` — PoI consensus records
- `urp_verification` — crown verification records

Under multi-worker (uvicorn `--workers N`) or multi-replica (K8s replicas > 1)
deployment, each worker/replica maintains its own `_DB`. This causes:

1. **Split-brain:** Worker A registers a model, Worker B doesn't see it
2. **Data loss on restart:** All state evaporates when the process exits
3. **Inconsistent reads:** Load balancer routes requests to random workers
4. **Testing false positives:** Single-worker tests pass, multi-worker fails

**Solution:** Introduce a `StorageBackend` abstraction that:
- Defaults to in-memory dict for tests (fast, isolated)
- Uses Redis for production (shared across workers/replicas)
- Optionally uses SQLite for single-node persistence without Redis

## Target Files

| File | Action |
|------|--------|
| `.tmp_prod_artifacts_v2/services/_shared/app/storage.py` | New: StorageBackend abstraction |
| `.tmp_prod_artifacts_v2/services/urp_registry/app/routers.py` | Update: use StorageBackend |
| `.tmp_prod_artifacts_v2/services/urp_knowledge_graph/app/routers.py` | Update: use StorageBackend |
| `.tmp_prod_artifacts_v2/services/urp_consensus/app/routers.py` | Update: use StorageBackend |
| `.tmp_prod_artifacts_v2/services/urp_verification/app/routers.py` | Update: use StorageBackend |
| `tests/services/test_storage_backend.py` | New: backend tests |

## Pseudocode

### services/_shared/app/storage.py

```pseudocode
"""StorageBackend — Pluggable persistence for URP services.

Usage:
    # In-memory (tests, development)
    store = MemoryBackend()

    # Redis (production, multi-worker)
    store = RedisBackend(url="redis://localhost:6379/0", prefix="urp_registry")

    # SQLite (single-node persistence without Redis)
    store = SQLiteBackend(path="/data/urp.db", table="registry")

All backends implement the same interface:
    get(key) -> value | None
    set(key, value) -> None
    delete(key) -> bool
    exists(key) -> bool
    list_keys(prefix) -> list[str]
    count() -> int
"""

IMPORT abc
IMPORT json
FROM typing IMPORT Any, Optional


CLASS StorageBackend(abc.ABC):
    """Abstract storage interface for URP services."""

    @abc.abstractmethod
    FUNCTION get(self, key: str) -> Optional[Any]:
        ...

    @abc.abstractmethod
    FUNCTION set(self, key: str, value: Any) -> None:
        ...

    @abc.abstractmethod
    FUNCTION delete(self, key: str) -> bool:
        ...

    @abc.abstractmethod
    FUNCTION exists(self, key: str) -> bool:
        ...

    @abc.abstractmethod
    FUNCTION list_keys(self, prefix: str = "") -> list[str]:
        ...

    @abc.abstractmethod
    FUNCTION count(self) -> int:
        ...


CLASS MemoryBackend(StorageBackend):
    """In-memory dict backend for tests and development."""

    FUNCTION __init__(self):
        self._data: dict[str, Any] = {}

    FUNCTION get(self, key: str) -> Optional[Any]:
        RETURN self._data.get(key)

    FUNCTION set(self, key: str, value: Any) -> None:
        self._data[key] = value

    FUNCTION delete(self, key: str) -> bool:
        RETURN self._data.pop(key, _SENTINEL) IS NOT _SENTINEL

    FUNCTION exists(self, key: str) -> bool:
        RETURN key IN self._data

    FUNCTION list_keys(self, prefix: str = "") -> list[str]:
        RETURN [k FOR k IN self._data IF k.startswith(prefix)]

    FUNCTION count(self) -> int:
        RETURN len(self._data)


CLASS RedisBackend(StorageBackend):
    """Redis-backed storage for multi-worker/multi-replica deployments.

    Values are JSON-serialized. Keys are prefixed with service name
    to allow multiple services to share one Redis instance.
    """

    FUNCTION __init__(self, url: str = "redis://localhost:6379/0", prefix: str = "urp"):
        IMPORT redis
        self._client = redis.from_url(url, decode_responses=True)
        self._prefix = prefix

    FUNCTION _key(self, key: str) -> str:
        RETURN f"{self._prefix}:{key}"

    FUNCTION get(self, key: str) -> Optional[Any]:
        raw = self._client.get(self._key(key))
        IF raw IS None:
            RETURN None
        RETURN json.loads(raw)

    FUNCTION set(self, key: str, value: Any) -> None:
        self._client.set(self._key(key), json.dumps(value))

    FUNCTION delete(self, key: str) -> bool:
        RETURN self._client.delete(self._key(key)) > 0

    FUNCTION exists(self, key: str) -> bool:
        RETURN self._client.exists(self._key(key)) > 0

    FUNCTION list_keys(self, prefix: str = "") -> list[str]:
        pattern = self._key(prefix) + "*"
        keys = self._client.keys(pattern)
        strip_len = len(self._prefix) + 1
        RETURN [k[strip_len:] FOR k IN keys]

    FUNCTION count(self) -> int:
        RETURN len(self._client.keys(self._key("") + "*"))


CLASS SQLiteBackend(StorageBackend):
    """SQLite-backed storage for single-node persistence without Redis."""

    FUNCTION __init__(self, path: str = ":memory:", table: str = "kv"):
        IMPORT sqlite3
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._table = table
        self._conn.execute(
            f"CREATE TABLE IF NOT EXISTS {table} "
            f"(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self._conn.commit()

    FUNCTION get(self, key: str) -> Optional[Any]:
        row = self._conn.execute(
            f"SELECT value FROM {self._table} WHERE key = ?", (key,)
        ).fetchone()
        RETURN json.loads(row[0]) IF row ELSE None

    FUNCTION set(self, key: str, value: Any) -> None:
        self._conn.execute(
            f"INSERT OR REPLACE INTO {self._table} (key, value) VALUES (?, ?)",
            (key, json.dumps(value)),
        )
        self._conn.commit()

    FUNCTION delete(self, key: str) -> bool:
        cursor = self._conn.execute(
            f"DELETE FROM {self._table} WHERE key = ?", (key,)
        )
        self._conn.commit()
        RETURN cursor.rowcount > 0

    FUNCTION exists(self, key: str) -> bool:
        row = self._conn.execute(
            f"SELECT 1 FROM {self._table} WHERE key = ?", (key,)
        ).fetchone()
        RETURN row IS NOT None

    FUNCTION list_keys(self, prefix: str = "") -> list[str]:
        rows = self._conn.execute(
            f"SELECT key FROM {self._table} WHERE key LIKE ?",
            (prefix + "%",),
        ).fetchall()
        RETURN [r[0] FOR r IN rows]

    FUNCTION count(self) -> int:
        row = self._conn.execute(
            f"SELECT COUNT(*) FROM {self._table}"
        ).fetchone()
        RETURN row[0]


FUNCTION create_backend(backend_type: str = "memory", **kwargs) -> StorageBackend:
    """Factory: create a storage backend by type name.

    Args:
        backend_type: "memory", "redis", or "sqlite"
        **kwargs: passed to backend constructor

    Env override: URP_STORAGE_BACKEND (takes precedence over backend_type arg)
    """
    IMPORT os
    effective_type = os.getenv("URP_STORAGE_BACKEND", backend_type).lower()

    IF effective_type == "memory":
        RETURN MemoryBackend()
    ELIF effective_type == "redis":
        url = kwargs.get("url", os.getenv("URP_REDIS_URL", "redis://localhost:6379/0"))
        prefix = kwargs.get("prefix", "urp")
        RETURN RedisBackend(url=url, prefix=prefix)
    ELIF effective_type == "sqlite":
        path = kwargs.get("path", os.getenv("URP_SQLITE_PATH", "/data/urp.db"))
        table = kwargs.get("table", "kv")
        RETURN SQLiteBackend(path=path, table=table)
    ELSE:
        RAISE ValueError(f"Unknown backend type: {effective_type}")
```

### URP Service Migration Pattern

```pseudocode
# Before (urp_registry/app/routers.py):
_DB: dict[str, Any] = {}

@router.post("/v1/registry/models")
async def register_model(body: dict, x_urp_admin: str = Header(None)):
    require_admin(x_urp_admin)
    _DB[body["model_id"]] = body
    return {"status": "registered"}

# After:
from app.storage import create_backend

_store = create_backend(prefix="registry")

@router.post("/v1/registry/models")
async def register_model(body: dict, x_urp_admin: str = Header(None)):
    require_admin(x_urp_admin)
    _store.set(body["model_id"], body)
    return {"status": "registered"}

@router.get("/v1/registry/models/{model_id}")
async def get_model(model_id: str):
    model = _store.get(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return model
```

## TDD Anchors

```pseudocode
# All tests run against all three backends via parametrize

@pytest.fixture(params=["memory", "sqlite"])
FUNCTION backend(request, tmp_path):
    IF request.param == "memory":
        RETURN MemoryBackend()
    ELIF request.param == "sqlite":
        RETURN SQLiteBackend(path=str(tmp_path / "test.db"))

TEST set_and_get(backend):
    backend.set("key1", {"name": "test"})
    ASSERT backend.get("key1") == {"name": "test"}

TEST get_missing_returns_none(backend):
    ASSERT backend.get("nonexistent") IS None

TEST delete_existing(backend):
    backend.set("key1", "value")
    ASSERT backend.delete("key1") IS True
    ASSERT backend.get("key1") IS None

TEST delete_missing_returns_false(backend):
    ASSERT backend.delete("nonexistent") IS False

TEST exists(backend):
    ASSERT backend.exists("key1") IS False
    backend.set("key1", "value")
    ASSERT backend.exists("key1") IS True

TEST list_keys_with_prefix(backend):
    backend.set("model:a", 1)
    backend.set("model:b", 2)
    backend.set("crown:x", 3)
    keys = backend.list_keys("model:")
    ASSERT sorted(keys) == ["model:a", "model:b"]

TEST count(backend):
    ASSERT backend.count() == 0
    backend.set("a", 1)
    backend.set("b", 2)
    ASSERT backend.count() == 2

TEST values_roundtrip_json(backend):
    """Complex nested values survive serialization."""
    value = {"nested": {"list": [1, 2, 3], "null": None, "bool": True}}
    backend.set("complex", value)
    ASSERT backend.get("complex") == value

TEST create_backend_factory:
    mem = create_backend("memory")
    ASSERT isinstance(mem, MemoryBackend)
    sqlite = create_backend("sqlite", path=":memory:")
    ASSERT isinstance(sqlite, SQLiteBackend)

TEST create_backend_env_override(monkeypatch):
    monkeypatch.setenv("URP_STORAGE_BACKEND", "sqlite")
    monkeypatch.setenv("URP_SQLITE_PATH", ":memory:")
    backend = create_backend("memory")  # arg says memory, env says sqlite
    ASSERT isinstance(backend, SQLiteBackend)

TEST multi_worker_simulation:
    """Two backends sharing same SQLite DB see each other's writes."""
    db_path = tmp_path / "shared.db"
    b1 = SQLiteBackend(path=str(db_path), table="kv")
    b2 = SQLiteBackend(path=str(db_path), table="kv")
    b1.set("key", "from_worker_1")
    ASSERT b2.get("key") == "from_worker_1"

TEST redis_backend_requires_redis:
    """RedisBackend gracefully fails if Redis unavailable."""
    pytest.importorskip("redis")
    # Use a port that's definitely not Redis
    WITH pytest.raises(Exception):
        backend = RedisBackend(url="redis://localhost:59999/0")
        backend.set("key", "value")  # Should fail on connection
```

## Acceptance Criteria

1. `StorageBackend` ABC with `MemoryBackend`, `RedisBackend`, `SQLiteBackend`
2. All 4 URP services migrated from `_DB` dict to `StorageBackend`
3. `URP_STORAGE_BACKEND` env var selects backend at runtime
4. Tests pass against memory and SQLite backends (Redis tested separately)
5. Multi-worker simulation test proves shared state works
6. No new required dependencies (redis is optional, sqlite3 is stdlib)
7. Full test suite GREEN

## Migration Path

1. Add `storage.py` to `_shared` (Phase 60)
2. Migrate URP services one at a time (Phase 60)
3. Default to `memory` for backward compat (Phase 60)
4. Switch production to `redis` via env var (Phase 61)
5. Add Redis health check to readiness probe (Phase 61)
