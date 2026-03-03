from __future__ import annotations

import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Optional

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    redis = None

DEFAULT_DB_PATH = "/tmp/reflex_cache.sqlite"
REDIS_HASH_KEY = "bizra:reflex_cache:v1"


def _resolve_db_path() -> Path:
    configured = (os.environ.get("REFLEX_CACHE_DB_PATH") or DEFAULT_DB_PATH).strip()
    return Path(configured)


class ReflexCache:
    """O(1) reflex lookup + persistence.

    Primary backend: Redis hash for multi-replica/shared state.
    Fallback backend: sqlite file for single-node/local durability.
    """

    def __init__(self) -> None:
        self._mem: dict[str, list[str]] = {}
        self._lock = threading.RLock()
        self._redis = self._connect_redis()
        self._conn = self._connect_sqlite()
        self._load()

    def backend(self) -> str:
        if self._redis is not None:
            return "redis"
        if self._conn is not None:
            return "sqlite"
        return "memory"

    def _connect_redis(self):
        redis_url = (os.environ.get("REDIS_URL") or "").strip()
        if not redis_url or redis is None:
            return None
        try:
            client = redis.Redis.from_url(
                redis_url,
                decode_responses=True,
                socket_connect_timeout=1.5,
                socket_timeout=1.5,
            )
            client.ping()
            return client
        except Exception:
            return None

    def _connect_sqlite(self):
        db_path = _resolve_db_path()
        try:
            db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(db_path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS reflexes (macro_state TEXT PRIMARY KEY, steps_json TEXT NOT NULL)"
            )
            conn.commit()
            return conn
        except Exception:
            return None

    def _load(self) -> None:
        if self._redis is not None:
            try:
                rows = self._redis.hgetall(REDIS_HASH_KEY)
                for macro, steps_json in rows.items():
                    try:
                        parsed = json.loads(steps_json)
                        if isinstance(parsed, list):
                            self._mem[macro] = [str(x) for x in parsed]
                    except Exception:
                        continue
                return
            except Exception:
                self._redis = None

        if self._conn is not None:
            try:
                cur = self._conn.execute("SELECT macro_state, steps_json FROM reflexes")
                for macro, steps_json in cur.fetchall():
                    try:
                        parsed = json.loads(steps_json)
                        if isinstance(parsed, list):
                            self._mem[macro] = [str(x) for x in parsed]
                    except Exception:
                        continue
            except Exception:
                pass

    def get(self, macro_state: str) -> Optional[list[str]]:
        with self._lock:
            if self._redis is not None:
                try:
                    raw = self._redis.hget(REDIS_HASH_KEY, macro_state)
                    if raw:
                        parsed = json.loads(raw)
                        if isinstance(parsed, list):
                            self._mem[macro_state] = [str(x) for x in parsed]
                except Exception:
                    self._redis = None
            value = self._mem.get(macro_state)
            return list(value) if isinstance(value, list) else None

    def put(self, macro_state: str, steps: list[str]) -> None:
        normalized = [str(step) for step in steps]
        with self._lock:
            self._mem[macro_state] = normalized

            if self._redis is not None:
                try:
                    self._redis.hset(
                        REDIS_HASH_KEY,
                        macro_state,
                        json.dumps(normalized, separators=(",", ":")),
                    )
                except Exception:
                    self._redis = None

            if self._conn is not None:
                try:
                    self._conn.execute(
                        "INSERT INTO reflexes(macro_state, steps_json) VALUES(?, ?) ON CONFLICT(macro_state) DO UPDATE SET steps_json=excluded.steps_json",
                        (macro_state, json.dumps(normalized)),
                    )
                    self._conn.commit()
                except Exception:
                    pass

    def count(self) -> int:
        with self._lock:
            if self._redis is not None:
                try:
                    return int(self._redis.hlen(REDIS_HASH_KEY))
                except Exception:
                    self._redis = None
            return len(self._mem)
