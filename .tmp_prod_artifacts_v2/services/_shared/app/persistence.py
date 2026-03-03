from __future__ import annotations

import json
import os
import threading

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    redis = None


def _connect_redis(redis_url_env: str = "REDIS_URL"):
    redis_url = (os.environ.get(redis_url_env) or "").strip()
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


class JsonHashStore:
    """Hash-based JSON store with Redis primary and in-memory fallback."""

    def __init__(self, namespace: str) -> None:
        self._namespace = namespace
        self._mem: dict[str, dict[str, object]] = {}
        self._lock = threading.RLock()
        self._redis = _connect_redis()
        if self._redis is not None:
            self._refresh_from_redis()

    def backend(self) -> str:
        return "redis" if self._redis is not None else "memory"

    def _refresh_from_redis(self) -> None:
        if self._redis is None:
            return
        try:
            rows = self._redis.hgetall(self._namespace)
        except Exception:
            self._redis = None
            return
        parsed: dict[str, dict[str, object]] = {}
        for key, raw_value in rows.items():
            try:
                value = json.loads(raw_value)
                if isinstance(value, dict):
                    parsed[str(key)] = value
            except Exception:
                continue
        self._mem = parsed

    def values(self) -> list[dict[str, object]]:
        with self._lock:
            if self._redis is not None:
                self._refresh_from_redis()
            return list(self._mem.values())

    def upsert(self, key: str, value: dict[str, object]) -> None:
        with self._lock:
            self._mem[key] = value
            if self._redis is None:
                return
            try:
                self._redis.hset(
                    self._namespace,
                    key,
                    json.dumps(value, separators=(",", ":"), sort_keys=True),
                )
            except Exception:
                self._redis = None

    def count(self) -> int:
        return len(self.values())


class JsonListStore:
    """Append-only JSON list with Redis primary and in-memory fallback."""

    def __init__(self, namespace: str) -> None:
        self._namespace = namespace
        self._mem: list[dict[str, object]] = []
        self._lock = threading.RLock()
        self._redis = _connect_redis()
        if self._redis is not None:
            self._refresh_from_redis()

    def backend(self) -> str:
        return "redis" if self._redis is not None else "memory"

    def _refresh_from_redis(self) -> None:
        if self._redis is None:
            return
        try:
            raw_items = self._redis.lrange(self._namespace, 0, -1)
        except Exception:
            self._redis = None
            return
        parsed: list[dict[str, object]] = []
        for raw_item in raw_items:
            try:
                item = json.loads(raw_item)
                if isinstance(item, dict):
                    parsed.append(item)
            except Exception:
                continue
        self._mem = parsed

    def append(self, value: dict[str, object]) -> None:
        with self._lock:
            self._mem.append(value)
            if self._redis is None:
                return
            try:
                self._redis.rpush(
                    self._namespace,
                    json.dumps(value, separators=(",", ":"), sort_keys=True),
                )
            except Exception:
                self._redis = None

    def values(self) -> list[dict[str, object]]:
        with self._lock:
            if self._redis is not None:
                self._refresh_from_redis()
            return list(self._mem)

    def count(self) -> int:
        return len(self.values())
