from __future__ import annotations

import os
import socket
import time
from typing import Callable

import httpx

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    redis = None

_START_TS = time.time()


HealthCheck = Callable[[], tuple[bool, str]]


def check_redis(redis_url_env: str = "REDIS_URL") -> HealthCheck:
    def _check() -> tuple[bool, str]:
        redis_url = (os.environ.get(redis_url_env) or "").strip()
        if not redis_url:
            return True, "skipped: REDIS_URL unset"
        if redis is None:
            return False, "redis dependency unavailable"
        try:
            client = redis.Redis.from_url(
                redis_url,
                decode_responses=True,
                socket_connect_timeout=1.5,
                socket_timeout=1.5,
            )
            if client.ping():
                return True, "ok"
            return False, "ping failed"
        except Exception as exc:
            return False, f"{type(exc).__name__}"

    return _check


def check_http(url_env: str, default: str) -> HealthCheck:
    def _check() -> tuple[bool, str]:
        base = (os.environ.get(url_env) or default).rstrip("/")
        try:
            response = httpx.get(f"{base}/health", timeout=2.0)
            ok = response.status_code == 200 and bool(response.json().get("ok"))
            return ok, f"http {response.status_code}"
        except Exception as exc:
            return False, f"{type(exc).__name__}"

    return _check


def build_health_payload(
    *,
    service: str,
    version: str,
    checks: dict[str, HealthCheck] | None = None,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    checks = checks or {}
    deps: dict[str, dict[str, object]] = {}
    overall_ok = True

    for name, check in checks.items():
        try:
            ok, detail = check()
        except Exception as exc:  # pragma: no cover - defensive
            ok, detail = False, f"{type(exc).__name__}"
        deps[name] = {"ok": bool(ok), "detail": detail}
        overall_ok = overall_ok and bool(ok)

    payload: dict[str, object] = {
        "ok": overall_ok,
        "service": service,
        "version": version,
        "hostname": socket.gethostname(),
        "uptime_s": round(time.time() - _START_TS, 3),
        "deps": deps,
    }
    if extra:
        payload.update(extra)
    return payload
