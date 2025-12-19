from __future__ import annotations

import json
import os
import socket
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from core.fate import FateEngine
from core.model_family import load_model_family


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw.strip())
    except Exception:
        return float(default)


def _sanitize_url(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        return ""
    try:
        p = urlparse(raw)
        host = p.hostname or ""
        port = f":{p.port}" if p.port else ""
        scheme = p.scheme or ""
        return f"{scheme}://{host}{port}"
    except Exception:
        return raw


def _parse_neo4j_auth(raw: str) -> Optional[tuple[str, str]]:
    raw = (raw or "").strip()
    if not raw or raw.lower() == "none":
        return None
    if "/" in raw:
        user, password = raw.split("/", 1)
    elif ":" in raw:
        user, password = raw.split(":", 1)
    else:
        return None
    user = user.strip()
    password = password.strip()
    if not user or not password:
        return None
    return user, password


def _redis_ping(url: str, timeout_s: float) -> None:
    p = urlparse(url)
    host = p.hostname or "127.0.0.1"
    port = int(p.port or 6379)
    with socket.create_connection((host, port), timeout=timeout_s) as sock:
        sock.settimeout(timeout_s)
        sock.sendall(b"*1\r\n$4\r\nPING\r\n")
        data = sock.recv(1024)
    if b"PONG" not in data:
        raise RuntimeError(f"unexpected_redis_response: {data[:64]!r}")


def _neo4j_verify(*, uri: str, auth: tuple[str, str], timeout_s: float) -> None:
    from neo4j import GraphDatabase  # type: ignore

    try:
        driver = GraphDatabase.driver(uri, auth=auth, connection_timeout=timeout_s)
    except TypeError:
        driver = GraphDatabase.driver(uri, auth=auth)
    try:
        driver.verify_connectivity()
    finally:
        driver.close()


def main() -> int:
    interval_s = max(1.0, _env_float("FATE_AUDITOR_INTERVAL_S", 30.0))
    redis_timeout_s = max(0.05, min(_env_float("FATE_AUDITOR_REDIS_TIMEOUT_S", 0.35), 2.0))
    neo4j_timeout_s = max(0.05, min(_env_float("FATE_AUDITOR_NEO4J_TIMEOUT_S", 0.55), 3.0))

    strict_raw = (os.getenv("BIZRA_FATE_STRICT") or os.getenv("FATE_LEVEL") or "1").strip()
    strict_mode = strict_raw != "0" and strict_raw.lower() not in {"lenient", "off"}
    fate = FateEngine(strict_mode=strict_mode, validator="Node0_FateAuditor")

    while True:
        report: Dict[str, Any] = {"time": utc_now_iso()}

        report["fate"] = {
            "strict_mode": strict_mode,
            "policy_loaded": fate.policy.loaded,
            "constitution_sha256": fate.policy.constitution_sha256,
        }

        try:
            mf = load_model_family()
            report["model_family"] = {
                "ok": True,
                "sealed": mf.sealed,
                "sealed_at_utc": mf.sealed_at_utc,
                "manifest_path": str(mf.path),
                "slots": sorted(mf.capability_slots.keys()),
            }
        except Exception as e:
            report["model_family"] = {"ok": False, "error": str(e)}

        synapse_url = os.getenv("SYNAPSE_URL", "").strip()
        if synapse_url:
            t0 = time.monotonic()
            try:
                _redis_ping(synapse_url, redis_timeout_s)
                report["synapse"] = {
                    "enabled": True,
                    "ok": True,
                    "url": _sanitize_url(synapse_url),
                    "latency_ms": round((time.monotonic() - t0) * 1000.0, 2),
                }
            except Exception as e:
                report["synapse"] = {
                    "enabled": True,
                    "ok": False,
                    "url": _sanitize_url(synapse_url),
                    "latency_ms": round((time.monotonic() - t0) * 1000.0, 2),
                    "error": str(e),
                }
        else:
            report["synapse"] = {"enabled": False}

        wisdom_url = os.getenv("WISDOM_URL", "").strip()
        neo4j_auth_raw = os.getenv("NEO4J_AUTH", "").strip()
        auth = _parse_neo4j_auth(neo4j_auth_raw) if (wisdom_url and neo4j_auth_raw) else None

        if wisdom_url and auth:
            t0 = time.monotonic()
            try:
                _neo4j_verify(uri=wisdom_url, auth=auth, timeout_s=neo4j_timeout_s)
                report["wisdom"] = {
                    "enabled": True,
                    "ok": True,
                    "url": _sanitize_url(wisdom_url),
                    "latency_ms": round((time.monotonic() - t0) * 1000.0, 2),
                }
            except Exception as e:
                report["wisdom"] = {
                    "enabled": True,
                    "ok": False,
                    "url": _sanitize_url(wisdom_url),
                    "latency_ms": round((time.monotonic() - t0) * 1000.0, 2),
                    "error": str(e),
                }
        elif wisdom_url:
            report["wisdom"] = {"enabled": True, "ok": False, "url": _sanitize_url(wisdom_url), "error": "invalid NEO4J_AUTH"}
        else:
            report["wisdom"] = {"enabled": False}

        print(json.dumps(report, ensure_ascii=False), flush=True)
        time.sleep(interval_s)


if __name__ == "__main__":
    raise SystemExit(main())

