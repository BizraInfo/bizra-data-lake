"""
Latency guardrail for desktop bridge control-plane methods.

Enforces <200ms P95 for non-tool requests under local test conditions.
"""

from __future__ import annotations

import asyncio
import json
import socket
import time
import uuid
from typing import Any

import pytest

from core.bridges.desktop_bridge import DesktopBridge, TokenBucket


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _payload(method: str, req_id: int, params: Any = None) -> bytes:
    msg: dict[str, Any] = {
        "jsonrpc": "2.0",
        "method": method,
        "id": req_id,
        "headers": {
            "X-BIZRA-TOKEN": "test-bridge-token",
            "X-BIZRA-TS": int(time.time() * 1000),
            "X-BIZRA-NONCE": uuid.uuid4().hex,
        },
    }
    if params is not None:
        msg["params"] = params
    return json.dumps(msg).encode() + b"\n"


async def _call(port: int, payload: bytes) -> tuple[dict[str, Any], float]:
    start = time.perf_counter()
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    try:
        writer.write(payload)
        await writer.drain()
        line = await asyncio.wait_for(reader.readline(), timeout=5.0)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return json.loads(line), elapsed_ms
    finally:
        writer.close()
        await writer.wait_closed()


def _p95(values_ms: list[float]) -> float:
    ordered = sorted(values_ms)
    idx = max(0, int(len(ordered) * 0.95) - 1)
    return ordered[idx]


@pytest.fixture(autouse=True)
def _bridge_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BIZRA_BRIDGE_TOKEN", "test-bridge-token")
    monkeypatch.setenv("BIZRA_RECEIPT_PRIVATE_KEY_HEX", "11" * 32)
    monkeypatch.setenv("BIZRA_NODE_ROLE", "node")


@pytest.mark.asyncio
async def test_non_tool_control_plane_p95_under_200ms() -> None:
    port = _free_port()
    bridge = DesktopBridge(host="127.0.0.1", port=port)
    bridge._validate_fate = lambda _op: {"passed": True, "overall": 0.99}  # type: ignore[assignment]
    bridge._get_gateway = lambda: None  # type: ignore[assignment]
    bridge._get_rust_constitution = lambda: None  # type: ignore[assignment]
    bridge._get_skill_router = lambda: None  # type: ignore[assignment]
    bridge._rate_limiter = TokenBucket(rate=1000.0, burst=1000.0)

    await bridge.start()
    try:
        latencies: dict[str, list[float]] = {"ping": [], "status": [], "list_skills": []}
        req_id = 0
        for method in ("ping", "status", "list_skills"):
            for _ in range(30):
                req_id += 1
                response, elapsed_ms = await _call(port, _payload(method, req_id))
                assert "error" not in response, f"{method} returned error: {response}"
                latencies[method].append(elapsed_ms)

        for method, values in latencies.items():
            assert _p95(values) < 200.0, f"{method} P95 exceeded 200ms: {_p95(values):.2f}ms"
    finally:
        await bridge.stop()
