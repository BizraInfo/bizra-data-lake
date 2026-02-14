"""
Tests for the BIZRA Desktop Bridge (core/bridges/desktop_bridge.py).

Covers:
- Server lifecycle (start, accept, stop)
- ping returns valid response
- status returns expected fields
- sovereign_query routes through FATE gate (mock InferenceGateway)
- Rate limiter rejects burst > 20 req/s
- Invalid JSON-RPC returns error response
- Server rejects bind to non-localhost address
- HookPhase.DESKTOP_INVOKE fires on bridge commands
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.bridges.desktop_bridge import (
    BRIDGE_HOST,
    BRIDGE_PORT,
    DesktopBridge,
    TokenBucket,
    _error,
    _ok,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _auth_headers(
    token: str = "test-bridge-token",
    ts_ms: int | None = None,
    nonce: str | None = None,
) -> dict[str, Any]:
    return {
        "X-BIZRA-TOKEN": token,
        "X-BIZRA-TS": ts_ms if ts_ms is not None else int(time.time() * 1000),
        "X-BIZRA-NONCE": nonce or uuid.uuid4().hex,
    }


def _jsonrpc(
    method: str,
    params: Any = None,
    id: int = 1,
    headers: dict[str, Any] | None = None,
) -> bytes:
    """Build a newline-delimited JSON-RPC request."""
    msg: dict[str, Any] = {
        "jsonrpc": "2.0",
        "method": method,
        "id": id,
        "headers": headers or _auth_headers(),
    }
    if params is not None:
        msg["params"] = params
    return json.dumps(msg).encode() + b"\n"


async def _send_recv(
    host: str, port: int, payload: bytes, timeout: float = 5.0
) -> dict[str, Any]:
    """Open TCP, send payload, read one line response, return parsed JSON."""
    reader, writer = await asyncio.wait_for(
        asyncio.open_connection(host, port), timeout=timeout
    )
    try:
        writer.write(payload)
        await writer.drain()
        line = await asyncio.wait_for(reader.readline(), timeout=timeout)
        return json.loads(line)
    finally:
        writer.close()
        await writer.wait_closed()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def free_port() -> int:
    """Get a free port to avoid collisions during parallel test runs."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(autouse=True)
def _bridge_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BIZRA_BRIDGE_TOKEN", "test-bridge-token")
    monkeypatch.setenv("BIZRA_RECEIPT_PRIVATE_KEY_HEX", "11" * 32)
    monkeypatch.setenv("BIZRA_NODE_ROLE", "node")


@pytest.fixture
async def bridge(free_port: int):
    """Start a DesktopBridge on a free port, yield it, then stop."""
    b = DesktopBridge(host="127.0.0.1", port=free_port)
    await b.start()
    yield b
    await b.stop()


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


class TestServerLifecycle:
    @pytest.mark.asyncio
    async def test_start_and_stop(self, free_port: int) -> None:
        b = DesktopBridge(host="127.0.0.1", port=free_port)
        assert not b.is_running
        await b.start()
        assert b.is_running
        assert b.uptime_s >= 0.0
        await b.stop()
        assert not b.is_running

    @pytest.mark.asyncio
    async def test_accepts_connection(self, bridge: DesktopBridge, free_port: int) -> None:
        resp = await _send_recv("127.0.0.1", free_port, _jsonrpc("ping"))
        assert resp["result"]["status"] == "alive"

    @pytest.mark.asyncio
    async def test_start_fails_without_signing_key(
        self, free_port: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("BIZRA_BRIDGE_TOKEN", "test-bridge-token")
        monkeypatch.delenv("BIZRA_RECEIPT_PRIVATE_KEY_HEX", raising=False)
        b = DesktopBridge(host="127.0.0.1", port=free_port)
        with pytest.raises(RuntimeError, match="BIZRA_RECEIPT_PRIVATE_KEY_HEX"):
            await b.start()

    @pytest.mark.asyncio
    async def test_start_fails_without_bridge_token(
        self, free_port: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("BIZRA_BRIDGE_TOKEN", raising=False)
        monkeypatch.setenv("BIZRA_RECEIPT_PRIVATE_KEY_HEX", "11" * 32)
        b = DesktopBridge(host="127.0.0.1", port=free_port)
        with pytest.raises(RuntimeError, match="BIZRA_BRIDGE_TOKEN"):
            await b.start()

    @pytest.mark.asyncio
    async def test_start_fails_in_node0_mode_without_genesis(
        self, free_port: int, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        import core.bridges.desktop_bridge as bridge_mod

        monkeypatch.setenv("BIZRA_BRIDGE_TOKEN", "test-bridge-token")
        monkeypatch.setenv("BIZRA_RECEIPT_PRIVATE_KEY_HEX", "11" * 32)
        monkeypatch.setenv("BIZRA_NODE_ROLE", "node0")
        monkeypatch.setattr(bridge_mod, "GENESIS_STATE_DIR", tmp_path / "missing_genesis")
        b = DesktopBridge(host="127.0.0.1", port=free_port)
        with pytest.raises(RuntimeError, match="Node0 genesis enforcement failed"):
            await b.start()


# ---------------------------------------------------------------------------
# ping
# ---------------------------------------------------------------------------


class TestPing:
    @pytest.mark.asyncio
    async def test_ping_returns_alive(self, bridge: DesktopBridge, free_port: int) -> None:
        resp = await _send_recv("127.0.0.1", free_port, _jsonrpc("ping"))
        assert "result" in resp
        result = resp["result"]
        assert result["status"] == "alive"
        assert "uptime_s" in result
        assert isinstance(result["uptime_s"], (int, float))
        assert "rust_available" in result
        assert isinstance(result["rust_available"], bool)
        assert "receipt" in result
        assert "receipt_id" in result["receipt"]

    @pytest.mark.asyncio
    async def test_ping_increments_request_count(
        self, bridge: DesktopBridge, free_port: int
    ) -> None:
        await _send_recv("127.0.0.1", free_port, _jsonrpc("ping", id=1))
        resp = await _send_recv("127.0.0.1", free_port, _jsonrpc("ping", id=2))
        assert resp["result"]["request_count"] >= 2


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


class TestStatus:
    @pytest.mark.asyncio
    async def test_status_returns_expected_fields(
        self, bridge: DesktopBridge, free_port: int
    ) -> None:
        resp = await _send_recv("127.0.0.1", free_port, _jsonrpc("status"))
        result = resp["result"]
        assert result["node"] == "node"
        assert "origin" in result
        assert result["origin"]["designation"] == "ephemeral_node"
        assert isinstance(result["origin"]["genesis_node"], bool)
        assert isinstance(result["origin"]["genesis_block"], bool)
        assert isinstance(result["origin"]["home_base_device"], bool)
        assert result["origin"]["authority_source"] == "genesis_files"
        assert isinstance(result["origin"]["hash_validated"], bool)
        assert "bridge_uptime_s" in result
        assert "fate_gate" in result
        assert "inference" in result
        assert "rust" in result
        assert isinstance(result["rust"]["available"], bool)
        assert "receipt" in result
        assert result["receipt"]["status"] in {"accepted", "rejected"}


# ---------------------------------------------------------------------------
# sovereign_query
# ---------------------------------------------------------------------------


@dataclass
class _MockInferenceResult:
    content: str = "mock answer"
    model: str = "mock-model"
    backend: str = "mock"
    tokens_generated: int = 5
    tokens_per_second: float = 100.0
    latency_ms: float = 10.0


class TestSovereignQuery:
    @pytest.mark.asyncio
    async def test_missing_query_returns_error(
        self, bridge: DesktopBridge, free_port: int
    ) -> None:
        resp = await _send_recv(
            "127.0.0.1", free_port, _jsonrpc("sovereign_query", {"no_query": True})
        )
        assert "error" in resp

    @pytest.mark.asyncio
    async def test_empty_query_returns_error(
        self, bridge: DesktopBridge, free_port: int
    ) -> None:
        resp = await _send_recv(
            "127.0.0.1", free_port, _jsonrpc("sovereign_query", {"query": "  "})
        )
        assert "error" in resp

    @pytest.mark.asyncio
    async def test_query_with_mock_gateway(self, free_port: int) -> None:
        mock_gw = AsyncMock()
        mock_gw.infer = AsyncMock(return_value=_MockInferenceResult())
        mock_gw.health = AsyncMock(return_value={"status": "ready"})

        b = DesktopBridge(host="127.0.0.1", port=free_port, gateway=mock_gw)
        await b.start()
        try:
            resp = await _send_recv(
                "127.0.0.1",
                free_port,
                _jsonrpc("sovereign_query", {"query": "What is sovereignty?"}),
            )
            result = resp["result"]
            # If FATE gate passes, we get content; if not, we get fate info
            if "content" in result:
                assert result["content"] == "mock answer"
                assert result["model"] == "mock-model"
                assert "latency_ms" in result
                assert "fate" in result
            else:
                # FATE blocked — still valid behavior
                assert "fate" in result
        finally:
            await b.stop()

    @pytest.mark.asyncio
    async def test_query_no_gateway_returns_error(
        self, free_port: int
    ) -> None:
        b = DesktopBridge(host="127.0.0.1", port=free_port, gateway=None)
        # Patch _get_gateway to return None (simulate no gateway available)
        b._get_gateway = lambda: None  # type: ignore[assignment]
        await b.start()
        try:
            resp = await _send_recv(
                "127.0.0.1",
                free_port,
                _jsonrpc("sovereign_query", {"query": "test"}),
            )
            result = resp["result"]
            # Either FATE blocks it, or gateway unavailable
            assert "error" in result or "fate" in result
        finally:
            await b.stop()


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------


class TestRateLimiter:
    def test_token_bucket_allows_burst(self) -> None:
        bucket = TokenBucket(rate=20.0, burst=30.0)
        allowed = sum(1 for _ in range(30) if bucket.allow())
        assert allowed == 30

    def test_token_bucket_rejects_over_burst(self) -> None:
        bucket = TokenBucket(rate=20.0, burst=20.0)
        # Drain all tokens
        for _ in range(20):
            bucket.allow()
        # Next should be rejected
        assert not bucket.allow()

    @pytest.mark.asyncio
    async def test_rate_limit_over_tcp(self, free_port: int) -> None:
        # Bridge with very low rate limit for testing
        b = DesktopBridge(host="127.0.0.1", port=free_port)
        b._rate_limiter = TokenBucket(rate=2.0, burst=3.0)
        await b.start()
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", free_port)
            # Send 5 rapid pings; first 3 should pass, rest should be rate-limited
            results = []
            for i in range(5):
                writer.write(_jsonrpc("ping", id=i))
                await writer.drain()
                line = await asyncio.wait_for(reader.readline(), timeout=5.0)
                results.append(json.loads(line))

            rate_limited = [r for r in results if "error" in r]
            assert len(rate_limited) >= 1, "Expected at least one rate-limited response"

            writer.close()
            await writer.wait_closed()
        finally:
            await b.stop()


# ---------------------------------------------------------------------------
# Invalid JSON-RPC
# ---------------------------------------------------------------------------


class TestInvalidRequests:
    @pytest.mark.asyncio
    async def test_malformed_json(self, bridge: DesktopBridge, free_port: int) -> None:
        resp = await _send_recv("127.0.0.1", free_port, b"not json\n")
        assert resp["error"]["code"] == -32700

    @pytest.mark.asyncio
    async def test_missing_jsonrpc_version(
        self, bridge: DesktopBridge, free_port: int
    ) -> None:
        payload = json.dumps({"method": "ping", "id": 1}).encode() + b"\n"
        resp = await _send_recv("127.0.0.1", free_port, payload)
        assert resp["error"]["code"] == -32600

    @pytest.mark.asyncio
    async def test_unknown_method(self, bridge: DesktopBridge, free_port: int) -> None:
        resp = await _send_recv(
            "127.0.0.1", free_port, _jsonrpc("nonexistent_method")
        )
        assert resp["error"]["code"] == -32601
        assert "not found" in resp["error"]["message"].lower()

    @pytest.mark.asyncio
    async def test_missing_method(self, bridge: DesktopBridge, free_port: int) -> None:
        payload = json.dumps({"jsonrpc": "2.0", "id": 1}).encode() + b"\n"
        resp = await _send_recv("127.0.0.1", free_port, payload)
        assert resp["error"]["code"] == -32600


# ---------------------------------------------------------------------------
# Security: localhost-only binding
# ---------------------------------------------------------------------------


class TestSecurity:
    def test_rejects_non_localhost_bind(self) -> None:
        with pytest.raises(ValueError, match="127.0.0.1"):
            DesktopBridge(host="0.0.0.0", port=9742)

    def test_rejects_external_address(self) -> None:
        with pytest.raises(ValueError, match="127.0.0.1"):
            DesktopBridge(host="192.168.1.100", port=9742)


# ---------------------------------------------------------------------------
# Security: auth envelope
# ---------------------------------------------------------------------------


class TestAuthEnvelope:
    @pytest.mark.asyncio
    async def test_missing_headers_rejected(self, bridge: DesktopBridge, free_port: int) -> None:
        payload = json.dumps({"jsonrpc": "2.0", "method": "ping", "id": 1}).encode() + b"\n"
        resp = await _send_recv("127.0.0.1", free_port, payload)
        assert resp["error"]["code"] == -32001
        assert resp["error"]["data"]["code"] == "AUTH_MISSING_HEADERS"
        assert "receipt" in resp["error"]["data"]

    @pytest.mark.asyncio
    async def test_wrong_token_rejected_all_methods(self, bridge: DesktopBridge, free_port: int) -> None:
        bad_headers = _auth_headers(token="wrong-token")
        methods = [
            ("ping", None),
            ("status", None),
            ("sovereign_query", {"query": "test"}),
            ("invoke_skill", {"skill": "test"}),
            ("list_skills", None),
            ("get_receipt", {"receipt_id": "missing"}),
        ]
        for i, (method, params) in enumerate(methods, start=1):
            resp = await _send_recv(
                "127.0.0.1",
                free_port,
                _jsonrpc(method, params=params, id=i, headers=bad_headers),
            )
            assert resp["error"]["code"] == -32002
            assert resp["error"]["data"]["code"] == "AUTH_INVALID_TOKEN"

    @pytest.mark.asyncio
    async def test_stale_timestamp_rejected(self, bridge: DesktopBridge, free_port: int) -> None:
        stale_headers = _auth_headers(ts_ms=946684800000)  # 2000-01-01 UTC
        resp = await _send_recv("127.0.0.1", free_port, _jsonrpc("ping", headers=stale_headers))
        assert resp["error"]["code"] == -32003
        assert resp["error"]["data"]["code"] == "AUTH_STALE_TIMESTAMP"

    @pytest.mark.asyncio
    async def test_nonce_replay_rejected(self, bridge: DesktopBridge, free_port: int) -> None:
        nonce = "fixed-nonce-for-replay-test"
        headers_1 = _auth_headers(nonce=nonce)
        headers_2 = _auth_headers(nonce=nonce)
        first = await _send_recv("127.0.0.1", free_port, _jsonrpc("ping", headers=headers_1))
        second = await _send_recv("127.0.0.1", free_port, _jsonrpc("ping", headers=headers_2))
        assert "result" in first
        assert second["error"]["code"] == -32004
        assert second["error"]["data"]["code"] == "AUTH_NONCE_REPLAY"

    @pytest.mark.asyncio
    async def test_unauthorized_request_does_not_increment_counter(
        self, bridge: DesktopBridge, free_port: int
    ) -> None:
        before = bridge._request_count
        bad_headers = _auth_headers(token="wrong-token")
        await _send_recv("127.0.0.1", free_port, _jsonrpc("ping", headers=bad_headers))
        assert bridge._request_count == before


# ---------------------------------------------------------------------------
# HookPhase.DESKTOP_INVOKE
# ---------------------------------------------------------------------------


class TestDesktopInvokeHook:
    @pytest.mark.asyncio
    async def test_desktop_invoke_phase_exists(self) -> None:
        from core.elite.hooks import HookPhase

        assert hasattr(HookPhase, "DESKTOP_INVOKE")
        assert HookPhase.DESKTOP_INVOKE.value == "desktop_invoke"

    @pytest.mark.asyncio
    async def test_hook_fires_on_ping(self, free_port: int) -> None:
        from core.elite.hooks import HookPhase, HookPriority, _get_global_registry

        hook_called = False
        hook_method = None

        def _test_hook(data: dict) -> dict:
            nonlocal hook_called, hook_method
            hook_called = True
            hook_method = data["metadata"].get("method")
            return data

        registry = _get_global_registry()
        registry.register(
            name="_test_desktop_hook",
            phase=HookPhase.DESKTOP_INVOKE,
            function=_test_hook,
            priority=HookPriority.NORMAL,
        )

        try:
            b = DesktopBridge(host="127.0.0.1", port=free_port)
            await b.start()
            try:
                await _send_recv("127.0.0.1", free_port, _jsonrpc("ping"))
                assert hook_called, "DESKTOP_INVOKE hook was not fired"
                assert hook_method == "ping"
            finally:
                await b.stop()
        finally:
            registry.unregister("_test_desktop_hook")


# ---------------------------------------------------------------------------
# JSON-RPC helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_ok_format(self) -> None:
        raw = _ok(1, {"a": "b"})
        parsed = json.loads(raw)
        assert parsed["jsonrpc"] == "2.0"
        assert parsed["id"] == 1
        assert parsed["result"] == {"a": "b"}
        assert raw.endswith(b"\n")

    def test_error_format(self) -> None:
        raw = _error(2, -32600, "bad request")
        parsed = json.loads(raw)
        assert parsed["jsonrpc"] == "2.0"
        assert parsed["id"] == 2
        assert parsed["error"]["code"] == -32600
        assert parsed["error"]["message"] == "bad request"
        assert raw.endswith(b"\n")
