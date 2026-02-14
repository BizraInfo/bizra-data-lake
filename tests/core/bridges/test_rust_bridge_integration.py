"""
Tests for Rust PyO3 integration in the BIZRA Desktop Bridge.

Verifies:
- Bridge detects Rust module availability (_RUST_AVAILABLE flag)
- GateChain verification runs when bizra module is present
- Constitution thresholds reported in status
- BLAKE3 domain_separated_digest used for content hashing
- Graceful fallback when Rust bindings are not compiled
- Rust gate chain blocks queries when scores below threshold
"""

from __future__ import annotations

import asyncio
import json
import socket
import time
import uuid
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from core.bridges.desktop_bridge import (
    BRIDGE_HOST,
    DesktopBridge,
    _RUST_AVAILABLE,
    _ok,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _jsonrpc(method: str, params: Any = None, id: int = 1) -> bytes:
    msg: dict[str, Any] = {
        "jsonrpc": "2.0",
        "method": method,
        "id": id,
        "headers": {
            "X-BIZRA-TOKEN": "test-bridge-token",
            "X-BIZRA-TS": int(time.time() * 1000),
            "X-BIZRA-NONCE": uuid.uuid4().hex,
        },
    }
    if params is not None:
        msg["params"] = params
    return json.dumps(msg).encode() + b"\n"


async def _tcp_call(port: int, method: str, params: Any = None) -> dict[str, Any]:
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(_jsonrpc(method, params))
    await writer.drain()
    line = await asyncio.wait_for(reader.readline(), timeout=5.0)
    writer.close()
    await writer.wait_closed()
    return json.loads(line)


@pytest.fixture(autouse=True)
def _bridge_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BIZRA_BRIDGE_TOKEN", "test-bridge-token")
    monkeypatch.setenv("BIZRA_RECEIPT_PRIVATE_KEY_HEX", "11" * 32)
    monkeypatch.setenv("BIZRA_NODE_ROLE", "node")


# ---------------------------------------------------------------------------
# Rust detection
# ---------------------------------------------------------------------------


class TestRustDetection:
    def test_rust_available_is_boolean(self) -> None:
        """_RUST_AVAILABLE is always a bool."""
        assert isinstance(_RUST_AVAILABLE, bool)

    @pytest.mark.asyncio
    async def test_ping_reports_rust_available(self) -> None:
        """ping response always includes rust_available flag."""
        port = _free_port()
        bridge = DesktopBridge(host="127.0.0.1", port=port)
        await bridge.start()
        try:
            resp = await _tcp_call(port, "ping")
            result = resp["result"]
            assert "rust_available" in result
            assert result["rust_available"] == _RUST_AVAILABLE
        finally:
            await bridge.stop()

    @pytest.mark.asyncio
    async def test_status_reports_rust_info(self) -> None:
        """status response includes rust section with availability."""
        port = _free_port()
        bridge = DesktopBridge(host="127.0.0.1", port=port)
        await bridge.start()
        try:
            resp = await _tcp_call(port, "status")
            result = resp["result"]
            assert "rust" in result
            assert result["rust"]["available"] == _RUST_AVAILABLE
        finally:
            await bridge.stop()


# ---------------------------------------------------------------------------
# Rust gate chain (mocked — tests work regardless of Rust compilation)
# ---------------------------------------------------------------------------


class TestRustGateChain:
    def test_validate_rust_gates_empty_when_unavailable(self) -> None:
        """_validate_rust_gates returns {} when Rust is not available."""
        bridge = DesktopBridge(host="127.0.0.1", port=9742)
        with patch("core.bridges.desktop_bridge._RUST_AVAILABLE", False):
            result = bridge._validate_rust_gates("test content")
            assert result == {}

    def test_validate_rust_gates_with_mock_chain(self) -> None:
        """_validate_rust_gates returns gate results with mocked GateChain."""
        bridge = DesktopBridge(host="127.0.0.1", port=9742)
        mock_chain = MagicMock()
        mock_chain.verify.return_value = [
            ("schema", True, "PASS"),
            ("ihsan", True, "PASS"),
            ("snr", True, "PASS"),
        ]
        bridge._rust_gate_chain = mock_chain

        with patch("core.bridges.desktop_bridge._RUST_AVAILABLE", True):
            result = bridge._validate_rust_gates("test content", 0.96, 0.97)

        assert result["passed"] is True
        assert result["engine"] == "rust"
        assert "schema" in result["gates"]
        assert "ihsan" in result["gates"]
        assert "snr" in result["gates"]
        mock_chain.verify.assert_called_once_with("test content", 0.96, 0.97)

    def test_validate_rust_gates_detects_failure(self) -> None:
        """_validate_rust_gates reports failed gates."""
        bridge = DesktopBridge(host="127.0.0.1", port=9742)
        mock_chain = MagicMock()
        mock_chain.verify.return_value = [
            ("schema", True, "PASS"),
            ("ihsan", False, "BELOW_THRESHOLD"),
            ("snr", True, "PASS"),
        ]
        bridge._rust_gate_chain = mock_chain

        with patch("core.bridges.desktop_bridge._RUST_AVAILABLE", True):
            result = bridge._validate_rust_gates("test", 0.90, 0.80)

        assert result["passed"] is False
        assert result["gates"]["ihsan"]["passed"] is False
        assert result["gates"]["ihsan"]["code"] == "BELOW_THRESHOLD"

    def test_validate_rust_gates_handles_exception(self) -> None:
        """_validate_rust_gates returns {} on GateChain exception."""
        bridge = DesktopBridge(host="127.0.0.1", port=9742)
        mock_chain = MagicMock()
        mock_chain.verify.side_effect = RuntimeError("Rust panic")
        bridge._rust_gate_chain = mock_chain

        with patch("core.bridges.desktop_bridge._RUST_AVAILABLE", True):
            result = bridge._validate_rust_gates("test")

        assert result == {}


# ---------------------------------------------------------------------------
# BLAKE3 digest
# ---------------------------------------------------------------------------


class TestBLAKE3Digest:
    def test_blake3_returns_none_when_unavailable(self) -> None:
        """_blake3_digest returns None when Rust is not available."""
        bridge = DesktopBridge(host="127.0.0.1", port=9742)
        with patch("core.bridges.desktop_bridge._RUST_AVAILABLE", False):
            assert bridge._blake3_digest("test") is None

    def test_blake3_with_mock_digest(self) -> None:
        """_blake3_digest calls domain_separated_digest correctly."""
        bridge = DesktopBridge(host="127.0.0.1", port=9742)
        mock_digest = MagicMock(return_value="abcdef1234567890")

        with (
            patch("core.bridges.desktop_bridge._RUST_AVAILABLE", True),
            patch("core.bridges.desktop_bridge.domain_separated_digest", mock_digest, create=True),
        ):
            result = bridge._blake3_digest("hello world")

        assert result == "abcdef1234567890"
        mock_digest.assert_called_once_with(b"hello world")

    def test_blake3_handles_exception(self) -> None:
        """_blake3_digest returns None on exception."""
        bridge = DesktopBridge(host="127.0.0.1", port=9742)
        mock_digest = MagicMock(side_effect=RuntimeError("FFI error"))

        with (
            patch("core.bridges.desktop_bridge._RUST_AVAILABLE", True),
            patch("core.bridges.desktop_bridge.domain_separated_digest", mock_digest, create=True),
        ):
            assert bridge._blake3_digest("test") is None


# ---------------------------------------------------------------------------
# Constitution
# ---------------------------------------------------------------------------


class TestRustConstitution:
    def test_constitution_none_when_unavailable(self) -> None:
        """_get_rust_constitution returns None when Rust is not available."""
        bridge = DesktopBridge(host="127.0.0.1", port=9742)
        with patch("core.bridges.desktop_bridge._RUST_AVAILABLE", False):
            assert bridge._get_rust_constitution() is None

    @pytest.mark.asyncio
    async def test_status_includes_constitution_thresholds(self) -> None:
        """status includes constitution version and thresholds when Rust is available."""
        port = _free_port()
        bridge = DesktopBridge(host="127.0.0.1", port=port)

        mock_const = MagicMock()
        mock_const.version = "1.0.0"
        mock_const.ihsan_threshold = 0.95
        mock_const.snr_threshold = 0.85
        bridge._rust_constitution = mock_const

        await bridge.start()
        try:
            with patch("core.bridges.desktop_bridge._RUST_AVAILABLE", True):
                resp = await _tcp_call(port, "status")
                rust = resp["result"]["rust"]
                assert rust["available"] is True
                assert rust["constitution_version"] == "1.0.0"
                assert rust["ihsan_threshold"] == 0.95
                assert rust["snr_threshold"] == 0.85
        finally:
            await bridge.stop()


# ---------------------------------------------------------------------------
# Full sovereign_query with Rust gates + BLAKE3
# ---------------------------------------------------------------------------


@dataclass
class _MockInferenceResult:
    content: str = "sovereign answer"
    model: str = "test-model"
    backend: str = "mock"
    tokens_generated: int = 10
    tokens_per_second: float = 100.0
    latency_ms: float = 15.0


class TestSovereignQueryWithRust:
    @pytest.mark.asyncio
    async def test_query_includes_rust_gates_and_hash(self) -> None:
        """sovereign_query response includes rust_gates and content_hash when mocked."""
        from unittest.mock import AsyncMock

        port = _free_port()
        mock_gw = AsyncMock()
        mock_gw.infer = AsyncMock(return_value=_MockInferenceResult())
        mock_gw.health = AsyncMock(return_value={"status": "ready"})

        bridge = DesktopBridge(host="127.0.0.1", port=port, gateway=mock_gw)

        # Mock Rust gate chain
        mock_chain = MagicMock()
        mock_chain.verify.return_value = [
            ("schema", True, "PASS"),
            ("ihsan", True, "PASS"),
            ("snr", True, "PASS"),
        ]
        bridge._rust_gate_chain = mock_chain

        # Mock FATE gate to pass (so Rust gates get exercised)
        bridge._fate_gate = MagicMock()
        mock_score = MagicMock()
        mock_score.to_dict.return_value = {"passed": True, "overall": 0.98}
        bridge._fate_gate.validate.return_value = mock_score

        await bridge.start()
        try:
            with (
                patch("core.bridges.desktop_bridge._RUST_AVAILABLE", True),
                patch(
                    "core.bridges.desktop_bridge.domain_separated_digest",
                    return_value="blake3hash123",
                    create=True,
                ),
            ):
                resp = await _tcp_call(
                    port,
                    "sovereign_query",
                    {"query": "What is sovereignty?"},
                )
                result = resp["result"]

                # Content from mock gateway
                assert "content" in result
                assert result["content"] == "sovereign answer"
                # Rust gate results
                assert result.get("rust_gates") is not None
                assert result["rust_gates"]["passed"] is True
                assert result["rust_gates"]["engine"] == "rust"
                # BLAKE3 hash
                assert result["content_hash"] == "blake3hash123"
        finally:
            await bridge.stop()

    @pytest.mark.asyncio
    async def test_query_blocked_by_rust_gates(self) -> None:
        """sovereign_query returns error when Rust gate chain blocks."""
        port = _free_port()
        bridge = DesktopBridge(host="127.0.0.1", port=port)

        mock_chain = MagicMock()
        mock_chain.verify.return_value = [
            ("schema", True, "PASS"),
            ("ihsan", False, "BELOW_THRESHOLD"),
        ]
        bridge._rust_gate_chain = mock_chain

        # Mock FATE gate to pass (so Rust gates get exercised)
        bridge._fate_gate = MagicMock()
        mock_score = MagicMock()
        mock_score.to_dict.return_value = {"passed": True, "overall": 0.98}
        bridge._fate_gate.validate.return_value = mock_score

        await bridge.start()
        try:
            with patch("core.bridges.desktop_bridge._RUST_AVAILABLE", True):
                resp = await _tcp_call(
                    port,
                    "sovereign_query",
                    {"query": "test query"},
                )
                result = resp["result"]
                assert "error" in result
                assert "Rust gate chain" in result["error"]
                assert result["rust_gates"]["passed"] is False
        finally:
            await bridge.stop()

    @pytest.mark.asyncio
    async def test_query_without_rust_still_works(self) -> None:
        """sovereign_query works correctly with no Rust available."""
        from unittest.mock import AsyncMock

        port = _free_port()
        mock_gw = AsyncMock()
        mock_gw.infer = AsyncMock(return_value=_MockInferenceResult())

        bridge = DesktopBridge(host="127.0.0.1", port=port, gateway=mock_gw)

        await bridge.start()
        try:
            with patch("core.bridges.desktop_bridge._RUST_AVAILABLE", False):
                resp = await _tcp_call(
                    port,
                    "sovereign_query",
                    {"query": "test query"},
                )
                result = resp["result"]
                if "content" in result:
                    assert result["content"] == "sovereign answer"
                    assert result.get("rust_gates") is None
                    assert result.get("content_hash") is None
        finally:
            await bridge.stop()
