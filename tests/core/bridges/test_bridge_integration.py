"""
Integration tests for DesktopBridge wired into SovereignLauncher.

Verifies:
- SovereignLauncher includes desktop_bridge in status
- enable_desktop_bridge=False skips bridge startup
- Bridge starts and stops with launcher lifecycle
- CLI bridge subcommand parse
"""

from __future__ import annotations

import asyncio
import json
import socket
import time
import uuid
from typing import Any

import pytest

from core.bridges.desktop_bridge import BRIDGE_HOST, DesktopBridge


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _jsonrpc(method: str, id: int = 1) -> bytes:
    return json.dumps(
        {
            "jsonrpc": "2.0",
            "method": method,
            "id": id,
            "headers": {
                "X-BIZRA-TOKEN": "test-bridge-token",
                "X-BIZRA-TS": int(time.time() * 1000),
                "X-BIZRA-NONCE": uuid.uuid4().hex,
            },
        }
    ).encode() + b"\n"


async def _tcp_call(port: int, method: str) -> dict[str, Any]:
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(_jsonrpc(method))
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
# SovereignLauncher integration
# ---------------------------------------------------------------------------


class TestLauncherIntegration:
    def test_launcher_accepts_bridge_params(self) -> None:
        """SovereignLauncher.__init__ accepts desktop bridge parameters."""
        from core.sovereign.launch import SovereignLauncher

        launcher = SovereignLauncher(
            enable_desktop_bridge=True,
            desktop_bridge_port=9999,
            enable_api=False,
            enable_autonomy=False,
        )
        assert launcher.enable_desktop_bridge is True
        assert launcher.desktop_bridge_port == 9999
        assert launcher.desktop_bridge is None  # Not started yet

    def test_launcher_disable_bridge(self) -> None:
        """SovereignLauncher with bridge disabled."""
        from core.sovereign.launch import SovereignLauncher

        launcher = SovereignLauncher(
            enable_desktop_bridge=False,
            enable_api=False,
            enable_autonomy=False,
        )
        assert launcher.enable_desktop_bridge is False
        assert launcher.desktop_bridge is None

    def test_launcher_status_includes_bridge_key(self) -> None:
        """status() includes desktop_bridge when bridge is set."""
        from core.sovereign.launch import SovereignLauncher

        launcher = SovereignLauncher(
            enable_desktop_bridge=True,
            enable_api=False,
            enable_autonomy=False,
        )
        # Before start, desktop_bridge is None so not in status
        status = launcher.status()
        assert "desktop_bridge" not in status

    @pytest.mark.asyncio
    async def test_launcher_status_with_running_bridge(self) -> None:
        """status() includes bridge info when bridge is running."""
        from core.sovereign.launch import SovereignLauncher

        port = _free_port()
        launcher = SovereignLauncher(
            enable_desktop_bridge=True,
            desktop_bridge_port=port,
            enable_api=False,
            enable_autonomy=False,
        )
        # Manually attach a bridge (simulating partial start)
        bridge = DesktopBridge(host="127.0.0.1", port=port)
        await bridge.start()
        launcher.desktop_bridge = bridge
        launcher.desktop_bridge_port = port

        try:
            status = launcher.status()
            assert "desktop_bridge" in status
            assert status["desktop_bridge"]["running"] is True
            assert status["desktop_bridge"]["port"] == port
        finally:
            await bridge.stop()


# ---------------------------------------------------------------------------
# Bridge standalone lifecycle
# ---------------------------------------------------------------------------


class TestBridgeLifecycle:
    @pytest.mark.asyncio
    async def test_start_ping_stop(self) -> None:
        """Full lifecycle: start -> ping -> stop."""
        port = _free_port()
        bridge = DesktopBridge(host="127.0.0.1", port=port)

        await bridge.start()
        assert bridge.is_running

        # Ping via TCP
        resp = await _tcp_call(port, "ping")
        assert resp["result"]["status"] == "alive"

        await bridge.stop()
        assert not bridge.is_running

    @pytest.mark.asyncio
    async def test_status_has_fate_gate(self) -> None:
        """status command returns fate_gate field."""
        port = _free_port()
        bridge = DesktopBridge(host="127.0.0.1", port=port)
        await bridge.start()
        try:
            resp = await _tcp_call(port, "status")
            result = resp["result"]
            assert "fate_gate" in result
            assert "inference" in result
            assert "rust" in result
            assert result["node"] == "node"
        finally:
            await bridge.stop()


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


class TestCLIParsing:
    def test_bridge_command_exists_in_help(self) -> None:
        """The 'bridge' subcommand is registered in the CLI."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "core.sovereign", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
            env={"PYTHONPATH": "/mnt/c/BIZRA-DATA-LAKE"},
        )
        assert "bridge" in result.stdout

    def test_launcher_cli_has_bridge_flags(self) -> None:
        """launch.py CLI accepts --no-bridge and --bridge-port."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "core.sovereign.launch", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
            env={"PYTHONPATH": "/mnt/c/BIZRA-DATA-LAKE"},
        )
        assert "--no-bridge" in result.stdout
        assert "--bridge-port" in result.stdout
