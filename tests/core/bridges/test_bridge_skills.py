"""
Tests for skill routing and receipt emission in the Desktop Bridge.

Covers:
- invoke_skill: success, unknown skill, missing param, FATE blocked, receipt attached
- list_skills: returns skills, filter by status, no registry
- get_receipt: found, not found
- _emit_receipt: returns dict, returns None when engine missing
- Existing commands (ping/status) emit receipts when engine is available
"""

from __future__ import annotations

import asyncio
import json
import socket
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.bridges.desktop_bridge import DesktopBridge


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
# Mock skill objects
# ---------------------------------------------------------------------------


@dataclass
class _MockManifest:
    name: str = "test-skill"
    description: str = "A test skill"
    agent: str = "sovereign-coder"
    tags: List[str] = field(default_factory=lambda: ["test"])


@dataclass
class _MockSkillEntry:
    manifest: _MockManifest = field(default_factory=_MockManifest)
    status: Any = None

    def __post_init__(self):
        if self.status is None:
            self.status = MagicMock(value="available")


@dataclass
class _MockInvocationResult:
    success: bool = True
    skill_name: str = "test-skill"
    output: str = "skill output"
    error: Optional[str] = None
    duration_ms: float = 42.0
    fate_score: float = 0.98
    ihsan_passed: bool = True
    execution_id: str = "exec-001"
    token_count: int = 50
    started_at: str = ""
    completed_at: str = ""
    agent_used: str = "sovereign-coder"
    tools_used: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "skill_name": self.skill_name,
            "output": self.output,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "fate_score": self.fate_score,
            "ihsan_passed": self.ihsan_passed,
            "execution_id": self.execution_id,
            "token_count": self.token_count,
            "agent_used": self.agent_used,
        }


# ---------------------------------------------------------------------------
# invoke_skill
# ---------------------------------------------------------------------------


class TestInvokeSkill:
    @pytest.mark.asyncio
    async def test_invoke_skill_success(self, tmp_path: Path) -> None:
        port = _free_port()
        bridge = DesktopBridge(host="127.0.0.1", port=port)

        # Mock FATE gate to pass
        bridge._fate_gate = MagicMock()
        mock_score = MagicMock()
        mock_score.to_dict.return_value = {"passed": True, "overall": 0.98}
        bridge._fate_gate.validate.return_value = mock_score

        # Mock skill router
        mock_router = AsyncMock()
        mock_router.invoke = AsyncMock(return_value=_MockInvocationResult())
        bridge._skill_router = mock_router

        # Mock receipt engine
        from core.bridges.bridge_receipt import BridgeReceiptEngine
        bridge._receipt_engine = BridgeReceiptEngine(receipt_dir=tmp_path / "receipts")

        await bridge.start()
        try:
            resp = await _tcp_call(
                port, "invoke_skill",
                {"skill": "test-skill", "inputs": {"query": "hello"}},
            )
            result = resp["result"]
            assert result["success"] is True
            assert result["skill_name"] == "test-skill"
            assert result["output"] == "skill output"
            assert "receipt" in result
            assert result["receipt"]["status"] == "accepted"
        finally:
            await bridge.stop()

    @pytest.mark.asyncio
    async def test_invoke_skill_missing_param(self) -> None:
        port = _free_port()
        bridge = DesktopBridge(host="127.0.0.1", port=port)
        await bridge.start()
        try:
            resp = await _tcp_call(port, "invoke_skill", {"no_skill": True})
            assert "error" in resp
        finally:
            await bridge.stop()

    @pytest.mark.asyncio
    async def test_invoke_skill_empty_name(self) -> None:
        port = _free_port()
        bridge = DesktopBridge(host="127.0.0.1", port=port)
        await bridge.start()
        try:
            resp = await _tcp_call(port, "invoke_skill", {"skill": "  "})
            assert "error" in resp
        finally:
            await bridge.stop()

    @pytest.mark.asyncio
    async def test_invoke_skill_no_router(self) -> None:
        """invoke_skill returns error when SkillRouter is not available."""
        port = _free_port()
        bridge = DesktopBridge(host="127.0.0.1", port=port)

        # Mock FATE gate to pass
        bridge._fate_gate = MagicMock()
        mock_score = MagicMock()
        mock_score.to_dict.return_value = {"passed": True, "overall": 0.98}
        bridge._fate_gate.validate.return_value = mock_score

        # No skill router
        bridge._get_skill_router = lambda: None

        await bridge.start()
        try:
            resp = await _tcp_call(
                port, "invoke_skill", {"skill": "test"}
            )
            result = resp["result"]
            assert "error" in result
            assert "SkillRouter" in result["error"]
        finally:
            await bridge.stop()

    @pytest.mark.asyncio
    async def test_invoke_skill_fate_blocked(self, tmp_path: Path) -> None:
        """invoke_skill returns error + receipt when FATE blocks."""
        port = _free_port()
        bridge = DesktopBridge(host="127.0.0.1", port=port)

        # Mock FATE gate to block
        bridge._fate_gate = MagicMock()
        mock_score = MagicMock()
        mock_score.to_dict.return_value = {"passed": False, "overall": 0.3}
        bridge._fate_gate.validate.return_value = mock_score

        from core.bridges.bridge_receipt import BridgeReceiptEngine
        bridge._receipt_engine = BridgeReceiptEngine(receipt_dir=tmp_path / "receipts")

        await bridge.start()
        try:
            resp = await _tcp_call(
                port, "invoke_skill", {"skill": "test"}
            )
            result = resp["result"]
            assert "error" in result
            assert "FATE" in result["error"]
            assert result.get("receipt") is not None
            assert result["receipt"]["status"] == "rejected"
        finally:
            await bridge.stop()


# ---------------------------------------------------------------------------
# list_skills
# ---------------------------------------------------------------------------


class TestListSkills:
    @pytest.mark.asyncio
    async def test_list_skills_returns_skills(self) -> None:
        port = _free_port()
        bridge = DesktopBridge(host="127.0.0.1", port=port)

        mock_registry = MagicMock()
        mock_registry.get_all.return_value = [
            _MockSkillEntry(_MockManifest("skill-a", "Skill A", "coder", ["dev"])),
            _MockSkillEntry(_MockManifest("skill-b", "Skill B", "researcher", ["research"])),
        ]
        mock_router = MagicMock()
        mock_router.registry = mock_registry
        bridge._skill_router = mock_router

        await bridge.start()
        try:
            resp = await _tcp_call(port, "list_skills")
            result = resp["result"]
            assert result["count"] == 2
            assert len(result["skills"]) == 2
            assert result["skills"][0]["name"] == "skill-a"
            assert result["skills"][1]["agent"] == "researcher"
        finally:
            await bridge.stop()

    @pytest.mark.asyncio
    async def test_list_skills_with_filter(self) -> None:
        port = _free_port()
        bridge = DesktopBridge(host="127.0.0.1", port=port)

        available = _MockSkillEntry(_MockManifest("active", "Active", "coder"))
        available.status = MagicMock(value="available")

        suspended = _MockSkillEntry(_MockManifest("paused", "Paused", "coder"))
        suspended.status = MagicMock(value="suspended")

        mock_registry = MagicMock()
        mock_registry.get_all.return_value = [available, suspended]
        mock_router = MagicMock()
        mock_router.registry = mock_registry
        bridge._skill_router = mock_router

        await bridge.start()
        try:
            resp = await _tcp_call(port, "list_skills", {"filter": "available"})
            result = resp["result"]
            assert result["count"] == 1
            assert result["skills"][0]["name"] == "active"
        finally:
            await bridge.stop()

    @pytest.mark.asyncio
    async def test_list_skills_no_registry(self) -> None:
        port = _free_port()
        bridge = DesktopBridge(host="127.0.0.1", port=port)
        bridge._get_skill_router = lambda: None

        await bridge.start()
        try:
            resp = await _tcp_call(port, "list_skills")
            result = resp["result"]
            assert result["skills"] == []
            assert "error" in result
        finally:
            await bridge.stop()


# ---------------------------------------------------------------------------
# get_receipt
# ---------------------------------------------------------------------------


class TestGetReceipt:
    @pytest.mark.asyncio
    async def test_get_receipt_found(self, tmp_path: Path) -> None:
        port = _free_port()
        bridge = DesktopBridge(host="127.0.0.1", port=port)

        from core.bridges.bridge_receipt import BridgeReceiptEngine
        engine = BridgeReceiptEngine(receipt_dir=tmp_path / "receipts")
        bridge._receipt_engine = engine

        # Create a receipt directly
        receipt = engine.create_receipt(
            method="test", query_data={}, result_data={},
            fate_score=1.0, snr_score=0.95, gate_passed="all",
            status="accepted",
        )

        await bridge.start()
        try:
            resp = await _tcp_call(
                port, "get_receipt",
                {"receipt_id": receipt["receipt_id"]},
            )
            result = resp["result"]
            assert result["receipt_id"] == receipt["receipt_id"]
            assert result["status"] == "accepted"
        finally:
            await bridge.stop()

    @pytest.mark.asyncio
    async def test_get_receipt_not_found(self, tmp_path: Path) -> None:
        port = _free_port()
        bridge = DesktopBridge(host="127.0.0.1", port=port)

        from core.bridges.bridge_receipt import BridgeReceiptEngine
        bridge._receipt_engine = BridgeReceiptEngine(receipt_dir=tmp_path / "receipts")

        await bridge.start()
        try:
            resp = await _tcp_call(
                port, "get_receipt",
                {"receipt_id": "br-nonexistent"},
            )
            assert "error" in resp
        finally:
            await bridge.stop()

    @pytest.mark.asyncio
    async def test_get_receipt_missing_param(self) -> None:
        port = _free_port()
        bridge = DesktopBridge(host="127.0.0.1", port=port)
        await bridge.start()
        try:
            resp = await _tcp_call(port, "get_receipt", {})
            assert "error" in resp
        finally:
            await bridge.stop()


# ---------------------------------------------------------------------------
# Receipt emission
# ---------------------------------------------------------------------------


class TestReceiptEmission:
    def test_emit_receipt_returns_dict(self, tmp_path: Path) -> None:
        bridge = DesktopBridge(host="127.0.0.1", port=9742)
        from core.bridges.bridge_receipt import BridgeReceiptEngine
        bridge._receipt_engine = BridgeReceiptEngine(receipt_dir=tmp_path / "receipts")

        result = bridge._emit_receipt(
            "ping", {"method": "ping"}, {"status": "alive"},
            "accepted", "all", 1.0
        )
        assert result is not None
        assert "receipt_id" in result
        assert result["status"] == "accepted"

    def test_emit_receipt_none_when_engine_missing(self) -> None:
        bridge = DesktopBridge(host="127.0.0.1", port=9742)
        bridge._get_receipt_engine = lambda: None

        result = bridge._emit_receipt(
            "ping", {}, {}, "accepted", "all"
        )
        assert result is None
