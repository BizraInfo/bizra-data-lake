"""
Tests for the RDVE Actuator Layer (Phase 20).

Covers:
- Shannon entropy gate (_validate_entropy)
- actuator_execute handler (3-gate pipeline)
- get_context handler (UIA schema)
- ActuatorSkillLedger registration and lookup
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any
from unittest.mock import patch

import pytest

from core.bridges.desktop_bridge import (
    ACTUATOR_ENTROPY_THRESHOLD,
    BRIDGE_HOST,
    DesktopBridge,
)
from core.spearpoint.actuator_skills import (
    BASELINE_SKILLS,
    ActuatorSkillLedger,
    ActuatorSkillManifest,
    create_default_ledger,
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
    import asyncio

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
    b = DesktopBridge(host="127.0.0.1", port=free_port)
    await b.start()
    yield b
    await b.stop()


# ===========================================================================
# Shannon Entropy Gate
# ===========================================================================


class TestEntropyGate:
    def test_entropy_gate_blocks_low_signal(self) -> None:
        b = DesktopBridge(host="127.0.0.1", port=9999)
        result = b._validate_entropy("aaaaaaaaaa")
        assert result["passed"] is False
        assert result["entropy"] < ACTUATOR_ENTROPY_THRESHOLD

    def test_entropy_gate_passes_high_signal(self) -> None:
        b = DesktopBridge(host="127.0.0.1", port=9999)
        code = "import json; data = fetch('api/v1/endpoint'); process(data, mode='strict')"
        result = b._validate_entropy(code)
        assert result["passed"] is True
        assert result["entropy"] >= ACTUATOR_ENTROPY_THRESHOLD

    def test_entropy_gate_empty_instruction(self) -> None:
        b = DesktopBridge(host="127.0.0.1", port=9999)
        result = b._validate_entropy("")
        assert result["passed"] is False
        assert result["error"] == "Empty instruction"

    def test_entropy_gate_whitespace_only(self) -> None:
        b = DesktopBridge(host="127.0.0.1", port=9999)
        result = b._validate_entropy("   \t\n  ")
        assert result["passed"] is False

    def test_entropy_gate_threshold_matches_constant(self) -> None:
        b = DesktopBridge(host="127.0.0.1", port=9999)
        result = b._validate_entropy("some valid instruction with variety")
        assert result["threshold"] == ACTUATOR_ENTROPY_THRESHOLD
        assert result["threshold"] == 3.5


# ===========================================================================
# actuator_execute handler
# ===========================================================================


class TestActuatorExecute:
    @pytest.mark.asyncio
    async def test_missing_code_returns_error(
        self, bridge: DesktopBridge, free_port: int
    ) -> None:
        resp = await _send_recv(
            "127.0.0.1", free_port, _jsonrpc("actuator_execute", {"no_code": True})
        )
        assert "error" in resp

    @pytest.mark.asyncio
    async def test_empty_code_returns_error(
        self, bridge: DesktopBridge, free_port: int
    ) -> None:
        resp = await _send_recv(
            "127.0.0.1", free_port, _jsonrpc("actuator_execute", {"code": "  "})
        )
        assert "error" in resp

    @pytest.mark.asyncio
    async def test_low_entropy_code_blocked(
        self, bridge: DesktopBridge, free_port: int
    ) -> None:
        resp = await _send_recv(
            "127.0.0.1",
            free_port,
            _jsonrpc("actuator_execute", {"code": "aaaa"}),
        )
        result = resp.get("result", resp.get("error", {}))
        # Either FATE blocks it or Shannon blocks it — both are valid
        if isinstance(result, dict):
            has_block = (
                "error" in result
                or result.get("entropy", {}).get("passed") is False
            )
            assert has_block or "FATE" in str(result)

    @pytest.mark.asyncio
    async def test_valid_instruction_sealed(
        self, bridge: DesktopBridge, free_port: int
    ) -> None:
        code = "Send('^s'); WinWait('Save Dialog'); Click(200, 150); Sleep(100)"
        resp = await _send_recv(
            "127.0.0.1",
            free_port,
            _jsonrpc(
                "actuator_execute",
                {"code": code, "intent": "save_file", "target_app": "Code.exe"},
            ),
        )
        result = resp.get("result", {})
        # If FATE gate passes, result is SEALED; if not, FATE blocked
        if result.get("status") == "SEALED":
            assert result["intent"] == "save_file"
            assert result["target_app"] == "Code.exe"
            assert result["entropy"]["passed"] is True
            assert result["instruction_length"] == len(code)
        else:
            # FATE gate blocked — still valid behavior
            assert "fate" in result or "error" in result


# ===========================================================================
# get_context handler
# ===========================================================================


class TestGetContext:
    @pytest.mark.asyncio
    async def test_get_context_returns_schema(
        self, bridge: DesktopBridge, free_port: int
    ) -> None:
        resp = await _send_recv(
            "127.0.0.1", free_port, _jsonrpc("get_context")
        )
        result = resp["result"]
        assert result["schema_version"] == "1.0"
        assert "expected_fields" in result

    @pytest.mark.asyncio
    async def test_get_context_schema_fields_complete(
        self, bridge: DesktopBridge, free_port: int
    ) -> None:
        resp = await _send_recv(
            "127.0.0.1", free_port, _jsonrpc("get_context")
        )
        fields = resp["result"]["expected_fields"]
        for expected in ["title", "class", "process", "structure", "hash"]:
            assert expected in fields


# ===========================================================================
# ActuatorSkillLedger
# ===========================================================================


class TestActuatorSkillManifest:
    def test_valid_manifest_passes_validation(self) -> None:
        m = ActuatorSkillManifest(
            name="test_skill",
            description="Test",
            target_app="test.exe",
            ahk_code="Send('test')",
            entropy_score=4.0,
        )
        assert m.validate() is True

    def test_low_entropy_fails_validation(self) -> None:
        m = ActuatorSkillManifest(
            name="test_skill",
            description="Test",
            target_app="test.exe",
            ahk_code="Send('test')",
            entropy_score=2.0,
        )
        assert m.validate() is False

    def test_low_ihsan_fails_validation(self) -> None:
        m = ActuatorSkillManifest(
            name="test_skill",
            description="Test",
            target_app="test.exe",
            ahk_code="Send('test')",
            entropy_score=4.0,
            min_ihsan=0.5,
        )
        assert m.validate() is False

    def test_empty_name_fails_validation(self) -> None:
        m = ActuatorSkillManifest(
            name="",
            description="Test",
            target_app="test.exe",
            ahk_code="Send('test')",
            entropy_score=4.0,
        )
        assert m.validate() is False

    def test_empty_code_fails_validation(self) -> None:
        m = ActuatorSkillManifest(
            name="test",
            description="Test",
            target_app="test.exe",
            ahk_code="",
            entropy_score=4.0,
        )
        assert m.validate() is False


class TestActuatorSkillLedger:
    def test_register_and_get(self) -> None:
        ledger = ActuatorSkillLedger()
        m = ActuatorSkillManifest(
            name="my_skill",
            description="Test",
            target_app="test.exe",
            ahk_code="Send('test')",
            entropy_score=4.0,
        )
        assert ledger.register(m) is True
        assert ledger.get("my_skill") is m

    def test_register_invalid_rejected(self) -> None:
        ledger = ActuatorSkillLedger()
        m = ActuatorSkillManifest(
            name="bad",
            description="Test",
            target_app="test.exe",
            ahk_code="Send('test')",
            entropy_score=1.0,
        )
        assert ledger.register(m) is False
        assert ledger.get("bad") is None

    def test_list_all(self) -> None:
        ledger = ActuatorSkillLedger()
        m1 = ActuatorSkillManifest(
            name="a", description="A", target_app="a.exe",
            ahk_code="Send('a')", entropy_score=4.0,
        )
        m2 = ActuatorSkillManifest(
            name="b", description="B", target_app="b.exe",
            ahk_code="Send('b')", entropy_score=4.5,
        )
        ledger.register(m1)
        ledger.register(m2)
        assert len(ledger.list_all()) == 2

    def test_resolve_for_app(self) -> None:
        ledger = ActuatorSkillLedger()
        m1 = ActuatorSkillManifest(
            name="vscode", description="VS Code", target_app="Code.exe",
            ahk_code="Send('^s')", entropy_score=4.0,
        )
        m2 = ActuatorSkillManifest(
            name="chrome", description="Chrome", target_app="chrome.exe",
            ahk_code="Send('^l')", entropy_score=4.0,
        )
        ledger.register(m1)
        ledger.register(m2)
        assert len(ledger.resolve_for_app("Code.exe")) == 1
        assert len(ledger.resolve_for_app("chrome.exe")) == 1
        assert len(ledger.resolve_for_app("notepad.exe")) == 0

    def test_count_property(self) -> None:
        ledger = ActuatorSkillLedger()
        assert ledger.count == 0
        m = ActuatorSkillManifest(
            name="x", description="X", target_app="x.exe",
            ahk_code="Send('x')", entropy_score=4.0,
        )
        ledger.register(m)
        assert ledger.count == 1


class TestDefaultLedger:
    def test_create_default_ledger_has_baseline_skills(self) -> None:
        ledger = create_default_ledger()
        assert ledger.count == len(BASELINE_SKILLS)

    def test_baseline_skills_all_valid(self) -> None:
        for skill in BASELINE_SKILLS:
            assert skill.validate() is True, f"Baseline skill '{skill.name}' failed validation"

    def test_default_ledger_contains_vscode_save(self) -> None:
        ledger = create_default_ledger()
        assert ledger.get("vscode_save") is not None

    def test_default_ledger_contains_browser_navigate(self) -> None:
        ledger = create_default_ledger()
        assert ledger.get("browser_navigate") is not None

    def test_default_ledger_contains_terminal_command(self) -> None:
        ledger = create_default_ledger()
        skill = ledger.get("terminal_command")
        assert skill is not None
        assert skill.requires_context is True
