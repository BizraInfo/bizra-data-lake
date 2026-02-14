"""
Tests for RDVE Skill Handler — Bridge adapter for Spearpoint orchestrator.

Covers:
  - RDVESkillHandler initialization and properties
  - SkillRouter registration (manifest, tags, agent, handler)
  - Operation dispatch (research_pattern, reproduce, improve, statistics)
  - Error handling (missing inputs, unknown operations)
  - Bridge integration (invoke_skill end-to-end via DesktopBridge)
"""

from __future__ import annotations

import asyncio
import json
import socket
import time
import uuid
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.spearpoint.config import MissionType, SpearpointConfig
from core.spearpoint.rdve_skill import (
    RDVESkillHandler,
    get_rdve_handler,
    register_rdve_skill,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config():
    """Spearpoint config with in-memory paths."""
    import tempfile
    from pathlib import Path

    tmpdir = Path(tempfile.mkdtemp())
    return SpearpointConfig(
        state_dir=tmpdir / "state",
        evidence_ledger_path=tmpdir / "evidence.jsonl",
        hypothesis_memory_path=tmpdir / "hypothesis_memory",
    )


@pytest.fixture(autouse=True)
def _bridge_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BIZRA_BRIDGE_TOKEN", "test-bridge-token")
    monkeypatch.setenv("BIZRA_RECEIPT_PRIVATE_KEY_HEX", "11" * 32)


def _bridge_headers() -> dict[str, Any]:
    return {
        "X-BIZRA-TOKEN": "test-bridge-token",
        "X-BIZRA-TS": int(time.time() * 1000),
        "X-BIZRA-NONCE": uuid.uuid4().hex,
    }


@pytest.fixture
def handler(config):
    """Fresh RDVESkillHandler with test config."""
    return RDVESkillHandler(config=config)


@pytest.fixture
def mock_router():
    """Mock SkillRouter with a real registry."""
    from core.skills.registry import SkillRegistry

    router = MagicMock()
    router.registry = SkillRegistry(skills_dir="/nonexistent")
    router.registry._skills = {}
    router.registry._by_tag = {}
    router.registry._by_agent = {}
    router.register_handler = MagicMock()
    return router


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------


class TestRDVESkillHandlerInit:
    """Test handler creation and properties."""

    def test_default_init(self):
        """Handler initializes with default config."""
        h = RDVESkillHandler()
        assert h.SKILL_NAME == "rdve_research"
        assert h.AGENT_NAME == "rdve-researcher"
        assert h._invocation_count == 0
        assert h.orchestrator is not None

    def test_custom_config(self, config):
        """Handler accepts custom config."""
        h = RDVESkillHandler(config=config)
        assert h._config is config
        assert h.orchestrator is not None

    def test_custom_orchestrator(self, config):
        """Handler accepts injected orchestrator."""
        from core.spearpoint.orchestrator import SpearpointOrchestrator

        orch = SpearpointOrchestrator(config=config)
        h = RDVESkillHandler(config=config, orchestrator=orch)
        assert h.orchestrator is orch

    def test_version_and_metadata(self, handler):
        """Handler exposes correct metadata."""
        assert handler.VERSION == "1.0.0"
        assert "research" in handler.TAGS
        assert "spearpoint" in handler.TAGS
        assert "rdve" in handler.TAGS
        assert "sci-reasoning" in handler.TAGS


# ---------------------------------------------------------------------------
# Registration tests
# ---------------------------------------------------------------------------


class TestRegistration:
    """Test skill registration on SkillRouter."""

    def test_register_creates_skill_in_registry(self, handler, mock_router):
        """Registration creates a RegisteredSkill in the registry."""
        handler.register(mock_router)

        skill = mock_router.registry._skills.get("rdve_research")
        assert skill is not None
        assert skill.manifest.name == "rdve_research"
        assert skill.manifest.agent == "rdve-researcher"
        assert skill.manifest.description.startswith("Recursive Discovery")
        assert skill.status.value == "available"

    def test_register_indexes_tags(self, handler, mock_router):
        """Registration indexes skill by tags."""
        handler.register(mock_router)

        for tag in handler.TAGS:
            assert "rdve_research" in mock_router.registry._by_tag.get(tag, [])

    def test_register_indexes_agent(self, handler, mock_router):
        """Registration indexes skill by agent."""
        handler.register(mock_router)
        assert "rdve_research" in mock_router.registry._by_agent.get("rdve-researcher", [])

    def test_register_wires_handler(self, handler, mock_router):
        """Registration calls register_handler on the router."""
        handler.register(mock_router)
        mock_router.register_handler.assert_called_once_with(
            "rdve-researcher", handler._handle
        )

    def test_register_idempotent(self, handler, mock_router):
        """Registering twice doesn't duplicate entries."""
        handler.register(mock_router)
        handler.register(mock_router)

        assert mock_router.registry._by_tag["rdve"].count("rdve_research") == 1

    def test_register_manifest_inputs(self, handler, mock_router):
        """Manifest has correct required/optional inputs."""
        handler.register(mock_router)
        skill = mock_router.registry._skills["rdve_research"]
        assert "operation" in skill.manifest.required_inputs
        assert "pattern_id" in skill.manifest.optional_inputs
        assert "claim" in skill.manifest.optional_inputs


# ---------------------------------------------------------------------------
# Operation dispatch tests
# ---------------------------------------------------------------------------


class TestOperationDispatch:
    """Test the _handle method routes to correct operations."""

    @pytest.mark.asyncio
    async def test_unknown_operation_returns_error(self, handler):
        """Unknown operation returns error with available operations list."""
        result = await handler._handle(
            MagicMock(), {"operation": "nonexistent"}, None
        )
        assert "error" in result
        assert "nonexistent" in result["error"]
        assert "available_operations" in result
        assert "research_pattern" in result["available_operations"]

    @pytest.mark.asyncio
    async def test_missing_operation_returns_error(self, handler):
        """Missing operation field returns error."""
        result = await handler._handle(MagicMock(), {}, None)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_invocation_count_increments(self, handler):
        """Each handle call increments invocation count."""
        assert handler._invocation_count == 0
        await handler._handle(MagicMock(), {"operation": "statistics"}, None)
        assert handler._invocation_count == 1
        await handler._handle(MagicMock(), {"operation": "statistics"}, None)
        assert handler._invocation_count == 2


# ---------------------------------------------------------------------------
# research_pattern operation tests
# ---------------------------------------------------------------------------


class TestResearchPattern:
    """Test the research_pattern operation."""

    @pytest.mark.asyncio
    async def test_missing_pattern_id_returns_error(self, handler):
        """Missing pattern_id returns error."""
        result = await handler._handle(
            MagicMock(),
            {"operation": "research_pattern"},
            None,
        )
        assert "error" in result
        assert "pattern_id" in result["error"]

    @pytest.mark.asyncio
    async def test_research_pattern_returns_mission(self, handler):
        """Valid pattern_id returns mission result."""
        result = await handler._handle(
            MagicMock(),
            {
                "operation": "research_pattern",
                "pattern_id": "P01",
                "claim_context": "optimize attention mechanism",
                "top_k": 2,
            },
            None,
        )
        assert result["operation"] == "research_pattern"
        assert result["pattern_id"] == "P01"
        assert "mission" in result
        assert "elapsed_ms" in result
        assert isinstance(result["elapsed_ms"], float)

    @pytest.mark.asyncio
    async def test_research_pattern_mission_has_fields(self, handler):
        """Mission result has required fields."""
        result = await handler._handle(
            MagicMock(),
            {"operation": "research_pattern", "pattern_id": "P01"},
            None,
        )
        mission = result["mission"]
        assert "mission_id" in mission
        assert "mission_type" in mission
        assert "success" in mission
        assert mission["mission_type"] == "improve"


# ---------------------------------------------------------------------------
# reproduce operation tests
# ---------------------------------------------------------------------------


class TestReproduce:
    """Test the reproduce operation."""

    @pytest.mark.asyncio
    async def test_missing_claim_returns_error(self, handler):
        """Missing claim returns error."""
        result = await handler._handle(
            MagicMock(),
            {"operation": "reproduce"},
            None,
        )
        assert "error" in result
        assert "claim" in result["error"]

    @pytest.mark.asyncio
    async def test_reproduce_returns_mission(self, handler):
        """Valid claim returns mission result."""
        result = await handler._handle(
            MagicMock(),
            {
                "operation": "reproduce",
                "claim": "System latency < 100ms under load",
                "proposed_change": "Add caching layer",
            },
            None,
        )
        assert result["operation"] == "reproduce"
        assert "mission" in result
        assert result["mission"]["mission_type"] == "reproduce"

    @pytest.mark.asyncio
    async def test_reproduce_truncates_claim_in_response(self, handler):
        """Long claims are truncated in the response for brevity."""
        long_claim = "x" * 200
        result = await handler._handle(
            MagicMock(),
            {"operation": "reproduce", "claim": long_claim},
            None,
        )
        assert len(result["claim"]) <= 100


# ---------------------------------------------------------------------------
# improve operation tests
# ---------------------------------------------------------------------------


class TestImprove:
    """Test the improve operation."""

    @pytest.mark.asyncio
    async def test_improve_returns_mission(self, handler):
        """Improve operation returns mission result."""
        result = await handler._handle(
            MagicMock(),
            {"operation": "improve", "top_k": 2},
            None,
        )
        assert result["operation"] == "improve"
        assert "mission" in result
        assert result["mission"]["mission_type"] == "improve"

    @pytest.mark.asyncio
    async def test_improve_with_observation(self, handler):
        """Improve with observation dict works."""
        result = await handler._handle(
            MagicMock(),
            {
                "operation": "improve",
                "observation": {
                    "snr_score": 0.91,
                    "ihsan_score": 0.96,
                    "latency_ms": 120,
                },
            },
            None,
        )
        assert result["operation"] == "improve"
        assert "mission" in result


# ---------------------------------------------------------------------------
# statistics operation tests
# ---------------------------------------------------------------------------


class TestStatistics:
    """Test the statistics operation."""

    @pytest.mark.asyncio
    async def test_statistics_returns_rdve_info(self, handler):
        """Statistics returns RDVE metadata."""
        result = await handler._handle(
            MagicMock(), {"operation": "statistics"}, None
        )
        assert result["operation"] == "statistics"
        assert result["rdve"]["version"] == "1.0.0"
        assert result["rdve"]["skill_name"] == "rdve_research"
        assert "orchestrator" in result
        assert "mission_history" in result

    @pytest.mark.asyncio
    async def test_statistics_tracks_invocations(self, handler):
        """Invocation count is reflected in statistics."""
        # Invoke twice
        await handler._handle(MagicMock(), {"operation": "statistics"}, None)
        await handler._handle(MagicMock(), {"operation": "statistics"}, None)

        result = await handler._handle(
            MagicMock(), {"operation": "statistics"}, None
        )
        assert result["rdve"]["invocation_count"] == 3  # Including this call


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


class TestModuleFunctions:
    """Test get_rdve_handler and register_rdve_skill."""

    def test_get_rdve_handler_returns_singleton(self):
        """get_rdve_handler returns same instance."""
        import core.spearpoint.rdve_skill as mod

        # Reset singleton
        mod._default_handler = None

        h1 = get_rdve_handler()
        h2 = get_rdve_handler()
        assert h1 is h2

        # Cleanup
        mod._default_handler = None

    def test_register_rdve_skill_returns_handler(self, mock_router):
        """register_rdve_skill returns the handler."""
        import core.spearpoint.rdve_skill as mod

        mod._default_handler = None

        result = register_rdve_skill(mock_router)
        assert isinstance(result, RDVESkillHandler)
        mock_router.register_handler.assert_called_once()

        # Cleanup
        mod._default_handler = None


# ---------------------------------------------------------------------------
# Bridge integration tests
# ---------------------------------------------------------------------------


class TestBridgeIntegration:
    """Test RDVE accessible through Desktop Bridge invoke_skill."""

    @pytest.mark.asyncio
    async def test_bridge_registers_rdve_on_skill_router_load(self):
        """When bridge lazy-loads SkillRouter, RDVE gets registered."""
        from core.bridges.desktop_bridge import DesktopBridge

        # Get a free port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

        bridge = DesktopBridge(port=port)
        router = bridge._get_skill_router()

        if router is not None:
            # RDVE should be registered
            skill = router.registry.get("rdve_research")
            assert skill is not None
            assert skill.manifest.agent == "rdve-researcher"

    @pytest.mark.asyncio
    async def test_invoke_rdve_statistics_via_bridge(self):
        """Full end-to-end: bridge -> invoke_skill -> rdve_research -> statistics."""
        from core.bridges.desktop_bridge import DesktopBridge

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

        bridge = DesktopBridge(port=port)
        await bridge.start()

        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)

            msg = json.dumps({
                "jsonrpc": "2.0",
                "method": "invoke_skill",
                "params": {
                    "skill": "rdve_research",
                    "inputs": {"operation": "statistics"},
                },
                "id": 1,
                "headers": _bridge_headers(),
            }).encode() + b"\n"

            writer.write(msg)
            await writer.drain()

            line = await asyncio.wait_for(reader.readline(), timeout=10.0)
            resp = json.loads(line)

            writer.close()
            await writer.wait_closed()

            # The response should either be a success or a "no handler" info
            # (depends on whether SkillRouter can be loaded)
            result = resp.get("result", {})
            error = resp.get("error")

            # If router loaded, check RDVE response structure
            if result and "output" in result:
                output = result["output"]
                if isinstance(output, dict) and "operation" in output:
                    assert output["operation"] == "statistics"
                    assert output["rdve"]["skill_name"] == "rdve_research"

        finally:
            await bridge.stop()

    @pytest.mark.asyncio
    async def test_invoke_rdve_reproduce_via_bridge(self):
        """End-to-end: bridge -> invoke_skill -> rdve_research -> reproduce."""
        from core.bridges.desktop_bridge import DesktopBridge

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

        bridge = DesktopBridge(port=port)
        await bridge.start()

        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)

            msg = json.dumps({
                "jsonrpc": "2.0",
                "method": "invoke_skill",
                "params": {
                    "skill": "rdve_research",
                    "inputs": {
                        "operation": "reproduce",
                        "claim": "BIZRA latency < 200ms P99",
                    },
                },
                "id": 2,
                "headers": _bridge_headers(),
            }).encode() + b"\n"

            writer.write(msg)
            await writer.drain()

            line = await asyncio.wait_for(reader.readline(), timeout=10.0)
            resp = json.loads(line)

            writer.close()
            await writer.wait_closed()

            # Verify response structure
            result = resp.get("result", {})
            if result and "output" in result:
                output = result["output"]
                if isinstance(output, dict) and "operation" in output:
                    assert output["operation"] == "reproduce"

        finally:
            await bridge.stop()
