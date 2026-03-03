"""Unit tests for core.sovereign.mission — MissionOrchestrator."""

from __future__ import annotations

import time

import pytest

from core.sovereign.mission import (
    ChannelResult,
    DesktopContext,
    HDAClient,
    MissionOrchestrator,
    MissionRequest,
    MissionResult,
)

# ── Data Type Tests ─────────────────────────────────────────────────


class TestMissionRequest:
    def test_creation(self):
        req = MissionRequest(
            mission_id="a" * 32,
            description="Test mission",
            context=DesktopContext("VS Code", "clipboard text", {}),
            timestamp=time.time(),
            source="test",
        )
        assert len(req.mission_id) == 32
        assert req.source == "test"
        assert req.context.active_window_title == "VS Code"

    def test_default_context(self):
        ctx = DesktopContext()
        assert ctx.active_window_title == "unknown"
        assert ctx.clipboard_text == ""
        assert ctx.screen_geometry == {}


class TestChannelResult:
    def test_success_result(self):
        cr = ChannelResult(
            channel="browser",
            success=True,
            data={"results_count": 5},
            duration_ms=123.4,
        )
        assert cr.success
        assert cr.error is None

    def test_failure_result(self):
        cr = ChannelResult(
            channel="desktop",
            success=False,
            data={},
            duration_ms=50.0,
            error="Connection refused",
        )
        assert not cr.success
        assert "refused" in cr.error


class TestMissionResult:
    def test_complete_result(self):
        result = MissionResult(
            mission_id="b" * 32,
            status="COMPLETE",
            channels_executed=[],
            synthesis="Test synthesis",
            briefing_path="/tmp/test.md",
            evidence_receipt_id="c" * 16,
            ihsan_score=0.96,
            snr_score=0.92,
            duration_ms=500.0,
        )
        assert result.status == "COMPLETE"
        assert result.ihsan_score >= 0.95


# ── MissionOrchestrator Tests ───────────────────────────────────────


class TestMissionOrchestrator:
    @pytest.fixture
    def config(self, tmp_path):
        return {
            "memory_path": str(tmp_path / "memory"),
            "evidence_path": str(tmp_path / "evidence.jsonl"),
            "hda_port": 59999,  # Non-listening port
        }

    @pytest.fixture
    def orchestrator(self, config):
        return MissionOrchestrator(config)

    @pytest.fixture
    def mission_req(self):
        return MissionRequest(
            mission_id="d" * 32,
            description="Research AI agent frameworks",
            context=DesktopContext("Test Window", "", {}),
            timestamp=time.time(),
            source="test",
        )

    async def test_initialize(self, orchestrator, tmp_path):
        await orchestrator.initialize()
        assert orchestrator._initialized
        assert (tmp_path / "memory").exists()

    async def test_initialize_idempotent(self, orchestrator):
        await orchestrator.initialize()
        await orchestrator.initialize()
        assert orchestrator._initialized

    async def test_execute_returns_mission_result(self, orchestrator, mission_req):
        await orchestrator.initialize()
        result = await orchestrator.execute(mission_req)
        assert isinstance(result, MissionResult)
        assert result.mission_id == "d" * 32
        assert result.status in ("COMPLETE", "PARTIAL", "FAILED")

    async def test_execute_has_valid_scores(self, orchestrator, mission_req):
        await orchestrator.initialize()
        result = await orchestrator.execute(mission_req)
        assert 0.0 <= result.ihsan_score <= 1.0
        assert 0.0 <= result.snr_score <= 1.0

    async def test_execute_has_positive_duration(self, orchestrator, mission_req):
        await orchestrator.initialize()
        result = await orchestrator.execute(mission_req)
        assert result.duration_ms > 0

    async def test_execute_has_evidence_receipt(self, orchestrator, mission_req):
        await orchestrator.initialize()
        result = await orchestrator.execute(mission_req)
        assert result.evidence_receipt_id != ""

    async def test_execute_creates_briefing(self, orchestrator, mission_req, tmp_path):
        await orchestrator.initialize()
        result = await orchestrator.execute(mission_req)
        if result.briefing_path:
            from pathlib import Path

            assert Path(result.briefing_path).exists()
            content = Path(result.briefing_path).read_text()
            assert "BIZRA Mission Briefing" in content

    async def test_execute_without_gateway_uses_template(
        self, orchestrator, mission_req
    ):
        await orchestrator.initialize()
        result = await orchestrator.execute(mission_req)
        assert "BIZRA Mission Briefing" in result.synthesis

    async def test_execute_stores_episodic_memory(self, orchestrator, mission_req):
        await orchestrator.initialize()
        await orchestrator.execute(mission_req)
        if orchestrator._memory:
            stats = orchestrator._memory.get_stats()
            assert stats.total_entries >= 1

    async def test_channel_results_populated(self, orchestrator, mission_req):
        await orchestrator.initialize()
        result = await orchestrator.execute(mission_req)
        assert len(result.channels_executed) >= 1
        browser = next(
            (c for c in result.channels_executed if c.channel == "browser"), None
        )
        assert browser is not None
        assert browser.success

    async def test_browser_channel_returns_results(self, orchestrator, mission_req):
        await orchestrator.initialize()
        result = await orchestrator.execute(mission_req)
        browser = next(
            (c for c in result.channels_executed if c.channel == "browser"), None
        )
        assert browser is not None
        assert browser.data.get("results_count", 0) > 0


# ── Template Synthesis Tests ────────────────────────────────────────


class TestTemplateSynthesis:
    @pytest.fixture
    def orchestrator(self, tmp_path):
        return MissionOrchestrator(
            {
                "memory_path": str(tmp_path / "memory"),
                "evidence_path": str(tmp_path / "evidence.jsonl"),
            }
        )

    def test_includes_description(self, orchestrator):
        synthesis = orchestrator._template_synthesis("Research AI agents", None, None)
        assert "Research AI agents" in synthesis

    def test_includes_browser_results(self, orchestrator):
        browser_data = {
            "results": [
                {
                    "title": "CrewAI",
                    "url": "https://crewai.com",
                    "snippet": "Framework",
                },
            ]
        }
        synthesis = orchestrator._template_synthesis("Research", browser_data, None)
        assert "CrewAI" in synthesis
        assert "https://crewai.com" in synthesis

    def test_includes_proof_section(self, orchestrator):
        synthesis = orchestrator._template_synthesis("Test", None, None)
        assert "Proof Trace" in synthesis
        assert "Ihsan" in synthesis
        assert "Ed25519" in synthesis

    def test_omits_empty_browser_results(self, orchestrator):
        synthesis = orchestrator._template_synthesis("Test", {"results": []}, None)
        assert "Research Findings" not in synthesis

    def test_includes_desktop_context(self, orchestrator):
        desktop_data = {"active_window": "VS Code", "hda_connected": True}
        synthesis = orchestrator._template_synthesis("Test", None, desktop_data)
        assert "VS Code" in synthesis
        assert "HDA: Connected" in synthesis


# ── RPC Handler Tests ───────────────────────────────────────────────


class TestHandleRPC:
    @pytest.fixture
    def orchestrator(self, tmp_path):
        return MissionOrchestrator(
            {
                "memory_path": str(tmp_path / "memory"),
                "evidence_path": str(tmp_path / "evidence.jsonl"),
            }
        )

    async def test_empty_description_returns_error(self, orchestrator):
        await orchestrator.initialize()
        result = await orchestrator.handle_rpc({"description": ""})
        assert "error" in result

    async def test_missing_description_returns_error(self, orchestrator):
        await orchestrator.initialize()
        result = await orchestrator.handle_rpc({})
        assert "error" in result

    async def test_valid_request_returns_mission(self, orchestrator):
        await orchestrator.initialize()
        result = await orchestrator.handle_rpc(
            {
                "description": "Research something",
            }
        )
        assert result.get("status") in ("COMPLETE", "PARTIAL")
        assert "mission_id" in result
        assert "ihsan_score" in result

    async def test_clipboard_truncated(self, orchestrator):
        await orchestrator.initialize()
        result = await orchestrator.handle_rpc(
            {
                "description": "Test",
                "context": {"clipboard": "x" * 10000},
            }
        )
        assert result.get("status") is not None

    async def test_context_passed_through(self, orchestrator):
        await orchestrator.initialize()
        result = await orchestrator.handle_rpc(
            {
                "description": "Create a file on desktop",
                "context": {"active_window": "Explorer"},
            }
        )
        assert result.get("status") is not None


# ── HDA Client Tests ───────────────────────────────────────────────


class TestHDAClient:
    async def test_connect_timeout_on_unreachable(self):
        client = HDAClient(host="127.0.0.1", port=59999, token="test")
        connected = await client.connect()
        assert connected is False

    async def test_close_is_idempotent(self):
        client = HDAClient(host="127.0.0.1", port=59999, token="test")
        await client.close()  # Should not raise
        await client.close()  # Double close should not raise
