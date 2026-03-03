"""Integration tests for the mission pipeline — full flow without AHK/LLM."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from core.sovereign.mission import (
    DesktopContext,
    MissionOrchestrator,
    MissionRequest,
)


@pytest.fixture
async def pipeline(tmp_path):
    """Boot a complete mission pipeline with isolated storage."""
    orch = MissionOrchestrator(
        {
            "memory_path": str(tmp_path / "memory"),
            "evidence_path": str(tmp_path / "evidence.jsonl"),
            "hda_port": 59999,
        }
    )
    await orch.initialize()
    return orch


def _make_request(mission_id: str, description: str) -> MissionRequest:
    return MissionRequest(
        mission_id=mission_id,
        description=description,
        context=DesktopContext("Test Window", "", {}),
        timestamp=time.time(),
        source="test",
    )


class TestMissionPipeline:
    """Full pipeline tests at Level 0 (no AHK, no LLM)."""

    async def test_browser_channel_returns_results(self, pipeline):
        request = _make_request("c" * 32, "Research distributed AI consensus")
        result = await pipeline.execute(request)
        browser = next(
            (c for c in result.channels_executed if c.channel == "browser"), None
        )
        assert browser is not None
        assert browser.success is True
        assert browser.data.get("results_count", 0) > 0

    async def test_desktop_channel_fallback_without_hda(self, pipeline):
        request = _make_request("d" * 32, "Create a summary file on desktop")
        result = await pipeline.execute(request)
        desktop = next(
            (c for c in result.channels_executed if c.channel == "desktop"), None
        )
        if desktop:
            assert desktop.data.get("fallback") == "python_file_io"

    async def test_briefing_file_created(self, pipeline):
        request = _make_request("e" * 32, "Research AI frameworks")
        result = await pipeline.execute(request)
        assert result.briefing_path is not None
        content = Path(result.briefing_path).read_text()
        assert "BIZRA Mission Briefing" in content

    async def test_evidence_chain_valid(self, pipeline, tmp_path):
        request = _make_request("f" * 32, "Research something")
        await pipeline.execute(request)
        if pipeline._evidence_ledger:
            valid, errors = pipeline._evidence_ledger.verify_chain()
            assert valid, f"Evidence chain errors: {errors}"

    async def test_multiple_missions_chain_evidence(self, pipeline):
        for i in range(3):
            request = _make_request(f"{i:032x}", f"Research topic {i}")
            await pipeline.execute(request)

        if pipeline._evidence_ledger:
            valid, errors = pipeline._evidence_ledger.verify_chain()
            assert valid, f"Chain errors: {errors}"

    async def test_memory_stored_after_mission(self, pipeline):
        request = _make_request("a1" * 16, "Research AI agent frameworks")
        await pipeline.execute(request)
        if pipeline._memory:
            stats = pipeline._memory.get_stats()
            assert stats.total_entries >= 1

    async def test_snr_scores_in_valid_range(self, pipeline):
        request = _make_request("b2" * 16, "Research consensus mechanisms")
        result = await pipeline.execute(request)
        assert 0.0 <= result.snr_score <= 1.0
        assert 0.0 <= result.ihsan_score <= 1.0

    async def test_channel_failure_isolated(self, pipeline):
        """One channel failing doesn't crash the entire mission."""
        # Force dispatcher to None so decompose falls back
        pipeline._dispatcher = None
        request = _make_request("c3" * 16, "Research something")
        result = await pipeline.execute(request)
        assert result.status in ("COMPLETE", "PARTIAL")

    async def test_rpc_end_to_end(self, pipeline):
        """RPC handler produces valid response structure."""
        result = await pipeline.handle_rpc(
            {
                "description": "Research the latest AI agent frameworks",
                "context": {"active_window": "Chrome"},
            }
        )
        assert "error" not in result
        assert result["status"] in ("COMPLETE", "PARTIAL")
        assert "mission_id" in result
        assert "ihsan_score" in result
        assert "snr_score" in result
        assert "channels" in result
        assert len(result["channels"]) >= 1

    async def test_mission_with_create_keyword_triggers_desktop(self, pipeline):
        """Description with 'create' keyword triggers desktop channel."""
        request = _make_request(
            "d4" * 16, "Research AI frameworks and create a briefing"
        )
        result = await pipeline.execute(request)
        channels = [c.channel for c in result.channels_executed]
        assert "browser" in channels
        # "create" keyword should trigger desktop channel
        assert "desktop" in channels
