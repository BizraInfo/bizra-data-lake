"""
Tests for SpearPointPipeline — the unified post-query cockpit.

Verifies that all 7 pipeline steps execute independently with proper
error isolation, and that the SpearPointResult reports per-step status.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.sovereign.spearpoint_pipeline import (
    SpearPointPipeline,
    SpearPointResult,
    StepResult,
)


# ---------------------------------------------------------------------------
# Fake SovereignResult / SovereignQuery for testing
# ---------------------------------------------------------------------------


@dataclass
class FakeResult:
    query_id: str = "test-001"
    success: bool = True
    response: str = "The answer is 42."
    reasoning_depth: int = 3
    thoughts: List[str] = field(
        default_factory=lambda: ["thought-1", "thought-2", "thought-3"]
    )
    ihsan_score: float = 0.96
    snr_score: float = 0.92
    snr_ok: bool = True
    validated: bool = False
    validation_passed: bool = True
    processing_time_ms: float = 150.0
    graph_hash: Optional[str] = "abc123def456"
    claim_tags: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    model_used: str = "test-model"


@dataclass
class FakeQuery:
    id: str = "q-001"
    text: str = "What is the meaning of life?"
    user_id: str = "user-001"


@dataclass
class FakeConfig:
    node_id: str = "BIZRA-TEST-NODE"
    ihsan_threshold: float = 0.95
    snr_threshold: float = 0.85


# ---------------------------------------------------------------------------
# Test: Pipeline with no subsystems (all steps skip gracefully)
# ---------------------------------------------------------------------------


class TestPipelineNoSubsystems:
    """Pipeline with no subsystems should skip all steps gracefully."""

    @pytest.fixture
    def pipeline(self):
        return SpearPointPipeline(config=FakeConfig())

    @pytest.mark.asyncio
    async def test_all_steps_skip(self, pipeline):
        result = await pipeline.execute(FakeResult(), FakeQuery())
        assert result.all_passed is True
        assert len(result.steps) == 8
        for step in result.steps:
            assert step.success is True
            assert "skipped" in step.detail

    @pytest.mark.asyncio
    async def test_returns_spearpoint_result(self, pipeline):
        result = await pipeline.execute(FakeResult(), FakeQuery())
        assert isinstance(result, SpearPointResult)
        assert result.total_duration_ms >= 0

    @pytest.mark.asyncio
    async def test_to_dict_serialization(self, pipeline):
        result = await pipeline.execute(FakeResult(), FakeQuery())
        d = result.to_dict()
        assert "all_passed" in d
        assert "steps" in d
        assert len(d["steps"]) == 8
        assert "total_duration_ms" in d


# ---------------------------------------------------------------------------
# Test: Individual step isolation (one failure doesn't block others)
# ---------------------------------------------------------------------------


class TestStepIsolation:
    """Each step is independently error-isolated."""

    @pytest.mark.asyncio
    async def test_evidence_failure_doesnt_block_others(self):
        """If evidence ledger throws, other steps still succeed."""
        mock_ledger = MagicMock()
        # Make emit_receipt raise an exception
        with patch(
            "core.sovereign.spearpoint_pipeline.SpearPointPipeline._step_evidence_receipt"
        ) as mock_step:
            mock_step.return_value = StepResult(
                name="evidence_receipt",
                success=False,
                error="simulated failure",
            )
            pipeline = SpearPointPipeline(
                evidence_ledger=mock_ledger,
                config=FakeConfig(),
            )
            # Override just the one step
            pipeline._step_evidence_receipt = mock_step

            result = await pipeline.execute(FakeResult(), FakeQuery())
            # Evidence failed, but others passed
            assert not result.all_passed
            assert "evidence_receipt" in result.failed_steps
            # All other steps should pass (skipped = success)
            passed = [s for s in result.steps if s.success]
            assert len(passed) == 7

    @pytest.mark.asyncio
    async def test_poi_failure_doesnt_block_judgment(self):
        """PoI failure doesn't prevent judgment telemetry."""
        mock_poi = MagicMock()
        mock_poi.register_contribution.side_effect = RuntimeError("PoI down")

        pipeline = SpearPointPipeline(
            poi_orchestrator=mock_poi,
            config=FakeConfig(),
        )
        result = await pipeline.execute(FakeResult(), FakeQuery())

        # PoI step should fail
        poi_step = next(s for s in result.steps if s.name == "poi_contribution")
        assert not poi_step.success
        assert "PoI down" in (poi_step.error or "")

        # Judgment step should still succeed (skipped due to no telemetry)
        judge_step = next(s for s in result.steps if s.name == "judgment_observe")
        assert judge_step.success


# ---------------------------------------------------------------------------
# Test: Graph artifact step
# ---------------------------------------------------------------------------


class TestGraphArtifactStep:
    """Test graph artifact storage."""

    @pytest.mark.asyncio
    async def test_stores_artifact(self):
        artifacts = {}
        mock_reasoner = MagicMock()
        mock_reasoner.to_artifact.return_value = {"graph": "data", "hash": "abc"}

        pipeline = SpearPointPipeline(
            graph_reasoner=mock_reasoner,
            graph_artifacts=artifacts,
            config=FakeConfig(),
        )
        result = await pipeline.execute(FakeResult(), FakeQuery())
        graph_step = next(s for s in result.steps if s.name == "graph_artifact")
        assert graph_step.success
        assert "q-001" in artifacts
        assert artifacts["q-001"]["hash"] == "abc"

    @pytest.mark.asyncio
    async def test_bounds_cache_at_100(self):
        artifacts = {f"old-{i}": {} for i in range(100)}
        mock_reasoner = MagicMock()
        mock_reasoner.to_artifact.return_value = {"new": True}

        pipeline = SpearPointPipeline(
            graph_reasoner=mock_reasoner,
            graph_artifacts=artifacts,
            config=FakeConfig(),
        )
        await pipeline.execute(FakeResult(), FakeQuery())
        assert len(artifacts) <= 100

    @pytest.mark.asyncio
    async def test_skips_when_no_graph_hash(self):
        mock_reasoner = MagicMock()
        pipeline = SpearPointPipeline(
            graph_reasoner=mock_reasoner,
            config=FakeConfig(),
        )
        r = FakeResult()
        r.graph_hash = None
        result = await pipeline.execute(r, FakeQuery())
        graph_step = next(s for s in result.steps if s.name == "graph_artifact")
        assert graph_step.success
        assert "skipped" in graph_step.detail


# ---------------------------------------------------------------------------
# Test: Judgment telemetry step
# ---------------------------------------------------------------------------


class TestJudgmentStep:
    """Test SJE Phase A observation."""

    @pytest.mark.asyncio
    async def test_promote_on_high_ihsan(self):
        mock_sje = MagicMock()
        pipeline = SpearPointPipeline(
            judgment_telemetry=mock_sje,
            config=FakeConfig(),
        )
        r = FakeResult(ihsan_score=0.97, snr_ok=True)
        await pipeline.execute(r, FakeQuery())
        mock_sje.observe.assert_called_once()

    @pytest.mark.asyncio
    async def test_demote_on_low_snr(self):
        mock_sje = MagicMock()
        pipeline = SpearPointPipeline(
            judgment_telemetry=mock_sje,
            config=FakeConfig(),
        )
        r = FakeResult(snr_ok=False, snr_score=0.4)
        await pipeline.execute(r, FakeQuery())
        mock_sje.observe.assert_called_once()


# ---------------------------------------------------------------------------
# Test: PoI contribution step
# ---------------------------------------------------------------------------


class TestPoIContributionStep:
    """Test PoI registration."""

    @pytest.mark.asyncio
    async def test_registers_on_success(self):
        mock_poi = MagicMock()
        pipeline = SpearPointPipeline(
            poi_orchestrator=mock_poi,
            config=FakeConfig(),
        )
        await pipeline.execute(FakeResult(), FakeQuery())
        mock_poi.register_contribution.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_on_failure(self):
        mock_poi = MagicMock()
        pipeline = SpearPointPipeline(
            poi_orchestrator=mock_poi,
            config=FakeConfig(),
        )
        r = FakeResult(success=False)
        result = await pipeline.execute(r, FakeQuery())
        mock_poi.register_contribution.assert_not_called()
        poi_step = next(s for s in result.steps if s.name == "poi_contribution")
        assert "skipped" in poi_step.detail


# ---------------------------------------------------------------------------
# Test: SpearPointResult properties
# ---------------------------------------------------------------------------


class TestSpearPointResult:
    """Test result aggregation."""

    def test_failed_steps_empty_on_all_pass(self):
        r = SpearPointResult(
            steps=[StepResult(name="a", success=True), StepResult(name="b", success=True)]
        )
        assert r.failed_steps == []
        assert r.all_passed is True

    def test_failed_steps_list(self):
        r = SpearPointResult(
            steps=[
                StepResult(name="a", success=True),
                StepResult(name="b", success=False, error="boom"),
            ]
        )
        assert r.failed_steps == ["b"]

    def test_step_summary_dict(self):
        r = SpearPointResult(
            steps=[
                StepResult(name="x", success=True),
                StepResult(name="y", success=False),
            ]
        )
        assert r.step_summary == {"x": True, "y": False}

    def test_to_dict_excludes_error_on_success(self):
        r = SpearPointResult(
            steps=[StepResult(name="a", success=True, detail="ok")]
        )
        d = r.to_dict()
        assert "error" not in d["steps"][0]

    def test_to_dict_includes_error_on_failure(self):
        r = SpearPointResult(
            steps=[StepResult(name="a", success=False, error="boom")]
        )
        d = r.to_dict()
        assert d["steps"][0]["error"] == "boom"


# ---------------------------------------------------------------------------
# Test: Living memory step (async)
# ---------------------------------------------------------------------------


class TestLivingMemoryStep:
    """Test async living memory encoding."""

    @pytest.mark.asyncio
    async def test_encodes_memory(self):
        mock_memory = AsyncMock()
        pipeline = SpearPointPipeline(
            living_memory=mock_memory,
            config=FakeConfig(),
        )
        result = await pipeline.execute(FakeResult(), FakeQuery())
        mem_step = next(s for s in result.steps if s.name == "living_memory")
        assert mem_step.success
        assert mem_step.detail == "encoded"
        mock_memory.encode.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_skips_on_empty_response(self):
        mock_memory = AsyncMock()
        pipeline = SpearPointPipeline(
            living_memory=mock_memory,
            config=FakeConfig(),
        )
        r = FakeResult(response="")
        result = await pipeline.execute(r, FakeQuery())
        mem_step = next(s for s in result.steps if s.name == "living_memory")
        assert "skipped" in mem_step.detail
        mock_memory.encode.assert_not_awaited()


# ---------------------------------------------------------------------------
# Test: SAT health check step
# ---------------------------------------------------------------------------


class TestSATHealthStep:
    """Test SAT ecosystem health observation."""

    @pytest.mark.asyncio
    async def test_skips_when_no_sat_controller(self):
        pipeline = SpearPointPipeline(config=FakeConfig())
        result = await pipeline.execute(FakeResult(), FakeQuery())
        sat_step = next(s for s in result.steps if s.name == "sat_health")
        assert sat_step.success
        assert "skipped" in sat_step.detail

    @pytest.mark.asyncio
    async def test_reports_healthy_gini(self):
        mock_sat = MagicMock()
        mock_snapshot = MagicMock()
        mock_snapshot.gini_coefficient = 0.25
        mock_sat.get_urp_snapshot.return_value = mock_snapshot
        mock_sat.config.gini_rebalance_threshold = 0.45

        pipeline = SpearPointPipeline(
            sat_controller=mock_sat,
            config=FakeConfig(),
        )
        result = await pipeline.execute(FakeResult(), FakeQuery())
        sat_step = next(s for s in result.steps if s.name == "sat_health")
        assert sat_step.success
        assert "healthy" in sat_step.detail
        assert "0.250" in sat_step.detail

    @pytest.mark.asyncio
    async def test_warns_on_high_gini(self):
        mock_sat = MagicMock()
        mock_snapshot = MagicMock()
        mock_snapshot.gini_coefficient = 0.60
        mock_sat.get_urp_snapshot.return_value = mock_snapshot
        mock_sat.config.gini_rebalance_threshold = 0.45

        pipeline = SpearPointPipeline(
            sat_controller=mock_sat,
            config=FakeConfig(),
        )
        result = await pipeline.execute(FakeResult(), FakeQuery())
        sat_step = next(s for s in result.steps if s.name == "sat_health")
        # Observational — always succeeds even on high Gini
        assert sat_step.success
        assert "warning" in sat_step.detail
        assert "0.600" in sat_step.detail
