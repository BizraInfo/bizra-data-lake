"""
Tests for SpearpointOrchestrator explicit handlers and heartbeat wiring.
"""

from pathlib import Path

import pytest

from core.autopoiesis.hypothesis_generator import SystemObservation
from core.spearpoint.config import MissionType, SpearpointConfig
from core.spearpoint.orchestrator import SpearpointOrchestrator


@pytest.fixture
def tmp_config(tmp_path: Path) -> SpearpointConfig:
    """Create orchestrator config with fast loop timing for tests."""
    return SpearpointConfig(
        state_dir=tmp_path / "spearpoint",
        evidence_ledger_path=tmp_path / "spearpoint" / "evidence.jsonl",
        hypothesis_memory_path=tmp_path / "spearpoint" / "hypothesis_memory",
        loop_interval_seconds=0.01,
        circuit_breaker_backoff_seconds=0.01,
    )


@pytest.fixture
def orchestrator(tmp_config: SpearpointConfig) -> SpearpointOrchestrator:
    """Create a ready orchestrator."""
    return SpearpointOrchestrator(config=tmp_config)


class TestExplicitHandlers:
    """Public handler methods map to correct mission pathways."""

    def test_reproduce_handler(self, orchestrator: SpearpointOrchestrator):
        """reproduce() routes to evaluator path and returns mission result."""
        result = orchestrator.reproduce(
            claim="The service maintains reliability above threshold with deterministic evaluation evidence across multiple windows",
            response="The service maintains reliability above threshold with deterministic evaluation evidence across repeated windows and checks",
            metrics={"accuracy": 0.95, "task_completion": 0.92},
            mission_id="reproduce_handler_test",
        )

        assert result.mission_id == "reproduce_handler_test"
        assert result.mission_type == MissionType.REPRODUCE
        assert len(result.evaluation_results) == 1

    def test_improve_handler(self, orchestrator: SpearpointOrchestrator):
        """improve() routes to researcher path and returns research results."""
        observation = SystemObservation(
            avg_latency_ms=900,
            cache_hit_rate=0.3,
            error_rate=0.09,
        )
        result = orchestrator.improve(
            observation=observation,
            top_k=2,
            mission_id="improve_handler_test",
        )

        assert result.mission_id == "improve_handler_test"
        assert result.mission_type == MissionType.IMPROVE
        assert len(result.research_results) >= 1


class TestHeartbeat:
    """Recursive heartbeat loop is callable from orchestrator."""

    @pytest.mark.asyncio
    async def test_run_heartbeat(self, orchestrator: SpearpointOrchestrator):
        """run_heartbeat executes configured number of cycles."""
        metrics = await orchestrator.run_heartbeat(max_cycles=1)
        assert metrics.cycles_completed == 1
        assert orchestrator.get_statistics()["heartbeat"] is not None

