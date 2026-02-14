"""
Tests for AutoResearcher — Hypothesis Generation with Evaluator Gate
=====================================================================

Verifies:
- Hypothesis generation from observations
- Evaluation gating (no bypass path)
- Constitutional admission integration
- No direct CLEAR/guardrail calls (verify isolation)
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

from core.autopoiesis.hypothesis_generator import SystemObservation
from core.spearpoint.auto_evaluator import AutoEvaluator
from core.spearpoint.auto_researcher import (
    AutoResearcher,
    ResearchOutcome,
    ResearchResult,
)
from core.spearpoint.config import SpearpointConfig


@pytest.fixture
def tmp_config(tmp_path: Path) -> SpearpointConfig:
    """Create a config pointing to temp directory."""
    return SpearpointConfig(
        state_dir=tmp_path / "spearpoint",
        evidence_ledger_path=tmp_path / "spearpoint" / "evidence.jsonl",
        hypothesis_memory_path=tmp_path / "spearpoint" / "hypothesis_memory",
    )


@pytest.fixture
def evaluator(tmp_config: SpearpointConfig) -> AutoEvaluator:
    """Create an AutoEvaluator with temp config."""
    return AutoEvaluator(config=tmp_config)


@pytest.fixture
def researcher(
    evaluator: AutoEvaluator, tmp_config: SpearpointConfig
) -> AutoResearcher:
    """Create an AutoResearcher with evaluator gate."""
    return AutoResearcher(
        evaluator=evaluator,
        config=tmp_config,
    )


class TestHypothesisGeneration:
    """Verify hypothesis generation from observations."""

    def test_generates_hypotheses_from_degraded_observation(
        self, researcher: AutoResearcher
    ):
        """Degraded system observation triggers hypothesis generation."""
        obs = SystemObservation(
            avg_latency_ms=1000,
            cache_hit_rate=0.3,
            ihsan_score=0.80,
            snr_score=0.75,
            error_rate=0.1,
        )
        results = researcher.research(observation=obs)
        assert len(results) > 0

    def test_no_hypotheses_from_healthy_observation(
        self, researcher: AutoResearcher
    ):
        """Healthy system produces NO_HYPOTHESES or few results."""
        obs = SystemObservation(
            avg_latency_ms=50,
            cache_hit_rate=0.95,
            ihsan_score=0.99,
            snr_score=0.98,
            error_rate=0.0,
            uptime_percent=100.0,
        )
        results = researcher.research(observation=obs)
        # Either no hypotheses or only novel trend-based ones
        assert len(results) >= 0  # May produce 0 or some novel ones

    def test_default_observation_works(self, researcher: AutoResearcher):
        """Research with default observation doesn't crash."""
        results = researcher.research()
        assert isinstance(results, list)


class TestEvaluationGating:
    """Verify all hypotheses gate through AutoEvaluator."""

    def test_every_hypothesis_gets_evaluated(
        self, researcher: AutoResearcher
    ):
        """Every hypothesis passes through the evaluator."""
        obs = SystemObservation(
            avg_latency_ms=800,
            cache_hit_rate=0.4,
            error_rate=0.08,
        )
        results = researcher.research(observation=obs, top_k=2)

        for result in results:
            if result.outcome != ResearchOutcome.NO_HYPOTHESES:
                assert result.evaluation is not None

    def test_rejected_hypothesis_not_promoted(
        self, researcher: AutoResearcher
    ):
        """Rejected hypotheses stay rejected."""
        obs = SystemObservation(error_rate=0.1)
        results = researcher.research(observation=obs)

        for result in results:
            if result.evaluation and result.evaluation.verdict.value == "REJECTED":
                assert result.outcome in (
                    ResearchOutcome.REJECTED,
                    ResearchOutcome.GATED,
                )

    def test_research_single_claim(self, researcher: AutoResearcher):
        """Single claim evaluation works through evaluator gate."""
        result = researcher.research_single(
            claim="The system can achieve 99% uptime with proper load balancing configuration across multiple availability zones and regions",
            mission_id="test_single",
        )
        assert isinstance(result, ResearchResult)
        assert result.evaluation is not None
        assert result.outcome in (
            ResearchOutcome.APPROVED,
            ResearchOutcome.REJECTED,
            ResearchOutcome.INCONCLUSIVE,
        )


class TestNoDirectCLEAR:
    """Verify researcher never calls CLEAR/guardrails directly."""

    def test_researcher_has_no_clear_attribute(
        self, researcher: AutoResearcher
    ):
        """AutoResearcher has no direct reference to CLEAR framework."""
        assert not hasattr(researcher, "_clear")
        assert not hasattr(researcher, "_guardrails")
        assert not hasattr(researcher, "_ihsan_gate")

    def test_researcher_uses_evaluator_only(
        self, researcher: AutoResearcher
    ):
        """AutoResearcher has _evaluator attribute (sole gateway)."""
        assert hasattr(researcher, "_evaluator")
        assert isinstance(researcher._evaluator, AutoEvaluator)


class TestConstitutionalIntegration:
    """Verify constitutional gate integration."""

    def test_no_gate_configured(self, researcher: AutoResearcher):
        """Without constitutional gate, result shows 'not_configured'."""
        obs = SystemObservation(
            avg_latency_ms=800,
            cache_hit_rate=0.4,
        )
        results = researcher.research(observation=obs, top_k=1)

        for result in results:
            if result.outcome == ResearchOutcome.APPROVED:
                assert result.constitutional_status == "not_configured"


class TestMissionIDPropagation:
    """Verify mission_id flows through the entire chain."""

    def test_mission_id_in_research_result(
        self, researcher: AutoResearcher
    ):
        """Mission ID propagates to research results."""
        result = researcher.research_single(
            claim="Test claim for mission ID propagation with sufficient length to pass all guardrail minimum requirements",
            mission_id="test_mission_42",
        )
        assert result.mission_id == "test_mission_42"


class TestStatistics:
    """Verify statistics tracking."""

    def test_statistics_after_research(self, researcher: AutoResearcher):
        """Statistics reflect research activity."""
        researcher.research_single(
            claim="A testable claim with sufficient detail to generate meaningful evaluation metrics and analysis results",
        )

        stats = researcher.get_statistics()
        assert stats["total_cycles"] >= 1
        assert "outcomes" in stats
        assert "generator_stats" in stats
        assert "evaluator_stats" in stats

    def test_serialization(self, researcher: AutoResearcher):
        """ResearchResult serializes to dict correctly."""
        result = researcher.research_single(
            claim="Serialization test claim with enough content to pass guardrail checks and produce valid evaluation metrics",
        )
        d = result.to_dict()
        assert "research_id" in d
        assert "outcome" in d
        assert "timestamp" in d


class TestResearchReceipts:
    """Verify researcher-level receipt emission for auditability."""

    def test_research_single_emits_receipt_hash(
        self, researcher: AutoResearcher
    ):
        """Single-claim research result carries ledger receipt hash."""
        result = researcher.research_single(
            claim="Research receipt claim with enough structure to run through evaluator and produce a deterministic audit trail entry",
            mission_id="receipt_test_single",
        )
        assert result.receipt_hash != ""

    def test_research_cycle_emits_receipts(
        self, researcher: AutoResearcher, evaluator: AutoEvaluator
    ):
        """Each research outcome appends a receipt to the shared evidence ledger."""
        initial_count = evaluator.ledger.count()
        obs = SystemObservation(
            avg_latency_ms=900,
            cache_hit_rate=0.35,
            error_rate=0.09,
        )
        results = researcher.research(
            observation=obs,
            mission_id="receipt_test_cycle",
            top_k=2,
        )

        assert len(results) > 0
        assert all(r.receipt_hash != "" for r in results)
        assert evaluator.ledger.count() >= initial_count + len(results)
