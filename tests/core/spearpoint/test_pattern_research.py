"""
Pattern-Aware Research Pipeline Tests — Sci-Reasoning Integration
=================================================================

Tests for `research_with_pattern()` in AutoResearcher and
`research_pattern()` in SpearpointOrchestrator, verifying the
integration of Li et al. (2025) thinking patterns into the
spearpoint evaluation pipeline.

Standing on Giants: Li et al. (Sci-Reasoning), Boyd (OODA)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from core.spearpoint.auto_evaluator import (
    AutoEvaluator,
    EvaluationResult,
    ExperimentDesign,
    TierDecision,
    Verdict,
)
from core.spearpoint.auto_researcher import (
    AutoResearcher,
    ResearchOutcome,
    ResearchResult,
)
from core.spearpoint.config import SpearpointConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _mock_evaluator(verdict: Verdict = Verdict.SUPPORTED) -> MagicMock:
    """Create a mock AutoEvaluator returning the given verdict."""
    evaluator = MagicMock(spec=AutoEvaluator)

    # Use MagicMock for tier_decision — we're testing research routing, not tier internals
    tier_decision = MagicMock(spec=TierDecision)
    tier_decision.to_dict.return_value = {"tier": "operational"}

    result = MagicMock(spec=EvaluationResult)
    result.verdict = verdict
    result.credibility_score = 0.92
    result.tier_decision = tier_decision
    result.clear_score = 0.90
    result.ihsan_score = 0.96
    result.guardrail_summary = {"passed": True}
    result.reason_codes = []
    result.receipt_hash = "abc123"
    result.mission_id = "mission_test"
    result.to_dict.return_value = {
        "evaluation_id": "eval_test",
        "verdict": verdict.value,
        "credibility_score": 0.92,
    }
    evaluator.evaluate.return_value = result
    evaluator.ledger = MagicMock()
    evaluator.get_statistics.return_value = {"total_evaluations": 0}
    return evaluator


def _mock_bridge(num_seeds: int = 2) -> MagicMock:
    """Create a mock SciReasoningBridge returning synthetic seeds."""
    bridge = MagicMock()
    bridge._loaded = True

    # Mock taxonomy pattern
    pattern = MagicMock()
    pattern.name = "Gap-Driven Reframing"
    pattern.cognitive_move = "Identify a gap and reframe the problem"
    bridge.taxonomy.get.return_value = pattern

    # Mock seed hypotheses
    seeds = []
    for i in range(num_seeds):
        seeds.append({
            "hypothesis_type": "pattern_exemplar",
            "pattern_id": "P01",
            "pattern_name": "Gap-Driven Reframing",
            "cognitive_move": "Identify a gap and reframe the problem",
            "exemplar_title": f"Test Paper {i+1}",
            "exemplar_conference": "NeurIPS",
            "exemplar_year": 2024,
            "exemplar_reasoning": "Novel reframing of the problem",
            "complementary_patterns": [
                {"id": "P02", "cooccurrence": 15}
            ],
            "learnable_insight": "Reframe the search space",
        })
    bridge.seed_hypotheses.return_value = seeds

    return bridge


# ---------------------------------------------------------------------------
# AutoResearcher.research_with_pattern() Tests
# ---------------------------------------------------------------------------


class TestResearchWithPattern:
    """Test AutoResearcher.research_with_pattern() method."""

    def test_returns_results_for_valid_pattern(self):
        """Valid pattern ID produces evaluation results."""
        evaluator = _mock_evaluator(Verdict.SUPPORTED)
        bridge = _mock_bridge(num_seeds=2)
        researcher = AutoResearcher(
            evaluator=evaluator,
            sci_reasoning_bridge=bridge,
        )

        results = researcher.research_with_pattern(
            pattern_id="P01",
            claim_context="Optimize inference latency",
            mission_id="mission_test",
            top_k=2,
        )

        assert len(results) == 2
        for r in results:
            assert isinstance(r, ResearchResult)
            assert r.outcome == ResearchOutcome.APPROVED
            assert r.mission_id == "mission_test"
            assert "P01" in r.reason

    def test_rejected_verdict_propagates(self):
        """REJECTED evaluation verdict produces REJECTED research outcome."""
        evaluator = _mock_evaluator(Verdict.REJECTED)
        evaluator.evaluate.return_value.reason_codes = ["LOW_QUALITY"]
        bridge = _mock_bridge(num_seeds=1)
        researcher = AutoResearcher(
            evaluator=evaluator,
            sci_reasoning_bridge=bridge,
        )

        results = researcher.research_with_pattern(
            pattern_id="P01",
            mission_id="mission_rej",
        )

        assert len(results) == 1
        assert results[0].outcome == ResearchOutcome.REJECTED

    def test_inconclusive_verdict_propagates(self):
        """INCONCLUSIVE evaluation verdict produces INCONCLUSIVE research outcome."""
        evaluator = _mock_evaluator(Verdict.INCONCLUSIVE)
        bridge = _mock_bridge(num_seeds=1)
        researcher = AutoResearcher(
            evaluator=evaluator,
            sci_reasoning_bridge=bridge,
        )

        results = researcher.research_with_pattern(
            pattern_id="P01",
        )

        assert len(results) == 1
        assert results[0].outcome == ResearchOutcome.INCONCLUSIVE

    def test_invalid_pattern_returns_no_hypotheses(self):
        """Invalid pattern ID returns NO_HYPOTHESES outcome."""
        evaluator = _mock_evaluator()
        bridge = _mock_bridge()
        researcher = AutoResearcher(
            evaluator=evaluator,
            sci_reasoning_bridge=bridge,
        )

        results = researcher.research_with_pattern(
            pattern_id="INVALID_PATTERN",
            mission_id="mission_bad",
        )

        assert len(results) == 1
        assert results[0].outcome == ResearchOutcome.NO_HYPOTHESES
        assert "Unknown pattern ID" in results[0].reason

    def test_no_seeds_returns_no_hypotheses(self):
        """Pattern with no exemplars returns NO_HYPOTHESES."""
        evaluator = _mock_evaluator()
        bridge = _mock_bridge(num_seeds=0)
        bridge.seed_hypotheses.return_value = []
        researcher = AutoResearcher(
            evaluator=evaluator,
            sci_reasoning_bridge=bridge,
        )

        results = researcher.research_with_pattern(
            pattern_id="P01",
        )

        assert len(results) == 1
        assert results[0].outcome == ResearchOutcome.NO_HYPOTHESES
        assert "No exemplars found" in results[0].reason

    def test_evaluator_called_with_pattern_claim(self):
        """Evaluator receives claims enriched with pattern metadata."""
        evaluator = _mock_evaluator()
        bridge = _mock_bridge(num_seeds=1)
        researcher = AutoResearcher(
            evaluator=evaluator,
            sci_reasoning_bridge=bridge,
        )

        researcher.research_with_pattern(
            pattern_id="P01",
            claim_context="Reduce memory usage",
        )

        call_args = evaluator.evaluate.call_args
        claim = call_args.kwargs.get("claim", call_args.args[0] if call_args.args else "")
        assert "Gap-Driven Reframing" in claim
        assert "Test Paper 1" in claim
        assert "Reduce memory usage" in claim

    def test_lazy_bridge_loading(self):
        """Bridge is lazily loaded when not provided at init."""
        evaluator = _mock_evaluator()
        researcher = AutoResearcher(evaluator=evaluator)

        # Bridge is None initially
        assert researcher._sci_bridge is None

        # Calling research_with_pattern should try to load it
        # Patch at the source so the dynamic import fails
        with patch.dict(
            "sys.modules",
            {"core.bridges.sci_reasoning_bridge": None},
        ):
            results = researcher.research_with_pattern(pattern_id="P01")
            assert len(results) == 1
            assert results[0].outcome == ResearchOutcome.NO_HYPOTHESES
            assert "not available" in results[0].reason

    def test_statistics_reflect_pattern_research(self):
        """get_statistics reports sci_reasoning_available correctly."""
        evaluator = _mock_evaluator()
        bridge = _mock_bridge()
        researcher_with = AutoResearcher(
            evaluator=evaluator,
            sci_reasoning_bridge=bridge,
        )
        researcher_without = AutoResearcher(evaluator=evaluator)

        assert researcher_with.get_statistics()["sci_reasoning_available"] is True
        assert researcher_without.get_statistics()["sci_reasoning_available"] is False

    def test_receipt_emitted_for_each_result(self):
        """Each research result gets a receipt hash."""
        evaluator = _mock_evaluator()
        bridge = _mock_bridge(num_seeds=2)
        researcher = AutoResearcher(
            evaluator=evaluator,
            sci_reasoning_bridge=bridge,
        )

        with patch.object(
            researcher, "_emit_research_receipt", return_value="receipt_hash_123"
        ):
            results = researcher.research_with_pattern(
                pattern_id="P01",
            )

            assert len(results) == 2
            for r in results:
                assert r.receipt_hash == "receipt_hash_123"

    def test_top_k_limits_results(self):
        """top_k parameter limits the number of seeds evaluated."""
        evaluator = _mock_evaluator()
        bridge = _mock_bridge(num_seeds=5)
        researcher = AutoResearcher(
            evaluator=evaluator,
            sci_reasoning_bridge=bridge,
        )

        # Bridge configured to return 5, but we only call with top_k=2
        researcher.research_with_pattern(pattern_id="P01", top_k=2)

        bridge.seed_hypotheses.assert_called_once()
        call_args = bridge.seed_hypotheses.call_args
        assert call_args[1].get("top_k", call_args[0][1] if len(call_args[0]) > 1 else None) == 2

    def test_all_15_patterns_accepted(self):
        """All 15 pattern IDs are valid inputs."""
        evaluator = _mock_evaluator()
        bridge = _mock_bridge(num_seeds=1)
        researcher = AutoResearcher(
            evaluator=evaluator,
            sci_reasoning_bridge=bridge,
        )

        for i in range(1, 16):
            pid = f"P{i:02d}"
            results = researcher.research_with_pattern(pattern_id=pid)
            # Should not fail with invalid pattern
            assert len(results) >= 1
            assert results[0].outcome != ResearchOutcome.NO_HYPOTHESES or "Unknown" not in results[0].reason


# ---------------------------------------------------------------------------
# SpearpointOrchestrator.research_pattern() Tests
# ---------------------------------------------------------------------------


class TestOrchestratorResearchPattern:
    """Test SpearpointOrchestrator.research_pattern() routing."""

    def test_research_pattern_returns_mission_result(self):
        """research_pattern returns a MissionResult."""
        from core.spearpoint.config import MissionType
        from core.spearpoint.orchestrator import MissionResult, SpearpointOrchestrator

        evaluator = _mock_evaluator()
        bridge = _mock_bridge(num_seeds=1)
        researcher = AutoResearcher(
            evaluator=evaluator,
            sci_reasoning_bridge=bridge,
        )

        orch = SpearpointOrchestrator(
            evaluator=evaluator,
            researcher=researcher,
        )

        result = orch.research_pattern(
            pattern_id="P01",
            claim_context="Test context",
            top_k=1,
        )

        assert isinstance(result, MissionResult)
        assert result.mission_type == MissionType.IMPROVE
        assert len(result.research_results) >= 1

    def test_research_pattern_tracks_in_history(self):
        """research_pattern results appear in mission history."""
        from core.spearpoint.orchestrator import SpearpointOrchestrator

        evaluator = _mock_evaluator()
        bridge = _mock_bridge(num_seeds=1)
        researcher = AutoResearcher(
            evaluator=evaluator,
            sci_reasoning_bridge=bridge,
        )

        orch = SpearpointOrchestrator(
            evaluator=evaluator,
            researcher=researcher,
        )

        orch.research_pattern(pattern_id="P01")

        history = orch.get_mission_history(limit=5)
        assert len(history) == 1
        assert history[0]["mission_type"] == "improve"

    def test_research_pattern_error_handling(self):
        """research_pattern handles exceptions gracefully."""
        from core.spearpoint.orchestrator import SpearpointOrchestrator

        evaluator = _mock_evaluator()
        researcher = MagicMock()
        researcher.research_with_pattern.side_effect = RuntimeError("boom")

        orch = SpearpointOrchestrator(
            evaluator=evaluator,
            researcher=researcher,
        )

        result = orch.research_pattern(pattern_id="P01")
        assert result.success is False
        assert result.error == "Pattern research failed"

    def test_research_pattern_custom_mission_id(self):
        """Custom mission_id is preserved."""
        from core.spearpoint.orchestrator import SpearpointOrchestrator

        evaluator = _mock_evaluator()
        bridge = _mock_bridge(num_seeds=1)
        researcher = AutoResearcher(
            evaluator=evaluator,
            sci_reasoning_bridge=bridge,
        )

        orch = SpearpointOrchestrator(
            evaluator=evaluator,
            researcher=researcher,
        )

        result = orch.research_pattern(
            pattern_id="P01",
            mission_id="custom_mission_42",
        )

        assert result.mission_id == "custom_mission_42"


# ---------------------------------------------------------------------------
# API Endpoint Tests
# ---------------------------------------------------------------------------


class TestSpearpointPatternEndpoint:
    """POST /v1/spearpoint/pattern API tests."""

    def test_pattern_endpoint_returns_200(self):
        """Pattern endpoint returns result from orchestrator.research_pattern."""
        from starlette.testclient import TestClient

        from core.sovereign.api import create_fastapi_app
        from core.sovereign.runtime_types import RuntimeMetrics
        from core.spearpoint.config import MissionType
        from core.spearpoint.orchestrator import MissionResult

        runtime = MagicMock()
        runtime.metrics = RuntimeMetrics(
            queries_processed=0,
            queries_succeeded=0,
            current_snr_score=0.95,
            current_ihsan_score=0.96,
            avg_query_time_ms=50.0,
        )
        runtime.status.return_value = {
            "health": {"status": "healthy"},
            "identity": {"version": "test"},
            "state": {"running": True},
            "autonomous": {"running": False},
        }

        pattern_result = MissionResult(
            mission_id="mission_pattern_01",
            mission_type=MissionType.IMPROVE,
            success=True,
            research_results=[
                {"outcome": "approved", "pattern_id": "P01"}
            ],
            elapsed_ms=85.0,
        )
        runtime._spearpoint_orchestrator = MagicMock()
        runtime._spearpoint_orchestrator.research_pattern.return_value = pattern_result

        app = create_fastapi_app(runtime)
        client = TestClient(app)

        resp = client.post(
            "/v1/spearpoint/pattern",
            json={"pattern_id": "P01", "claim_context": "Optimize latency"},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["mission_id"] == "mission_pattern_01"
        assert data["success"] is True
        assert data["mission_type"] == "improve"
        assert data["research_count"] == 1

    def test_pattern_endpoint_passes_all_fields(self):
        """All request fields are forwarded to orchestrator.research_pattern()."""
        from starlette.testclient import TestClient

        from core.sovereign.api import create_fastapi_app
        from core.sovereign.runtime_types import RuntimeMetrics
        from core.spearpoint.config import MissionType
        from core.spearpoint.orchestrator import MissionResult

        runtime = MagicMock()
        runtime.metrics = RuntimeMetrics(
            queries_processed=0, queries_succeeded=0,
            current_snr_score=0.95, current_ihsan_score=0.96,
            avg_query_time_ms=50.0,
        )
        runtime.status.return_value = {
            "health": {"status": "healthy"},
            "identity": {"version": "test"},
            "state": {"running": True},
            "autonomous": {"running": False},
        }

        result = MissionResult(
            mission_id="m1",
            mission_type=MissionType.IMPROVE,
            success=True,
            elapsed_ms=10.0,
        )
        runtime._spearpoint_orchestrator = MagicMock()
        runtime._spearpoint_orchestrator.research_pattern.return_value = result

        app = create_fastapi_app(runtime)
        client = TestClient(app)

        client.post(
            "/v1/spearpoint/pattern",
            json={
                "pattern_id": "P02",
                "claim_context": "Cross-domain synthesis",
                "top_k": 5,
            },
        )

        runtime._spearpoint_orchestrator.research_pattern.assert_called_once_with(
            pattern_id="P02",
            claim_context="Cross-domain synthesis",
            top_k=5,
        )

    def test_pattern_endpoint_503_when_orchestrator_missing(self):
        """Returns 503 when orchestrator not wired."""
        from starlette.testclient import TestClient

        from core.sovereign.api import create_fastapi_app
        from core.sovereign.runtime_types import RuntimeMetrics

        runtime = MagicMock()
        runtime.metrics = RuntimeMetrics(
            queries_processed=0, queries_succeeded=0,
            current_snr_score=0.95, current_ihsan_score=0.96,
            avg_query_time_ms=50.0,
        )
        runtime.status.return_value = {
            "health": {"status": "healthy"},
            "identity": {"version": "test"},
            "state": {"running": True},
            "autonomous": {"running": False},
        }
        runtime._spearpoint_orchestrator = None

        app = create_fastapi_app(runtime)
        client = TestClient(app)

        resp = client.post(
            "/v1/spearpoint/pattern",
            json={"pattern_id": "P01"},
        )
        assert resp.status_code == 503
        assert "not available" in resp.json()["error"]

    def test_pattern_endpoint_defaults(self):
        """Pattern endpoint works with only required fields."""
        from starlette.testclient import TestClient

        from core.sovereign.api import create_fastapi_app
        from core.sovereign.runtime_types import RuntimeMetrics
        from core.spearpoint.config import MissionType
        from core.spearpoint.orchestrator import MissionResult

        runtime = MagicMock()
        runtime.metrics = RuntimeMetrics(
            queries_processed=0, queries_succeeded=0,
            current_snr_score=0.95, current_ihsan_score=0.96,
            avg_query_time_ms=50.0,
        )
        runtime.status.return_value = {
            "health": {"status": "healthy"},
            "identity": {"version": "test"},
            "state": {"running": True},
            "autonomous": {"running": False},
        }

        result = MissionResult(
            mission_id="m1",
            mission_type=MissionType.IMPROVE,
            success=True,
            elapsed_ms=10.0,
        )
        runtime._spearpoint_orchestrator = MagicMock()
        runtime._spearpoint_orchestrator.research_pattern.return_value = result

        app = create_fastapi_app(runtime)
        client = TestClient(app)

        resp = client.post(
            "/v1/spearpoint/pattern",
            json={"pattern_id": "P01"},
        )
        assert resp.status_code == 200

        # Verify defaults were passed
        runtime._spearpoint_orchestrator.research_pattern.assert_called_once_with(
            pattern_id="P01",
            claim_context="",
            top_k=3,
        )


__all__ = [
    "TestResearchWithPattern",
    "TestOrchestratorResearchPattern",
    "TestSpearpointPatternEndpoint",
]
