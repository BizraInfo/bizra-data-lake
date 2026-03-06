"""Tests for BIZRA Mission Pipeline — Complete Heartbeat Lifecycle."""

import os
import sys
import tempfile
import pytest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("BIZRA_CONSTITUTION_PATH",
                       str(Path(__file__).parent.parent / "constitution.toml"))

from mission_pipeline import (
    MissionPipeline, Mission, MissionStatus, PatAgent,
)


@pytest.fixture
def pipeline(tmp_path):
    return MissionPipeline(
        evidence_path=tmp_path / "test_evidence.jsonl",
        cache_path=tmp_path / "test_cache.json",
    )


class TestMissionExecution:
    """End-to-end mission lifecycle tests."""

    def test_simple_mission_completes(self, pipeline):
        m = pipeline.execute("Hello world")
        assert m.status == MissionStatus.COMPLETE

    def test_mission_has_id(self, pipeline):
        m = pipeline.execute("test")
        assert len(m.mission_id) > 0

    def test_mission_has_output(self, pipeline):
        m = pipeline.execute("What is AI?")
        assert len(m.output_text) > 0

    def test_mission_has_classification(self, pipeline):
        m = pipeline.execute("test query")
        assert m.classification is not None
        assert m.classification.tier is not None

    def test_mission_has_ihsan_score(self, pipeline):
        m = pipeline.execute("test")
        assert m.ihsan_score is not None
        assert 0.0 <= m.ihsan_score.composite <= 1.0

    def test_mission_has_snr(self, pipeline):
        m = pipeline.execute("test")
        assert m.mission_snr is not None
        assert m.mission_snr.snr_normalized >= 0

    def test_mission_has_evidence_receipt(self, pipeline):
        m = pipeline.execute("test")
        assert m.evidence_receipt is not None
        assert len(m.evidence_receipt.receipt_id) == 64

    def test_mission_has_agent_trace(self, pipeline):
        m = pipeline.execute("test")
        assert len(m.agent_trace) >= 1

    def test_timing_is_tracked(self, pipeline):
        m = pipeline.execute("test")
        assert m.total_ms > 0
        assert m.classify_ms >= 0
        assert m.execute_ms >= 0
        assert m.gate_ms >= 0


class TestTrustCompiler:
    """PAT agent pipeline tests — trust must increase monotonically."""

    def test_seven_agents_in_pipeline(self, pipeline):
        assert len(pipeline.agents) == 7

    def test_agent_names_match_constitution(self, pipeline):
        names = [a.name for a in pipeline.agents]
        assert names == [
            "Planner", "Researcher", "Coder",
            "Evaluator", "Ethicist", "Publisher", "Integrator",
        ]

    def test_trust_stages_sequential(self, pipeline):
        stages = [a.trust_stage for a in pipeline.agents]
        expected = [
            "abstracting", "gathering", "executing",
            "attesting", "certifying", "publishing", "chaining",
        ]
        assert stages == expected

    def test_full_pipeline_runs_all_agents(self, pipeline):
        m = pipeline.execute("run all agents")
        agent_names = [a.get("agent") for a in m.agent_trace]
        assert "Planner" in agent_names
        assert "Integrator" in agent_names or "ReflexCache" in agent_names


class TestEvidenceChain:
    """Evidence chain integrity across multiple missions."""

    def test_first_receipt_is_genesis(self, pipeline):
        m = pipeline.execute("first")
        assert m.evidence_receipt.previous_hash == "0" * 64

    def test_second_receipt_links_to_first(self, pipeline):
        m1 = pipeline.execute("first")
        m2 = pipeline.execute("second")
        assert m2.evidence_receipt.previous_hash == m1.evidence_receipt.receipt_id

    def test_chain_of_five_is_valid(self, pipeline):
        for i in range(5):
            pipeline.execute(f"mission-{i}")
        valid, count, errors = pipeline.evidence_ledger.verify_chain()
        assert valid is True
        assert count == 5
        assert len(errors) == 0

    def test_receipt_contains_ihsan_tensor(self, pipeline):
        m = pipeline.execute("tensor test")
        tensor = m.evidence_receipt.ihsan_tensor
        assert isinstance(tensor, dict)
        assert len(tensor) == 6
        assert "moral_clarity" in tensor


class TestGateEnforcement:
    """Constitutional gate pass/fail behavior."""

    def test_normal_output_passes_gate(self, pipeline):
        m = pipeline.execute("A reasonable well-formed question")
        assert m.ihsan_score.passes is True

    def test_gate_minimum_from_constitution(self, pipeline):
        assert pipeline.ihsan_gate.gate_minimum == 0.85

    def test_bloom_eligible_tracked(self, pipeline):
        m = pipeline.execute("test bloom")
        if m.ihsan_score.composite >= 0.90:
            assert m.bloom_eligible is True


class TestReflexPrecipitation:
    """Reflex cache integration — precipitation and cache hits."""

    def test_no_reflex_on_first_call(self, pipeline):
        m = pipeline.execute("unique query xyz")
        assert m.reflex_hit is False

    def test_precipitation_after_three_repeats(self, pipeline):
        for _ in range(3):
            pipeline.execute("repeat me please")
        # 4th call should hit cache
        m = pipeline.execute("repeat me please")
        assert m.reflex_hit is True

    def test_cache_hit_faster_than_full_pipeline(self, pipeline):
        for _ in range(3):
            pipeline.execute("speed test")
        m_cached = pipeline.execute("speed test")
        m_fresh = pipeline.execute("completely new unique query abc")
        # Both should be fast in test mode, but cache should have fewer agents
        cached_agents = len(m_cached.agent_trace)
        fresh_agents = len(m_fresh.agent_trace)
        assert cached_agents <= fresh_agents


class TestPipelineStats:
    """Aggregate statistics tracking."""

    def test_completed_count(self, pipeline):
        pipeline.execute("a")
        pipeline.execute("b")
        assert pipeline.stats.missions_completed == 2

    def test_gate_pass_rate(self, pipeline):
        pipeline.execute("test")
        assert pipeline.stats.gate_pass_rate > 0

    def test_evidence_receipt_count(self, pipeline):
        pipeline.execute("x")
        pipeline.execute("y")
        assert pipeline.stats.evidence_receipts == 2


class TestHealthReport:
    """Pipeline health introspection."""

    def test_health_contains_version(self, pipeline):
        h = pipeline.health()
        assert "constitution_version" in h

    def test_health_contains_stats(self, pipeline):
        pipeline.execute("health check")
        h = pipeline.health()
        assert h["pipeline_stats"]["missions_completed"] == 1

    def test_health_evidence_chain_valid(self, pipeline):
        pipeline.execute("chain check")
        h = pipeline.health()
        assert h["evidence_chain_valid"] is True

    def test_health_agents_listed(self, pipeline):
        h = pipeline.health()
        assert len(h["agents"]) == 7


class TestMissionAsEvidence:
    """Mission evidence output format."""

    def test_as_evidence_fields(self, pipeline):
        m = pipeline.execute("evidence format test")
        ev = m.as_evidence()
        assert "mission_id" in ev
        assert "status" in ev
        assert "ihsan_composite" in ev
        assert "snr_normalized" in ev
        assert "bloom_eligible" in ev
        assert "receipt_id" in ev

    def test_bloom_eligible_field(self, pipeline):
        m = pipeline.execute("test")
        ev = m.as_evidence()
        assert isinstance(ev["bloom_eligible"], bool)


class TestLLMIntegration:
    """Custom LLM function integration."""

    def test_custom_llm_fn(self, tmp_path):
        def mock_llm(text):
            return f"LLM response to: {text}"

        p = MissionPipeline(
            evidence_path=tmp_path / "llm_test.jsonl",
            llm_fn=mock_llm,
        )
        m = p.execute("use the LLM")
        assert "LLM response to:" in m.output_text

    def test_llm_fallback_on_error(self, tmp_path):
        def broken_llm(text):
            raise RuntimeError("LLM down")

        p = MissionPipeline(
            evidence_path=tmp_path / "fallback_test.jsonl",
            llm_fn=broken_llm,
        )
        m = p.execute("handle the error")
        # Should fallback to template response
        assert m.status == MissionStatus.COMPLETE
        assert len(m.output_text) > 0


class TestShutdown:
    """Graceful shutdown behavior."""

    def test_shutdown_persists_cache(self, tmp_path):
        cache_path = tmp_path / "shutdown_cache.json"
        p = MissionPipeline(
            evidence_path=tmp_path / "shutdown.jsonl",
            cache_path=cache_path,
        )
        for _ in range(3):
            p.execute("persist on shutdown")
        p.shutdown()
        assert cache_path.exists()
