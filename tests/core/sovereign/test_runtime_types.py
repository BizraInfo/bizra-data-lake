"""
Tests for runtime_types — TypedDicts, Protocols, Enums, Dataclasses
====================================================================
Comprehensive coverage for every exported type in core.sovereign.runtime_types:
TypedDicts, runtime-checkable Protocols, Enums, RuntimeConfig classmethods,
RuntimeMetrics rolling-average arithmetic, and SovereignQuery/SovereignResult
serialisation.

Standing on Giants: Shannon + Lamport + Besta + Vaswani + Anthropic
"""

from __future__ import annotations

import os
import uuid
from dataclasses import fields as dc_fields
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from core.sovereign.runtime_types import (
    # TypedDicts
    AutonomousCycleResult,
    LoopStatus,
    ReasoningResult,
    SNRResult,
    ValidationResult,
    # Protocols
    AutonomousLoopProtocol,
    GraphReasonerProtocol,
    GuardianProtocol,
    ImpactTrackerProtocol,
    SNROptimizerProtocol,
    # Enums
    HealthStatus,
    RuntimeMode,
    # Config / Metrics / Query / Result
    RuntimeConfig,
    RuntimeMetrics,
    SovereignQuery,
    SovereignResult,
)


# =============================================================================
# TYPED DICTS
# =============================================================================


class TestReasoningResult:
    """ReasoningResult — total=False TypedDict."""

    def test_empty_construction(self) -> None:
        """total=False means all keys are optional; empty dict is valid."""
        result: ReasoningResult = {}
        assert isinstance(result, dict)

    def test_full_construction(self) -> None:
        result: ReasoningResult = {
            "thoughts": ["step-1", "step-2"],
            "conclusion": "answer",
            "confidence": 0.92,
            "depth_reached": 3,
        }
        assert result["thoughts"] == ["step-1", "step-2"]
        assert result["conclusion"] == "answer"
        assert result["confidence"] == 0.92
        assert result["depth_reached"] == 3

    def test_partial_construction(self) -> None:
        result: ReasoningResult = {"confidence": 0.5}
        assert result["confidence"] == 0.5
        assert "thoughts" not in result


class TestSNRResult:
    """SNRResult — total=False TypedDict."""

    def test_empty_construction(self) -> None:
        result: SNRResult = {}
        assert isinstance(result, dict)

    def test_full_construction(self) -> None:
        result: SNRResult = {
            "original_length": 500,
            "snr_score": 0.93,
            "meets_threshold": True,
            "optimized": "clean text",
        }
        assert result["original_length"] == 500
        assert result["snr_score"] == 0.93
        assert result["meets_threshold"] is True
        assert result["optimized"] == "clean text"

    def test_optimized_none(self) -> None:
        result: SNRResult = {"optimized": None}
        assert result["optimized"] is None


class TestValidationResult:
    """ValidationResult — total=False TypedDict."""

    def test_full_construction(self) -> None:
        result: ValidationResult = {
            "is_valid": True,
            "confidence": 0.99,
            "issues": [],
        }
        assert result["is_valid"] is True
        assert result["confidence"] == 0.99
        assert result["issues"] == []


class TestAutonomousCycleResult:
    """AutonomousCycleResult — total=False TypedDict."""

    def test_full_construction(self) -> None:
        result: AutonomousCycleResult = {
            "cycle": 7,
            "decisions": 3,
            "actions": 2,
        }
        assert result["cycle"] == 7


class TestLoopStatus:
    """LoopStatus — total=False TypedDict."""

    def test_full_construction(self) -> None:
        status: LoopStatus = {"running": True, "cycle": 42}
        assert status["running"] is True
        assert status["cycle"] == 42

    def test_empty_construction(self) -> None:
        status: LoopStatus = {}
        assert isinstance(status, dict)


# =============================================================================
# PROTOCOLS (runtime_checkable)
# =============================================================================


class _StubGraphReasoner:
    async def reason(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        max_depth: int = 5,
    ) -> ReasoningResult:
        return {"confidence": 1.0}


class _StubSNROptimizer:
    async def optimize(self, text: str) -> SNRResult:
        return {"snr_score": 0.99}


class _StubGuardian:
    async def validate(
        self, content: str, context: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        return {"is_valid": True, "confidence": 1.0, "issues": []}


class _StubAutonomousLoop:
    async def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def status(self) -> LoopStatus:
        return {"running": False, "cycle": 0}


class _StubImpactTracker:
    @property
    def node_id(self) -> str:
        return "node-test"

    @property
    def sovereignty_score(self) -> float:
        return 0.8

    @property
    def sovereignty_tier(self) -> Any:
        return "seedling"

    @property
    def total_bloom(self) -> float:
        return 42.0

    @property
    def achievements(self) -> List[str]:
        return ["first_query"]

    def record_event(
        self,
        category: str,
        action: str,
        bloom: float,
        uers: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        return None

    def get_progress(self) -> Any:
        return {}

    def flush(self) -> None:
        pass


class _NotAReasoner:
    """Missing the reason() method entirely."""

    def think(self, query: str) -> str:
        return "hmm"


class TestGraphReasonerProtocol:
    def test_isinstance_positive(self) -> None:
        assert isinstance(_StubGraphReasoner(), GraphReasonerProtocol)

    def test_isinstance_negative(self) -> None:
        assert not isinstance(_NotAReasoner(), GraphReasonerProtocol)

    def test_isinstance_negative_plain_object(self) -> None:
        assert not isinstance(object(), GraphReasonerProtocol)


class TestSNROptimizerProtocol:
    def test_isinstance_positive(self) -> None:
        assert isinstance(_StubSNROptimizer(), SNROptimizerProtocol)

    def test_isinstance_negative(self) -> None:
        assert not isinstance(object(), SNROptimizerProtocol)


class TestGuardianProtocol:
    def test_isinstance_positive(self) -> None:
        assert isinstance(_StubGuardian(), GuardianProtocol)

    def test_isinstance_negative(self) -> None:
        assert not isinstance(object(), GuardianProtocol)


class TestAutonomousLoopProtocol:
    def test_isinstance_positive(self) -> None:
        assert isinstance(_StubAutonomousLoop(), AutonomousLoopProtocol)

    def test_isinstance_negative(self) -> None:
        assert not isinstance(object(), AutonomousLoopProtocol)


class TestImpactTrackerProtocol:
    def test_isinstance_positive(self) -> None:
        assert isinstance(_StubImpactTracker(), ImpactTrackerProtocol)

    def test_isinstance_negative(self) -> None:
        assert not isinstance(object(), ImpactTrackerProtocol)


# =============================================================================
# ENUMS
# =============================================================================


class TestRuntimeMode:
    def test_all_members_exist(self) -> None:
        members = {m.name for m in RuntimeMode}
        assert members == {"MINIMAL", "STANDARD", "AUTONOMOUS", "DEBUG", "DEVELOPMENT"}

    def test_auto_values_are_unique(self) -> None:
        values = [m.value for m in RuntimeMode]
        assert len(values) == len(set(values))

    def test_identity(self) -> None:
        assert RuntimeMode.MINIMAL is RuntimeMode.MINIMAL
        assert RuntimeMode.STANDARD is not RuntimeMode.AUTONOMOUS


class TestHealthStatus:
    def test_string_values(self) -> None:
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.UNKNOWN.value == "unknown"

    def test_member_count(self) -> None:
        assert len(HealthStatus) == 4

    def test_lookup_by_value(self) -> None:
        assert HealthStatus("healthy") is HealthStatus.HEALTHY


# =============================================================================
# RUNTIME CONFIG
# =============================================================================


class TestRuntimeConfig:
    def test_default_values(self) -> None:
        cfg = RuntimeConfig()
        assert cfg.ihsan_threshold == 0.95
        assert cfg.snr_threshold == 0.85
        assert cfg.mode is RuntimeMode.STANDARD
        assert cfg.max_reasoning_depth == 5
        assert cfg.enable_graph_reasoning is True
        assert cfg.enable_snr_optimization is True
        assert cfg.enable_guardian_validation is True
        assert cfg.enable_autonomous_loop is False
        assert cfg.enable_proactive_kernel is False

    def test_node_id_auto_generated(self) -> None:
        cfg = RuntimeConfig()
        assert cfg.node_id.startswith("node-")
        assert len(cfg.node_id) == len("node-") + 8  # 8-char hex suffix

    def test_node_id_unique_per_instance(self) -> None:
        a = RuntimeConfig()
        b = RuntimeConfig()
        assert a.node_id != b.node_id

    def test_minimal_classmethod(self) -> None:
        cfg = RuntimeConfig.minimal()
        assert cfg.mode is RuntimeMode.MINIMAL
        assert cfg.enable_graph_reasoning is False
        assert cfg.enable_snr_optimization is False
        assert cfg.enable_guardian_validation is False
        assert cfg.enable_autonomous_loop is False
        assert cfg.enable_proactive_kernel is False
        assert cfg.enable_zpk_preflight is False

    def test_standard_classmethod(self) -> None:
        cfg = RuntimeConfig.standard()
        assert cfg.mode is RuntimeMode.STANDARD
        # Standard should keep default feature flags
        assert cfg.enable_graph_reasoning is True
        assert cfg.enable_snr_optimization is True
        assert cfg.enable_guardian_validation is True

    def test_observer_classmethod(self) -> None:
        cfg = RuntimeConfig.observer()
        assert cfg.mode is RuntimeMode.STANDARD
        assert cfg.enable_proactive_kernel is True
        # Observer: auto-execute threshold is effectively disabled
        assert cfg.proactive_kernel_auto_execute_tau == 0.99
        assert cfg.proactive_kernel_min_auto_confidence == 0.95

    def test_autonomous_classmethod(self) -> None:
        cfg = RuntimeConfig.autonomous()
        assert cfg.mode is RuntimeMode.AUTONOMOUS
        assert cfg.enable_autonomous_loop is True
        assert cfg.enable_proactive_kernel is True

    def test_lm_studio_url_defaults(self) -> None:
        # Clear env vars to guarantee defaults
        env_patch = {
            "LMSTUDIO_URL": "",
            "LMSTUDIO_HOST": "",
            "LMSTUDIO_PORT": "",
        }
        saved = {}
        for k in env_patch:
            saved[k] = os.environ.pop(k, None)
        try:
            cfg = RuntimeConfig()
            assert "192.168.56.1" in cfg.lm_studio_url
            assert "1234" in cfg.lm_studio_url
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v

    def test_ollama_url_default(self) -> None:
        saved_url = os.environ.pop("OLLAMA_URL", None)
        saved_host = os.environ.pop("OLLAMA_HOST", None)
        try:
            cfg = RuntimeConfig()
            assert cfg.ollama_url == "http://localhost:11434"
        finally:
            if saved_url is not None:
                os.environ["OLLAMA_URL"] = saved_url
            if saved_host is not None:
                os.environ["OLLAMA_HOST"] = saved_host

    def test_state_dir_is_path(self) -> None:
        cfg = RuntimeConfig()
        assert isinstance(cfg.state_dir, Path)
        assert cfg.state_dir == Path("sovereign_state")

    def test_default_model(self) -> None:
        cfg = RuntimeConfig()
        assert cfg.default_model == "qwen2.5:7b"

    def test_zpk_defaults(self) -> None:
        cfg = RuntimeConfig()
        assert cfg.enable_zpk_preflight is False
        assert cfg.zpk_manifest_uri == ""
        assert cfg.zpk_allowed_versions == []
        assert cfg.zpk_min_policy_version == 1
        assert cfg.zpk_min_ihsan_policy == 0.95


# =============================================================================
# RUNTIME METRICS
# =============================================================================


class TestRuntimeMetrics:
    def test_initial_values(self) -> None:
        m = RuntimeMetrics()
        assert m.queries_processed == 0
        assert m.queries_succeeded == 0
        assert m.queries_failed == 0
        assert m.avg_query_time_ms == 0.0
        assert m.reasoning_calls == 0
        assert m.reasoning_avg_depth == 0.0
        assert m.snr_optimizations == 0
        assert m.snr_avg_improvement == 0.0
        assert m.validations == 0
        assert m.validation_pass_rate == 0.0
        assert m.started_at is None

    # -- update_query_stats --

    def test_query_stats_first_success(self) -> None:
        m = RuntimeMetrics()
        m.update_query_stats(success=True, duration_ms=100.0)
        assert m.queries_processed == 1
        assert m.queries_succeeded == 1
        assert m.queries_failed == 0
        assert m.avg_query_time_ms == 100.0

    def test_query_stats_first_failure(self) -> None:
        m = RuntimeMetrics()
        m.update_query_stats(success=False, duration_ms=50.0)
        assert m.queries_processed == 1
        assert m.queries_succeeded == 0
        assert m.queries_failed == 1
        assert m.avg_query_time_ms == 50.0

    def test_query_stats_rolling_average(self) -> None:
        m = RuntimeMetrics()
        m.update_query_stats(True, 100.0)
        m.update_query_stats(True, 200.0)
        assert m.queries_processed == 2
        assert m.avg_query_time_ms == pytest.approx(150.0)

    def test_query_stats_multiple_calls(self) -> None:
        m = RuntimeMetrics()
        durations = [10.0, 20.0, 30.0, 40.0]
        for d in durations:
            m.update_query_stats(True, d)
        expected_avg = sum(durations) / len(durations)
        assert m.avg_query_time_ms == pytest.approx(expected_avg)
        assert m.queries_processed == 4
        assert m.queries_succeeded == 4

    def test_query_stats_mixed_success_failure(self) -> None:
        m = RuntimeMetrics()
        m.update_query_stats(True, 100.0)
        m.update_query_stats(False, 200.0)
        m.update_query_stats(True, 300.0)
        assert m.queries_succeeded == 2
        assert m.queries_failed == 1
        assert m.queries_processed == 3

    # -- update_reasoning_stats --

    def test_reasoning_stats_first_call(self) -> None:
        m = RuntimeMetrics()
        m.update_reasoning_stats(depth=3)
        assert m.reasoning_calls == 1
        assert m.reasoning_avg_depth == 3.0

    def test_reasoning_stats_rolling_average(self) -> None:
        m = RuntimeMetrics()
        m.update_reasoning_stats(2)
        m.update_reasoning_stats(4)
        m.update_reasoning_stats(6)
        assert m.reasoning_calls == 3
        assert m.reasoning_avg_depth == pytest.approx(4.0)

    def test_reasoning_stats_zero_depth(self) -> None:
        m = RuntimeMetrics()
        m.update_reasoning_stats(0)
        assert m.reasoning_avg_depth == 0.0

    # -- update_snr_stats --

    def test_snr_stats_first_call(self) -> None:
        m = RuntimeMetrics()
        m.update_snr_stats(improvement=0.15)
        assert m.snr_optimizations == 1
        assert m.snr_avg_improvement == pytest.approx(0.15)

    def test_snr_stats_rolling_average(self) -> None:
        m = RuntimeMetrics()
        m.update_snr_stats(0.10)
        m.update_snr_stats(0.20)
        assert m.snr_optimizations == 2
        assert m.snr_avg_improvement == pytest.approx(0.15)

    def test_snr_stats_zero_improvement(self) -> None:
        m = RuntimeMetrics()
        m.update_snr_stats(0.0)
        assert m.snr_avg_improvement == 0.0

    # -- update_validation_stats --

    def test_validation_stats_first_pass(self) -> None:
        m = RuntimeMetrics()
        m.update_validation_stats(passed=True)
        assert m.validations == 1
        assert m.validation_pass_rate == pytest.approx(1.0)

    def test_validation_stats_first_fail(self) -> None:
        m = RuntimeMetrics()
        m.update_validation_stats(passed=False)
        assert m.validations == 1
        assert m.validation_pass_rate == pytest.approx(0.0)

    def test_validation_stats_mixed(self) -> None:
        m = RuntimeMetrics()
        m.update_validation_stats(True)
        m.update_validation_stats(False)
        m.update_validation_stats(True)
        assert m.validations == 3
        # 2 passes out of 3
        assert m.validation_pass_rate == pytest.approx(2.0 / 3.0)

    def test_validation_stats_all_pass(self) -> None:
        m = RuntimeMetrics()
        for _ in range(10):
            m.update_validation_stats(True)
        assert m.validation_pass_rate == pytest.approx(1.0)

    def test_validation_stats_all_fail(self) -> None:
        m = RuntimeMetrics()
        for _ in range(5):
            m.update_validation_stats(False)
        assert m.validation_pass_rate == pytest.approx(0.0)

    # -- to_dict --

    def test_to_dict_structure(self) -> None:
        m = RuntimeMetrics()
        d = m.to_dict()
        assert "queries" in d
        assert "reasoning" in d
        assert "snr" in d
        assert "validation" in d
        assert "autonomous" in d
        assert "quality" in d
        assert "uptime_seconds" in d

    def test_to_dict_queries_subkeys(self) -> None:
        m = RuntimeMetrics()
        m.update_query_stats(True, 123.456)
        d = m.to_dict()
        q = d["queries"]
        assert q["processed"] == 1
        assert q["succeeded"] == 1
        assert q["failed"] == 0
        assert q["avg_time_ms"] == 123.46  # rounded to 2 decimal places

    def test_to_dict_reasoning_subkeys(self) -> None:
        m = RuntimeMetrics()
        m.update_reasoning_stats(3)
        d = m.to_dict()
        assert d["reasoning"]["calls"] == 1
        assert d["reasoning"]["avg_depth"] == 3.0

    def test_to_dict_snr_subkeys(self) -> None:
        m = RuntimeMetrics()
        m.update_snr_stats(0.1234567)
        d = m.to_dict()
        assert d["snr"]["optimizations"] == 1
        assert d["snr"]["avg_improvement"] == 0.123  # rounded to 3 dp

    def test_to_dict_validation_subkeys(self) -> None:
        m = RuntimeMetrics()
        m.update_validation_stats(True)
        d = m.to_dict()
        assert d["validation"]["count"] == 1
        assert d["validation"]["pass_rate"] == 1.0

    def test_to_dict_autonomous_subkeys(self) -> None:
        m = RuntimeMetrics()
        m.autonomous_cycles = 5
        m.autonomous_decisions = 10
        m.autonomous_actions = 3
        d = m.to_dict()
        assert d["autonomous"]["cycles"] == 5
        assert d["autonomous"]["decisions"] == 10
        assert d["autonomous"]["actions"] == 3

    def test_to_dict_quality_subkeys(self) -> None:
        m = RuntimeMetrics()
        m.current_ihsan_score = 0.97123
        m.current_snr_score = 0.88456
        d = m.to_dict()
        assert d["quality"]["ihsan_score"] == 0.971
        assert d["quality"]["snr_score"] == 0.885

    def test_to_dict_uptime(self) -> None:
        m = RuntimeMetrics()
        m.total_uptime_seconds = 123.456789
        d = m.to_dict()
        assert d["uptime_seconds"] == 123.5

    # -- computed properties (P1 regression guards) --

    def test_total_queries_alias(self) -> None:
        """total_queries must always equal queries_processed."""
        m = RuntimeMetrics()
        assert m.total_queries == 0
        m.update_query_stats(True, 10.0)
        assert m.total_queries == 1
        assert m.total_queries == m.queries_processed

    def test_success_rate_zero_queries(self) -> None:
        """success_rate returns 0.0 when no queries have been processed."""
        m = RuntimeMetrics()
        assert m.success_rate() == 0.0

    def test_success_rate_all_success(self) -> None:
        m = RuntimeMetrics()
        for _ in range(5):
            m.update_query_stats(True, 10.0)
        assert m.success_rate() == pytest.approx(1.0)

    def test_success_rate_mixed(self) -> None:
        m = RuntimeMetrics()
        m.update_query_stats(True, 10.0)
        m.update_query_stats(False, 10.0)
        assert m.success_rate() == pytest.approx(0.5)

    def test_health_score_zero_state(self) -> None:
        """health_score is 0.0 when everything is zeroed."""
        m = RuntimeMetrics()
        assert m.health_score == pytest.approx(0.0)

    def test_health_score_perfect_state(self) -> None:
        """health_score ≈ 1.0 when all sub-signals are 1.0."""
        m = RuntimeMetrics()
        m.current_snr_score = 1.0
        m.current_ihsan_score = 1.0
        m.update_query_stats(True, 10.0)
        # 0.4*1 + 0.4*1 + 0.2*1 = 1.0
        assert m.health_score == pytest.approx(1.0)

    def test_health_score_composition(self) -> None:
        """health_score = 0.4*SNR + 0.4*Ihsan + 0.2*success_rate."""
        m = RuntimeMetrics()
        m.current_snr_score = 0.8
        m.current_ihsan_score = 0.6
        m.update_query_stats(True, 10.0)
        m.update_query_stats(False, 10.0)
        # success_rate = 0.5
        expected = 0.4 * 0.8 + 0.4 * 0.6 + 0.2 * 0.5
        assert m.health_score == pytest.approx(expected)

    def test_cache_hit_rate_zero_lookups(self) -> None:
        m = RuntimeMetrics()
        assert m.cache_hit_rate == 0.0

    def test_cache_hit_rate_mixed(self) -> None:
        m = RuntimeMetrics(cache_hits=3, cache_misses=1)
        assert m.cache_hit_rate == pytest.approx(0.75)

    def test_to_prometheus_with_help(self) -> None:
        m = RuntimeMetrics(
            queries_processed=10,
            queries_succeeded=8,
            current_snr_score=0.91,
            current_ihsan_score=0.96,
            avg_query_time_ms=123.4,
        )
        text = m.to_prometheus(include_help=True)
        assert "# HELP sovereign_queries_total Total queries processed" in text
        assert "# TYPE sovereign_queries_total counter" in text
        assert "sovereign_queries_total 10" in text
        assert "sovereign_query_success_rate 0.8000" in text
        assert "sovereign_snr_score 0.9100" in text
        assert "sovereign_ihsan_score 0.9600" in text
        assert "sovereign_health_score 0.9080" in text

    def test_to_prometheus_without_help(self) -> None:
        m = RuntimeMetrics(queries_processed=1, queries_succeeded=1)
        text = m.to_prometheus(include_help=False)
        assert "# HELP " not in text
        assert "# TYPE " not in text
        assert "sovereign_queries_total 1" in text


# =============================================================================
# SOVEREIGN QUERY
# =============================================================================


class TestSovereignQuery:
    def test_default_construction(self) -> None:
        q = SovereignQuery()
        assert len(q.id) == 8
        assert q.text == ""
        assert q.context == {}
        assert q.require_reasoning is True
        assert q.require_snr is True
        assert q.require_validation is False
        assert q.timeout == 60.0
        assert isinstance(q.created_at, datetime)

    def test_unique_ids(self) -> None:
        a = SovereignQuery()
        b = SovereignQuery()
        assert a.id != b.id

    def test_custom_text_and_context(self) -> None:
        q = SovereignQuery(text="What is BIZRA?", context={"source": "cli"})
        assert q.text == "What is BIZRA?"
        assert q.context["source"] == "cli"

    def test_user_id_default_empty(self) -> None:
        q = SovereignQuery()
        assert q.user_id == ""

    def test_user_id_set(self) -> None:
        q = SovereignQuery(user_id="user-42")
        assert q.user_id == "user-42"


# =============================================================================
# SOVEREIGN RESULT
# =============================================================================


class TestSovereignResult:
    def test_default_construction(self) -> None:
        r = SovereignResult()
        assert r.query_id == ""
        assert r.success is False
        assert r.response == ""
        assert r.reasoning_used is False
        assert r.reasoning_depth == 0
        assert r.thoughts == []
        assert r.ihsan_score == 0.0
        assert r.snr_score == 0.0
        assert r.snr_ok is False
        assert r.validated is False
        assert r.validation_passed is False
        assert r.validation_issues == []
        assert r.processing_time_ms == 0.0
        assert r.model_used is None
        assert r.graph_hash is None
        assert r.claim_tags == {}
        assert r.user_id == ""
        assert r.error is None
        assert isinstance(r.created_at, datetime)

    def test_to_dict_structure(self) -> None:
        r = SovereignResult(query_id="abc", success=True, response="42")
        d = r.to_dict()
        assert d["query_id"] == "abc"
        assert d["success"] is True
        assert d["response"] == "42"
        assert "reasoning" in d
        assert "quality" in d
        assert "validation" in d
        assert "processing_time_ms" in d
        assert "graph_hash" in d
        assert "user_id" in d
        assert "error" in d

    def test_to_dict_reasoning_subdict(self) -> None:
        r = SovereignResult(
            reasoning_used=True,
            reasoning_depth=4,
            thoughts=["a", "b"],
        )
        d = r.to_dict()
        assert d["reasoning"]["used"] is True
        assert d["reasoning"]["depth"] == 4
        assert d["reasoning"]["thoughts"] == ["a", "b"]

    def test_to_dict_quality_subdict(self) -> None:
        r = SovereignResult(ihsan_score=0.97123, snr_score=0.88456, snr_ok=True)
        d = r.to_dict()
        assert d["quality"]["ihsan_score"] == 0.971
        assert d["quality"]["snr_score"] == 0.885
        assert d["quality"]["snr_ok"] is True

    def test_to_dict_validation_subdict(self) -> None:
        r = SovereignResult(
            validated=True,
            validation_passed=False,
            validation_issues=["issue-1"],
        )
        d = r.to_dict()
        assert d["validation"]["performed"] is True
        assert d["validation"]["passed"] is False
        assert d["validation"]["issues"] == ["issue-1"]

    def test_to_dict_graph_hash_and_user_id(self) -> None:
        r = SovereignResult(graph_hash="sha256:abc", user_id="u-7")
        d = r.to_dict()
        assert d["graph_hash"] == "sha256:abc"
        assert d["user_id"] == "u-7"

    def test_to_dict_error_field(self) -> None:
        r = SovereignResult(error="timeout exceeded")
        d = r.to_dict()
        assert d["error"] == "timeout exceeded"

    def test_to_dict_processing_time_rounded(self) -> None:
        r = SovereignResult(processing_time_ms=123.456789)
        d = r.to_dict()
        assert d["processing_time_ms"] == 123.46


# =============================================================================
# MODULE __all__ EXPORT CHECK
# =============================================================================


class TestModuleExports:
    """Verify __all__ is exhaustive and importable."""

    def test_all_exports_importable(self) -> None:
        from core.sovereign import runtime_types

        for name in runtime_types.__all__:
            assert hasattr(runtime_types, name), f"{name} listed in __all__ but not defined"

    def test_expected_names_in_all(self) -> None:
        from core.sovereign import runtime_types

        expected = {
            "ReasoningResult",
            "SNRResult",
            "ValidationResult",
            "AutonomousCycleResult",
            "LoopStatus",
            "GraphReasonerProtocol",
            "SNROptimizerProtocol",
            "GuardianProtocol",
            "AutonomousLoopProtocol",
            "RuntimeMode",
            "HealthStatus",
            "RuntimeConfig",
            "RuntimeMetrics",
            "SovereignQuery",
            "SovereignResult",
        }
        assert expected.issubset(set(runtime_types.__all__))
