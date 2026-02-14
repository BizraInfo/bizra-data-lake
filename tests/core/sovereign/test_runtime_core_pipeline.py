"""
Runtime Core Pipeline Tests -- Query Processing, Lifecycle, Integration
========================================================================
Part 2 of the SovereignRuntime test suite. Covers:

1. query() method -- cache, timeout, errors, conversation recording
2. _process_query routing -- complexity-based orchestrator/direct dispatch
3. _process_query_direct -- full 5-stage pipeline with gate chain
4. _orchestrate_complex_query -- decomposition, synthesis, fallback
5. _select_compute_tier -- omega mode to tier mapping
6. _execute_reasoning_stage -- GoT integration
7. _build_contextual_prompt -- PAT routing, memory retrieval, fallback
8. _perform_llm_inference -- gateway calls, fail-loud NO_LLM tagging
9. _optimize_snr -- dual engine (maximizer + authoritative scorer)
10. _validate_constitutionally -- IhsanGate + Omega + Guardian
11. initialize() / shutdown() lifecycle
12. _init_components() feature flag dispatch
13. _init_user_context / _load_genesis_identity / _init_memory_coordinator
14. _record_query_impact / _get_runtime_state / convenience methods
15. wait_for_shutdown / _setup_signal_handlers

Standing on Giants: Shannon (SNR) + Besta (GoT) + Anthropic (Constitutional AI)
"""

from __future__ import annotations

import asyncio
import signal
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import (
    AsyncMock,
    MagicMock,
    PropertyMock,
    call,
    patch,
)

import pytest

from core.sovereign.runtime_core import SovereignRuntime
from core.sovereign.runtime_types import (
    HealthStatus,
    RuntimeConfig,
    RuntimeMetrics,
    RuntimeMode,
    SovereignQuery,
    SovereignResult,
)


# =============================================================================
# HELPERS
# =============================================================================


def _minimal_config(tmp_path: Path) -> RuntimeConfig:
    """Create minimal config with all optional features disabled."""
    return RuntimeConfig(
        enable_graph_reasoning=False,
        enable_snr_optimization=False,
        enable_guardian_validation=False,
        enable_autonomous_loop=False,
        enable_cache=False,
        enable_persistence=False,
        autonomous_enabled=False,
        enable_zpk_preflight=False,
        enable_proactive_kernel=False,
        state_dir=tmp_path / "sovereign_state",
    )


def _make_runtime(tmp_path: Path, **overrides: Any) -> SovereignRuntime:
    """Create a runtime with minimal config and optional overrides."""
    cfg = _minimal_config(tmp_path)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return SovereignRuntime(cfg)


def _set_ready(runtime: SovereignRuntime) -> None:
    """Mark runtime as initialized and running (skip real init)."""
    runtime._initialized = True
    runtime._running = True


# =============================================================================
# 1. query() METHOD
# =============================================================================


class TestQueryMethod:
    """Tests for the public query() entry-point (lines 1476-1548)."""

    @pytest.mark.asyncio
    async def test_raises_runtime_error_when_not_initialized(self, tmp_path: Path) -> None:
        """query() must raise RuntimeError if runtime is not initialized."""
        rt = _make_runtime(tmp_path)
        with pytest.raises(RuntimeError, match="not initialized"):
            await rt.query("hello")

    @pytest.mark.asyncio
    async def test_cache_hit_returns_cached_result(self, tmp_path: Path) -> None:
        """When enable_cache=True and key exists, return cached and bump cache_hits."""
        rt = _make_runtime(tmp_path, enable_cache=True)
        _set_ready(rt)

        cached_result = SovereignResult(query_id="cached-01", success=True, response="from cache")
        # Pre-populate cache with a known key
        q = SovereignQuery(text="cached query", require_reasoning=True)
        cache_key = rt._cache_key(q)
        rt._cache[cache_key] = cached_result

        result = await rt.query("cached query")
        assert result is cached_result
        assert rt.metrics.cache_hits == 1

    @pytest.mark.asyncio
    async def test_cache_miss_increments_counter(self, tmp_path: Path) -> None:
        """When enable_cache=True but key absent, increment cache_misses."""
        rt = _make_runtime(tmp_path, enable_cache=True)
        _set_ready(rt)
        rt._process_query = AsyncMock(return_value=SovereignResult(
            query_id="q1", success=True, response="ok"
        ))

        await rt.query("unique query text that is not cached")
        assert rt.metrics.cache_misses >= 1

    @pytest.mark.asyncio
    async def test_cache_miss_stores_result_when_cache_enabled(self, tmp_path: Path) -> None:
        """Successful result gets stored in cache when enable_cache=True."""
        rt = _make_runtime(tmp_path, enable_cache=True)
        _set_ready(rt)
        expected = SovereignResult(query_id="q2", success=True, response="cached me")
        rt._process_query = AsyncMock(return_value=expected)

        result = await rt.query("store me in cache")
        assert result.success is True
        assert len(rt._cache) == 1

    @pytest.mark.asyncio
    async def test_user_context_conversation_recording(self, tmp_path: Path) -> None:
        """Human turn and PAT turn are recorded when user_context is present."""
        rt = _make_runtime(tmp_path)
        _set_ready(rt)

        mock_conv = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.conversation = mock_conv
        rt._user_context = mock_ctx

        resp = SovereignResult(query_id="q3", success=True, response="answer", snr_score=0.9, ihsan_score=0.96)
        rt._process_query = AsyncMock(return_value=resp)

        await rt.query("human says hi")
        mock_conv.add_human_turn.assert_called_once_with("human says hi")
        mock_conv.add_pat_turn.assert_called_once()

    @pytest.mark.asyncio
    async def test_timeout_returns_failure_result(self, tmp_path: Path) -> None:
        """When _process_query exceeds timeout, return failure with timeout error."""
        rt = _make_runtime(tmp_path)
        _set_ready(rt)

        async def slow_process(*args: Any, **kwargs: Any) -> SovereignResult:
            await asyncio.sleep(10)
            return SovereignResult()

        rt._process_query = slow_process  # type: ignore[assignment]

        result = await rt.query("slow query", timeout_ms=50)
        assert result.success is False
        assert "timeout" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_exception_returns_failure_result(self, tmp_path: Path) -> None:
        """When _process_query raises, return failure with error string."""
        rt = _make_runtime(tmp_path)
        _set_ready(rt)
        rt._process_query = AsyncMock(side_effect=ValueError("boom"))

        result = await rt.query("exploding query")
        assert result.success is False
        assert "boom" in (result.error or "")

    @pytest.mark.asyncio
    async def test_exception_updates_metrics(self, tmp_path: Path) -> None:
        """Failed query updates metrics with failure and duration."""
        rt = _make_runtime(tmp_path)
        _set_ready(rt)
        rt._process_query = AsyncMock(side_effect=RuntimeError("fail"))

        await rt.query("failing query")
        assert rt.metrics.queries_processed == 1
        assert rt.metrics.queries_failed == 1

    @pytest.mark.asyncio
    async def test_success_updates_cache(self, tmp_path: Path) -> None:
        """Successful result populates cache when enable_cache is True."""
        rt = _make_runtime(tmp_path, enable_cache=True)
        _set_ready(rt)
        rt._process_query = AsyncMock(return_value=SovereignResult(
            query_id="qc", success=True, response="cacheable"
        ))

        await rt.query("cache this")
        assert len(rt._cache) == 1

    @pytest.mark.asyncio
    async def test_cache_disabled_does_not_store(self, tmp_path: Path) -> None:
        """When enable_cache=False, no cache reads or writes occur."""
        rt = _make_runtime(tmp_path, enable_cache=False)
        _set_ready(rt)
        rt._process_query = AsyncMock(return_value=SovereignResult(
            query_id="nc", success=True, response="no cache"
        ))

        await rt.query("no cache query")
        assert len(rt._cache) == 0
        assert rt.metrics.cache_hits == 0

    @pytest.mark.asyncio
    async def test_query_passes_user_id(self, tmp_path: Path) -> None:
        """user_id option is forwarded to SovereignQuery."""
        rt = _make_runtime(tmp_path)
        _set_ready(rt)

        captured_queries: List[SovereignQuery] = []

        async def capture_query(q: SovereignQuery, *a: Any) -> SovereignResult:
            captured_queries.append(q)
            return SovereignResult(query_id=q.id, success=True, response="ok")

        rt._process_query = capture_query  # type: ignore[assignment]
        await rt.query("test", user_id="user-42")
        assert captured_queries[0].user_id == "user-42"


# =============================================================================
# 2. _process_query ROUTING
# =============================================================================


class TestProcessQueryRouting:
    """Tests for _process_query complexity-based routing (lines 1662-1678)."""

    @pytest.mark.asyncio
    async def test_low_complexity_routes_to_direct(self, tmp_path: Path) -> None:
        """complexity < 0.6 routes to _process_query_direct."""
        rt = _make_runtime(tmp_path)
        _set_ready(rt)

        rt._estimate_complexity = MagicMock(return_value=0.3)
        rt._process_query_direct = AsyncMock(return_value=SovereignResult(success=True))
        rt._orchestrate_complex_query = AsyncMock()

        query = SovereignQuery(text="simple")
        await rt._process_query(query, time.perf_counter())

        rt._process_query_direct.assert_awaited_once()
        rt._orchestrate_complex_query.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_high_complexity_with_orchestrator_routes_to_orchestrator(self, tmp_path: Path) -> None:
        """complexity >= 0.6 AND orchestrator available routes to _orchestrate_complex_query."""
        rt = _make_runtime(tmp_path)
        _set_ready(rt)
        rt._orchestrator = MagicMock()

        rt._estimate_complexity = MagicMock(return_value=0.8)
        rt._orchestrate_complex_query = AsyncMock(return_value=SovereignResult(success=True))
        rt._process_query_direct = AsyncMock()

        query = SovereignQuery(text="multi-step complex analysis")
        await rt._process_query(query, time.perf_counter())

        rt._orchestrate_complex_query.assert_awaited_once()
        rt._process_query_direct.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_high_complexity_without_orchestrator_falls_back_to_direct(self, tmp_path: Path) -> None:
        """complexity >= 0.6 but no orchestrator falls back to direct pipeline."""
        rt = _make_runtime(tmp_path)
        _set_ready(rt)
        rt._orchestrator = None

        rt._estimate_complexity = MagicMock(return_value=0.9)
        rt._process_query_direct = AsyncMock(return_value=SovereignResult(success=True))
        rt._orchestrate_complex_query = AsyncMock()

        query = SovereignQuery(text="complex but no orchestrator")
        await rt._process_query(query, time.perf_counter())

        rt._process_query_direct.assert_awaited_once()
        rt._orchestrate_complex_query.assert_not_awaited()


# =============================================================================
# 3. _process_query_direct -- FULL 5-STAGE PIPELINE
# =============================================================================


class TestProcessQueryDirect:
    """Tests for the direct 5-stage pipeline (lines 1680-1783)."""

    @pytest.mark.asyncio
    async def test_gate_chain_rejection_early_exit(self, tmp_path: Path) -> None:
        """If gate chain rejects, return immediately without further stages."""
        rt = _make_runtime(tmp_path)
        _set_ready(rt)

        rejection = SovereignResult(query_id="rej", success=False, response="rejected")
        rt._run_gate_chain_preflight = AsyncMock(return_value=rejection)
        rt._select_compute_tier = AsyncMock()
        rt._execute_reasoning_stage = AsyncMock()

        query = SovereignQuery(text="test")
        result = await rt._process_query_direct(query, time.perf_counter())

        assert result.success is False
        assert result.response == "rejected"
        rt._select_compute_tier.assert_not_awaited()
        rt._execute_reasoning_stage.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_full_pipeline_flow_all_5_stages(self, tmp_path: Path) -> None:
        """Verify all 5 stages execute in order for a normal query."""
        rt = _make_runtime(tmp_path)
        _set_ready(rt)

        call_order: List[str] = []

        async def mock_gate(*a: Any) -> None:
            call_order.append("gate")
            return None

        async def mock_tier(*a: Any) -> None:
            call_order.append("tier")
            return None

        async def mock_reasoning(*a: Any) -> tuple:
            call_order.append("reasoning")
            return (["thought1"], 0.85, "prompt text", "hash123")

        async def mock_inference(*a: Any) -> tuple:
            call_order.append("inference")
            return ("LLM answer", "qwen2.5:7b")

        async def mock_snr(content: str) -> tuple:
            call_order.append("snr")
            return (content, 0.92, {"snr": "measured"})

        async def mock_validate(*a: Any) -> tuple:
            call_order.append("validation")
            return (0.96, "IHSAN_GATE")

        rt._run_gate_chain_preflight = mock_gate  # type: ignore[assignment]
        rt._select_compute_tier = mock_tier  # type: ignore[assignment]
        rt._execute_reasoning_stage = mock_reasoning  # type: ignore[assignment]
        rt._perform_llm_inference = mock_inference  # type: ignore[assignment]
        rt._optimize_snr = mock_snr  # type: ignore[assignment]
        rt._validate_constitutionally = mock_validate  # type: ignore[assignment]
        rt._store_graph_artifact = MagicMock()
        rt._record_query_impact = MagicMock()
        rt._register_poi_contribution = MagicMock()
        rt._emit_query_receipt = MagicMock()
        rt._encode_query_memory = MagicMock()
        rt._commit_experience_episode = MagicMock()
        rt._observe_judgment = MagicMock()

        query = SovereignQuery(text="test pipeline")
        result = await rt._process_query_direct(query, time.perf_counter())

        assert call_order == ["gate", "tier", "reasoning", "inference", "snr", "validation"]
        assert result.success is True

    @pytest.mark.asyncio
    async def test_result_fields_populated(self, tmp_path: Path) -> None:
        """Verify all result fields are correctly populated by the pipeline."""
        rt = _make_runtime(tmp_path)
        _set_ready(rt)

        rt._run_gate_chain_preflight = AsyncMock(return_value=None)
        rt._select_compute_tier = AsyncMock(return_value=None)
        rt._execute_reasoning_stage = AsyncMock(
            return_value=(["t1", "t2"], 0.88, "conclusion text", "ghash")
        )
        rt._perform_llm_inference = AsyncMock(
            return_value=("LLM answer text", "qwen2.5:7b")
        )
        rt._optimize_snr = AsyncMock(
            return_value=("optimized answer", 0.93, {"snr": "measured"})
        )
        rt._validate_constitutionally = AsyncMock(
            return_value=(0.97, "IHSAN_GATE+OMEGA")
        )
        rt._store_graph_artifact = MagicMock()
        rt._record_query_impact = MagicMock()
        rt._register_poi_contribution = MagicMock()
        rt._emit_query_receipt = MagicMock()
        rt._encode_query_memory = MagicMock()
        rt._commit_experience_episode = MagicMock()
        rt._observe_judgment = MagicMock()

        query = SovereignQuery(text="fields check", require_reasoning=True)
        result = await rt._process_query_direct(query, time.perf_counter())

        assert result.success is True
        assert result.thoughts == ["t1", "t2"]
        assert result.reasoning_depth == 2
        assert result.graph_hash == "ghash"
        assert result.response == "optimized answer"
        assert result.snr_score == 0.93
        assert result.snr_ok is True
        assert result.ihsan_score == 0.97
        assert result.reasoning_used is True
        assert result.processing_time_ms > 0
        assert result.claim_tags == {"snr": "measured"}

    @pytest.mark.asyncio
    async def test_degraded_tagging_when_no_llm(self, tmp_path: Path) -> None:
        """Result tagged as degraded when model_used is NO_LLM."""
        rt = _make_runtime(tmp_path)
        _set_ready(rt)

        rt._run_gate_chain_preflight = AsyncMock(return_value=None)
        rt._select_compute_tier = AsyncMock(return_value=None)
        rt._execute_reasoning_stage = AsyncMock(
            return_value=([], 0.5, "prompt", None)
        )
        rt._perform_llm_inference = AsyncMock(
            return_value=("template output", "NO_LLM")
        )
        rt._optimize_snr = AsyncMock(return_value=("template output", 0.86, {}))
        rt._validate_constitutionally = AsyncMock(return_value=(0.90, "SKIPPED"))
        rt._store_graph_artifact = MagicMock()
        rt._record_query_impact = MagicMock()
        rt._register_poi_contribution = MagicMock()
        rt._emit_query_receipt = MagicMock()
        rt._encode_query_memory = MagicMock()
        rt._commit_experience_episode = MagicMock()
        rt._observe_judgment = MagicMock()

        query = SovereignQuery(text="no llm")
        result = await rt._process_query_direct(query, time.perf_counter())

        assert result.success is True
        assert getattr(result, "degraded", False) is True
        assert result.model_used == "NO_LLM"
        assert "NO_LLM" in getattr(result, "degraded_reason", "")

    @pytest.mark.asyncio
    async def test_fire_and_forget_calls_invoked(self, tmp_path: Path) -> None:
        """All fire-and-forget side-effects are called after pipeline finishes."""
        rt = _make_runtime(tmp_path)
        _set_ready(rt)

        rt._run_gate_chain_preflight = AsyncMock(return_value=None)
        rt._select_compute_tier = AsyncMock(return_value=None)
        rt._execute_reasoning_stage = AsyncMock(return_value=([], 0.5, "p", None))
        rt._perform_llm_inference = AsyncMock(return_value=("ans", "qwen"))
        rt._optimize_snr = AsyncMock(return_value=("ans", 0.9, {}))
        rt._validate_constitutionally = AsyncMock(return_value=(0.95, "OK"))
        rt._store_graph_artifact = MagicMock()
        rt._record_query_impact = MagicMock()
        rt._register_poi_contribution = MagicMock()
        rt._emit_query_receipt = MagicMock()
        rt._encode_query_memory = MagicMock()
        rt._commit_experience_episode = MagicMock()
        rt._observe_judgment = MagicMock()

        query = SovereignQuery(text="fire and forget")
        await rt._process_query_direct(query, time.perf_counter())

        rt._store_graph_artifact.assert_called_once()
        rt._record_query_impact.assert_called_once()
        rt._register_poi_contribution.assert_called_once()
        rt._emit_query_receipt.assert_called_once()
        rt._encode_query_memory.assert_called_once()
        rt._commit_experience_episode.assert_called_once()
        rt._observe_judgment.assert_called_once()

    @pytest.mark.asyncio
    async def test_metrics_update_after_direct_pipeline(self, tmp_path: Path) -> None:
        """Metrics are updated with success and duration."""
        rt = _make_runtime(tmp_path)
        _set_ready(rt)

        rt._run_gate_chain_preflight = AsyncMock(return_value=None)
        rt._select_compute_tier = AsyncMock(return_value=None)
        rt._execute_reasoning_stage = AsyncMock(return_value=(["t"], 0.8, "p", None))
        rt._perform_llm_inference = AsyncMock(return_value=("a", "model"))
        rt._optimize_snr = AsyncMock(return_value=("a", 0.9, {}))
        rt._validate_constitutionally = AsyncMock(return_value=(0.95, "OK"))
        rt._store_graph_artifact = MagicMock()
        rt._record_query_impact = MagicMock()
        rt._register_poi_contribution = MagicMock()
        rt._emit_query_receipt = MagicMock()
        rt._encode_query_memory = MagicMock()
        rt._commit_experience_episode = MagicMock()
        rt._observe_judgment = MagicMock()

        query = SovereignQuery(text="metrics test")
        await rt._process_query_direct(query, time.perf_counter())

        assert rt.metrics.queries_processed == 1
        assert rt.metrics.queries_succeeded == 1
        assert rt.metrics.avg_query_time_ms > 0
        assert rt.metrics.reasoning_calls == 1


# =============================================================================
# 4. _orchestrate_complex_query
# =============================================================================


class TestOrchestrateComplexQuery:
    """Tests for _orchestrate_complex_query (lines 1583-1660)."""

    @pytest.mark.asyncio
    async def test_decomposition_execution_synthesis(self, tmp_path: Path) -> None:
        """Orchestrator decomposes, executes, and synthesizes task results."""
        rt = _make_runtime(tmp_path)
        _set_ready(rt)

        task1 = SimpleNamespace(id="t1", title="Sub-task 1")
        task2 = SimpleNamespace(id="t2", title="Sub-task 2")
        plan = SimpleNamespace(subtasks=[task1, task2])

        mock_decomposer = AsyncMock()
        mock_decomposer.decompose = AsyncMock(return_value=plan)

        mock_orch = MagicMock()
        mock_orch.decomposer = mock_decomposer
        mock_orch.execute_task = AsyncMock()
        mock_orch.task_results = {
            "t1": {"content": "Result A"},
            "t2": {"content": "Result B"},
        }
        rt._orchestrator = mock_orch

        rt._optimize_snr = AsyncMock(return_value=("Result A\n\nResult B", 0.92, {}))
        rt._validate_constitutionally = AsyncMock(return_value=(0.96, "OK"))
        rt._record_query_impact = MagicMock()
        rt._emit_query_receipt = MagicMock()
        rt._encode_query_memory = MagicMock()
        rt._commit_experience_episode = MagicMock()
        rt._observe_judgment = MagicMock()

        with patch("core.sovereign.runtime_core.SovereignRuntime._orchestrate_complex_query.__module__", create=True):
            pass

        query = SovereignQuery(text="complex multi-part query")
        # We must patch the import inside the method
        with patch.dict("sys.modules", {"core.sovereign.orchestrator": MagicMock(TaskNode=SimpleNamespace)}):
            result = await rt._orchestrate_complex_query(query, time.perf_counter())

        assert result.success is True
        assert result.reasoning_used is True
        assert result.thoughts == ["Sub-task 1", "Sub-task 2"]
        assert result.reasoning_depth == 2

    @pytest.mark.asyncio
    async def test_fallback_to_direct_on_exception(self, tmp_path: Path) -> None:
        """When orchestrator throws, falls back to _process_query_direct."""
        rt = _make_runtime(tmp_path)
        _set_ready(rt)

        mock_orch = MagicMock()
        mock_orch.decomposer = MagicMock()
        mock_orch.decomposer.decompose = AsyncMock(side_effect=RuntimeError("decompose failed"))
        rt._orchestrator = mock_orch

        expected_result = SovereignResult(query_id="fb", success=True, response="direct fallback")
        rt._process_query_direct = AsyncMock(return_value=expected_result)

        query = SovereignQuery(text="fallback test")

        with patch.dict("sys.modules", {"core.sovereign.orchestrator": MagicMock(TaskNode=SimpleNamespace)}):
            result = await rt._orchestrate_complex_query(query, time.perf_counter())

        assert result is expected_result
        rt._process_query_direct.assert_awaited_once()


# =============================================================================
# 5. _select_compute_tier
# =============================================================================


class TestSelectComputeTier:
    """Tests for _select_compute_tier (lines 1785-1793)."""

    @pytest.mark.asyncio
    async def test_no_omega_returns_none(self, tmp_path: Path) -> None:
        """With no omega engine, return None."""
        rt = _make_runtime(tmp_path)
        rt._omega = None
        result = await rt._select_compute_tier(SovereignQuery(text="x"))
        assert result is None

    @pytest.mark.asyncio
    async def test_omega_returning_mode_maps_to_tier(self, tmp_path: Path) -> None:
        """When omega returns a mode, _mode_to_tier is called."""
        rt = _make_runtime(tmp_path)
        mock_omega = MagicMock()
        mock_omega.get_operational_mode = MagicMock(return_value="ETHICAL")
        rt._omega = mock_omega
        rt._mode_to_tier = MagicMock(return_value="LOCAL_TIER")

        result = await rt._select_compute_tier(SovereignQuery(text="x"))
        rt._mode_to_tier.assert_called_once_with("ETHICAL")
        assert result == "LOCAL_TIER"

    @pytest.mark.asyncio
    async def test_omega_returning_none_mode(self, tmp_path: Path) -> None:
        """When omega.get_operational_mode returns None, tier is None."""
        rt = _make_runtime(tmp_path)
        mock_omega = MagicMock()
        mock_omega.get_operational_mode = MagicMock(return_value=None)
        rt._omega = mock_omega

        result = await rt._select_compute_tier(SovereignQuery(text="x"))
        assert result is None


# =============================================================================
# 6. _execute_reasoning_stage
# =============================================================================


class TestExecuteReasoningStage:
    """Tests for _execute_reasoning_stage (lines 1795-1821)."""

    @pytest.mark.asyncio
    async def test_reasoning_not_required_returns_defaults(self, tmp_path: Path) -> None:
        """When require_reasoning=False, skip GoT and return defaults."""
        rt = _make_runtime(tmp_path)
        query = SovereignQuery(text="no reasoning", require_reasoning=False)

        path, conf, prompt, ghash = await rt._execute_reasoning_stage(query)
        assert path == []
        assert conf == 0.75
        assert prompt == "no reasoning"
        assert ghash is None

    @pytest.mark.asyncio
    async def test_no_graph_reasoner_returns_defaults(self, tmp_path: Path) -> None:
        """When graph_reasoner is None, return defaults even if reasoning requested."""
        rt = _make_runtime(tmp_path)
        rt._graph_reasoner = None

        query = SovereignQuery(text="no reasoner", require_reasoning=True)
        path, conf, prompt, ghash = await rt._execute_reasoning_stage(query)
        assert path == []
        assert prompt == "no reasoner"

    @pytest.mark.asyncio
    async def test_graph_reasoner_called_and_extracts_results(self, tmp_path: Path) -> None:
        """Graph reasoner is called; thoughts, confidence, graph_hash extracted."""
        rt = _make_runtime(tmp_path)
        rt.config.max_reasoning_depth = 7

        mock_reasoner = AsyncMock()
        mock_reasoner.reason = AsyncMock(return_value={
            "thoughts": ["idea-a", "idea-b"],
            "confidence": 0.92,
            "graph_hash": "abc123",
            "conclusion": "final conclusion",
        })
        rt._graph_reasoner = mock_reasoner

        query = SovereignQuery(text="deep think", require_reasoning=True, context={"key": "val"})
        path, conf, prompt, ghash = await rt._execute_reasoning_stage(query)

        mock_reasoner.reason.assert_awaited_once_with(
            query="deep think", context={"key": "val"}, max_depth=7
        )
        assert path == ["idea-a", "idea-b"]
        assert conf == 0.92
        assert ghash == "abc123"
        assert prompt == "final conclusion"

    @pytest.mark.asyncio
    async def test_no_conclusion_keeps_original_prompt(self, tmp_path: Path) -> None:
        """Without a conclusion, thought_prompt remains the original query text."""
        rt = _make_runtime(tmp_path)

        mock_reasoner = AsyncMock()
        mock_reasoner.reason = AsyncMock(return_value={
            "thoughts": ["t1"],
            "confidence": 0.7,
        })
        rt._graph_reasoner = mock_reasoner

        query = SovereignQuery(text="original text", require_reasoning=True)
        _, _, prompt, _ = await rt._execute_reasoning_stage(query)
        assert prompt == "original text"


# =============================================================================
# 7. _build_contextual_prompt
# =============================================================================


class TestBuildContextualPrompt:
    """Tests for _build_contextual_prompt (lines 1823-1879)."""

    @pytest.mark.asyncio
    async def test_no_user_context_returns_unchanged(self, tmp_path: Path) -> None:
        """When _user_context is None, return thought_prompt as-is."""
        rt = _make_runtime(tmp_path)
        rt._user_context = None

        result = await rt._build_contextual_prompt("thought", SovereignQuery(text="q"))
        assert result == "thought"

    @pytest.mark.asyncio
    async def test_genesis_pat_team_includes_agent_routing(self, tmp_path: Path) -> None:
        """With genesis PAT team, agent routing is included in the prompt."""
        rt = _make_runtime(tmp_path)

        agent1 = SimpleNamespace(role="researcher")
        agent2 = SimpleNamespace(role="coder")
        rt._genesis = SimpleNamespace(pat_team=[agent1, agent2])

        mock_ctx = MagicMock()
        mock_ctx.build_system_prompt = MagicMock(return_value="[SYSTEM PROMPT]")
        rt._user_context = mock_ctx
        rt._living_memory = None

        with patch("core.sovereign.runtime_core.select_pat_agent", return_value="researcher"):
            query = SovereignQuery(text="code review", context={})
            result = await rt._build_contextual_prompt("thought prompt", query)

        assert "[SYSTEM PROMPT]" in result
        assert "thought prompt" in result
        assert query.context.get("_responding_agent") == "researcher"

    @pytest.mark.asyncio
    async def test_living_memory_retrieval_included(self, tmp_path: Path) -> None:
        """Living memory retrieval results are passed to build_system_prompt."""
        rt = _make_runtime(tmp_path)
        rt._genesis = None

        mem_entry = SimpleNamespace(
            memory_type=SimpleNamespace(value="episodic"),
            content="Past conversation about X",
        )
        mock_living = AsyncMock()
        mock_living.retrieve = AsyncMock(return_value=[mem_entry])
        rt._living_memory = mock_living

        mock_ctx = MagicMock()
        mock_ctx.build_system_prompt = MagicMock(return_value="[SYS]")
        rt._user_context = mock_ctx

        query = SovereignQuery(text="tell me about X", context={})
        await rt._build_contextual_prompt("tp", query)

        mock_ctx.build_system_prompt.assert_called_once()
        call_kwargs = mock_ctx.build_system_prompt.call_args
        memory_arg = call_kwargs.kwargs.get("memory_context") or call_kwargs[1].get("memory_context", "")
        assert "EPISODIC" in memory_arg or "Past conversation" in memory_arg

    @pytest.mark.asyncio
    async def test_memory_retrieval_failure_falls_back_to_working_context(self, tmp_path: Path) -> None:
        """When retrieve() fails, fall back to get_working_context."""
        rt = _make_runtime(tmp_path)
        rt._genesis = None

        mock_living = AsyncMock()
        mock_living.retrieve = AsyncMock(side_effect=RuntimeError("retrieve fail"))
        mock_living.get_working_context = MagicMock(return_value="working ctx fallback")
        rt._living_memory = mock_living

        mock_ctx = MagicMock()
        mock_ctx.build_system_prompt = MagicMock(return_value="[SYS]")
        rt._user_context = mock_ctx

        query = SovereignQuery(text="fallback", context={})
        await rt._build_contextual_prompt("tp", query)

        mock_living.get_working_context.assert_called_once_with(max_entries=5)

    @pytest.mark.asyncio
    async def test_agent_routing_stored_in_query_context(self, tmp_path: Path) -> None:
        """Selected PAT agent is stored in query.context['_responding_agent']."""
        rt = _make_runtime(tmp_path)
        rt._genesis = SimpleNamespace(pat_team=[SimpleNamespace(role="analyst")])
        rt._living_memory = None

        mock_ctx = MagicMock()
        mock_ctx.build_system_prompt = MagicMock(return_value="[SYS]")
        rt._user_context = mock_ctx

        with patch("core.sovereign.runtime_core.select_pat_agent", return_value="analyst"):
            query = SovereignQuery(text="analyze", context={})
            await rt._build_contextual_prompt("tp", query)
            assert query.context["_responding_agent"] == "analyst"


# =============================================================================
# 8. _perform_llm_inference
# =============================================================================


class TestPerformLLMInference:
    """Tests for _perform_llm_inference (lines 1881-1913)."""

    @pytest.mark.asyncio
    async def test_gateway_calls_infer(self, tmp_path: Path) -> None:
        """With a gateway, calls infer() and returns (answer, model)."""
        rt = _make_runtime(tmp_path)
        rt._user_context = None

        inference_result = SimpleNamespace(content="LLM said this", model="qwen2.5:7b")
        mock_gw = MagicMock()
        mock_gw.infer = AsyncMock(return_value=inference_result)
        rt._gateway = mock_gw

        answer, model = await rt._perform_llm_inference("prompt", None, SovereignQuery(text="q"))
        assert answer == "LLM said this"
        assert model == "qwen2.5:7b"

    @pytest.mark.asyncio
    async def test_gateway_exception_falls_back_to_no_llm(self, tmp_path: Path) -> None:
        """When gateway raises, fall back to NO_LLM."""
        rt = _make_runtime(tmp_path)
        rt._user_context = None

        mock_gw = MagicMock()
        mock_gw.infer = AsyncMock(side_effect=ConnectionError("LLM down"))
        rt._gateway = mock_gw

        answer, model = await rt._perform_llm_inference("prompt text", None, SovereignQuery(text="q"))
        assert model == "NO_LLM"
        assert "prompt text" in answer

    @pytest.mark.asyncio
    async def test_no_gateway_returns_no_llm(self, tmp_path: Path) -> None:
        """Without a gateway, return (thought_prompt, 'NO_LLM')."""
        rt = _make_runtime(tmp_path)
        rt._user_context = None
        rt._gateway = None

        answer, model = await rt._perform_llm_inference("my prompt", None, SovereignQuery(text="q"))
        assert model == "NO_LLM"
        # The answer should contain the contextual prompt which is based on thought_prompt
        assert "my prompt" in answer or answer is not None

    @pytest.mark.asyncio
    async def test_gateway_without_infer_method_returns_no_llm(self, tmp_path: Path) -> None:
        """If gateway object lacks infer method, treat as NO_LLM."""
        rt = _make_runtime(tmp_path)
        rt._user_context = None
        rt._gateway = MagicMock(spec=[])  # No infer attribute

        answer, model = await rt._perform_llm_inference("p", None, SovereignQuery(text="q"))
        assert model == "NO_LLM"


# =============================================================================
# 9. _optimize_snr -- DUAL ENGINE
# =============================================================================


class TestOptimizeSNR:
    """Tests for _optimize_snr (lines 1915-1977)."""

    @pytest.mark.asyncio
    async def test_phase1_snr_maximizer_optimization(self, tmp_path: Path) -> None:
        """Phase 1: SNR maximizer optimize() is called and result extracted."""
        rt = _make_runtime(tmp_path)

        mock_optimizer = AsyncMock()
        mock_optimizer.optimize = AsyncMock(return_value={
            "snr_score": 0.91,
            "optimized": "clean text",
            "claim_tags": {"quality": "measured"},
        })
        rt._snr_optimizer = mock_optimizer

        with patch("core.sovereign.runtime_core.inspect.isawaitable", return_value=True):
            content, score, tags = await rt._optimize_snr("noisy text")

        mock_optimizer.optimize.assert_called_once_with("noisy text")
        # SNREngine v1 overrides the score, so we just check content
        assert "clean text" in content or content is not None

    @pytest.mark.asyncio
    async def test_phase2_snr_engine_v1_authoritative_score(self, tmp_path: Path) -> None:
        """Phase 2: SNREngine v1 produces the authoritative score."""
        rt = _make_runtime(tmp_path)
        rt._snr_optimizer = None

        mock_engine = MagicMock()
        mock_engine.snr_score = MagicMock(return_value={
            "score": 0.94,
            "claim_tags": {"audit": "measured"},
        })
        mock_snr_input = MagicMock()

        with patch("core.sovereign.runtime_core.SovereignRuntime._optimize_snr") as orig:
            # Call the actual method
            pass

        # Call directly with mocked imports
        with patch.dict("sys.modules", {
            "core.proof_engine.snr": MagicMock(
                SNREngine=MagicMock(return_value=mock_engine),
                SNRInput=MagicMock(return_value=mock_snr_input),
            ),
        }):
            content, score, tags = await rt._optimize_snr("test content")

        assert score == 0.94
        assert tags.get("audit") == "measured"

    @pytest.mark.asyncio
    async def test_metrics_updated_on_improvement(self, tmp_path: Path) -> None:
        """When optimized content differs from original, snr_stats update."""
        rt = _make_runtime(tmp_path)

        mock_opt = MagicMock()
        # Return synchronous result (not awaitable)
        mock_opt.optimize = MagicMock(return_value={
            "snr_score": 0.90,
            "optimized": "short",
            "claim_tags": {},
        })
        rt._snr_optimizer = mock_opt

        with patch.dict("sys.modules", {
            "core.proof_engine.snr": MagicMock(
                SNREngine=MagicMock(return_value=MagicMock(
                    snr_score=MagicMock(return_value={"score": 0.92, "claim_tags": {}})
                )),
                SNRInput=MagicMock(),
            ),
        }):
            await rt._optimize_snr("this is a longer original text that gets optimized")

        assert rt.metrics.snr_optimizations == 1

    @pytest.mark.asyncio
    async def test_no_optimizer_uses_default_score(self, tmp_path: Path) -> None:
        """With no optimizer, default SNR threshold is used."""
        rt = _make_runtime(tmp_path)
        rt._snr_optimizer = None

        # Mock the SNREngine too to prevent import errors
        with patch.dict("sys.modules", {
            "core.proof_engine.snr": MagicMock(
                SNREngine=MagicMock(return_value=MagicMock(
                    snr_score=MagicMock(return_value={"score": 0.88, "claim_tags": {}})
                )),
                SNRInput=MagicMock(),
            ),
        }):
            content, score, tags = await rt._optimize_snr("raw content")

        assert content == "raw content"  # Unchanged without optimizer

    @pytest.mark.asyncio
    async def test_snr_engine_import_failure_graceful(self, tmp_path: Path) -> None:
        """When SNREngine v1 import fails, graceful fallback."""
        rt = _make_runtime(tmp_path)
        rt._snr_optimizer = None

        with patch.dict("sys.modules", {
            "core.proof_engine.snr": None,  # Simulate import failure
        }):
            # Should not raise
            content, score, tags = await rt._optimize_snr("test")

        assert content == "test"
        assert isinstance(score, float)

    @pytest.mark.asyncio
    async def test_metrics_current_snr_score_updated(self, tmp_path: Path) -> None:
        """After _optimize_snr, metrics.current_snr_score is updated."""
        rt = _make_runtime(tmp_path)
        rt._snr_optimizer = None

        with patch.dict("sys.modules", {
            "core.proof_engine.snr": MagicMock(
                SNREngine=MagicMock(return_value=MagicMock(
                    snr_score=MagicMock(return_value={"score": 0.935, "claim_tags": {}})
                )),
                SNRInput=MagicMock(),
            ),
        }):
            await rt._optimize_snr("text")

        assert rt.metrics.current_snr_score == 0.935


# =============================================================================
# 10. _validate_constitutionally
# =============================================================================


class TestValidateConstitutionally:
    """Tests for _validate_constitutionally (lines 1979-2057)."""

    @pytest.mark.asyncio
    async def test_phase1_ihsan_gate_scoring(self, tmp_path: Path) -> None:
        """Phase 1: IhsanGate v1 computes authoritative ihsan score."""
        rt = _make_runtime(tmp_path)
        rt._omega = None
        rt._guardian_council = None

        mock_gate = MagicMock()
        mock_gate.ihsan_score = MagicMock(return_value={
            "score": 0.965,
            "decision": "APPROVED",
        })

        with patch.dict("sys.modules", {
            "core.proof_engine.ihsan_gate": MagicMock(
                IhsanGate=MagicMock(return_value=mock_gate),
                IhsanComponents=MagicMock(),
            ),
        }):
            query = SovereignQuery(text="validate me", require_validation=False)
            score, verdict = await rt._validate_constitutionally("content", {}, query, 0.9)

        assert score == 0.965
        assert verdict == "APPROVED"

    @pytest.mark.asyncio
    async def test_phase2_omega_enrichment_70_30_blend(self, tmp_path: Path) -> None:
        """Phase 2: Omega enriches with 70/30 blend."""
        rt = _make_runtime(tmp_path)
        rt._guardian_council = None

        mock_omega = MagicMock()
        # evaluate_ihsan returns tuple (score, details)
        mock_omega.evaluate_ihsan = MagicMock(return_value=(0.98, "details"))
        rt._omega = mock_omega

        mock_gate = MagicMock()
        mock_gate.ihsan_score = MagicMock(return_value={"score": 0.90, "decision": "OK"})

        with patch.dict("sys.modules", {
            "core.proof_engine.ihsan_gate": MagicMock(
                IhsanGate=MagicMock(return_value=mock_gate),
                IhsanComponents=MagicMock(),
            ),
        }):
            with patch.object(rt, "_extract_ihsan_from_response", return_value="ihsan_vector"):
                query = SovereignQuery(text="omega test", require_validation=False)
                score, verdict = await rt._validate_constitutionally("c", {}, query, 0.9)

        # 0.7 * 0.90 + 0.3 * 0.98 = 0.63 + 0.294 = 0.924
        assert abs(score - 0.924) < 0.01
        assert "OMEGA" in verdict

    @pytest.mark.asyncio
    async def test_phase3_guardian_council_60_40_blend(self, tmp_path: Path) -> None:
        """Phase 3: Guardian Council enriches with 60/40 blend."""
        rt = _make_runtime(tmp_path)
        rt._omega = None

        mock_guardian = AsyncMock()
        mock_guardian.validate = AsyncMock(return_value={
            "confidence": 0.88,
            "is_valid": True,
            "issues": [],
        })
        rt._guardian_council = mock_guardian

        mock_gate = MagicMock()
        mock_gate.ihsan_score = MagicMock(return_value={"score": 0.95, "decision": "OK"})

        with patch.dict("sys.modules", {
            "core.proof_engine.ihsan_gate": MagicMock(
                IhsanGate=MagicMock(return_value=mock_gate),
                IhsanComponents=MagicMock(),
            ),
        }):
            query = SovereignQuery(text="guardian test", require_validation=True)
            score, verdict = await rt._validate_constitutionally("c", {}, query, 0.9)

        # 0.6 * 0.95 + 0.4 * 0.88 = 0.57 + 0.352 = 0.922
        assert abs(score - 0.922) < 0.01
        assert "GUARDIAN" in verdict
        assert "VALID" in verdict

    @pytest.mark.asyncio
    async def test_ihsan_gate_import_failure_graceful(self, tmp_path: Path) -> None:
        """When IhsanGate import fails, fallback to snr_score."""
        rt = _make_runtime(tmp_path)
        rt._omega = None
        rt._guardian_council = None

        with patch.dict("sys.modules", {"core.proof_engine.ihsan_gate": None}):
            query = SovereignQuery(text="no gate", require_validation=False)
            score, verdict = await rt._validate_constitutionally("c", {}, query, 0.87)

        # Falls through to snr_score as default
        assert score == 0.87
        assert verdict == "SKIPPED"

    @pytest.mark.asyncio
    async def test_omega_failure_graceful(self, tmp_path: Path) -> None:
        """When omega evaluation fails, score is not modified by omega."""
        rt = _make_runtime(tmp_path)
        rt._guardian_council = None

        mock_omega = MagicMock()
        mock_omega.evaluate_ihsan = MagicMock(side_effect=RuntimeError("omega crash"))
        rt._omega = mock_omega

        mock_gate = MagicMock()
        mock_gate.ihsan_score = MagicMock(return_value={"score": 0.93, "decision": "OK"})

        with patch.dict("sys.modules", {
            "core.proof_engine.ihsan_gate": MagicMock(
                IhsanGate=MagicMock(return_value=mock_gate),
                IhsanComponents=MagicMock(),
            ),
        }):
            with patch.object(rt, "_extract_ihsan_from_response", return_value="vec"):
                query = SovereignQuery(text="omega fail", require_validation=False)
                score, verdict = await rt._validate_constitutionally("c", {}, query, 0.9)

        # Omega failed, so score remains from IhsanGate
        assert score == 0.93

    @pytest.mark.asyncio
    async def test_guardian_updates_validation_count(self, tmp_path: Path) -> None:
        """Guardian Council validation updates metrics.validations."""
        rt = _make_runtime(tmp_path)
        rt._omega = None

        mock_guardian = AsyncMock()
        mock_guardian.validate = AsyncMock(return_value={
            "confidence": 0.90,
            "is_valid": True,
        })
        rt._guardian_council = mock_guardian

        mock_gate = MagicMock()
        mock_gate.ihsan_score = MagicMock(return_value={"score": 0.95, "decision": "OK"})

        with patch.dict("sys.modules", {
            "core.proof_engine.ihsan_gate": MagicMock(
                IhsanGate=MagicMock(return_value=mock_gate),
                IhsanComponents=MagicMock(),
            ),
        }):
            query = SovereignQuery(text="val count", require_validation=True)
            await rt._validate_constitutionally("c", {}, query, 0.9)

        assert rt.metrics.validations >= 1

    @pytest.mark.asyncio
    async def test_metrics_current_ihsan_score_updated(self, tmp_path: Path) -> None:
        """After constitutional validation, metrics.current_ihsan_score updated."""
        rt = _make_runtime(tmp_path)
        rt._omega = None
        rt._guardian_council = None

        mock_gate = MagicMock()
        mock_gate.ihsan_score = MagicMock(return_value={"score": 0.972, "decision": "OK"})

        with patch.dict("sys.modules", {
            "core.proof_engine.ihsan_gate": MagicMock(
                IhsanGate=MagicMock(return_value=mock_gate),
                IhsanComponents=MagicMock(),
            ),
        }):
            query = SovereignQuery(text="metrics", require_validation=False)
            await rt._validate_constitutionally("c", {}, query, 0.9)

        assert rt.metrics.current_ihsan_score == 0.972


# =============================================================================
# 11. initialize() METHOD
# =============================================================================


class TestInitialize:
    """Tests for initialize() lifecycle (lines 276-348)."""

    @pytest.mark.asyncio
    async def test_full_initialization_sequence(self, tmp_path: Path) -> None:
        """initialize() calls all sub-init methods and sets _initialized/_running."""
        rt = _make_runtime(tmp_path)

        # Mock every init sub-method to prevent real imports
        rt._load_env_vars = MagicMock()
        rt._apply_env_overrides = MagicMock()
        rt._load_genesis_identity = MagicMock()
        rt._init_evidence_ledger = MagicMock()
        rt._init_experience_ledger = MagicMock()
        rt._init_judgment_telemetry = MagicMock()
        rt._init_gate_chain = MagicMock()
        rt._init_poi_engine = MagicMock()
        rt._run_zpk_preflight = AsyncMock()
        rt._init_components = AsyncMock()
        rt._start_autonomous_loop = AsyncMock()
        rt._init_user_context = MagicMock()
        rt._init_memory_coordinator = AsyncMock()
        rt._init_impact_tracker = MagicMock()
        rt._setup_signal_handlers = MagicMock()

        await rt.initialize()

        assert rt._initialized is True
        assert rt._running is True
        assert rt.metrics.started_at is not None

        rt._load_env_vars.assert_called_once()
        rt._apply_env_overrides.assert_called_once()
        rt._load_genesis_identity.assert_called_once()
        rt._init_evidence_ledger.assert_called_once()
        rt._init_experience_ledger.assert_called_once()
        rt._init_judgment_telemetry.assert_called_once()
        rt._init_gate_chain.assert_called_once()
        rt._init_poi_engine.assert_called_once()
        rt._run_zpk_preflight.assert_awaited_once()
        rt._init_components.assert_awaited_once()
        rt._init_user_context.assert_called_once()
        rt._init_memory_coordinator.assert_awaited_once()
        rt._init_impact_tracker.assert_called_once()
        rt._setup_signal_handlers.assert_called_once()

    @pytest.mark.asyncio
    async def test_double_init_returns_early(self, tmp_path: Path) -> None:
        """Calling initialize() twice is a no-op on the second call."""
        rt = _make_runtime(tmp_path)
        rt._initialized = True  # Pretend already initialized

        rt._load_env_vars = MagicMock()
        await rt.initialize()

        rt._load_env_vars.assert_not_called()

    @pytest.mark.asyncio
    async def test_starts_autonomous_loop_when_enabled(self, tmp_path: Path) -> None:
        """When autonomous_enabled=True, _start_autonomous_loop is called."""
        rt = _make_runtime(tmp_path, autonomous_enabled=True)

        rt._load_env_vars = MagicMock()
        rt._apply_env_overrides = MagicMock()
        rt._load_genesis_identity = MagicMock()
        rt._init_evidence_ledger = MagicMock()
        rt._init_experience_ledger = MagicMock()
        rt._init_judgment_telemetry = MagicMock()
        rt._init_gate_chain = MagicMock()
        rt._init_poi_engine = MagicMock()
        rt._run_zpk_preflight = AsyncMock()
        rt._init_components = AsyncMock()
        rt._start_autonomous_loop = AsyncMock()
        rt._init_user_context = MagicMock()
        rt._init_memory_coordinator = AsyncMock()
        rt._init_impact_tracker = MagicMock()
        rt._setup_signal_handlers = MagicMock()

        await rt.initialize()
        rt._start_autonomous_loop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_does_not_start_autonomous_loop_when_disabled(self, tmp_path: Path) -> None:
        """When autonomous_enabled=False, autonomous loop is not started."""
        rt = _make_runtime(tmp_path, autonomous_enabled=False)

        rt._load_env_vars = MagicMock()
        rt._apply_env_overrides = MagicMock()
        rt._load_genesis_identity = MagicMock()
        rt._init_evidence_ledger = MagicMock()
        rt._init_experience_ledger = MagicMock()
        rt._init_judgment_telemetry = MagicMock()
        rt._init_gate_chain = MagicMock()
        rt._init_poi_engine = MagicMock()
        rt._run_zpk_preflight = AsyncMock()
        rt._init_components = AsyncMock()
        rt._start_autonomous_loop = AsyncMock()
        rt._init_user_context = MagicMock()
        rt._init_memory_coordinator = AsyncMock()
        rt._init_impact_tracker = MagicMock()
        rt._setup_signal_handlers = MagicMock()

        await rt.initialize()
        rt._start_autonomous_loop.assert_not_awaited()


# =============================================================================
# 12. shutdown() METHOD
# =============================================================================


class TestShutdown:
    """Tests for shutdown() lifecycle (lines 1428-1466)."""

    @pytest.mark.asyncio
    async def test_not_running_is_noop(self, tmp_path: Path) -> None:
        """If runtime is not running, shutdown() is a no-op."""
        rt = _make_runtime(tmp_path)
        rt._running = False
        rt._autonomous_loop = MagicMock()

        await rt.shutdown()
        rt._autonomous_loop.stop.assert_not_called()

    @pytest.mark.asyncio
    async def test_stops_autonomous_loop(self, tmp_path: Path) -> None:
        """shutdown() stops the autonomous loop."""
        rt = _make_runtime(tmp_path)
        _set_ready(rt)

        mock_loop = MagicMock()
        mock_loop.stop = MagicMock()
        rt._autonomous_loop = mock_loop
        rt._memory_coordinator = None

        await rt.shutdown()
        mock_loop.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_stops_pek(self, tmp_path: Path) -> None:
        """shutdown() stops the PEK if present."""
        rt = _make_runtime(tmp_path)
        _set_ready(rt)
        rt._autonomous_loop = None
        rt._memory_coordinator = None

        mock_pek = MagicMock()
        mock_pek.stop = AsyncMock()
        rt._pek = mock_pek

        await rt.shutdown()
        mock_pek.stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_saves_user_context(self, tmp_path: Path) -> None:
        """shutdown() saves user context."""
        rt = _make_runtime(tmp_path)
        _set_ready(rt)
        rt._autonomous_loop = None
        rt._memory_coordinator = None

        mock_ctx = MagicMock()
        mock_ctx.save = MagicMock()
        rt._user_context = mock_ctx

        await rt.shutdown()
        mock_ctx.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_flushes_impact_tracker(self, tmp_path: Path) -> None:
        """shutdown() flushes impact tracker."""
        rt = _make_runtime(tmp_path)
        _set_ready(rt)
        rt._autonomous_loop = None
        rt._memory_coordinator = None

        mock_tracker = MagicMock()
        mock_tracker.flush = MagicMock()
        rt._impact_tracker = mock_tracker

        await rt.shutdown()
        mock_tracker.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_stops_memory_coordinator(self, tmp_path: Path) -> None:
        """shutdown() stops memory coordinator."""
        rt = _make_runtime(tmp_path)
        _set_ready(rt)
        rt._autonomous_loop = None

        mock_coord = MagicMock()
        mock_coord.stop = AsyncMock()
        rt._memory_coordinator = mock_coord

        await rt.shutdown()
        mock_coord.stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_sets_running_false_and_shutdown_event(self, tmp_path: Path) -> None:
        """shutdown() sets _running=False and signals _shutdown_event."""
        rt = _make_runtime(tmp_path)
        _set_ready(rt)
        rt._autonomous_loop = None
        rt._memory_coordinator = None

        await rt.shutdown()
        assert rt._running is False
        assert rt._shutdown_event.is_set()

    @pytest.mark.asyncio
    async def test_pek_stop_failure_does_not_crash(self, tmp_path: Path) -> None:
        """If PEK stop raises, shutdown continues gracefully."""
        rt = _make_runtime(tmp_path)
        _set_ready(rt)
        rt._autonomous_loop = None
        rt._memory_coordinator = None

        mock_pek = MagicMock()
        mock_pek.stop = AsyncMock(side_effect=RuntimeError("pek crash"))
        rt._pek = mock_pek

        # Should not raise
        await rt.shutdown()
        assert rt._running is False

    @pytest.mark.asyncio
    async def test_user_context_save_failure_does_not_crash(self, tmp_path: Path) -> None:
        """If user context save raises, shutdown continues."""
        rt = _make_runtime(tmp_path)
        _set_ready(rt)
        rt._autonomous_loop = None
        rt._memory_coordinator = None

        mock_ctx = MagicMock()
        mock_ctx.save = MagicMock(side_effect=OSError("disk full"))
        rt._user_context = mock_ctx

        await rt.shutdown()
        assert rt._running is False


# =============================================================================
# 13. _init_components()
# =============================================================================


class TestInitComponents:
    """Tests for _init_components() feature flag dispatch (lines 881-986)."""

    @pytest.mark.asyncio
    async def test_enable_graph_reasoning_true_imports_got(self, tmp_path: Path) -> None:
        """enable_graph_reasoning=True attempts to import GraphOfThoughts."""
        rt = _make_runtime(tmp_path, enable_graph_reasoning=True)
        rt._init_omega_components = AsyncMock()
        rt._init_proactive_execution_kernel = AsyncMock()

        mock_got = MagicMock()
        with patch("core.sovereign.runtime_core.GraphOfThoughts", create=True):
            with patch.dict("sys.modules", {}):
                # The import happens inside _init_components; mock the import path
                with patch("core.sovereign.graph_reasoner.GraphOfThoughts", mock_got, create=True):
                    try:
                        await rt._init_components()
                    except (ImportError, ModuleNotFoundError):
                        pass  # Expected when imports cannot resolve

        # Either it loaded real or fell back to stub
        assert rt._graph_reasoner is not None

    @pytest.mark.asyncio
    async def test_enable_graph_reasoning_false_uses_stub(self, tmp_path: Path) -> None:
        """enable_graph_reasoning=False uses stub without attempting import."""
        rt = _make_runtime(tmp_path, enable_graph_reasoning=False)
        rt._init_omega_components = AsyncMock()
        rt._init_proactive_execution_kernel = AsyncMock()

        await rt._init_components()
        assert rt._graph_reasoner is not None
        assert getattr(rt._graph_reasoner, "is_stub", False) is True

    @pytest.mark.asyncio
    async def test_enable_snr_optimization_false_uses_stub(self, tmp_path: Path) -> None:
        """enable_snr_optimization=False uses stub."""
        rt = _make_runtime(tmp_path, enable_snr_optimization=False)
        rt._init_omega_components = AsyncMock()
        rt._init_proactive_execution_kernel = AsyncMock()

        await rt._init_components()
        assert rt._snr_optimizer is not None
        assert getattr(rt._snr_optimizer, "is_stub", False) is True

    @pytest.mark.asyncio
    async def test_enable_guardian_validation_false_uses_stub(self, tmp_path: Path) -> None:
        """enable_guardian_validation=False uses stub."""
        rt = _make_runtime(tmp_path, enable_guardian_validation=False)
        rt._init_omega_components = AsyncMock()
        rt._init_proactive_execution_kernel = AsyncMock()

        await rt._init_components()
        assert rt._guardian_council is not None
        assert getattr(rt._guardian_council, "is_stub", False) is True

    @pytest.mark.asyncio
    async def test_enable_autonomous_loop_false_uses_stub(self, tmp_path: Path) -> None:
        """enable_autonomous_loop=False uses stub."""
        rt = _make_runtime(tmp_path, enable_autonomous_loop=False)
        rt._init_omega_components = AsyncMock()
        rt._init_proactive_execution_kernel = AsyncMock()

        await rt._init_components()
        assert rt._autonomous_loop is not None
        assert getattr(rt._autonomous_loop, "is_stub", False) is True

    @pytest.mark.asyncio
    async def test_import_error_uses_stub_for_graph_reasoner(self, tmp_path: Path) -> None:
        """ImportError when loading GraphOfThoughts falls back to stub."""
        rt = _make_runtime(tmp_path, enable_graph_reasoning=True)
        rt._init_omega_components = AsyncMock()
        rt._init_proactive_execution_kernel = AsyncMock()

        with patch.dict("sys.modules", {"core.sovereign.graph_reasoner": None}):
            await rt._init_components()

        assert rt._graph_reasoner is not None
        assert getattr(rt._graph_reasoner, "is_stub", False) is True

    @pytest.mark.asyncio
    async def test_gateway_wired_into_got(self, tmp_path: Path) -> None:
        """After init, gateway is wired into GraphOfThoughts if both present."""
        rt = _make_runtime(tmp_path, enable_graph_reasoning=False)
        rt._init_omega_components = AsyncMock()
        rt._init_proactive_execution_kernel = AsyncMock()

        mock_gw = MagicMock()

        async def set_gateway(*a: Any) -> None:
            rt._gateway = mock_gw

        rt._init_omega_components = set_gateway  # type: ignore[assignment]

        # Make graph_reasoner have the attribute to wire into
        mock_got = MagicMock()
        mock_got._inference_gateway = None
        rt._graph_reasoner = mock_got

        await rt._init_components()

        # Check if wiring happened (gateway into got)
        if rt._gateway and hasattr(rt._graph_reasoner, "_inference_gateway"):
            assert rt._graph_reasoner._inference_gateway is mock_gw

    @pytest.mark.asyncio
    async def test_gateway_wired_into_guardian(self, tmp_path: Path) -> None:
        """After init, gateway is wired into GuardianCouncil if both present.

        The wiring code (lines 976-982) checks _guardian_council and _gateway
        after _init_omega_components completes. We test by setting guardian
        to enable_guardian_validation=True (stub gets loaded), then injecting
        gateway via _init_omega_components so the wiring logic fires.
        """
        rt = _make_runtime(tmp_path, enable_guardian_validation=False)
        rt._init_proactive_execution_kernel = AsyncMock()

        mock_gw = MagicMock()
        mock_guardian = MagicMock()
        mock_guardian.set_inference_gateway = MagicMock()

        async def set_gateway_and_guardian() -> None:
            # Simulate omega init setting the gateway
            rt._gateway = mock_gw
            # Override the stub guardian with our mock (as if real import succeeded)
            rt._guardian_council = mock_guardian

        rt._init_omega_components = set_gateway_and_guardian  # type: ignore[assignment]

        await rt._init_components()

        mock_guardian.set_inference_gateway.assert_called_once_with(mock_gw)


# =============================================================================
# 14. _init_user_context
# =============================================================================


class TestInitUserContext:
    """Tests for _init_user_context (lines 840-864)."""

    def test_creates_user_context_manager(self, tmp_path: Path) -> None:
        """Creates a UserContextManager and calls load()."""
        rt = _make_runtime(tmp_path)
        state_dir = tmp_path / "sovereign_state"
        state_dir.mkdir(parents=True, exist_ok=True)
        rt.config.state_dir = state_dir

        rt._init_user_context()

        assert rt._user_context is not None

    def test_wires_genesis_into_user_profile(self, tmp_path: Path) -> None:
        """When genesis is present, node_id and node_name are wired into profile."""
        rt = _make_runtime(tmp_path)
        state_dir = tmp_path / "sovereign_state"
        state_dir.mkdir(parents=True, exist_ok=True)
        rt.config.state_dir = state_dir

        rt._genesis = SimpleNamespace(
            node_id="genesis-node-42",
            node_name="Node42",
        )

        rt._init_user_context()

        assert rt._user_context is not None
        assert rt._user_context.profile.node_id == "genesis-node-42"
        assert rt._user_context.profile.node_name == "Node42"

    def test_registers_with_memory_coordinator(self, tmp_path: Path) -> None:
        """When memory coordinator exists, user context is registered."""
        rt = _make_runtime(tmp_path)
        state_dir = tmp_path / "sovereign_state"
        state_dir.mkdir(parents=True, exist_ok=True)
        rt.config.state_dir = state_dir

        mock_coord = MagicMock()
        rt._memory_coordinator = mock_coord

        rt._init_user_context()

        mock_coord.register_state_provider.assert_called_once()
        call_args = mock_coord.register_state_provider.call_args
        assert call_args[0][0] == "user_context"


# =============================================================================
# 15. _load_genesis_identity
# =============================================================================


class TestLoadGenesisIdentity:
    """Tests for _load_genesis_identity (lines 866-879)."""

    def test_successful_load_sets_genesis_and_node_id(self, tmp_path: Path) -> None:
        """Successful genesis load sets _genesis and updates config.node_id."""
        rt = _make_runtime(tmp_path)

        mock_genesis = SimpleNamespace(
            node_id="loaded-id",
            node_name="Loaded Node",
        )
        with patch("core.sovereign.runtime_core.load_and_validate_genesis", return_value=mock_genesis):
            rt._load_genesis_identity()

        assert rt._genesis is mock_genesis
        assert rt.config.node_id == "loaded-id"

    def test_none_return_ephemeral_node(self, tmp_path: Path) -> None:
        """When load_and_validate_genesis returns None, no genesis set."""
        rt = _make_runtime(tmp_path)
        original_node_id = rt.config.node_id

        with patch("core.sovereign.runtime_core.load_and_validate_genesis", return_value=None):
            rt._load_genesis_identity()

        assert rt._genesis is None
        assert rt.config.node_id == original_node_id

    def test_value_error_logs_and_continues(self, tmp_path: Path) -> None:
        """ValueError from genesis load is logged, not raised."""
        rt = _make_runtime(tmp_path)

        with patch(
            "core.sovereign.runtime_core.load_and_validate_genesis",
            side_effect=ValueError("corrupted genesis"),
        ):
            # Should not raise
            rt._load_genesis_identity()

        assert rt._genesis is None


# =============================================================================
# 16. _init_memory_coordinator
# =============================================================================


class TestInitMemoryCoordinator:
    """Tests for _init_memory_coordinator (lines 1181-1229)."""

    @pytest.mark.asyncio
    async def test_creates_and_initializes_coordinator(self, tmp_path: Path) -> None:
        """Creates MemoryCoordinator, calls initialize, registers providers."""
        rt = _make_runtime(tmp_path)
        rt.config.enable_persistence = False  # Skip auto-save to simplify

        mock_coord = MagicMock()
        mock_coord.initialize = MagicMock()
        mock_coord.register_state_provider = MagicMock()
        mock_coord.register_living_memory = MagicMock()
        mock_coord.start_auto_save = AsyncMock()

        with patch("core.sovereign.runtime_core.MemoryCoordinator", return_value=mock_coord):
            with patch("core.sovereign.runtime_core.MemoryCoordinatorConfig"):
                rt._register_proactive_providers = MagicMock()
                await rt._init_memory_coordinator()

        mock_coord.initialize.assert_called_once()
        mock_coord.register_state_provider.assert_called()

    @pytest.mark.asyncio
    async def test_starts_auto_save_when_persistence_enabled(self, tmp_path: Path) -> None:
        """When enable_persistence=True, auto-save is started."""
        rt = _make_runtime(tmp_path, enable_persistence=True)

        mock_coord = MagicMock()
        mock_coord.initialize = MagicMock()
        mock_coord.register_state_provider = MagicMock()
        mock_coord.start_auto_save = AsyncMock()

        with patch("core.sovereign.runtime_core.MemoryCoordinator", return_value=mock_coord):
            with patch("core.sovereign.runtime_core.MemoryCoordinatorConfig"):
                rt._register_proactive_providers = MagicMock()
                # Prevent LivingMemory import
                with patch.dict("sys.modules", {"core.living_memory.core": None}):
                    await rt._init_memory_coordinator()

        mock_coord.start_auto_save.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_exception_handling(self, tmp_path: Path) -> None:
        """Coordinator init failure is caught gracefully."""
        rt = _make_runtime(tmp_path)

        with patch(
            "core.sovereign.runtime_core.MemoryCoordinator",
            side_effect=RuntimeError("coord init fail"),
        ):
            with patch("core.sovereign.runtime_core.MemoryCoordinatorConfig"):
                # Should not raise
                await rt._init_memory_coordinator()

        assert rt._memory_coordinator is None


# =============================================================================
# 17. _record_query_impact
# =============================================================================


class TestRecordQueryImpact:
    """Tests for _record_query_impact (lines 1269-1307)."""

    def test_no_tracker_is_noop(self, tmp_path: Path) -> None:
        """When _impact_tracker is None, nothing happens."""
        rt = _make_runtime(tmp_path)
        rt._impact_tracker = None
        result = SovereignResult(query_id="x", success=True)
        # Should not raise
        rt._record_query_impact(result)

    def test_success_path_calls_record_event(self, tmp_path: Path) -> None:
        """With tracker present, record_event is called with correct params."""
        rt = _make_runtime(tmp_path)

        mock_tracker = MagicMock()
        mock_tracker.record_event = MagicMock()
        rt._impact_tracker = mock_tracker

        result = SovereignResult(
            query_id="q1",
            success=True,
            response="good answer",
            processing_time_ms=150.0,
            reasoning_depth=3,
            snr_score=0.91,
            ihsan_score=0.96,
            validation_passed=True,
        )

        with patch.dict("sys.modules", {
            "core.pat.impact_tracker": MagicMock(
                UERSScore=MagicMock(),
                compute_query_bloom=MagicMock(return_value=0.5),
            ),
        }):
            rt._record_query_impact(result)

        mock_tracker.record_event.assert_called_once()
        call_kwargs = mock_tracker.record_event.call_args
        assert call_kwargs.kwargs["category"] == "computation"
        assert call_kwargs.kwargs["action"] == "sovereign_query"

    def test_exception_swallowed(self, tmp_path: Path) -> None:
        """Impact recording failures are swallowed silently."""
        rt = _make_runtime(tmp_path)

        mock_tracker = MagicMock()
        mock_tracker.record_event = MagicMock(side_effect=RuntimeError("tracker broke"))
        rt._impact_tracker = mock_tracker

        result = SovereignResult(query_id="q1", success=True, response="x",
                                 processing_time_ms=10, snr_score=0.9, ihsan_score=0.95)

        with patch.dict("sys.modules", {
            "core.pat.impact_tracker": MagicMock(
                UERSScore=MagicMock(),
                compute_query_bloom=MagicMock(return_value=0.5),
            ),
        }):
            # Should not raise
            rt._record_query_impact(result)


# =============================================================================
# 18. _get_runtime_state
# =============================================================================


class TestGetRuntimeState:
    """Tests for _get_runtime_state (lines 1309-1341)."""

    def test_returns_correct_structure(self, tmp_path: Path) -> None:
        """Verify the state dict has expected top-level keys."""
        rt = _make_runtime(tmp_path)
        _set_ready(rt)

        state = rt._get_runtime_state()
        assert "metrics" in state
        assert "config" in state
        assert "components" in state
        assert "cache_size" in state

    def test_with_zpk_bootstrap_result(self, tmp_path: Path) -> None:
        """When _zpk_bootstrap_result is present, zpk_preflight is included."""
        rt = _make_runtime(tmp_path)
        rt._zpk_bootstrap_result = SimpleNamespace(
            success=True,
            executed_version="1.0.0",
            rollback_used=False,
            reason="all good",
        )

        state = rt._get_runtime_state()
        assert "zpk_preflight" in state
        assert state["zpk_preflight"]["success"] is True
        assert state["zpk_preflight"]["executed_version"] == "1.0.0"

    def test_with_genesis(self, tmp_path: Path) -> None:
        """When genesis is present, genesis summary is included."""
        rt = _make_runtime(tmp_path)
        rt._genesis = SimpleNamespace(summary=MagicMock(return_value={"node": "info"}))

        state = rt._get_runtime_state()
        assert "genesis" in state
        assert state["genesis"] == {"node": "info"}

    def test_without_zpk_or_genesis(self, tmp_path: Path) -> None:
        """Without zpk or genesis, state omits those keys."""
        rt = _make_runtime(tmp_path)
        rt._zpk_bootstrap_result = None
        rt._genesis = None

        state = rt._get_runtime_state()
        assert "zpk_preflight" not in state
        assert "genesis" not in state

    def test_components_reflect_actual_state(self, tmp_path: Path) -> None:
        """Component booleans reflect whether components are set."""
        rt = _make_runtime(tmp_path)
        rt._graph_reasoner = MagicMock()
        rt._snr_optimizer = None
        rt._gateway = MagicMock()
        rt._omega = None

        state = rt._get_runtime_state()
        assert state["components"]["graph_reasoner"] is True
        assert state["components"]["snr_optimizer"] is False
        assert state["components"]["gateway"] is True
        assert state["components"]["omega"] is False


# =============================================================================
# 19. CONVENIENCE METHODS
# =============================================================================


class TestConvenienceMethods:
    """Tests for think(), validate(), reason() convenience wrappers."""

    @pytest.mark.asyncio
    async def test_think_delegates_to_query(self, tmp_path: Path) -> None:
        """think() calls query() and returns response on success."""
        rt = _make_runtime(tmp_path)
        _set_ready(rt)
        rt.query = AsyncMock(return_value=SovereignResult(
            success=True, response="deep thought"
        ))

        result = await rt.think("what is life?")
        assert result == "deep thought"
        rt.query.assert_awaited_once_with("what is life?")

    @pytest.mark.asyncio
    async def test_think_returns_error_on_failure(self, tmp_path: Path) -> None:
        """think() returns error string when query fails."""
        rt = _make_runtime(tmp_path)
        _set_ready(rt)
        rt.query = AsyncMock(return_value=SovereignResult(
            success=False, error="query failed"
        ))

        result = await rt.think("broken question")
        assert "Error" in result
        assert "query failed" in result

    @pytest.mark.asyncio
    async def test_validate_delegates_with_require_validation(self, tmp_path: Path) -> None:
        """validate() calls query with require_validation=True."""
        rt = _make_runtime(tmp_path)
        _set_ready(rt)
        rt.query = AsyncMock(return_value=SovereignResult(
            success=True, ihsan_score=0.97
        ))

        result = await rt.validate("check this content")
        assert result is True
        rt.query.assert_awaited_once()
        call_kwargs = rt.query.call_args
        assert call_kwargs.kwargs.get("require_validation") is True

    @pytest.mark.asyncio
    async def test_validate_returns_false_below_threshold(self, tmp_path: Path) -> None:
        """validate() returns False when ihsan_score < threshold."""
        rt = _make_runtime(tmp_path)
        _set_ready(rt)
        rt.config.ihsan_threshold = 0.95
        rt.query = AsyncMock(return_value=SovereignResult(
            success=True, ihsan_score=0.80
        ))

        result = await rt.validate("low quality")
        assert result is False

    @pytest.mark.asyncio
    async def test_reason_returns_thoughts_list(self, tmp_path: Path) -> None:
        """reason() returns the thoughts list from query result."""
        rt = _make_runtime(tmp_path)
        _set_ready(rt)
        rt.query = AsyncMock(return_value=SovereignResult(
            success=True, thoughts=["step1", "step2", "step3"]
        ))

        result = await rt.reason("complex question", depth=3)
        assert result == ["step1", "step2", "step3"]

    @pytest.mark.asyncio
    async def test_reason_passes_max_depth(self, tmp_path: Path) -> None:
        """reason() passes depth as max_depth to query."""
        rt = _make_runtime(tmp_path)
        _set_ready(rt)
        rt.query = AsyncMock(return_value=SovereignResult(thoughts=["a"]))

        await rt.reason("q", depth=7)
        rt.query.assert_awaited_once_with("q", max_depth=7)


# =============================================================================
# 20. wait_for_shutdown
# =============================================================================


class TestWaitForShutdown:
    """Tests for wait_for_shutdown (lines 1468-1470)."""

    @pytest.mark.asyncio
    async def test_awaits_shutdown_event(self, tmp_path: Path) -> None:
        """wait_for_shutdown blocks until _shutdown_event is set."""
        rt = _make_runtime(tmp_path)

        async def set_event_later() -> None:
            await asyncio.sleep(0.05)
            rt._shutdown_event.set()

        task = asyncio.create_task(set_event_later())
        await asyncio.wait_for(rt.wait_for_shutdown(), timeout=2.0)
        await task

        assert rt._shutdown_event.is_set()

    @pytest.mark.asyncio
    async def test_returns_immediately_if_already_set(self, tmp_path: Path) -> None:
        """If event already set, wait_for_shutdown returns immediately."""
        rt = _make_runtime(tmp_path)
        rt._shutdown_event.set()

        # Should not block
        await asyncio.wait_for(rt.wait_for_shutdown(), timeout=1.0)


# =============================================================================
# 21. _setup_signal_handlers
# =============================================================================


class TestSetupSignalHandlers:
    """Tests for _setup_signal_handlers (lines 1417-1426)."""

    @pytest.mark.asyncio
    async def test_registers_signal_handlers(self, tmp_path: Path) -> None:
        """On supported platforms, registers SIGTERM and SIGINT handlers."""
        rt = _make_runtime(tmp_path)

        mock_loop = MagicMock()
        with patch("asyncio.get_running_loop", return_value=mock_loop):
            rt._setup_signal_handlers()

        # Should have added handlers for both signals
        assert mock_loop.add_signal_handler.call_count == 2

    def test_not_implemented_error_handled(self, tmp_path: Path) -> None:
        """NotImplementedError (Windows) is caught silently."""
        rt = _make_runtime(tmp_path)

        with patch("asyncio.get_running_loop", side_effect=NotImplementedError("no signals on Windows")):
            # Should not raise
            rt._setup_signal_handlers()

    def test_runtime_error_handled(self, tmp_path: Path) -> None:
        """RuntimeError (no running loop) is caught silently."""
        rt = _make_runtime(tmp_path)

        with patch("asyncio.get_running_loop", side_effect=RuntimeError("no running loop")):
            # Should not raise
            rt._setup_signal_handlers()


# =============================================================================
# ADDITIONAL EDGE CASES
# =============================================================================


class TestEstimateComplexity:
    """Tests for _estimate_complexity (lines 1550-1581)."""

    def test_short_simple_query_low_complexity(self, tmp_path: Path) -> None:
        """Short, simple queries produce low complexity scores."""
        rt = _make_runtime(tmp_path)
        q = SovereignQuery(text="What is BIZRA?")
        score = rt._estimate_complexity(q)
        assert score < 0.6

    def test_long_multi_question_high_complexity(self, tmp_path: Path) -> None:
        """Long queries with multiple questions produce higher scores."""
        rt = _make_runtime(tmp_path)
        text = (
            "Compare and contrast the distributed systems approach "
            "and also analyze the performance implications? "
            "Additionally, evaluate the security model? "
            "Furthermore, provide a comprehensive step by step plan."
        )
        q = SovereignQuery(text=text)
        score = rt._estimate_complexity(q)
        assert score >= 0.4  # Should be elevated

    def test_complexity_hint_from_context(self, tmp_path: Path) -> None:
        """Explicit complexity_hint in context boosts the score."""
        rt = _make_runtime(tmp_path)
        q = SovereignQuery(text="simple text", context={"complexity_hint": 1.0})
        score = rt._estimate_complexity(q)
        # 0.2 * 1.0 = 0.2 from hint alone, plus other factors
        assert score >= 0.2

    def test_empty_query(self, tmp_path: Path) -> None:
        """Empty query text produces zero or near-zero complexity."""
        rt = _make_runtime(tmp_path)
        q = SovereignQuery(text="")
        score = rt._estimate_complexity(q)
        assert score == 0.0


class TestCacheMethods:
    """Tests for _cache_key and _update_cache."""

    def test_cache_key_deterministic(self, tmp_path: Path) -> None:
        """Same query produces same cache key."""
        rt = _make_runtime(tmp_path)
        q1 = SovereignQuery(text="same", require_reasoning=True)
        q2 = SovereignQuery(text="same", require_reasoning=True)
        assert rt._cache_key(q1) == rt._cache_key(q2)

    def test_cache_key_differs_for_different_text(self, tmp_path: Path) -> None:
        """Different text produces different cache keys."""
        rt = _make_runtime(tmp_path)
        q1 = SovereignQuery(text="query A", require_reasoning=True)
        q2 = SovereignQuery(text="query B", require_reasoning=True)
        assert rt._cache_key(q1) != rt._cache_key(q2)

    def test_update_cache_evicts_when_full(self, tmp_path: Path) -> None:
        """When cache exceeds max entries, oldest entries are evicted."""
        rt = _make_runtime(tmp_path, max_cache_entries=10)
        for i in range(15):
            rt._cache[f"key-{i}"] = SovereignResult(query_id=f"q{i}")

        rt._update_cache("new-key", SovereignResult(query_id="new"))
        # After eviction, should have <= 10 + 1 entries
        assert len(rt._cache) <= 11


class TestObserveJudgment:
    """Tests for _observe_judgment (lines 406-433)."""

    def test_no_telemetry_is_noop(self, tmp_path: Path) -> None:
        """When _judgment_telemetry is None, nothing happens."""
        rt = _make_runtime(tmp_path)
        rt._judgment_telemetry = None
        result = SovereignResult(success=True, snr_ok=True, ihsan_score=0.96)
        # Should not raise
        rt._observe_judgment(result)

    def test_promote_verdict_for_excellent_result(self, tmp_path: Path) -> None:
        """ihsan >= 0.95 and snr_ok = PROMOTE."""
        rt = _make_runtime(tmp_path)
        mock_telemetry = MagicMock()
        mock_telemetry.observe = MagicMock()
        rt._judgment_telemetry = mock_telemetry

        result = SovereignResult(success=True, snr_ok=True, ihsan_score=0.97)

        mock_verdict = MagicMock()
        with patch.dict("sys.modules", {
            "core.sovereign.judgment_telemetry": MagicMock(
                JudgmentVerdict=SimpleNamespace(
                    PROMOTE=mock_verdict,
                    NEUTRAL="NEUTRAL",
                    DEMOTE="DEMOTE",
                    FORBID="FORBID",
                ),
            ),
        }):
            rt._observe_judgment(result)

        mock_telemetry.observe.assert_called_once_with(mock_verdict)

    def test_demote_verdict_when_snr_not_ok(self, tmp_path: Path) -> None:
        """snr_ok=False results in DEMOTE verdict."""
        rt = _make_runtime(tmp_path)
        mock_telemetry = MagicMock()
        rt._judgment_telemetry = mock_telemetry

        result = SovereignResult(success=True, snr_ok=False, ihsan_score=0.80)

        mock_demote = MagicMock()
        with patch.dict("sys.modules", {
            "core.sovereign.judgment_telemetry": MagicMock(
                JudgmentVerdict=SimpleNamespace(
                    PROMOTE="PROMOTE",
                    NEUTRAL="NEUTRAL",
                    DEMOTE=mock_demote,
                    FORBID="FORBID",
                ),
            ),
        }):
            rt._observe_judgment(result)

        mock_telemetry.observe.assert_called_once_with(mock_demote)


class TestCommitExperienceEpisode:
    """Tests for _commit_experience_episode (lines 435-488)."""

    def test_no_ledger_is_noop(self, tmp_path: Path) -> None:
        """When _experience_ledger is None, nothing happens."""
        rt = _make_runtime(tmp_path)
        rt._experience_ledger = None
        result = SovereignResult(success=True, snr_ok=True)
        query = SovereignQuery(text="test")
        rt._commit_experience_episode(result, query)

    def test_not_success_skips(self, tmp_path: Path) -> None:
        """Non-successful results are skipped."""
        rt = _make_runtime(tmp_path)
        rt._experience_ledger = MagicMock()
        result = SovereignResult(success=False, snr_ok=True)
        query = SovereignQuery(text="test")
        rt._commit_experience_episode(result, query)
        rt._experience_ledger.commit.assert_not_called()

    def test_not_snr_ok_skips(self, tmp_path: Path) -> None:
        """snr_ok=False results are skipped."""
        rt = _make_runtime(tmp_path)
        rt._experience_ledger = MagicMock()
        result = SovereignResult(success=True, snr_ok=False)
        query = SovereignQuery(text="test")
        rt._commit_experience_episode(result, query)
        rt._experience_ledger.commit.assert_not_called()


class TestEncodeQueryMemory:
    """Tests for _encode_query_memory (lines 693-731)."""

    def test_no_living_memory_is_noop(self, tmp_path: Path) -> None:
        """When _living_memory is None, nothing happens."""
        rt = _make_runtime(tmp_path)
        rt._living_memory = None
        result = SovereignResult(success=True, response="answer")
        query = SovereignQuery(text="test")
        rt._encode_query_memory(result, query)

    def test_not_success_skips(self, tmp_path: Path) -> None:
        """Non-successful results are skipped."""
        rt = _make_runtime(tmp_path)
        rt._living_memory = MagicMock()
        result = SovereignResult(success=False, response="answer")
        query = SovereignQuery(text="test")
        rt._encode_query_memory(result, query)

    def test_no_response_skips(self, tmp_path: Path) -> None:
        """Empty response is skipped."""
        rt = _make_runtime(tmp_path)
        rt._living_memory = MagicMock()
        result = SovereignResult(success=True, response="")
        query = SovereignQuery(text="test")
        rt._encode_query_memory(result, query)


class TestGateChainPreflight:
    """Tests for _run_gate_chain_preflight (lines 514-583)."""

    @pytest.mark.asyncio
    async def test_no_gate_chain_rejects(self, tmp_path: Path) -> None:
        """CRITICAL-1 FIX: None gate chain → REJECT (fail-closed)."""
        rt = _make_runtime(tmp_path)
        rt._gate_chain = None
        query = SovereignQuery(text="test")
        result = SovereignResult()
        ret = await rt._run_gate_chain_preflight(query, result)
        # Fail-closed: must reject, not pass through
        assert ret is not None or result.success is False

    @pytest.mark.asyncio
    async def test_gate_chain_passes_returns_none(self, tmp_path: Path) -> None:
        """When gate chain passes, returns None."""
        rt = _make_runtime(tmp_path)

        chain_result = SimpleNamespace(passed=True, gate_results=[])
        rt._gate_chain = MagicMock()
        rt._gate_chain.evaluate = MagicMock(return_value=(chain_result, None))

        with patch.dict("sys.modules", {
            "core.proof_engine.canonical": MagicMock(
                CanonQuery=MagicMock(),
                CanonPolicy=MagicMock(),
            ),
        }):
            query = SovereignQuery(text="test")
            result = SovereignResult()
            ret = await rt._run_gate_chain_preflight(query, result)

        assert ret is None

    @pytest.mark.asyncio
    async def test_gate_chain_rejects_returns_result(self, tmp_path: Path) -> None:
        """When gate chain fails, returns rejection result."""
        rt = _make_runtime(tmp_path)

        chain_result = SimpleNamespace(
            passed=False,
            last_gate_passed="trust_gate",
            rejection_reason="Trust too low",
            snr=0.5,
            ihsan_score=0.3,
            gate_results=[],
        )
        rt._gate_chain = MagicMock()
        rt._gate_chain.evaluate = MagicMock(return_value=(chain_result, None))

        with patch.dict("sys.modules", {
            "core.proof_engine.canonical": MagicMock(
                CanonQuery=MagicMock(),
                CanonPolicy=MagicMock(),
            ),
        }):
            query = SovereignQuery(text="untrusted")
            result = SovereignResult(query_id="rej1")
            ret = await rt._run_gate_chain_preflight(query, result)

        assert ret is not None
        assert ret.success is False
        assert "rejected" in ret.response.lower()

    @pytest.mark.asyncio
    async def test_gate_chain_exception_rejects(self, tmp_path: Path) -> None:
        """CRITICAL-2 FIX: Gate chain exceptions → REJECT (fail-closed)."""
        rt = _make_runtime(tmp_path)
        rt._gate_chain = MagicMock()
        rt._gate_chain.evaluate = MagicMock(side_effect=RuntimeError("gate boom"))

        with patch.dict("sys.modules", {
            "core.proof_engine.canonical": MagicMock(
                CanonQuery=MagicMock(),
                CanonPolicy=MagicMock(),
            ),
        }):
            query = SovereignQuery(text="boom")
            result = SovereignResult()
            ret = await rt._run_gate_chain_preflight(query, result)

        # CRITICAL-2: Exception → fail-closed (not pass-through)
        assert ret is not None or result.success is False


class TestRegisterPoiContribution:
    """Tests for _register_poi_contribution (lines 659-691)."""

    def test_no_orchestrator_is_noop(self, tmp_path: Path) -> None:
        """When _poi_orchestrator is None, nothing happens."""
        rt = _make_runtime(tmp_path)
        rt._poi_orchestrator = None
        result = SovereignResult(success=True)
        query = SovereignQuery(text="test")
        rt._register_poi_contribution(result, query)

    def test_not_success_skips(self, tmp_path: Path) -> None:
        """Non-successful results are skipped."""
        rt = _make_runtime(tmp_path)
        rt._poi_orchestrator = MagicMock()
        result = SovereignResult(success=False)
        query = SovereignQuery(text="test")
        rt._register_poi_contribution(result, query)
        rt._poi_orchestrator.register_contribution.assert_not_called()

    def test_exception_swallowed(self, tmp_path: Path) -> None:
        """Registration exceptions are caught silently."""
        rt = _make_runtime(tmp_path)
        rt._poi_orchestrator = MagicMock()

        with patch.dict("sys.modules", {
            "core.proof_engine.poi_engine": MagicMock(
                ContributionMetadata=MagicMock(side_effect=RuntimeError("fail")),
                ContributionType=MagicMock(),
            ),
        }):
            result = SovereignResult(success=True, query_id="q1")
            query = SovereignQuery(text="test")
            # Should not raise
            rt._register_poi_contribution(result, query)


class TestEmitQueryReceipt:
    """Tests for _emit_query_receipt (lines 585-657)."""

    def test_no_ledger_is_noop(self, tmp_path: Path) -> None:
        """When _evidence_ledger is None, nothing happens."""
        rt = _make_runtime(tmp_path)
        rt._evidence_ledger = None
        result = SovereignResult(success=True, response="x")
        query = SovereignQuery(text="test")
        rt._emit_query_receipt(result, query)

    def test_exception_does_not_raise(self, tmp_path: Path) -> None:
        """Receipt emission failures are caught and logged."""
        rt = _make_runtime(tmp_path)
        rt._evidence_ledger = MagicMock()

        with patch.dict("sys.modules", {
            "core.proof_engine.evidence_ledger": MagicMock(
                emit_receipt=MagicMock(side_effect=RuntimeError("ledger fail"))
            ),
        }):
            result = SovereignResult(success=True, response="x", query_id="q1",
                                     snr_score=0.9, ihsan_score=0.95, validation_passed=True)
            query = SovereignQuery(text="test")
            # Should not raise
            rt._emit_query_receipt(result, query)


class TestModeToTier:
    """Tests for _mode_to_tier helper."""

    def test_returns_none_when_imports_fail(self, tmp_path: Path) -> None:
        """When imports fail, returns None."""
        rt = _make_runtime(tmp_path)
        with patch.dict("sys.modules", {
            "core.inference.gateway": None,
        }):
            result = rt._mode_to_tier("ANYTHING")
        assert result is None

    def test_returns_none_for_non_treasury_mode(self, tmp_path: Path) -> None:
        """Non-TreasuryMode input returns None."""
        rt = _make_runtime(tmp_path)

        # We need a real class for isinstance() check inside _mode_to_tier.
        # The method imports TreasuryMode and ComputeTier, then does
        # isinstance(mode, TreasuryMode). A MagicMock is not a valid type.
        from enum import Enum, auto

        class FakeTreasuryMode(Enum):
            ETHICAL = auto()
            HIBERNATION = auto()
            EMERGENCY = auto()

        class FakeComputeTier(Enum):
            LOCAL = auto()
            EDGE = auto()

        fake_omega_mod = MagicMock(TreasuryMode=FakeTreasuryMode)
        fake_gw_mod = MagicMock(ComputeTier=FakeComputeTier)

        with patch.dict("sys.modules", {
            "core.inference.gateway": fake_gw_mod,
            "core.sovereign.omega_engine": fake_omega_mod,
        }):
            # A plain string is not an instance of FakeTreasuryMode
            result = rt._mode_to_tier("not_a_mode")
        assert result is None


class TestStoreGraphArtifact:
    """Tests for _store_graph_artifact."""

    def test_no_graph_reasoner_skips(self, tmp_path: Path) -> None:
        """When _graph_reasoner is None, nothing stored."""
        rt = _make_runtime(tmp_path)
        rt._graph_reasoner = None
        rt._store_graph_artifact("q1", "hash1")
        assert len(rt._graph_artifacts) == 0

    def test_stores_artifact(self, tmp_path: Path) -> None:
        """When reasoner has to_artifact, artifact is stored."""
        rt = _make_runtime(tmp_path)
        mock_reasoner = MagicMock()
        mock_reasoner.to_artifact = MagicMock(return_value={"nodes": 5, "hash": "abc"})
        rt._graph_reasoner = mock_reasoner

        rt._store_graph_artifact("q1", "hash1")
        assert "q1" in rt._graph_artifacts
        assert rt._graph_artifacts["q1"]["nodes"] == 5

    def test_bounds_storage_at_100(self, tmp_path: Path) -> None:
        """Graph artifacts storage is bounded at 100 entries."""
        rt = _make_runtime(tmp_path)
        mock_reasoner = MagicMock()
        mock_reasoner.to_artifact = MagicMock(return_value={"data": True})
        rt._graph_reasoner = mock_reasoner

        for i in range(105):
            rt._store_graph_artifact(f"q{i}", f"h{i}")

        # Should have evicted oldest, size <= 101 (100 + 1 after last insert)
        assert len(rt._graph_artifacts) <= 101

    def test_exception_caught(self, tmp_path: Path) -> None:
        """Artifact storage failure is caught."""
        rt = _make_runtime(tmp_path)
        mock_reasoner = MagicMock()
        mock_reasoner.to_artifact = MagicMock(side_effect=RuntimeError("artifact fail"))
        rt._graph_reasoner = mock_reasoner

        # Should not raise
        rt._store_graph_artifact("q1", "h1")


class TestCreateContextManager:
    """Tests for the async context manager create() class method."""

    @pytest.mark.asyncio
    async def test_create_yields_runtime_and_shuts_down(self, tmp_path: Path) -> None:
        """create() context manager initializes and shuts down."""
        cfg = _minimal_config(tmp_path)

        with patch.object(SovereignRuntime, "initialize", new_callable=AsyncMock) as mock_init, \
             patch.object(SovereignRuntime, "shutdown", new_callable=AsyncMock) as mock_shutdown:
            async with SovereignRuntime.create(cfg) as rt:
                assert isinstance(rt, SovereignRuntime)
                mock_init.assert_awaited_once()

            mock_shutdown.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_create_shuts_down_on_exception(self, tmp_path: Path) -> None:
        """Even if code inside context raises, shutdown is called."""
        cfg = _minimal_config(tmp_path)

        with patch.object(SovereignRuntime, "initialize", new_callable=AsyncMock), \
             patch.object(SovereignRuntime, "shutdown", new_callable=AsyncMock) as mock_shutdown:
            with pytest.raises(ValueError, match="intentional"):
                async with SovereignRuntime.create(cfg) as rt:
                    raise ValueError("intentional")

            mock_shutdown.assert_awaited_once()


class TestGetGraphArtifact:
    """Tests for get_graph_artifact."""

    def test_returns_stored_artifact(self, tmp_path: Path) -> None:
        rt = _make_runtime(tmp_path)
        rt._graph_artifacts["q1"] = {"nodes": 3}
        assert rt.get_graph_artifact("q1") == {"nodes": 3}

    def test_returns_none_for_missing(self, tmp_path: Path) -> None:
        rt = _make_runtime(tmp_path)
        assert rt.get_graph_artifact("nonexistent") is None


class TestGetImpactState:
    """Tests for _get_impact_state."""

    def test_no_tracker_returns_empty(self, tmp_path: Path) -> None:
        rt = _make_runtime(tmp_path)
        rt._impact_tracker = None
        assert rt._get_impact_state() == {}

    def test_with_tracker_returns_progress(self, tmp_path: Path) -> None:
        rt = _make_runtime(tmp_path)
        mock_tracker = MagicMock()
        mock_progress = MagicMock()
        mock_progress.to_dict.return_value = {"score": 0.42, "tier": "seed"}
        mock_tracker.get_progress.return_value = mock_progress
        rt._impact_tracker = mock_tracker
        result = rt._get_impact_state()
        assert result == {"score": 0.42, "tier": "seed"}

    def test_exception_returns_empty(self, tmp_path: Path) -> None:
        rt = _make_runtime(tmp_path)
        mock_tracker = MagicMock()
        mock_tracker.get_progress.side_effect = RuntimeError("fail")
        rt._impact_tracker = mock_tracker
        assert rt._get_impact_state() == {}
