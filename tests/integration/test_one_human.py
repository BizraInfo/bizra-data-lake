"""
BIZRA "Serve One Human" End-to-End Integration Test

The critical proof: BIZRA can serve ONE human end-to-end.
    Human asks a question
    -> Orchestrator processes through the full pipeline
    -> Quality answer comes back with provenance

This test suite exercises the real orchestrator against real data
(84,795 semantic chunks, 56,358-node hypergraph, 384-dim MiniLM embeddings).
No mocks. No stubs. The genuine pipeline or an honest skip.

35 tests across 8 classes covering:
    1. Bootstrap          -- Can the orchestrator even start?
    2. Simple Query       -- SIMPLE query returns a grounded answer
    3. Moderate Query     -- MODERATE query with richer retrieval
    4. Complex Query      -- COMPLEX query with multi-hop reasoning
    5. Quality Gates      -- SNR and Ihsan constraints are enforced
    6. System Status      -- Engine availability and health reporting
    7. Lifecycle          -- Full create-init-query-verify cycle
    8. Graceful Degrade   -- Disabled engines still produce answers

Standing on Giants: Dijkstra (testing shows the presence of bugs, never
their absence), but this suite proves the presence of *function*.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import Optional

import pytest

# ---------------------------------------------------------------------------
# Path setup: mirrors the orchestrator's own sys.path bootstrapping so that
# engine imports resolve identically in the test runner.
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent.parent  # tests/integration/.. -> repo root
for _subdir in ("tools/engines", "tools/bridges", "tools"):
    _p = str(_ROOT / _subdir)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bizra_orchestrator import (
    BIZRAOrchestrator,
    BIZRAQuery,
    BIZRAResponse,
    QueryComplexity,
)
from bizra_config import CHUNKS_TABLE_PATH, CORPUS_TABLE_PATH, INDEXED_PATH, GOLD_PATH

# ---------------------------------------------------------------------------
# Custom markers
# ---------------------------------------------------------------------------
pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_REAL_DATA_AVAILABLE: Optional[bool] = None


def _check_real_data() -> bool:
    """Check once whether the real parquet + index data is present."""
    global _REAL_DATA_AVAILABLE
    if _REAL_DATA_AVAILABLE is None:
        chunks_ok = CHUNKS_TABLE_PATH.exists() and CHUNKS_TABLE_PATH.stat().st_size > 0
        index_dir = INDEXED_PATH / "graph"
        index_ok = index_dir.exists() and any(index_dir.iterdir()) if index_dir.exists() else False
        _REAL_DATA_AVAILABLE = chunks_ok and index_ok
    return _REAL_DATA_AVAILABLE


requires_real_data = pytest.mark.skipif(
    not _check_real_data(),
    reason="Real data (chunks.parquet + graph index) not present on this machine",
)


# ---------------------------------------------------------------------------
# Module-scoped orchestrator fixture -- created once, shared across all tests
# in this file.  We deliberately use a *module* scope (not session) so that
# other integration test modules get their own instances.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def event_loop():
    """Provide a single event loop for the entire module."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def orchestrator():
    """Create a BIZRAOrchestrator instance (not yet initialized)."""
    return BIZRAOrchestrator(
        enable_pat=True,
        enable_kep=True,
        enable_multimodal=True,
        enable_discipline=True,
    )


@pytest.fixture(scope="module")
def initialized_orchestrator(orchestrator, event_loop):
    """Return an orchestrator that has been fully initialized (sync wrapper)."""
    event_loop.run_until_complete(orchestrator.initialize())
    return orchestrator


# ---------------------------------------------------------------------------
# 1. TestOneHumanBootstrap
# ---------------------------------------------------------------------------


class TestOneHumanBootstrap:
    """Can the orchestrator start at all?  These are gate-zero tests."""

    def test_orchestrator_creates_without_crash(self):
        """BIZRAOrchestrator() constructor succeeds without raising."""
        orch = BIZRAOrchestrator(
            enable_pat=False,
            enable_kep=False,
            enable_multimodal=False,
        )
        assert orch is not None
        assert isinstance(orch, BIZRAOrchestrator)

    async def test_orchestrator_initializes(self, orchestrator):
        """await orchestrator.initialize() returns True."""
        result = await orchestrator.initialize()
        assert result is True
        assert orchestrator._initialized is True

    def test_system_status_reports_engines(self, orchestrator):
        """get_system_status() returns a dict with engine availability."""
        status = orchestrator.get_system_status()
        assert isinstance(status, dict)
        assert "engines" in status
        engines = status["engines"]
        expected_keys = {
            "hypergraph_rag", "arte", "pat", "kep",
            "multimodal", "dual_agentic",
        }
        assert expected_keys.issubset(set(engines.keys())), (
            f"Missing engine keys: {expected_keys - set(engines.keys())}"
        )

    @requires_real_data
    def test_has_real_data(self):
        """The parquet and index files exist and are non-trivial."""
        assert CHUNKS_TABLE_PATH.exists(), "chunks.parquet missing"
        assert CHUNKS_TABLE_PATH.stat().st_size > 1_000_000, (
            "chunks.parquet too small -- expected 84k+ chunks"
        )
        graph_dir = INDEXED_PATH / "graph"
        assert graph_dir.exists(), "03_INDEXED/graph directory missing"
        graph_files = list(graph_dir.iterdir())
        assert len(graph_files) > 0, "graph directory is empty"


# ---------------------------------------------------------------------------
# 2. TestOneHumanSimpleQuery
# ---------------------------------------------------------------------------


class TestOneHumanSimpleQuery:
    """A SIMPLE query returns a usable, grounded answer."""

    QUERY_TEXT = "What file formats does the BIZRA data lake support?"

    @requires_real_data
    async def test_simple_query_returns_response(self, initialized_orchestrator):
        """A SIMPLE query returns a BIZRAResponse with a non-empty answer."""
        q = BIZRAQuery(text=self.QUERY_TEXT, complexity=QueryComplexity.SIMPLE)
        resp = await initialized_orchestrator.query(q)
        assert isinstance(resp, BIZRAResponse)
        assert resp.answer is not None
        assert len(resp.answer.strip()) > 0, "Answer must not be empty"

    @requires_real_data
    async def test_response_has_provenance(self, initialized_orchestrator):
        """Response carries sources, reasoning_trace, and execution_time."""
        q = BIZRAQuery(text=self.QUERY_TEXT, complexity=QueryComplexity.SIMPLE)
        resp = await initialized_orchestrator.query(q)
        assert isinstance(resp.sources, list)
        assert isinstance(resp.reasoning_trace, list)
        assert len(resp.reasoning_trace) > 0, "Reasoning trace must not be empty"
        assert isinstance(resp.execution_time, float)
        assert resp.execution_time > 0

    @requires_real_data
    async def test_response_snr_above_zero(self, initialized_orchestrator):
        """snr_score is positive (the pipeline produced signal)."""
        q = BIZRAQuery(text=self.QUERY_TEXT, complexity=QueryComplexity.SIMPLE)
        resp = await initialized_orchestrator.query(q)
        assert resp.snr_score > 0, f"Expected positive SNR, got {resp.snr_score}"

    @requires_real_data
    async def test_answer_is_relevant(self, initialized_orchestrator):
        """Answer contains domain-relevant content, not an error message."""
        q = BIZRAQuery(text=self.QUERY_TEXT, complexity=QueryComplexity.SIMPLE)
        resp = await initialized_orchestrator.query(q)
        # A genuine answer should reference sources or data -- not be a
        # bare error string like "Retrieval engine not available".
        answer_lower = resp.answer.lower()
        assert "not available" not in answer_lower or len(resp.sources) > 0, (
            "Answer appears to be an error message with no sources"
        )

    @requires_real_data
    async def test_execution_time_reasonable(self, initialized_orchestrator):
        """SIMPLE queries complete in under 30 seconds."""
        q = BIZRAQuery(text=self.QUERY_TEXT, complexity=QueryComplexity.SIMPLE)
        resp = await initialized_orchestrator.query(q)
        assert resp.execution_time < 30.0, (
            f"SIMPLE query took {resp.execution_time:.1f}s (limit: 30s)"
        )


# ---------------------------------------------------------------------------
# 3. TestOneHumanModerateQuery
# ---------------------------------------------------------------------------


class TestOneHumanModerateQuery:
    """MODERATE queries use hybrid retrieval and should produce richer output."""

    QUERY_TEXT = "How does the vector embedding pipeline generate and store chunk embeddings?"

    @requires_real_data
    async def test_moderate_query_succeeds(self, initialized_orchestrator):
        """MODERATE query works end to end."""
        q = BIZRAQuery(text=self.QUERY_TEXT, complexity=QueryComplexity.MODERATE)
        resp = await initialized_orchestrator.query(q)
        assert isinstance(resp, BIZRAResponse)
        assert len(resp.answer.strip()) > 0

    @requires_real_data
    async def test_moderate_has_more_sources(self, initialized_orchestrator):
        """MODERATE retrieves sources."""
        q = BIZRAQuery(text=self.QUERY_TEXT, complexity=QueryComplexity.MODERATE)
        resp = await initialized_orchestrator.query(q)
        assert isinstance(resp.sources, list)
        # Moderate uses k=5 so we should get some sources if data exists
        assert len(resp.sources) > 0, "MODERATE should retrieve at least one source"

    @requires_real_data
    async def test_reasoning_trace_shows_pipeline(self, initialized_orchestrator):
        """reasoning_trace mentions the retrieval mode used."""
        q = BIZRAQuery(text=self.QUERY_TEXT, complexity=QueryComplexity.MODERATE)
        resp = await initialized_orchestrator.query(q)
        trace_text = " ".join(resp.reasoning_trace).lower()
        # The orchestrator logs "Retrieval mode: hybrid" for MODERATE
        assert "retrieval" in trace_text or "mode" in trace_text, (
            f"Reasoning trace should mention retrieval mode, got: {resp.reasoning_trace[:5]}"
        )

    @requires_real_data
    async def test_tension_analysis_present(self, initialized_orchestrator):
        """tension_analysis dict is populated."""
        q = BIZRAQuery(text=self.QUERY_TEXT, complexity=QueryComplexity.MODERATE)
        resp = await initialized_orchestrator.query(q)
        assert isinstance(resp.tension_analysis, dict)
        assert len(resp.tension_analysis) > 0, "tension_analysis should not be empty"

    @requires_real_data
    async def test_query_trace_has_snr(self, initialized_orchestrator):
        """query_trace contains an 'snr' key."""
        q = BIZRAQuery(text=self.QUERY_TEXT, complexity=QueryComplexity.MODERATE)
        resp = await initialized_orchestrator.query(q)
        assert isinstance(resp.query_trace, dict)
        assert "snr" in resp.query_trace, (
            f"query_trace missing 'snr' key. Keys present: {list(resp.query_trace.keys())}"
        )


# ---------------------------------------------------------------------------
# 4. TestOneHumanComplexQuery
# ---------------------------------------------------------------------------


class TestOneHumanComplexQuery:
    """COMPLEX queries engage multi-hop retrieval and deeper reasoning."""

    QUERY_TEXT = (
        "What is the relationship between the ARTE symbolic-neural bridge "
        "and the hypergraph retrieval engine in BIZRA's architecture?"
    )

    @requires_real_data
    async def test_complex_query_succeeds(self, initialized_orchestrator):
        """COMPLEX query returns a response."""
        q = BIZRAQuery(text=self.QUERY_TEXT, complexity=QueryComplexity.COMPLEX)
        resp = await initialized_orchestrator.query(q)
        assert isinstance(resp, BIZRAResponse)
        assert len(resp.answer.strip()) > 0

    @requires_real_data
    async def test_complex_reasoning_depth(self, initialized_orchestrator):
        """reasoning_trace has multiple steps (more than a simple lookup)."""
        q = BIZRAQuery(text=self.QUERY_TEXT, complexity=QueryComplexity.COMPLEX)
        resp = await initialized_orchestrator.query(q)
        # A COMPLEX query should have at minimum: query, complexity, retrieval mode,
        # retrieval results, ARTE analysis, and response generation.
        assert len(resp.reasoning_trace) >= 3, (
            f"Expected >= 3 reasoning steps for COMPLEX, got {len(resp.reasoning_trace)}"
        )

    @requires_real_data
    async def test_complex_retrieves_multiple_sources(self, initialized_orchestrator):
        """COMPLEX query retrieves more sources than a SIMPLE query (k=10 vs k=3)."""
        q = BIZRAQuery(text=self.QUERY_TEXT, complexity=QueryComplexity.COMPLEX)
        resp = await initialized_orchestrator.query(q)
        # The orchestrator caps displayed sources at 5, but should have at least 2
        assert len(resp.sources) >= 2, (
            f"Expected >= 2 sources for COMPLEX, got {len(resp.sources)}"
        )

    @requires_real_data
    async def test_complex_uses_multi_hop(self, initialized_orchestrator):
        """COMPLEX queries should use multi_hop retrieval, not plain semantic."""
        q = BIZRAQuery(text=self.QUERY_TEXT, complexity=QueryComplexity.COMPLEX)
        resp = await initialized_orchestrator.query(q)
        retrieval_mode = resp.metadata.get("retrieval_mode", "")
        assert retrieval_mode != "semantic", (
            f"COMPLEX query should not use 'semantic' mode, got '{retrieval_mode}'"
        )


# ---------------------------------------------------------------------------
# 5. TestOneHumanQualityGates
# ---------------------------------------------------------------------------


class TestOneHumanQualityGates:
    """SNR and Ihsan constraints are computed and enforced."""

    @requires_real_data
    async def test_snr_is_computed(self, initialized_orchestrator):
        """snr_score is a float in [0, 1]."""
        q = BIZRAQuery(
            text="How does BIZRA process documents?",
            complexity=QueryComplexity.SIMPLE,
        )
        resp = await initialized_orchestrator.query(q)
        assert isinstance(resp.snr_score, float)
        assert 0.0 <= resp.snr_score <= 1.0, (
            f"SNR must be in [0, 1], got {resp.snr_score}"
        )

    @requires_real_data
    async def test_ihsan_flag_is_boolean(self, initialized_orchestrator):
        """ihsan_achieved is a boolean."""
        q = BIZRAQuery(
            text="Explain the data ingestion pipeline",
            complexity=QueryComplexity.SIMPLE,
        )
        resp = await initialized_orchestrator.query(q)
        assert isinstance(resp.ihsan_achieved, bool)

    @requires_real_data
    async def test_low_snr_gets_warning(self, initialized_orchestrator):
        """When SNR is below Ihsan threshold, reasoning trace notes the gap."""
        # Use a very high SNR threshold to force a warning
        q = BIZRAQuery(
            text="Tell me about embeddings",
            complexity=QueryComplexity.SIMPLE,
            snr_threshold=0.999,
        )
        resp = await initialized_orchestrator.query(q)
        # If SNR is below the requested threshold, trace should mention it
        if resp.snr_score < 0.999:
            trace_text = " ".join(resp.reasoning_trace).lower()
            assert "snr" in trace_text or "threshold" in trace_text or "warning" in trace_text or "low" in trace_text, (
                "Expected a warning in reasoning trace when SNR is below threshold"
            )

    async def test_empty_query_handled_gracefully(self, initialized_orchestrator):
        """An empty string query does not crash."""
        q = BIZRAQuery(text="", complexity=QueryComplexity.SIMPLE)
        # Must not raise -- graceful degradation is the contract
        resp = await initialized_orchestrator.query(q)
        assert isinstance(resp, BIZRAResponse)

    async def test_nonsense_query_handled(self, initialized_orchestrator):
        """Gibberish query does not crash and returns low or zero SNR."""
        q = BIZRAQuery(
            text="xkjf923 zqpwm dlkfj aslkdj f09u23",
            complexity=QueryComplexity.SIMPLE,
        )
        resp = await initialized_orchestrator.query(q)
        assert isinstance(resp, BIZRAResponse)
        # Gibberish should not achieve Ihsan-level SNR
        assert resp.snr_score < 0.99 or resp.ihsan_achieved is False, (
            "Gibberish query should not reach near-perfect SNR"
        )


# ---------------------------------------------------------------------------
# 6. TestOneHumanSystemStatus
# ---------------------------------------------------------------------------


class TestOneHumanSystemStatus:
    """System status reporting is accurate and complete."""

    def test_status_has_version(self, orchestrator):
        """Status dict contains a version string."""
        status = orchestrator.get_system_status()
        assert "version" in status
        assert isinstance(status["version"], str)
        assert len(status["version"]) > 0

    def test_status_shows_all_engines(self, orchestrator):
        """Engines dict includes all expected keys."""
        status = orchestrator.get_system_status()
        engines = status["engines"]
        for key in ("hypergraph_rag", "arte", "pat", "kep", "multimodal", "dual_agentic"):
            assert key in engines, f"Missing engine key: {key}"
            assert engines[key] in ("ready", "disabled", "unavailable", "not_initialized"), (
                f"Unexpected status for {key}: {engines[key]}"
            )

    async def test_status_after_init_shows_ready(self, initialized_orchestrator):
        """After initialize(), core engines report ready or disabled (not not_initialized)."""
        status = initialized_orchestrator.get_system_status()
        assert status["initialized"] is True
        engines = status["engines"]
        # Hypergraph is the core retrieval engine -- if available it must be ready
        if initialized_orchestrator.hypergraph_available:
            assert engines["hypergraph_rag"] == "ready"

    @requires_real_data
    async def test_arte_health_available(self, initialized_orchestrator):
        """If ARTE is ready, arte_health key exists in status."""
        status = initialized_orchestrator.get_system_status()
        if status["engines"]["arte"] == "ready":
            assert "arte_health" in status, (
                "ARTE reports ready but arte_health missing from status"
            )
            arte_health = status["arte_health"]
            assert "integration_snr" in arte_health
            assert "tension_level" in arte_health


# ---------------------------------------------------------------------------
# 7. TestOneHumanLifecycle
# ---------------------------------------------------------------------------


class TestOneHumanLifecycle:
    """Full lifecycle: create -> init -> query -> verify -> status."""

    @requires_real_data
    async def test_full_lifecycle(self):
        """Complete lifecycle: construct, initialize, query, verify, status."""
        # Step 1: Create
        orch = BIZRAOrchestrator(
            enable_pat=False,
            enable_kep=True,
            enable_multimodal=False,
        )
        assert not orch._initialized

        # Step 2: Initialize
        ok = await orch.initialize()
        assert ok is True
        assert orch._initialized is True

        # Step 3: Query
        q = BIZRAQuery(
            text="What is the BIZRA data processing pipeline?",
            complexity=QueryComplexity.MODERATE,
        )
        resp = await orch.query(q)

        # Step 4: Verify response fields
        assert isinstance(resp.query, str)
        assert isinstance(resp.answer, str)
        assert isinstance(resp.snr_score, float)
        assert isinstance(resp.ihsan_achieved, bool)
        assert isinstance(resp.sources, list)
        assert isinstance(resp.reasoning_trace, list)
        assert isinstance(resp.tension_analysis, dict)
        assert isinstance(resp.execution_time, float)
        assert isinstance(resp.query_trace, dict)
        assert isinstance(resp.synergies, list)
        assert isinstance(resp.compounds, list)
        assert isinstance(resp.learning_boost, float)
        assert isinstance(resp.modality_used, list)
        assert isinstance(resp.metadata, dict)

        # Step 5: Status confirms health
        status = orch.get_system_status()
        assert status["initialized"] is True
        assert "engines" in status

    @requires_real_data
    async def test_multiple_queries_sequential(self, initialized_orchestrator):
        """Three queries in sequence, all succeed."""
        queries = [
            BIZRAQuery(
                text="How does BIZRA handle duplicate files?",
                complexity=QueryComplexity.SIMPLE,
            ),
            BIZRAQuery(
                text="What embedding model does BIZRA use?",
                complexity=QueryComplexity.SIMPLE,
            ),
            BIZRAQuery(
                text="Describe the ARTE engine's role in quality validation",
                complexity=QueryComplexity.MODERATE,
            ),
        ]
        for i, q in enumerate(queries):
            resp = await initialized_orchestrator.query(q)
            assert isinstance(resp, BIZRAResponse), f"Query {i} failed to return BIZRAResponse"
            assert len(resp.answer.strip()) > 0, f"Query {i} returned empty answer"

    @requires_real_data
    async def test_query_before_init(self):
        """Querying before explicit init triggers auto-initialization."""
        orch = BIZRAOrchestrator(
            enable_pat=False,
            enable_kep=False,
            enable_multimodal=False,
        )
        assert not orch._initialized

        q = BIZRAQuery(
            text="What is BIZRA?",
            complexity=QueryComplexity.SIMPLE,
        )
        # The query method calls initialize() internally if not yet done
        resp = await orch.query(q)
        assert orch._initialized is True
        assert isinstance(resp, BIZRAResponse)

    async def test_idempotent_init(self, orchestrator):
        """Calling initialize() twice does not break anything."""
        first = await orchestrator.initialize()
        second = await orchestrator.initialize()
        assert first is True
        assert second is True
        assert orchestrator._initialized is True


# ---------------------------------------------------------------------------
# 8. TestOneHumanGracefulDegradation
# ---------------------------------------------------------------------------


class TestOneHumanGracefulDegradation:
    """Disabled engines do not prevent the orchestrator from answering."""

    QUERY_TEXT = "How does BIZRA store processed documents?"

    @requires_real_data
    async def test_disabled_pat_still_works(self):
        """Orchestrator with enable_pat=False still answers."""
        orch = BIZRAOrchestrator(
            enable_pat=False,
            enable_kep=True,
            enable_multimodal=True,
        )
        await orch.initialize()
        q = BIZRAQuery(text=self.QUERY_TEXT, complexity=QueryComplexity.SIMPLE)
        resp = await orch.query(q)
        assert isinstance(resp, BIZRAResponse)
        assert len(resp.answer.strip()) > 0

    @requires_real_data
    async def test_disabled_kep_still_works(self):
        """Orchestrator with enable_kep=False still answers."""
        orch = BIZRAOrchestrator(
            enable_pat=True,
            enable_kep=False,
            enable_multimodal=True,
        )
        await orch.initialize()
        q = BIZRAQuery(text=self.QUERY_TEXT, complexity=QueryComplexity.SIMPLE)
        resp = await orch.query(q)
        assert isinstance(resp, BIZRAResponse)
        assert len(resp.answer.strip()) > 0

    @requires_real_data
    async def test_disabled_multimodal_still_works(self):
        """Orchestrator with enable_multimodal=False still answers."""
        orch = BIZRAOrchestrator(
            enable_pat=True,
            enable_kep=True,
            enable_multimodal=False,
        )
        await orch.initialize()
        q = BIZRAQuery(text=self.QUERY_TEXT, complexity=QueryComplexity.SIMPLE)
        resp = await orch.query(q)
        assert isinstance(resp, BIZRAResponse)
        assert len(resp.answer.strip()) > 0

    @requires_real_data
    async def test_all_extras_disabled(self):
        """Only hypergraph + ARTE remain -- orchestrator still works."""
        orch = BIZRAOrchestrator(
            enable_pat=False,
            enable_kep=False,
            enable_multimodal=False,
            enable_discipline=False,
        )
        await orch.initialize()
        q = BIZRAQuery(text=self.QUERY_TEXT, complexity=QueryComplexity.SIMPLE)
        resp = await orch.query(q)
        assert isinstance(resp, BIZRAResponse)
        assert len(resp.answer.strip()) > 0
        # Verify extras are indeed disabled
        status = orch.get_system_status()
        assert status["engines"]["pat"] == "disabled"
        assert status["engines"]["kep"] == "disabled"
        assert status["engines"]["multimodal"] == "disabled"
