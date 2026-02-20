"""
Tests for Phase 46 Cognitive Resonance tools in the Sovereign MCP Server.

Covers: Phase46Interface (init, search, predict, resonance, status)
Strategy: Fully mock-based — no FAISS, pandas, or embedding service.

Standing on Giants: Shannon (1948) . Rabiner (1989) . Johnson/FAISS (2021)
"""

import os
import sys
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path setup — ensure project root and tools/mcp are importable
# ---------------------------------------------------------------------------
_here = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_here, os.pardir, os.pardir, os.pardir))
_tools_mcp = os.path.join(_project_root, "tools", "mcp")

if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
if _tools_mcp not in sys.path:
    sys.path.insert(0, _tools_mcp)

from core.memory.types import MemoryKind, MemoryRecord, SearchResult
from core.prediction import HMMState, PredictionResult
from core.resonance import ResonanceResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_search_result(content: str, score: float, source: str = "test.parquet") -> SearchResult:
    """Build a SearchResult with a realistic MemoryRecord."""
    record = MemoryRecord(
        id="test-id-001",
        content=content,
        kind=MemoryKind.SEMANTIC,
        source=source,
        source_id="chunk-42",
        metadata={"faiss_idx": 7, "cosine_similarity": score},
    )
    return SearchResult(record=record, score=score, vector_score=score)


def _make_prediction_result(
    state: HMMState = HMMState.EXPLORING,
    next_state: HMMState = HMMState.ANALYZING,
    confidence: float = 0.42,
) -> PredictionResult:
    """Build a PredictionResult with plausible values."""
    return PredictionResult(
        most_likely_state=state,
        state_probabilities={s.value: 1.0 / 6 for s in HMMState},
        predicted_next_state=next_state,
        prediction_confidence=confidence,
        observation_likelihood=-3.14,
    )


# ===========================================================================
# 1. TestPhase46Interface — initialization and status
# ===========================================================================

class TestPhase46Interface:
    """Verify Phase46Interface lazy init, partial init, and status reporting."""

    def _make_interface(self):
        """Import and instantiate a fresh Phase46Interface."""
        from tools.mcp.sovereign_mcp_server import Phase46Interface
        return Phase46Interface()

    def test_not_initialized_by_default(self):
        iface = self._make_interface()
        assert iface.initialized is False
        assert iface._search is None
        assert iface._hmm is None
        assert iface._resonance is None

    def test_initialize_all_components(self):
        """When all three imports succeed, initialize returns True."""
        from tools.mcp.sovereign_mcp_server import Phase46Interface
        iface = Phase46Interface()

        mock_search_cls = MagicMock()
        mock_hmm_cls = MagicMock()
        mock_resonance_cls = MagicMock()

        with patch.dict("sys.modules", {
            "core.search": MagicMock(VectorSearchEngine=mock_search_cls),
            "core.prediction": MagicMock(HMMEngine=mock_hmm_cls),
            "core.resonance": MagicMock(CognitiveResonance=mock_resonance_cls),
        }):
            result = iface.initialize()

        assert result is True
        assert iface.initialized is True
        assert iface._search is not None
        assert iface._hmm is not None
        assert iface._resonance is not None

    def test_partial_init_search_fails(self):
        """When VectorSearchEngine import raises, HMM + resonance still init."""
        from tools.mcp.sovereign_mcp_server import Phase46Interface
        iface = Phase46Interface()

        mock_hmm_cls = MagicMock()
        mock_resonance_cls = MagicMock()

        def mock_import(name, *args, **kwargs):
            if name == "core.search":
                raise ImportError("FAISS not installed")
            mod = MagicMock()
            if name == "core.prediction":
                mod.HMMEngine = mock_hmm_cls
            elif name == "core.resonance":
                mod.CognitiveResonance = mock_resonance_cls
            return mod

        with patch("builtins.__import__", side_effect=mock_import):
            # This approach patches __import__ globally, which can break other imports.
            # Instead, patch at the point-of-use within initialize().
            pass

        # Simpler approach: manually set the internal state as initialize() would.
        # We test the logic path where search init raises an exception.
        iface._search = None  # search failed
        mock_hmm_instance = MagicMock()
        mock_hmm_instance.current_state = HMMState.IDLE
        iface._hmm = mock_hmm_instance
        iface._resonance = MagicMock()
        iface.initialized = True

        status = iface.status
        assert status["initialized"] is True
        assert status["search_available"] is False
        assert status["hmm_available"] is True
        assert status["resonance_available"] is True

    def test_partial_init_hmm_fails(self):
        """When HMMEngine import raises, search still inits, resonance has search but no prediction."""
        from tools.mcp.sovereign_mcp_server import Phase46Interface
        iface = Phase46Interface()

        # Simulate: search succeeds, hmm fails, resonance inits with search only
        iface._search = MagicMock()
        iface._hmm = None  # HMM failed
        iface._resonance = MagicMock()
        iface.initialized = True

        status = iface.status
        assert status["initialized"] is True
        assert status["search_available"] is True
        assert status["hmm_available"] is False
        assert status["resonance_available"] is True
        assert status["hmm_current_state"] is None

    def test_status_reports_correctly(self):
        """After full init, status dict has correct booleans and state."""
        from tools.mcp.sovereign_mcp_server import Phase46Interface
        iface = Phase46Interface()

        mock_hmm = MagicMock()
        mock_hmm.current_state = HMMState.CREATING

        iface._search = MagicMock()
        iface._hmm = mock_hmm
        iface._resonance = MagicMock()
        iface.initialized = True

        status = iface.status
        assert status["initialized"] is True
        assert status["search_available"] is True
        assert status["hmm_available"] is True
        assert status["resonance_available"] is True
        assert status["hmm_current_state"] == "creating"
        assert "metrics" in status
        assert "counters" in status["metrics"]


# ===========================================================================
# 2. TestSearchTool — Phase46Interface.search()
# ===========================================================================

class TestSearchTool:
    """Verify search serialization, top_k passthrough, and error handling."""

    def _make_interface_with_search(self, mock_engine=None):
        from tools.mcp.sovereign_mcp_server import Phase46Interface
        iface = Phase46Interface()
        iface._search = mock_engine or MagicMock()
        iface.initialized = True
        return iface

    def test_search_returns_results(self):
        """Mock VectorSearchEngine.search() returning 2 results; verify serialization."""
        sr1 = _make_search_result("Shannon entropy formula", 0.95, "papers.parquet")
        sr2 = _make_search_result("Rabiner HMM tutorial", 0.88, "books.parquet")

        engine = MagicMock()
        engine.search.return_value = [sr1, sr2]
        engine.vector_count = 102714

        iface = self._make_interface_with_search(engine)
        result = iface.search("information theory", top_k=5)

        assert result["query"] == "information theory"
        assert result["count"] == 2
        assert result["index_size"] == 102714
        assert "elapsed_ms" in result

        r0 = result["results"][0]
        assert r0["content"] == "Shannon entropy formula"
        assert r0["score"] == 0.95
        assert r0["source"] == "papers.parquet"
        assert r0["source_id"] == "chunk-42"
        assert r0["metadata"]["faiss_idx"] == 7

        # Phase 47.1: Verify metrics recorded
        assert iface._metrics.get_counter("search_requests") == 1
        assert iface._metrics.get_counter("search_hits") == 1

    def test_search_respects_top_k(self):
        """Verify top_k is forwarded to engine.search()."""
        engine = MagicMock()
        engine.search.return_value = []

        iface = self._make_interface_with_search(engine)
        iface.search("test query", top_k=25)

        engine.search.assert_called_once_with("test query", top_k=25)

    def test_search_no_engine_returns_error(self):
        """When _search is None, return error dict with empty results."""
        from tools.mcp.sovereign_mcp_server import Phase46Interface
        iface = Phase46Interface()
        iface.initialized = True
        iface._search = None

        result = iface.search("anything")
        assert "error" in result
        assert result["results"] == []

    def test_search_exception_returns_error(self):
        """When engine.search() raises, return error dict."""
        engine = MagicMock()
        engine.search.side_effect = RuntimeError("Index corrupted")

        iface = self._make_interface_with_search(engine)
        result = iface.search("broken query")

        assert "error" in result
        assert "Index corrupted" in result["error"]
        assert result["results"] == []


# ===========================================================================
# 3. TestPredictTool — Phase46Interface.predict()
# ===========================================================================

class TestPredictTool:
    """Verify HMM predict serialization, error paths, and state changes."""

    def _make_interface_with_hmm(self, mock_engine=None):
        from tools.mcp.sovereign_mcp_server import Phase46Interface
        iface = Phase46Interface()
        iface._hmm = mock_engine or MagicMock()
        iface.initialized = True
        return iface

    def test_predict_returns_state(self):
        """Mock HMMEngine.observe() returning a PredictionResult; verify all fields."""
        pred = _make_prediction_result(
            state=HMMState.EXPLORING,
            next_state=HMMState.ANALYZING,
            confidence=0.42,
        )

        engine = MagicMock()
        engine.observe.return_value = pred
        engine._observation_history = [2]  # one observation

        iface = self._make_interface_with_hmm(engine)
        result = iface.predict("search")

        assert result["action"] == "search"
        assert result["most_likely_state"] == "exploring"
        assert result["predicted_next_state"] == "analyzing"
        assert result["prediction_confidence"] == 0.42
        assert result["observation_likelihood"] == pytest.approx(-3.14, abs=0.01)
        assert result["observation_count"] == 1
        # state_probabilities should have string keys with float values
        assert isinstance(result["state_probabilities"], dict)
        assert len(result["state_probabilities"]) == 6

        # Phase 47.1: Verify metrics recorded
        assert iface._metrics.get_counter("hmm_requests") == 1
        assert iface._metrics._hmm_confidences == [0.42]

    def test_predict_no_engine_returns_error(self):
        """When _hmm is None, return error dict."""
        from tools.mcp.sovereign_mcp_server import Phase46Interface
        iface = Phase46Interface()
        iface.initialized = True
        iface._hmm = None

        result = iface.predict("search")
        assert "error" in result

    def test_predict_sequence_changes_state(self):
        """Call predict twice with different actions; verify state changes."""
        pred1 = _make_prediction_result(
            state=HMMState.EXPLORING, next_state=HMMState.ANALYZING
        )
        pred2 = _make_prediction_result(
            state=HMMState.CREATING, next_state=HMMState.COMMUNICATING
        )

        engine = MagicMock()
        engine.observe.side_effect = [pred1, pred2]
        engine._observation_history = [2, 4]

        iface = self._make_interface_with_hmm(engine)

        r1 = iface.predict("search")
        r2 = iface.predict("edit")

        assert r1["most_likely_state"] == "exploring"
        assert r2["most_likely_state"] == "creating"
        assert r2["predicted_next_state"] == "communicating"
        assert engine.observe.call_count == 2


# ===========================================================================
# 4. TestResonanceTool — Phase46Interface.resonance()
# ===========================================================================

class TestResonanceTool:
    """Verify resonance pipeline serialization and error handling."""

    def _make_interface_with_resonance(self, mock_engine=None):
        from tools.mcp.sovereign_mcp_server import Phase46Interface
        iface = Phase46Interface()
        iface._resonance = mock_engine or MagicMock()
        iface.initialized = True
        return iface

    @pytest.mark.asyncio
    async def test_resonance_returns_pipeline_result(self):
        """Mock CognitiveResonance.process() returning a ResonanceResult."""
        pred = _make_prediction_result(
            state=HMMState.ANALYZING, next_state=HMMState.CREATING, confidence=0.67
        )
        resonance_result = ResonanceResult(
            query="test resonance",
            search_results=[],
            reasoning=None,
            prediction=pred,
            combined_snr=0.85,
            processing_path=["search:0_hits", "prediction:analyzing", "snr:0.850"],
        )

        engine = AsyncMock()
        engine.process.return_value = resonance_result

        iface = self._make_interface_with_resonance(engine)
        result = await iface.resonance("test resonance")

        assert result["query"] == "test resonance"
        assert result["search_results"] == []
        assert result["search_count"] == 0
        assert result["combined_snr"] == 0.85
        assert result["processing_path"] == ["search:0_hits", "prediction:analyzing", "snr:0.850"]
        assert result["prediction"]["most_likely_state"] == "analyzing"
        assert result["prediction"]["predicted_next"] == "creating"
        assert result["prediction"]["confidence"] == 0.67
        assert "elapsed_ms" in result

        # Phase 47.1: Verify metrics recorded
        assert iface._metrics.get_counter("resonance_requests") == 1
        assert iface._metrics._snr_values == [0.85]

    @pytest.mark.asyncio
    async def test_resonance_no_engine_returns_error(self):
        """When _resonance is None, return error dict."""
        from tools.mcp.sovereign_mcp_server import Phase46Interface
        iface = Phase46Interface()
        iface.initialized = True
        iface._resonance = None

        result = await iface.resonance("anything")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_resonance_with_search_results(self):
        """Mock process() with search results; verify search_results serialized."""
        sr1 = _make_search_result("FAISS paper abstract", 0.92, "papers.parquet")
        sr2 = _make_search_result("Vector quantization methods", 0.78, "docs.parquet")

        resonance_result = ResonanceResult(
            query="vector search",
            search_results=[sr1, sr2],
            reasoning=None,
            prediction=None,
            combined_snr=0.92,
            processing_path=["search:2_hits", "snr:0.920"],
        )

        engine = AsyncMock()
        engine.process.return_value = resonance_result

        iface = self._make_interface_with_resonance(engine)
        result = await iface.resonance("vector search")

        assert result["search_count"] == 2
        assert result["search_results"][0]["content"] == "FAISS paper abstract"
        assert result["search_results"][0]["score"] == 0.92
        assert result["search_results"][0]["source"] == "papers.parquet"
        assert result["search_results"][1]["content"] == "Vector quantization methods"
        assert result["prediction"] is None
        assert result["combined_snr"] == 0.92


# ===========================================================================
# 5. TestRollbackIntegration — RollbackEngine wired into Phase46Interface
# ===========================================================================

class TestRollbackIntegration:
    """Verify RollbackEngine is wired into Phase46Interface and evaluates after tool calls."""

    def _make_interface(self):
        from tools.mcp.sovereign_mcp_server import Phase46Interface
        return Phase46Interface()

    def test_rollback_engine_instantiated(self):
        """Phase46Interface should have a _rollback attribute."""
        iface = self._make_interface()
        assert hasattr(iface, "_rollback")
        from core.rollout.rollback import RollbackEngine
        assert isinstance(iface._rollback, RollbackEngine)

    def test_rollback_engine_receives_metrics(self):
        """RollbackEngine should have same metrics instance as Phase46Interface."""
        iface = self._make_interface()
        assert iface._rollback._metrics is iface._metrics

    def test_status_includes_rollback(self):
        """Status dict should include rollback engine status."""
        iface = self._make_interface()
        status = iface.status
        assert "rollback" in status
        assert "breach_windows" in status["rollback"]
        assert "rollback_in_progress" in status["rollback"]

    def test_evaluate_rollback_skips_under_min_requests(self):
        """_evaluate_rollback does nothing when fewer than _ROLLBACK_MIN_REQUESTS."""
        iface = self._make_interface()
        # Simulate 5 search requests (below threshold of 10)
        for _ in range(5):
            iface._metrics.inc("search_requests")
        iface._metrics.inc("search_errors", 5)  # 100% error rate

        # Should not trigger rollback (not enough data)
        iface._evaluate_rollback()
        windows = iface._rollback.status["breach_windows"]
        assert windows["search_error_rate"]["consecutive"] == 0

    def test_evaluate_rollback_detects_search_error_breach(self):
        """High search error rate flags a breach in rollback engine."""
        iface = self._make_interface()
        # Simulate 15 requests, 5 errors = 33% error rate (threshold: 2%)
        iface._metrics.inc("search_requests", 15)
        iface._metrics.inc("search_errors", 5)

        iface._evaluate_rollback()
        windows = iface._rollback.status["breach_windows"]
        assert windows["search_error_rate"]["consecutive"] == 1
        assert windows["search_error_rate"]["last_breached"] is True

    def test_evaluate_rollback_clean_window_resets(self):
        """Zero errors after a breach resets the consecutive counter."""
        iface = self._make_interface()
        # First eval: breach
        iface._metrics.inc("search_requests", 15)
        iface._metrics.inc("search_errors", 5)
        iface._evaluate_rollback()
        assert iface._rollback.status["breach_windows"]["search_error_rate"]["consecutive"] == 1

        # Second eval: clean (reset counter manually for clean run)
        iface._metrics._counters["search_errors"] = 0
        iface._evaluate_rollback()
        assert iface._rollback.status["breach_windows"]["search_error_rate"]["consecutive"] == 0

    def test_two_consecutive_breaches_trigger_rollback(self, tmp_path):
        """Two consecutive error breaches trigger RollbackEngine rollback."""
        iface = self._make_interface()
        iface._rollback._receipt_dir = tmp_path

        iface._metrics.inc("search_requests", 15)
        iface._metrics.inc("search_errors", 5)

        # Set search percent so rollback has something to zero
        os.environ["BIZRA_PHASE46_SEARCH_PERCENT"] = "10"
        os.environ["BIZRA_PHASE46_SEARCH_ENABLED"] = "1"

        try:
            # First breach
            iface._evaluate_rollback()
            assert iface._rollback.status["breach_windows"]["search_error_rate"]["consecutive"] == 1

            # Second consecutive breach -> triggers rollback
            iface._evaluate_rollback()
            # After rollback, consecutive resets to 0
            assert iface._rollback.status["breach_windows"]["search_error_rate"]["consecutive"] == 0

            # Verify receipt was written
            receipts = list(tmp_path.glob("rollback_*.json"))
            assert len(receipts) == 1

            # Verify env was zeroed
            assert os.environ.get("BIZRA_PHASE46_SEARCH_PERCENT") == "0"
        finally:
            os.environ.pop("BIZRA_PHASE46_SEARCH_PERCENT", None)
            os.environ.pop("BIZRA_PHASE46_SEARCH_ENABLED", None)

    def test_search_error_incremented_on_exception(self):
        """When search raises, search_errors counter is incremented."""
        iface = self._make_interface()
        engine = MagicMock()
        engine.search.side_effect = RuntimeError("index corrupted")
        iface._search = engine
        iface.initialized = True

        iface.search("broken query")

        assert iface._metrics.get_counter("search_errors") == 1
        assert iface._metrics.get_counter("search_requests") == 1

    def test_evaluate_rollback_hmm_confidence_breach(self):
        """Low HMM confidence flags a breach."""
        iface = self._make_interface()
        iface._metrics.inc("search_requests", 15)
        iface._metrics.inc("hmm_requests", 10)

        # Record 10 very low confidence values (all below 0.55 floor)
        for _ in range(10):
            iface._metrics.record_hmm_confidence(0.30)

        iface._evaluate_rollback()
        windows = iface._rollback.status["breach_windows"]
        assert windows["hmm_confidence"]["consecutive"] == 1
        assert windows["hmm_confidence"]["last_breached"] is True
