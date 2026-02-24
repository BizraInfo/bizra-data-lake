"""Tests for Phase 46 CognitiveResonance orchestration facade.

Validates the search -> reason -> predict pipeline, combined-SNR policy,
graceful degradation when components are unavailable, and observe() shortcut.

Standing on Giants: Shannon (1948) . Besta (GoT, 2024) . Rabiner (HMM, 1989)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.memory.types import MemoryKind, MemoryRecord, SearchResult
from core.resonance import CognitiveResonance, ResonanceResult, _compute_combined_snr

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(
    content: str = "test content", source: str = "unit_test"
) -> MemoryRecord:
    return MemoryRecord(
        id="rec-1", content=content, kind=MemoryKind.SEMANTIC, source=source
    )


def _make_search_result(score: float = 0.9, vector_score: float = 0.0) -> SearchResult:
    return SearchResult(record=_make_record(), score=score, vector_score=vector_score)


@dataclass(frozen=True)
class _FakePrediction:
    """Minimal stand-in for PredictionResult (avoids numpy import)."""

    most_likely_state: Any  # HMMState-like
    state_probabilities: dict
    predicted_next_state: Any
    prediction_confidence: float
    observation_likelihood: float


class _FakeState:
    """HMMState-like enum stand-in with a .value attribute."""

    def __init__(self, value: str = "exploring"):
        self.value = value


@dataclass
class _FakeGoTBridgeResult:
    """Minimal stand-in for GoTBridgeResult."""

    answer: str = "got answer"
    snr_score: float = 0.92
    converged: bool = True
    hypotheses_explored: int = 5
    reasoning_depth: int = 3
    convergence_path: list = None

    def __post_init__(self):
        if self.convergence_path is None:
            self.convergence_path = ["step1", "step2"]


# =========================================================================
# 1. ResonanceResult frozen dataclass
# =========================================================================


class TestResonanceResult:
    """ResonanceResult must be an immutable frozen dataclass."""

    def test_frozen_fields(self):
        result = ResonanceResult(
            query="hello",
            search_results=[],
            reasoning=None,
            prediction=None,
            combined_snr=0.5,
            processing_path=["snr:0.500"],
        )
        assert result.query == "hello"
        assert result.combined_snr == 0.5

    def test_immutable(self):
        result = ResonanceResult(
            query="q",
            search_results=[],
            reasoning=None,
            prediction=None,
            combined_snr=0.0,
            processing_path=[],
        )
        with pytest.raises(AttributeError):
            result.query = "changed"  # type: ignore[misc]


# =========================================================================
# 2. No components -- graceful degradation
# =========================================================================


class TestNoComponents:
    """CognitiveResonance with no components returns a minimal result."""

    async def test_empty_resonance(self):
        cr = CognitiveResonance()
        result = await cr.process("hello world")
        assert isinstance(result, ResonanceResult)
        assert result.query == "hello world"
        assert result.search_results == []
        assert result.reasoning is None
        assert result.prediction is None
        assert result.combined_snr == 0.0

    async def test_processing_path_contains_snr(self):
        cr = CognitiveResonance()
        result = await cr.process("test")
        assert any("snr:" in p for p in result.processing_path)


# =========================================================================
# 3. Search-only pipeline
# =========================================================================


class TestSearchOnly:
    """Pipeline with only a search engine wired in."""

    def _make_search_engine(self, results: List[SearchResult]) -> MagicMock:
        engine = MagicMock()
        engine.search.return_value = results
        return engine

    async def test_search_populates_results(self):
        hits = [_make_search_result(score=0.85)]
        cr = CognitiveResonance(search=self._make_search_engine(hits))
        result = await cr.process("find something")
        assert len(result.search_results) == 1
        assert result.search_results[0].score == 0.85

    async def test_combined_snr_equals_max_vector_score(self):
        hits = [_make_search_result(score=0.80), _make_search_result(score=0.95)]
        cr = CognitiveResonance(search=self._make_search_engine(hits))
        result = await cr.process("find")
        assert result.combined_snr == pytest.approx(0.95)

    async def test_search_error_degrades_gracefully(self):
        engine = MagicMock()
        engine.search.side_effect = RuntimeError("FAISS unavailable")
        cr = CognitiveResonance(search=engine)
        result = await cr.process("boom")
        assert result.combined_snr == 0.0
        assert any("search:error" in p for p in result.processing_path)


# =========================================================================
# 4. Reasoning-only pipeline
# =========================================================================


class TestReasoningOnly:
    """Pipeline with only a GoT bridge wired in (no search)."""

    def _make_reasoning(self, snr: float = 0.91) -> MagicMock:
        engine = MagicMock()
        bridge_result = _FakeGoTBridgeResult(snr_score=snr)
        engine.reason = AsyncMock(return_value=bridge_result)
        engine.reason_with_evidence = AsyncMock(return_value=bridge_result)
        return engine

    async def test_reasoning_result_propagated(self):
        cr = CognitiveResonance(reasoning=self._make_reasoning(0.91))
        result = await cr.process("why is the sky blue")
        assert result.reasoning is not None
        assert result.reasoning.snr_score == 0.91

    async def test_combined_snr_equals_reasoning_snr(self):
        cr = CognitiveResonance(reasoning=self._make_reasoning(0.88))
        result = await cr.process("why")
        assert result.combined_snr == pytest.approx(0.88)

    async def test_reason_called_without_evidence(self):
        engine = self._make_reasoning()
        cr = CognitiveResonance(reasoning=engine)
        await cr.process("query")
        engine.reason.assert_awaited_once()
        engine.reason_with_evidence.assert_not_awaited()


# =========================================================================
# 5. Full pipeline (search + reasoning + prediction)
# =========================================================================


class TestFullPipeline:
    """All three components wired in."""

    def _build_all(self, search_score: float = 0.90, reasoning_snr: float = 0.92):
        search = MagicMock()
        search.search.return_value = [_make_search_result(score=search_score)]

        reasoning = MagicMock()
        bridge_result = _FakeGoTBridgeResult(snr_score=reasoning_snr)
        reasoning.reason_with_evidence = AsyncMock(return_value=bridge_result)
        reasoning.reason = AsyncMock(return_value=bridge_result)

        prediction = MagicMock()
        pred_result = _FakePrediction(
            most_likely_state=_FakeState("exploring"),
            state_probabilities={"exploring": 0.7, "idle": 0.3},
            predicted_next_state=_FakeState("analyzing"),
            prediction_confidence=0.65,
            observation_likelihood=-2.3,
        )
        prediction.observe.return_value = pred_result

        return search, reasoning, prediction

    async def test_combined_snr_formula(self):
        search, reasoning, prediction = self._build_all(
            search_score=0.90, reasoning_snr=0.92
        )
        cr = CognitiveResonance(
            search=search, reasoning=reasoning, prediction=prediction
        )
        result = await cr.process("test query")
        expected = 0.7 * 0.92 + 0.3 * 0.90
        assert result.combined_snr == pytest.approx(expected)

    async def test_reason_with_evidence_called(self):
        search, reasoning, prediction = self._build_all()
        cr = CognitiveResonance(
            search=search, reasoning=reasoning, prediction=prediction
        )
        await cr.process("query")
        reasoning.reason_with_evidence.assert_awaited_once()
        reasoning.reason.assert_not_awaited()

    async def test_prediction_populated(self):
        search, reasoning, prediction = self._build_all()
        cr = CognitiveResonance(
            search=search, reasoning=reasoning, prediction=prediction
        )
        result = await cr.process("search for files")
        assert result.prediction is not None
        assert result.prediction.most_likely_state.value == "exploring"


# =========================================================================
# 6. Combined-SNR policy (exact math, all 4 branches)
# =========================================================================


class TestCombinedSNRPolicy:
    """Verify _compute_combined_snr covers all four branches."""

    def test_both_reasoning_and_search(self):
        reasoning = _FakeGoTBridgeResult(snr_score=0.92)
        assert _compute_combined_snr(reasoning, 0.85) == pytest.approx(
            0.7 * 0.92 + 0.3 * 0.85
        )

    def test_reasoning_only(self):
        reasoning = _FakeGoTBridgeResult(snr_score=0.88)
        assert _compute_combined_snr(reasoning, 0.0) == pytest.approx(0.88)

    def test_search_only(self):
        assert _compute_combined_snr(None, 0.75) == pytest.approx(0.75)

    def test_neither(self):
        assert _compute_combined_snr(None, 0.0) == pytest.approx(0.0)

    def test_reasoning_without_snr_attr_treated_as_none(self):
        """An object without .snr_score is treated as 'no reasoning'."""
        fake = MagicMock(spec=[])  # empty spec -> no attributes
        assert _compute_combined_snr(fake, 0.7) == pytest.approx(0.7)


# =========================================================================
# 7. Processing path audit trail
# =========================================================================


class TestProcessingPath:
    """Verify processing_path contains expected stage markers."""

    async def test_search_stage_recorded(self):
        search = MagicMock()
        search.search.return_value = [_make_search_result(score=0.8)]
        cr = CognitiveResonance(search=search)
        result = await cr.process("q")
        assert any(p.startswith("search:") for p in result.processing_path)

    async def test_reasoning_stage_recorded(self):
        reasoning = MagicMock()
        reasoning.reason = AsyncMock(return_value=_FakeGoTBridgeResult())
        cr = CognitiveResonance(reasoning=reasoning)
        result = await cr.process("q")
        assert "reasoning:ok" in result.processing_path

    async def test_prediction_stage_recorded(self):
        prediction = MagicMock()
        pred = _FakePrediction(
            most_likely_state=_FakeState("idle"),
            state_probabilities={},
            predicted_next_state=_FakeState("idle"),
            prediction_confidence=0.5,
            observation_likelihood=-1.0,
        )
        prediction.observe.return_value = pred
        cr = CognitiveResonance(prediction=prediction)
        result = await cr.process("idle query")
        assert any(p.startswith("prediction:") for p in result.processing_path)

    async def test_snr_always_last(self):
        cr = CognitiveResonance()
        result = await cr.process("anything")
        assert result.processing_path[-1].startswith("snr:")


# =========================================================================
# 8. observe() convenience method
# =========================================================================


class TestObserve:
    """Test the observe-only shortcut (HMM only, no search/reason)."""

    def test_observe_without_prediction_returns_none(self):
        cr = CognitiveResonance()
        assert cr.observe("search") is None

    def test_observe_with_prediction_delegates(self):
        prediction = MagicMock()
        pred = _FakePrediction(
            most_likely_state=_FakeState("analyzing"),
            state_probabilities={},
            predicted_next_state=_FakeState("creating"),
            prediction_confidence=0.8,
            observation_likelihood=-0.5,
        )
        prediction.observe.return_value = pred
        cr = CognitiveResonance(prediction=prediction)
        result = cr.observe("review")
        prediction.observe.assert_called_once_with("review")
        assert result.most_likely_state.value == "analyzing"

    def test_observe_error_returns_none(self):
        prediction = MagicMock()
        prediction.observe.side_effect = RuntimeError("HMM broken")
        cr = CognitiveResonance(prediction=prediction)
        assert cr.observe("edit") is None
