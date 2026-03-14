"""
Tests for HMMEngine — Phase 46 Cognitive Resonance.

Standing on Giants: Rabiner (1989) · Viterbi (1967) · Friston (Active Inference, 2010)
"""

from __future__ import annotations

import json
import math
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pytest

from core.prediction.hmm_engine import (
    DEFAULT_OBSERVATION_SYMBOLS,
    HMMEngine,
    HMMState,
    PredictionResult,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def engine() -> HMMEngine:
    """Default HMM engine with 6 states and 12 observation symbols."""
    return HMMEngine()


@pytest.fixture
def small_engine() -> HMMEngine:
    """Minimal 2-state, 3-symbol engine for deterministic testing."""
    return HMMEngine(n_hidden=2, observation_symbols=["a", "b", "c"])


# ═══════════════════════════════════════════════════════════════════════════════
# HMMState Enum
# ═══════════════════════════════════════════════════════════════════════════════


class TestHMMState:
    """HMMState enum has correct members and serialization."""

    def test_all_six_states_exist(self):
        assert len(HMMState) == 6

    def test_string_values(self):
        assert HMMState.IDLE.value == "idle"
        assert HMMState.EXPLORING.value == "exploring"
        assert HMMState.ORGANIZING.value == "organizing"
        assert HMMState.CREATING.value == "creating"
        assert HMMState.ANALYZING.value == "analyzing"
        assert HMMState.COMMUNICATING.value == "communicating"

    def test_is_str_enum(self):
        assert isinstance(HMMState.IDLE, str)
        assert HMMState.IDLE == "idle"


# ═══════════════════════════════════════════════════════════════════════════════
# PredictionResult
# ═══════════════════════════════════════════════════════════════════════════════


class TestPredictionResult:
    """PredictionResult is frozen and has correct fields."""

    def test_frozen(self):
        result = PredictionResult(
            most_likely_state=HMMState.IDLE,
            state_probabilities={"idle": 1.0},
            predicted_next_state=HMMState.EXPLORING,
            prediction_confidence=0.5,
            observation_likelihood=-1.0,
        )
        with pytest.raises(AttributeError):
            result.most_likely_state = HMMState.CREATING  # type: ignore[misc]

    def test_json_serializable_keys(self):
        result = PredictionResult(
            most_likely_state=HMMState.IDLE,
            state_probabilities={"idle": 0.5, "exploring": 0.5},
            predicted_next_state=HMMState.EXPLORING,
            prediction_confidence=0.5,
            observation_likelihood=-1.0,
        )
        # state_probabilities keys must be plain strings
        for key in result.state_probabilities:
            assert isinstance(key, str)
        # The whole thing should be JSON-serializable via manual conversion
        d = {
            "most_likely_state": result.most_likely_state.value,
            "state_probabilities": result.state_probabilities,
            "predicted_next_state": result.predicted_next_state.value,
            "prediction_confidence": result.prediction_confidence,
            "observation_likelihood": result.observation_likelihood,
        }
        assert json.dumps(d)  # Does not raise


# ═══════════════════════════════════════════════════════════════════════════════
# Initialization
# ═══════════════════════════════════════════════════════════════════════════════


class TestHMMEngineInit:
    """Constructor validation and matrix invariants."""

    def test_default_init(self, engine: HMMEngine):
        A = engine.transition_matrix
        B = engine.emission_matrix
        assert A.shape == (6, 6)
        assert B.shape == (6, 12)

    def test_transition_rows_sum_to_one(self, engine: HMMEngine):
        A = engine.transition_matrix
        np.testing.assert_allclose(A.sum(axis=1), np.ones(6), atol=1e-10)

    def test_emission_rows_sum_to_one(self, engine: HMMEngine):
        B = engine.emission_matrix
        np.testing.assert_allclose(B.sum(axis=1), np.ones(6), atol=1e-10)

    def test_transition_nonnegative(self, engine: HMMEngine):
        assert np.all(engine.transition_matrix >= 0)

    def test_emission_nonnegative(self, engine: HMMEngine):
        assert np.all(engine.emission_matrix >= 0)

    def test_self_transition_dominant(self, engine: HMMEngine):
        """Diagonal should have the highest probability in each row."""
        A = engine.transition_matrix
        diag = np.diag(A)
        for i in range(6):
            assert diag[i] == A[i].max(), f"State {i}: self-transition is not dominant"

    def test_custom_n_hidden(self):
        e = HMMEngine(n_hidden=3, observation_symbols=["x", "y"])
        assert e.transition_matrix.shape == (3, 3)
        assert e.emission_matrix.shape == (3, 2)

    def test_invalid_n_hidden_raises(self):
        with pytest.raises(ValueError, match="n_hidden must be >= 1"):
            HMMEngine(n_hidden=0)

    def test_empty_symbols_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            HMMEngine(observation_symbols=[])

    def test_default_symbols(self, engine: HMMEngine):
        # Should use the DEFAULT_OBSERVATION_SYMBOLS
        assert len(DEFAULT_OBSERVATION_SYMBOLS) == 12

    def test_initial_state_is_uniform(self, engine: HMMEngine):
        result = engine.predict_next()
        probs = list(result.state_probabilities.values())
        # All should be approximately equal (uniform prior)
        assert max(probs) - min(probs) < 0.01

    def test_matrices_are_copies(self, engine: HMMEngine):
        """transition_matrix and emission_matrix return copies, not refs."""
        A1 = engine.transition_matrix
        A2 = engine.transition_matrix
        assert A1 is not A2
        A1[0, 0] = 999.0
        assert engine.transition_matrix[0, 0] != 999.0


# ═══════════════════════════════════════════════════════════════════════════════
# Observe
# ═══════════════════════════════════════════════════════════════════════════════


class TestObserve:
    """observe() updates belief state and returns valid predictions."""

    def test_single_observe(self, engine: HMMEngine):
        result = engine.observe("search")
        assert isinstance(result, PredictionResult)
        assert isinstance(result.most_likely_state, HMMState)

    def test_observe_updates_state(self, engine: HMMEngine):
        # Before: uniform
        pre = engine.predict_next()
        pre_probs = list(pre.state_probabilities.values())
        assert max(pre_probs) - min(pre_probs) < 0.01

        # After observing search, EXPLORING should dominate
        result = engine.observe("search")
        assert (
            result.state_probabilities["exploring"] > result.state_probabilities["idle"]
        )

    def test_observe_search_favors_exploring(self, engine: HMMEngine):
        """Repeated 'search' observations should push belief toward EXPLORING."""
        for _ in range(5):
            result = engine.observe("search")
        assert result.most_likely_state == HMMState.EXPLORING

    def test_observe_edit_favors_creating(self, engine: HMMEngine):
        """Repeated 'edit' observations should push belief toward CREATING."""
        for _ in range(5):
            result = engine.observe("edit")
        assert result.most_likely_state == HMMState.CREATING

    def test_observe_idle_favors_idle(self, engine: HMMEngine):
        for _ in range(5):
            result = engine.observe("idle")
        assert result.most_likely_state == HMMState.IDLE

    def test_observe_chat_favors_communicating(self, engine: HMMEngine):
        for _ in range(5):
            result = engine.observe("chat")
        assert result.most_likely_state == HMMState.COMMUNICATING

    def test_observe_review_favors_analyzing(self, engine: HMMEngine):
        for _ in range(5):
            result = engine.observe("review")
        assert result.most_likely_state == HMMState.ANALYZING

    def test_observe_organize_favors_organizing(self, engine: HMMEngine):
        for _ in range(5):
            result = engine.observe("organize")
        assert result.most_likely_state == HMMState.ORGANIZING

    def test_unknown_symbol_raises(self, engine: HMMEngine):
        with pytest.raises(ValueError, match="Unknown observation symbol"):
            engine.observe("unknown_symbol")

    def test_probabilities_sum_to_one(self, engine: HMMEngine):
        result = engine.observe("edit")
        total = sum(result.state_probabilities.values())
        assert abs(total - 1.0) < 1e-10

    def test_confidence_in_range(self, engine: HMMEngine):
        result = engine.observe("search")
        assert 0.0 <= result.prediction_confidence <= 1.0

    def test_observation_window_bounded(self, engine: HMMEngine):
        """History should not exceed HMM_OBSERVATION_WINDOW."""
        for _ in range(100):
            engine.observe("search")
        state = engine.to_dict()
        assert len(state["observation_history"]) <= 50

    def test_log_likelihood_decreases(self, engine: HMMEngine):
        """Log-likelihood is always <= 0 and accumulates."""
        result = engine.observe("search")
        # After one observation, likelihood should be negative
        assert result.observation_likelihood <= 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Predict Next
# ═══════════════════════════════════════════════════════════════════════════════


class TestPredictNext:
    """predict_next() does not consume an observation."""

    def test_predict_next_returns_result(self, engine: HMMEngine):
        result = engine.predict_next()
        assert isinstance(result, PredictionResult)

    def test_predict_next_idempotent(self, engine: HMMEngine):
        r1 = engine.predict_next()
        r2 = engine.predict_next()
        assert r1.state_probabilities == r2.state_probabilities
        assert r1.predicted_next_state == r2.predicted_next_state

    def test_predict_after_observe_reflects_last_state(self, engine: HMMEngine):
        for _ in range(5):
            engine.observe("search")
        result = engine.predict_next()
        # After heavy "search" observations, exploring should dominate
        assert result.state_probabilities["exploring"] > 0.3


# ═══════════════════════════════════════════════════════════════════════════════
# Decode (Viterbi)
# ═══════════════════════════════════════════════════════════════════════════════


class TestDecode:
    """Viterbi decoding returns a plausible state sequence."""

    def test_decode_returns_correct_length(self, engine: HMMEngine):
        obs = ["search", "navigate", "search", "edit", "compile"]
        states = engine.decode(obs)
        assert len(states) == len(obs)

    def test_decode_returns_hmm_states(self, engine: HMMEngine):
        states = engine.decode(["idle", "idle", "idle"])
        for s in states:
            assert isinstance(s, HMMState)

    def test_decode_idle_sequence(self, engine: HMMEngine):
        """All-idle observations should produce mostly IDLE states."""
        states = engine.decode(["idle"] * 10)
        idle_count = sum(1 for s in states if s == HMMState.IDLE)
        assert idle_count >= 7, f"Expected mostly IDLE, got {states}"

    def test_decode_search_sequence(self, engine: HMMEngine):
        """All-search observations should produce mostly EXPLORING states."""
        states = engine.decode(["search"] * 10)
        exploring_count = sum(1 for s in states if s == HMMState.EXPLORING)
        assert exploring_count >= 7, f"Expected mostly EXPLORING, got {states}"

    def test_decode_mixed_sequence(self, engine: HMMEngine):
        """A realistic sequence should produce a sensible state path."""
        obs = [
            "idle",
            "search",
            "search",
            "navigate",
            "edit",
            "edit",
            "compile",
            "test",
            "review",
        ]
        states = engine.decode(obs)
        # First state should lean IDLE, middle should lean EXPLORING/CREATING
        assert states[0] == HMMState.IDLE
        assert len(states) == 9

    def test_decode_empty_raises(self, engine: HMMEngine):
        with pytest.raises(ValueError, match="non-empty"):
            engine.decode([])

    def test_decode_unknown_symbol_raises(self, engine: HMMEngine):
        with pytest.raises(ValueError, match="Unknown observation symbol"):
            engine.decode(["search", "INVALID"])


# ═══════════════════════════════════════════════════════════════════════════════
# Likelihood (Forward Algorithm)
# ═══════════════════════════════════════════════════════════════════════════════


class TestLikelihood:
    """Forward algorithm log-likelihood."""

    def test_likelihood_returns_float(self, engine: HMMEngine):
        ll = engine.likelihood(["search", "edit", "compile"])
        assert isinstance(ll, float)

    def test_likelihood_negative(self, engine: HMMEngine):
        """Log-likelihood should always be <= 0."""
        ll = engine.likelihood(["search", "edit"])
        assert ll <= 0.0

    def test_longer_sequences_lower_likelihood(self, engine: HMMEngine):
        """Longer sequences generally have lower (more negative) log-likelihood."""
        ll_short = engine.likelihood(["search"])
        ll_long = engine.likelihood(["search"] * 20)
        assert ll_long < ll_short

    def test_consistent_observations_higher_likelihood(self, engine: HMMEngine):
        """Observations matching one state should have higher likelihood
        than random observations."""
        # All search (consistent with EXPLORING)
        ll_consistent = engine.likelihood(["search"] * 5)
        # Random mix
        ll_random = engine.likelihood(["search", "idle", "deploy", "edit", "organize"])
        # Consistent should be higher (less negative)
        assert ll_consistent > ll_random

    def test_likelihood_empty_raises(self, engine: HMMEngine):
        with pytest.raises(ValueError, match="non-empty"):
            engine.likelihood([])


# ═══════════════════════════════════════════════════════════════════════════════
# Learn (Phase 47 Stub)
# ═══════════════════════════════════════════════════════════════════════════════


class TestLearn:
    """learn() performs Baum-Welch EM training (Phase 47 implemented)."""

    def test_learn_returns_log_likelihood(self, engine: HMMEngine):
        """learn() now returns a finite log-likelihood after EM convergence."""
        ll = engine.learn(["search", "edit", "compile", "test", "review"])
        assert isinstance(ll, float)
        assert math.isfinite(ll)
        assert ll <= 0.0  # log-likelihood is always non-positive

    def test_learn_empty_raises(self, engine: HMMEngine):
        with pytest.raises(ValueError, match="non-empty"):
            engine.learn([])

    def test_learn_improves_likelihood(self, engine: HMMEngine):
        """Training on repeated data should not decrease likelihood."""
        obs = ["search", "navigate", "edit", "compile", "test"] * 3
        ll_before = engine.likelihood(obs)
        engine.learn(obs)
        ll_after = engine.likelihood(obs)
        # After training on this data, likelihood should not decrease
        assert ll_after >= ll_before - 1e-6


# ═══════════════════════════════════════════════════════════════════════════════
# Serialization
# ═══════════════════════════════════════════════════════════════════════════════


class TestSerialization:
    """to_dict / from_dict round-trip."""

    def test_round_trip(self, engine: HMMEngine):
        engine.observe("search")
        engine.observe("edit")
        engine.observe("compile")

        data = engine.to_dict()
        restored = HMMEngine.from_dict(data)

        np.testing.assert_allclose(restored.transition_matrix, engine.transition_matrix)
        np.testing.assert_allclose(restored.emission_matrix, engine.emission_matrix)
        assert restored.to_dict()["observation_history"] == data["observation_history"]

    def test_to_dict_is_json_serializable(self, engine: HMMEngine):
        engine.observe("idle")
        data = engine.to_dict()
        serialized = json.dumps(data)
        assert isinstance(serialized, str)
        # Round-trip through JSON
        parsed = json.loads(serialized)
        assert parsed["n_hidden"] == 6

    def test_from_dict_preserves_state(self, engine: HMMEngine):
        for _ in range(3):
            engine.observe("search")
        r1 = engine.predict_next()

        data = engine.to_dict()
        restored = HMMEngine.from_dict(data)
        r2 = restored.predict_next()

        assert r1.most_likely_state == r2.most_likely_state
        for key in r1.state_probabilities:
            assert (
                abs(r1.state_probabilities[key] - r2.state_probabilities[key]) < 1e-10
            )

    def test_dict_has_required_keys(self, engine: HMMEngine):
        data = engine.to_dict()
        required_keys = {
            "n_hidden",
            "n_obs",
            "observation_symbols",
            "pi",
            "A",
            "B",
            "current_state_dist",
            "observation_history",
            "log_likelihood_accum",
            "convergence_threshold",
            "max_iterations",
            "observation_window",
        }
        assert required_keys.issubset(set(data.keys()))


# ═══════════════════════════════════════════════════════════════════════════════
# Current State Property
# ═══════════════════════════════════════════════════════════════════════════════


class TestCurrentState:
    """current_state property reflects belief state."""

    def test_initial_state(self, engine: HMMEngine):
        # With uniform prior, current_state is the first state (tie-break)
        state = engine.current_state
        assert isinstance(state, HMMState)

    def test_state_updates_after_observations(self, engine: HMMEngine):
        for _ in range(10):
            engine.observe("search")
        assert engine.current_state == HMMState.EXPLORING


# ═══════════════════════════════════════════════════════════════════════════════
# Thread Safety
# ═══════════════════════════════════════════════════════════════════════════════


class TestThreadSafety:
    """Concurrent observe() calls do not corrupt internal state."""

    def test_concurrent_observe(self, engine: HMMEngine):
        """Many threads observing in parallel should not raise or corrupt."""
        symbols = list(DEFAULT_OBSERVATION_SYMBOLS)
        errors: list[Exception] = []

        def worker(sym: str, n: int) -> None:
            try:
                for _ in range(n):
                    result = engine.observe(sym)
                    # Basic invariant: probabilities sum to 1
                    total = sum(result.state_probabilities.values())
                    assert abs(total - 1.0) < 1e-6, f"Prob sum = {total}"
            except Exception as e:
                errors.append(e)

        threads = []
        for sym in symbols[:6]:
            t = threading.Thread(target=worker, args=(sym, 20))
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Thread errors: {errors}"

    def test_concurrent_observe_and_predict(self, engine: HMMEngine):
        """Observe and predict_next interleaved across threads."""
        errors: list[Exception] = []

        def observer() -> None:
            try:
                for _ in range(30):
                    engine.observe("edit")
            except Exception as e:
                errors.append(e)

        def predictor() -> None:
            try:
                for _ in range(30):
                    result = engine.predict_next()
                    assert isinstance(result, PredictionResult)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=observer),
            threading.Thread(target=predictor),
            threading.Thread(target=observer),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Thread errors: {errors}"


# ═══════════════════════════════════════════════════════════════════════════════
# Numerical Stability
# ═══════════════════════════════════════════════════════════════════════════════


class TestNumericalStability:
    """Log-space computation handles edge cases without NaN/Inf."""

    def test_long_sequence_no_nan(self, engine: HMMEngine):
        """100+ observations should not produce NaN in probabilities."""
        for _ in range(150):
            result = engine.observe("search")
        probs = list(result.state_probabilities.values())
        assert all(not math.isnan(p) for p in probs), f"NaN in probs: {probs}"
        assert all(not math.isinf(p) for p in probs), f"Inf in probs: {probs}"

    def test_viterbi_long_sequence(self, engine: HMMEngine):
        """Viterbi on a long sequence should not produce errors."""
        obs = ["search", "navigate", "edit", "compile", "test"] * 20
        states = engine.decode(obs)
        assert len(states) == 100
        assert all(isinstance(s, HMMState) for s in states)

    def test_likelihood_long_sequence(self, engine: HMMEngine):
        """Forward algorithm on long sequence should return finite value."""
        obs = ["idle"] * 100
        ll = engine.likelihood(obs)
        assert math.isfinite(ll)
        assert ll < 0


# ═══════════════════════════════════════════════════════════════════════════════
# Integration: Emission Priors Sanity
# ═══════════════════════════════════════════════════════════════════════════════


class TestEmissionPriors:
    """Verify that informed emission priors match state semantics."""

    def test_idle_state_prefers_idle_symbol(self, engine: HMMEngine):
        B = engine.emission_matrix
        idle_state_idx = 0
        idle_symbol_idx = DEFAULT_OBSERVATION_SYMBOLS.index("idle")
        assert B[idle_state_idx, idle_symbol_idx] == B[idle_state_idx].max()

    def test_exploring_state_prefers_search(self, engine: HMMEngine):
        B = engine.emission_matrix
        exploring_idx = 1
        search_idx = DEFAULT_OBSERVATION_SYMBOLS.index("search")
        assert B[exploring_idx, search_idx] == B[exploring_idx].max()

    def test_creating_state_prefers_edit(self, engine: HMMEngine):
        B = engine.emission_matrix
        creating_idx = 3
        edit_idx = DEFAULT_OBSERVATION_SYMBOLS.index("edit")
        assert B[creating_idx, edit_idx] == B[creating_idx].max()

    def test_analyzing_state_prefers_review(self, engine: HMMEngine):
        B = engine.emission_matrix
        analyzing_idx = 4
        review_idx = DEFAULT_OBSERVATION_SYMBOLS.index("review")
        assert B[analyzing_idx, review_idx] == B[analyzing_idx].max()

    def test_communicating_state_prefers_chat(self, engine: HMMEngine):
        B = engine.emission_matrix
        comm_idx = 5
        chat_idx = DEFAULT_OBSERVATION_SYMBOLS.index("chat")
        assert B[comm_idx, chat_idx] == B[comm_idx].max()
