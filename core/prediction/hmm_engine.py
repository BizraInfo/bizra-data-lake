"""
HMM Engine — Hidden Markov Model for Cognitive State Forecasting.

Phase 46: Cognitive Resonance (observe / likelihood / Viterbi / persistence)
Phase 47: Baum-Welch training (deferred)

Timescale T1 (Reactive / Cerebellum): predicts next cognitive micro-state
from an observation stream of desktop events.  Feeds into T2 (GoT diffusion)
to bias hypothesis generation, and into T3 (federated memory) via Takaful
bootstrap — new nodes inherit transition priors from behaviorally similar
peers (KL-divergence grouping).

Architecture:
    lambda = (pi, A, B)
    pi — initial state distribution (n_hidden,)
    A  — transition matrix          (n_hidden, n_hidden)  [informed priors]
    B  — emission matrix            (n_hidden, n_obs)     [informed priors]

Numerical stability: all probability chains computed in log-space to prevent
underflow on long observation sequences.

Thread-safe: all state-mutating operations acquire self._lock, matching
the pattern established in core/proof_engine/evidence_ledger.py and
core/hashtable/skill_cache.py.

Standing on Giants:
    Rabiner (1989) — A Tutorial on Hidden Markov Models
    Viterbi (1967) — Error Bounds for Convolutional Codes
    Friston (2010) — The Free-Energy Principle: A Unified Brain Theory?
    Shannon (1948) — A Mathematical Theory of Communication
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

import numpy as np

from core.integration.constants import (
    HMM_CONVERGENCE_THRESHOLD,
    HMM_MAX_EM_ITERATIONS,
    HMM_NUM_HIDDEN_STATES,
    HMM_OBSERVATION_WINDOW,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Feature Flag
# ═══════════════════════════════════════════════════════════════════════════════
PHASE46_HMM_ENABLED: Final[bool] = (
    os.getenv("BIZRA_PHASE46_HMM_ENABLED", "1").lower() in ("1", "true", "yes")
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Default Observation Vocabulary
# ═══════════════════════════════════════════════════════════════════════════════
DEFAULT_OBSERVATION_SYMBOLS: Final[list[str]] = [
    "file_open",
    "file_save",
    "search",
    "navigate",
    "edit",
    "compile",
    "test",
    "chat",
    "organize",
    "review",
    "deploy",
    "idle",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Hidden States — Cognitive Micro-States
# ═══════════════════════════════════════════════════════════════════════════════
class HMMState(str, Enum):
    """Six cognitive micro-states of a sovereign node operator."""

    IDLE = "idle"
    EXPLORING = "exploring"
    ORGANIZING = "organizing"
    CREATING = "creating"
    ANALYZING = "analyzing"
    COMMUNICATING = "communicating"


# Ordered list matching numpy index positions
_STATE_LIST: Final[list[HMMState]] = list(HMMState)
_STATE_NAMES: Final[list[str]] = [s.value for s in _STATE_LIST]


# ═══════════════════════════════════════════════════════════════════════════════
# Prediction Result — Immutable Output
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class PredictionResult:
    """Immutable result of an HMM prediction step.

    Attributes:
        most_likely_state: Current MAP state estimate.
        state_probabilities: Mapping from state name (str) to probability.
            Uses str keys (not HMMState) for JSON serializability.
        predicted_next_state: One-step-ahead MAP prediction via A.
        prediction_confidence: Max probability of the predicted next state.
        observation_likelihood: Log-likelihood of the observation sequence
            under the current model parameters.
    """

    most_likely_state: HMMState
    state_probabilities: dict[str, float]
    predicted_next_state: HMMState
    prediction_confidence: float
    observation_likelihood: float


# ═══════════════════════════════════════════════════════════════════════════════
# Informed Priors — Transition Matrix
# ═══════════════════════════════════════════════════════════════════════════════
def _build_informed_transition(n: int) -> np.ndarray:
    """Build an informed transition matrix for 6 cognitive states.

    Priors reflect realistic user behavior:
    - Self-transition ~0.50 (users tend to stay in the same state)
    - Natural transitions have elevated probability
    - All rows sum to 1.0

    State ordering: IDLE(0), EXPLORING(1), ORGANIZING(2),
                    CREATING(3), ANALYZING(4), COMMUNICATING(5)
    """
    # Start with a small base probability for all transitions
    A = np.full((n, n), 0.04, dtype=np.float64)

    # Self-transitions (diagonal) — high stickiness
    np.fill_diagonal(A, 0.50)

    # IDLE -> EXPLORING (user begins a session)
    A[0, 1] = 0.20
    # IDLE -> COMMUNICATING (user checks messages)
    A[0, 5] = 0.10

    # EXPLORING -> ANALYZING (found something, now studying it)
    A[1, 4] = 0.15
    # EXPLORING -> ORGANIZING (curating discoveries)
    A[1, 2] = 0.10
    # EXPLORING -> CREATING (inspired to build)
    A[1, 3] = 0.09

    # ORGANIZING -> CREATING (structured thought leads to output)
    A[2, 3] = 0.15
    # ORGANIZING -> ANALYZING (review organized material)
    A[2, 4] = 0.10
    # ORGANIZING -> IDLE (wrap-up)
    A[2, 0] = 0.09

    # CREATING -> ANALYZING (test/review what was built)
    A[3, 4] = 0.18
    # CREATING -> COMMUNICATING (share creation)
    A[3, 5] = 0.10

    # ANALYZING -> CREATING (analysis reveals what to build next)
    A[4, 3] = 0.15
    # ANALYZING -> COMMUNICATING (report findings)
    A[4, 5] = 0.10
    # ANALYZING -> ORGANIZING (file results)
    A[4, 2] = 0.09

    # COMMUNICATING -> EXPLORING (conversation sparks curiosity)
    A[5, 1] = 0.12
    # COMMUNICATING -> CREATING (feedback leads to iteration)
    A[5, 3] = 0.10
    # COMMUNICATING -> IDLE (conversation ends)
    A[5, 0] = 0.12

    # Normalize each row to sum to 1.0
    row_sums = A.sum(axis=1, keepdims=True)
    A = A / row_sums

    return A


# ═══════════════════════════════════════════════════════════════════════════════
# Informed Priors — Emission Matrix
# ═══════════════════════════════════════════════════════════════════════════════
def _build_informed_emission(n_hidden: int, n_obs: int) -> np.ndarray:
    """Build emission matrix B (n_hidden x n_obs) with informed priors.

    Maps observation symbols to cognitive states logically.

    Observation indices (DEFAULT_OBSERVATION_SYMBOLS):
        0: file_open, 1: file_save, 2: search, 3: navigate,
        4: edit, 5: compile, 6: test, 7: chat,
        8: organize, 9: review, 10: deploy, 11: idle

    State indices:
        0: IDLE, 1: EXPLORING, 2: ORGANIZING,
        3: CREATING, 4: ANALYZING, 5: COMMUNICATING
    """
    # Start with uniform low base
    B = np.full((n_hidden, n_obs), 0.03, dtype=np.float64)

    # IDLE: idle(11) dominant, file_open(0) moderate
    B[0, 11] = 0.55
    B[0, 0] = 0.12

    # EXPLORING: search(2), navigate(3), file_open(0) dominant
    B[1, 2] = 0.28
    B[1, 3] = 0.25
    B[1, 0] = 0.15

    # ORGANIZING: organize(8), file_save(1) dominant
    B[2, 8] = 0.30
    B[2, 1] = 0.25
    B[2, 0] = 0.10

    # CREATING: edit(4), compile(5) dominant
    B[3, 4] = 0.30
    B[3, 5] = 0.20
    B[3, 1] = 0.12

    # ANALYZING: review(9), test(6) dominant
    B[4, 9] = 0.28
    B[4, 6] = 0.25
    B[4, 0] = 0.10

    # COMMUNICATING: chat(7), deploy(10) dominant
    B[5, 7] = 0.32
    B[5, 10] = 0.22
    B[5, 9] = 0.10

    # Normalize each row to sum to 1.0
    row_sums = B.sum(axis=1, keepdims=True)
    B = B / row_sums

    return B


# ═══════════════════════════════════════════════════════════════════════════════
# Log-Space Utilities
# ═══════════════════════════════════════════════════════════════════════════════
_LOG_ZERO: Final[float] = -1e10  # Sentinel for log(0) to avoid -inf


def _safe_log(x: np.ndarray) -> np.ndarray:
    """Element-wise log with floor at _LOG_ZERO for zero-valued entries."""
    with np.errstate(divide="ignore"):
        result = np.log(x)
    result[np.isneginf(result)] = _LOG_ZERO
    return result


def _log_sum_exp(log_probs: np.ndarray) -> float:
    """Numerically stable log-sum-exp over a 1-D array."""
    max_val = np.max(log_probs)
    if max_val <= _LOG_ZERO:
        return _LOG_ZERO
    return float(max_val + np.log(np.sum(np.exp(log_probs - max_val))))


# ═══════════════════════════════════════════════════════════════════════════════
# HMM Engine
# ═══════════════════════════════════════════════════════════════════════════════
class HMMEngine:
    """Hidden Markov Model engine for cognitive state forecasting.

    Phase 46 capabilities:
        observe()       — ingest a single observation, update belief state
        predict_next()  — one-step-ahead prediction (no observation update)
        decode()        — Viterbi decoding of an observation sequence
        likelihood()    — Forward algorithm log-likelihood
        to_dict()       — full parameter serialization
        from_dict()     — deserialization / reconstruction

    Phase 47 (deferred):
        learn()         — Baum-Welch (EM) parameter re-estimation

    Thread-safe: all state-mutating methods acquire self._lock.

    Usage:
        >>> engine = HMMEngine()
        >>> result = engine.observe("search")
        >>> result.most_likely_state
        <HMMState.EXPLORING: 'exploring'>
    """

    __slots__ = (
        "_n_hidden",
        "_n_obs",
        "_pi",
        "_A",
        "_B",
        "_observation_symbols",
        "_symbol_to_index",
        "_observation_history",
        "_observation_window",
        "_current_state_dist",
        "_log_likelihood_accum",
        "_convergence_threshold",
        "_max_iterations",
        "_lock",
    )

    def __init__(
        self,
        n_hidden: int = HMM_NUM_HIDDEN_STATES,
        observation_symbols: list[str] | None = None,
        convergence_threshold: float = HMM_CONVERGENCE_THRESHOLD,
        max_iterations: int = HMM_MAX_EM_ITERATIONS,
    ) -> None:
        if n_hidden < 1:
            raise ValueError(f"n_hidden must be >= 1, got {n_hidden}")

        self._n_hidden: int = n_hidden
        self._convergence_threshold: float = convergence_threshold
        self._max_iterations: int = max_iterations
        self._observation_window: int = HMM_OBSERVATION_WINDOW
        self._lock = threading.Lock()

        # Observation vocabulary
        symbols = observation_symbols if observation_symbols is not None else list(DEFAULT_OBSERVATION_SYMBOLS)
        if not symbols:
            raise ValueError("observation_symbols must be a non-empty list")
        self._observation_symbols: list[str] = symbols
        self._symbol_to_index: dict[str, int] = {s: i for i, s in enumerate(symbols)}
        self._n_obs: int = len(symbols)

        # HMM parameters: lambda = (pi, A, B)
        # pi — uniform initial state distribution
        self._pi: np.ndarray = np.full(n_hidden, 1.0 / n_hidden, dtype=np.float64)

        # A — transition matrix with informed priors
        if n_hidden == HMM_NUM_HIDDEN_STATES:
            self._A: np.ndarray = _build_informed_transition(n_hidden)
        else:
            # Custom state count: slightly sticky uniform
            A = np.full((n_hidden, n_hidden), 1.0 / n_hidden, dtype=np.float64)
            np.fill_diagonal(A, 0.5)
            A = A / A.sum(axis=1, keepdims=True)
            self._A = A

        # B — emission matrix with informed priors
        if n_hidden == HMM_NUM_HIDDEN_STATES and self._n_obs == len(DEFAULT_OBSERVATION_SYMBOLS):
            self._B: np.ndarray = _build_informed_emission(n_hidden, self._n_obs)
        else:
            # Custom configuration: uniform emission
            self._B = np.full((n_hidden, self._n_obs), 1.0 / self._n_obs, dtype=np.float64)

        # Running state
        self._current_state_dist: np.ndarray = self._pi.copy()
        self._observation_history: list[int] = []
        self._log_likelihood_accum: float = 0.0

        logger.debug(
            "HMMEngine initialized: %d hidden states, %d observation symbols",
            n_hidden,
            self._n_obs,
        )

    # ───────────────────────────────────────────────────────────────────────
    # Properties
    # ───────────────────────────────────────────────────────────────────────

    @property
    def current_state(self) -> HMMState:
        """Return the MAP (maximum a posteriori) state estimate."""
        idx = int(np.argmax(self._current_state_dist))
        if idx < len(_STATE_LIST):
            return _STATE_LIST[idx]
        return _STATE_LIST[0]

    @property
    def transition_matrix(self) -> np.ndarray:
        """Return a copy of the transition matrix A."""
        return self._A.copy()

    @property
    def emission_matrix(self) -> np.ndarray:
        """Return a copy of the emission matrix B."""
        return self._B.copy()

    # ───────────────────────────────────────────────────────────────────────
    # Core Methods
    # ───────────────────────────────────────────────────────────────────────

    def observe(self, symbol: str) -> PredictionResult:
        """Ingest a single observation and update the belief state.

        Performs a single forward step:
            alpha_t = B[:, obs] * (A^T @ alpha_{t-1})
            alpha_t = alpha_t / sum(alpha_t)    [normalization]

        Then predicts next state via:
            next_dist = A^T @ alpha_t

        Args:
            symbol: An observation symbol (must be in the vocabulary).

        Returns:
            PredictionResult with current state estimate, next-state
            prediction, confidence, and cumulative log-likelihood.

        Raises:
            ValueError: If symbol is not in the observation vocabulary.
        """
        obs_idx = self._symbol_to_index.get(symbol)
        if obs_idx is None:
            raise ValueError(
                f"Unknown observation symbol: {symbol!r}. "
                f"Valid symbols: {self._observation_symbols}"
            )

        with self._lock:
            # Append to bounded history
            self._observation_history.append(obs_idx)
            if len(self._observation_history) > self._observation_window:
                self._observation_history = self._observation_history[-self._observation_window :]

            # Forward step: alpha_t = B[:, obs] * (A^T @ alpha_{t-1})
            prediction = self._A.T @ self._current_state_dist
            alpha = self._B[:, obs_idx] * prediction
            alpha_sum = alpha.sum()

            if alpha_sum > 0:
                self._log_likelihood_accum += np.log(alpha_sum)
                alpha /= alpha_sum
            else:
                # Degenerate case: reset to uniform
                logger.warning("Forward step yielded zero mass; resetting to uniform.")
                alpha = np.full(self._n_hidden, 1.0 / self._n_hidden, dtype=np.float64)

            self._current_state_dist = alpha

            # Build result
            return self._build_result()

    def predict_next(self) -> PredictionResult:
        """One-step-ahead prediction without consuming an observation.

        Computes:
            next_dist = A^T @ current_state_dist

        Returns:
            PredictionResult reflecting the predicted next state.
        """
        with self._lock:
            return self._build_result()

    def decode(self, observations: list[str]) -> list[HMMState]:
        """Viterbi decoding — most likely hidden state sequence.

        Uses log-space arithmetic for numerical stability on long
        sequences.

        Args:
            observations: Sequence of observation symbols.

        Returns:
            List of HMMState enums representing the most likely state
            sequence.

        Raises:
            ValueError: If any symbol is not in the vocabulary.
            ValueError: If observations is empty.
        """
        if not observations:
            raise ValueError("observations must be a non-empty list")

        obs_indices = self._resolve_observation_indices(observations)
        n_obs = len(obs_indices)
        n = self._n_hidden

        # Log-space parameters
        log_pi = _safe_log(self._pi)
        log_A = _safe_log(self._A)
        log_B = _safe_log(self._B)

        # Viterbi tables
        viterbi = np.full((n_obs, n), _LOG_ZERO, dtype=np.float64)
        backptr = np.zeros((n_obs, n), dtype=np.int64)

        # Initialization: t=0
        viterbi[0, :] = log_pi + log_B[:, obs_indices[0]]

        # Recursion
        for t in range(1, n_obs):
            for j in range(n):
                # candidates[i] = viterbi[t-1, i] + log_A[i, j]
                candidates = viterbi[t - 1, :] + log_A[:, j]
                best_i = int(np.argmax(candidates))
                viterbi[t, j] = candidates[best_i] + log_B[j, obs_indices[t]]
                backptr[t, j] = best_i

        # Termination: backtrack
        path = [0] * n_obs
        path[-1] = int(np.argmax(viterbi[-1, :]))

        for t in range(n_obs - 2, -1, -1):
            path[t] = int(backptr[t + 1, path[t + 1]])

        return [_STATE_LIST[i] if i < len(_STATE_LIST) else _STATE_LIST[0] for i in path]

    def likelihood(self, observations: list[str]) -> float:
        """Forward algorithm — log-likelihood P(observations | model).

        Uses log-space with log-sum-exp for numerical stability.

        Args:
            observations: Sequence of observation symbols.

        Returns:
            Log-likelihood (float, always <= 0).

        Raises:
            ValueError: If any symbol is not in the vocabulary.
            ValueError: If observations is empty.
        """
        if not observations:
            raise ValueError("observations must be a non-empty list")

        obs_indices = self._resolve_observation_indices(observations)
        n = self._n_hidden

        # Log-space parameters
        log_pi = _safe_log(self._pi)
        log_A = _safe_log(self._A)
        log_B = _safe_log(self._B)

        # Initialization: log_alpha[i] = log(pi[i]) + log(B[i, o_0])
        log_alpha = log_pi + log_B[:, obs_indices[0]]

        # Induction
        for t in range(1, len(obs_indices)):
            log_alpha_new = np.full(n, _LOG_ZERO, dtype=np.float64)
            for j in range(n):
                # log_alpha_new[j] = log( sum_i alpha[i] * A[i,j] ) + log B[j, o_t]
                terms = log_alpha + log_A[:, j]
                log_alpha_new[j] = _log_sum_exp(terms) + log_B[j, obs_indices[t]]
            log_alpha = log_alpha_new

        # Termination: log P(O | model) = log sum_i alpha_T[i]
        return _log_sum_exp(log_alpha)

    def learn(self, observations: list[str]) -> None:
        """Baum-Welch (EM) parameter re-estimation.

        Deferred to Phase 47 — Cognitive Resonance Training.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("Baum-Welch training deferred to Phase 47")

    # ───────────────────────────────────────────────────────────────────────
    # Serialization
    # ───────────────────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize all HMM parameters to a JSON-safe dictionary.

        numpy arrays are converted to nested Python lists.
        HMMState values are serialized as strings.

        Returns:
            Dictionary suitable for ``json.dumps()``.
        """
        with self._lock:
            return {
                "n_hidden": self._n_hidden,
                "n_obs": self._n_obs,
                "observation_symbols": list(self._observation_symbols),
                "pi": self._pi.tolist(),
                "A": self._A.tolist(),
                "B": self._B.tolist(),
                "current_state_dist": self._current_state_dist.tolist(),
                "observation_history": list(self._observation_history),
                "log_likelihood_accum": self._log_likelihood_accum,
                "convergence_threshold": self._convergence_threshold,
                "max_iterations": self._max_iterations,
                "observation_window": self._observation_window,
            }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HMMEngine:
        """Reconstruct an HMMEngine from a serialized dictionary.

        Args:
            data: Dictionary produced by ``to_dict()``.

        Returns:
            Fully reconstructed HMMEngine with restored parameters
            and observation history.
        """
        engine = cls(
            n_hidden=data["n_hidden"],
            observation_symbols=data["observation_symbols"],
            convergence_threshold=data.get("convergence_threshold", HMM_CONVERGENCE_THRESHOLD),
            max_iterations=data.get("max_iterations", HMM_MAX_EM_ITERATIONS),
        )

        # Restore learned / accumulated parameters
        engine._pi = np.array(data["pi"], dtype=np.float64)
        engine._A = np.array(data["A"], dtype=np.float64)
        engine._B = np.array(data["B"], dtype=np.float64)
        engine._current_state_dist = np.array(data["current_state_dist"], dtype=np.float64)
        engine._observation_history = list(data.get("observation_history", []))
        engine._log_likelihood_accum = float(data.get("log_likelihood_accum", 0.0))
        engine._observation_window = int(data.get("observation_window", HMM_OBSERVATION_WINDOW))

        return engine

    # ───────────────────────────────────────────────────────────────────────
    # Internal Helpers
    # ───────────────────────────────────────────────────────────────────────

    def _resolve_observation_indices(self, observations: list[str]) -> list[int]:
        """Map a list of observation symbols to their integer indices.

        Raises:
            ValueError: If any symbol is not in the vocabulary.
        """
        indices: list[int] = []
        for sym in observations:
            idx = self._symbol_to_index.get(sym)
            if idx is None:
                raise ValueError(
                    f"Unknown observation symbol: {sym!r}. "
                    f"Valid symbols: {self._observation_symbols}"
                )
            indices.append(idx)
        return indices

    def _build_result(self) -> PredictionResult:
        """Build a PredictionResult from the current internal state.

        Must be called while holding self._lock.
        """
        dist = self._current_state_dist

        # Current MAP state
        current_idx = int(np.argmax(dist))
        current_state = _STATE_LIST[current_idx] if current_idx < len(_STATE_LIST) else _STATE_LIST[0]

        # State probabilities (string keys for JSON serializability)
        state_probs: dict[str, float] = {}
        for i, name in enumerate(_STATE_NAMES):
            if i < len(dist):
                state_probs[name] = float(dist[i])
            else:
                state_probs[name] = 0.0

        # One-step prediction: next_dist = A^T @ current_dist
        next_dist = self._A.T @ dist
        next_sum = next_dist.sum()
        if next_sum > 0:
            next_dist = next_dist / next_sum

        next_idx = int(np.argmax(next_dist))
        predicted_next = _STATE_LIST[next_idx] if next_idx < len(_STATE_LIST) else _STATE_LIST[0]
        confidence = float(next_dist[next_idx])

        return PredictionResult(
            most_likely_state=current_state,
            state_probabilities=state_probs,
            predicted_next_state=predicted_next,
            prediction_confidence=confidence,
            observation_likelihood=self._log_likelihood_accum,
        )
