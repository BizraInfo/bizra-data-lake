"""SNR Dual Verification — Amended Definition 1.5.

Extends the SNR protocol with dual-stage verification:
  V_gate(pi) — local gate verification (product of 5 constitutional gates)
  V_pool(pi) — network pool consensus verification

Amended SNR formula:
  SNR(N) = DeltaK * V_gate(pi) * V_pool(pi) / (H(S|O) + Sigma_tension + R_halluc)

Standing on Giants: Shannon (1948) | Meyer (DbC, 1986) | Phase 60 Step 4 (assert_snr_normalized)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from core.integration.constants import UNIFIED_SNR_THRESHOLD  # noqa: F401

# The 5 constitutional gates from the proof engine pipeline.
CONSTITUTIONAL_GATES: tuple[str, ...] = (
    "alpha4_fallback",
    "alpha7_verification",
    "alpha8_dark_matter",
    "alpha9_attestation",
    "alpha10_binary",
)


@dataclass(frozen=True)
class DualVerificationScore:
    """Combined verification score: V_gate * V_pool.

    Properties:
      P1: combined in [0, 1]  (product of [0,1] values)
      P2: gate_score = 0 => combined = 0  (single failed gate kills SNR)
      P3: pool_score = 0 => combined = 0  (consensus failure kills SNR)
    """

    gate_score: float  # V_gate: product of 5 constitutional gate scores
    pool_score: float  # V_pool: consensus approval ratio

    def __post_init__(self) -> None:
        if not (0.0 <= self.gate_score <= 1.0):
            raise ValueError(
                f"gate_score={self.gate_score} outside normalized range [0, 1]"
            )
        if not (0.0 <= self.pool_score <= 1.0):
            raise ValueError(
                f"pool_score={self.pool_score} outside normalized range [0, 1]"
            )

    @property
    def combined(self) -> float:
        """V_gate * V_pool — the dual verification score."""
        return self.gate_score * self.pool_score

    @property
    def is_valid(self) -> bool:
        """Both scores must be in [0, 1] for the score to be valid."""
        return 0.0 <= self.gate_score <= 1.0 and 0.0 <= self.pool_score <= 1.0

    @classmethod
    def from_gate_scores(
        cls,
        gate_scores: dict[str, float],
    ) -> DualVerificationScore:
        """Construct from individual gate scores.

        Computes V_gate as the product of 5 constitutional gate values.
        Missing gates default to 1.0 (identity — gate is assumed passing).

        Args:
            gate_scores: Mapping of gate name to score in [0, 1].
                Example: {"alpha4_fallback": 0.98, "alpha7_verification": 0.95, ...}

        Returns:
            DualVerificationScore with computed gate_score and pool_score=0.0.
        """
        v_gate = 1.0
        for gate_name in CONSTITUTIONAL_GATES:
            score = gate_scores.get(gate_name, 1.0)
            v_gate *= score

        # Clamp to [0, 1] — product of [0,1] values is in [0,1] by construction,
        # but floating-point arithmetic can drift.
        v_gate = max(0.0, min(1.0, v_gate))

        return cls(gate_score=v_gate, pool_score=0.0)

    @classmethod
    def from_pool_votes(
        cls,
        honest: int,
        total: int,
        gate_score: float,
    ) -> DualVerificationScore:
        """Construct from pool vote counts and a pre-computed gate score.

        Args:
            honest: Number of honest/approving votes.
            total: Total number of validators in the quorum.
            gate_score: Pre-computed V_gate score in [0, 1].

        Returns:
            DualVerificationScore with the given gate_score and computed pool_score.
        """
        pool_score = honest / total if total > 0 else 0.0
        return cls(gate_score=gate_score, pool_score=pool_score)


def compute_snr_dual(
    knowledge_gain: float,
    gate_score: float,
    pool_score: float,
    conditional_entropy: float,
    stress_tension: float,
    hallucination_rate: float,
) -> dict[str, object]:
    """Compute SNR with dual verification (Amended Definition 1.5).

    Formula:
        numerator   = knowledge_gain * gate_score * pool_score
        denominator = max(conditional_entropy + stress_tension + hallucination_rate, 1e-10)
        snr_raw     = numerator / denominator
        snr_normalized = 1 / (1 + exp(-snr_raw))   (logistic sigmoid)

    Args:
        knowledge_gain: DeltaK — knowledge gain from the mission.
        gate_score: V_gate — local gate verification score in [0, 1].
        pool_score: V_pool — pool consensus verification score in [0, 1].
        conditional_entropy: H(S|O) — remaining uncertainty.
        stress_tension: Sigma_tension — epistemic tension.
        hallucination_rate: R_halluc — hallucination residual.

    Returns:
        Dict with keys: snr_raw, snr_normalized, verification_dual,
        knowledge_gain, conditional_entropy, stress_tension, hallucination_rate.
    """
    numerator = knowledge_gain * gate_score * pool_score
    denominator = max(
        conditional_entropy + stress_tension + hallucination_rate,
        1e-10,
    )
    snr_raw = numerator / denominator

    # Logistic sigmoid normalization: maps (-inf, +inf) -> (0, 1)
    snr_normalized = 1.0 / (1.0 + math.exp(-snr_raw))

    verification_dual = DualVerificationScore(
        gate_score=gate_score,
        pool_score=pool_score,
    )

    return {
        "snr_raw": snr_raw,
        "snr_normalized": snr_normalized,
        "verification_dual": verification_dual,
        "knowledge_gain": knowledge_gain,
        "conditional_entropy": conditional_entropy,
        "stress_tension": stress_tension,
        "hallucination_rate": hallucination_rate,
    }
