"""Tests for SNR Dual Verification — Phase 61 Step 4.

10 TDD anchors covering DualVerificationScore and compute_snr_dual.

Standing on Giants: Shannon (1948) | Meyer (DbC, 1986)
"""

from __future__ import annotations

import math

import pytest

from core.iaas.snr_dual_verification import (
    CONSTITUTIONAL_GATES,
    DualVerificationScore,
    compute_snr_dual,
)


# ---------------------------------------------------------------------------
# DualVerificationScore tests
# ---------------------------------------------------------------------------


class TestDualVerificationScore:
    """Tests for the DualVerificationScore frozen dataclass."""

    def test_dual_score_is_product(self) -> None:
        """Combined = gate * pool."""
        dv = DualVerificationScore(gate_score=0.9, pool_score=0.8)
        assert abs(dv.combined - 0.72) < 1e-6

    def test_dual_score_bounded_0_1(self) -> None:
        """P1: Combined always in [0, 1] for all valid inputs."""
        for g in [0.0, 0.25, 0.5, 0.75, 1.0]:
            for p in [0.0, 0.25, 0.5, 0.75, 1.0]:
                dv = DualVerificationScore(gate_score=g, pool_score=p)
                assert 0.0 <= dv.combined <= 1.0, (
                    f"combined={dv.combined} out of range for g={g}, p={p}"
                )

    def test_zero_gate_zero_combined(self) -> None:
        """P2: Failed gate zeroes everything."""
        dv = DualVerificationScore(gate_score=0.0, pool_score=1.0)
        assert dv.combined == 0.0

    def test_zero_pool_zero_combined(self) -> None:
        """P3: No consensus zeroes everything."""
        dv = DualVerificationScore(gate_score=1.0, pool_score=0.0)
        assert dv.combined == 0.0

    def test_perfect_scores_equal_1(self) -> None:
        """Perfect gate and pool scores produce combined = 1.0."""
        dv = DualVerificationScore(gate_score=1.0, pool_score=1.0)
        assert dv.combined == 1.0

    def test_rejects_gate_out_of_range(self) -> None:
        """Gate score outside [0, 1] raises ValueError."""
        with pytest.raises(ValueError, match="outside normalized range"):
            DualVerificationScore(gate_score=1.5, pool_score=0.5)

    def test_rejects_pool_out_of_range(self) -> None:
        """Pool score outside [0, 1] raises ValueError."""
        with pytest.raises(ValueError, match="outside normalized range"):
            DualVerificationScore(gate_score=0.5, pool_score=-0.1)

    def test_is_valid_for_valid_scores(self) -> None:
        """is_valid returns True when both scores are in [0, 1]."""
        dv = DualVerificationScore(gate_score=0.5, pool_score=0.5)
        assert dv.is_valid is True

    def test_frozen(self) -> None:
        """Dataclass is frozen — attributes cannot be mutated."""
        dv = DualVerificationScore(gate_score=0.5, pool_score=0.5)
        with pytest.raises(AttributeError):
            dv.gate_score = 0.9  # type: ignore[misc]


class TestFromGateScores:
    """Tests for DualVerificationScore.from_gate_scores classmethod."""

    def test_from_gate_scores_product(self) -> None:
        """Product of 5 gates at 0.9 each -> 0.9^5."""
        gates = {g: 0.9 for g in CONSTITUTIONAL_GATES}
        dv = DualVerificationScore.from_gate_scores(gates)
        expected = 0.9**5
        assert abs(dv.gate_score - expected) < 1e-6

    def test_from_gate_scores_missing_defaults_to_1(self) -> None:
        """Missing gates default to 1.0 (identity for multiplication)."""
        gates = {"alpha4_fallback": 0.8}
        dv = DualVerificationScore.from_gate_scores(gates)
        # Only alpha4_fallback contributes; others are 1.0
        assert abs(dv.gate_score - 0.8) < 1e-6

    def test_from_gate_scores_zero_gate_zeros_product(self) -> None:
        """A single zero gate zeroes the entire product."""
        gates = {g: 1.0 for g in CONSTITUTIONAL_GATES}
        gates["alpha7_verification"] = 0.0
        dv = DualVerificationScore.from_gate_scores(gates)
        assert dv.gate_score == 0.0

    def test_from_gate_scores_pool_score_is_zero(self) -> None:
        """from_gate_scores sets pool_score to 0.0 (no pool info)."""
        gates = {g: 1.0 for g in CONSTITUTIONAL_GATES}
        dv = DualVerificationScore.from_gate_scores(gates)
        assert dv.pool_score == 0.0


class TestFromPoolVotes:
    """Tests for DualVerificationScore.from_pool_votes classmethod."""

    def test_from_pool_votes_ratio(self) -> None:
        """pool_score = honest / total."""
        dv = DualVerificationScore.from_pool_votes(
            honest=8, total=10, gate_score=0.9
        )
        assert abs(dv.pool_score - 0.8) < 1e-6
        assert abs(dv.gate_score - 0.9) < 1e-6

    def test_from_pool_votes_zero_total(self) -> None:
        """Zero total validators yields pool_score = 0.0."""
        dv = DualVerificationScore.from_pool_votes(
            honest=0, total=0, gate_score=1.0
        )
        assert dv.pool_score == 0.0

    def test_from_pool_votes_unanimous(self) -> None:
        """Unanimous approval yields pool_score = 1.0."""
        dv = DualVerificationScore.from_pool_votes(
            honest=5, total=5, gate_score=0.95
        )
        assert dv.pool_score == 1.0


# ---------------------------------------------------------------------------
# compute_snr_dual tests
# ---------------------------------------------------------------------------


class TestComputeSNRDual:
    """Tests for the compute_snr_dual function."""

    def test_snr_dual_positive_for_good_inputs(self) -> None:
        """Positive knowledge gain with passing verification produces positive SNR."""
        result = compute_snr_dual(
            knowledge_gain=10.0,
            gate_score=0.9,
            pool_score=0.8,
            conditional_entropy=2.0,
            stress_tension=1.0,
            hallucination_rate=0.5,
        )
        assert result["snr_raw"] > 0.0
        assert result["snr_normalized"] > 0.5  # sigmoid(positive) > 0.5

    def test_snr_dual_zero_when_gate_zero(self) -> None:
        """Zero gate score produces snr_raw = 0."""
        result = compute_snr_dual(
            knowledge_gain=10.0,
            gate_score=0.0,
            pool_score=0.8,
            conditional_entropy=2.0,
            stress_tension=1.0,
            hallucination_rate=0.5,
        )
        assert result["snr_raw"] == 0.0
        # sigmoid(0) = 0.5
        assert abs(result["snr_normalized"] - 0.5) < 1e-6

    def test_snr_normalized_in_0_1(self) -> None:
        """snr_normalized is always in (0, 1) for any finite inputs."""
        test_cases = [
            # (knowledge_gain, gate, pool, entropy, tension, halluc)
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (5.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            (0.001, 0.5, 0.5, 10.0, 10.0, 10.0),
            (50.0, 0.95, 0.9, 1.0, 0.5, 0.3),
        ]
        for kg, gs, ps, ce, st, hr in test_cases:
            result = compute_snr_dual(
                knowledge_gain=kg,
                gate_score=gs,
                pool_score=ps,
                conditional_entropy=ce,
                stress_tension=st,
                hallucination_rate=hr,
            )
            snr_norm = result["snr_normalized"]
            assert 0.0 < snr_norm < 1.0, (
                f"snr_normalized={snr_norm} not in (0,1) for inputs "
                f"kg={kg}, gs={gs}, ps={ps}, ce={ce}, st={st}, hr={hr}"
            )

    def test_snr_dual_returns_dict_with_all_keys(self) -> None:
        """Return dict contains all required keys."""
        result = compute_snr_dual(
            knowledge_gain=5.0,
            gate_score=0.9,
            pool_score=0.8,
            conditional_entropy=2.0,
            stress_tension=1.0,
            hallucination_rate=0.5,
        )
        expected_keys = {
            "snr_raw",
            "snr_normalized",
            "verification_dual",
            "knowledge_gain",
            "conditional_entropy",
            "stress_tension",
            "hallucination_rate",
        }
        assert set(result.keys()) == expected_keys

    def test_snr_dual_verification_dual_is_score(self) -> None:
        """verification_dual field is a DualVerificationScore instance."""
        result = compute_snr_dual(
            knowledge_gain=5.0,
            gate_score=0.9,
            pool_score=0.8,
            conditional_entropy=2.0,
            stress_tension=1.0,
            hallucination_rate=0.5,
        )
        dual = result["verification_dual"]
        assert isinstance(dual, DualVerificationScore)
        assert abs(dual.gate_score - 0.9) < 1e-6
        assert abs(dual.pool_score - 0.8) < 1e-6

    def test_snr_dual_formula_correctness(self) -> None:
        """Verify the raw SNR formula: kg * gs * ps / max(ce + st + hr, 1e-10)."""
        result = compute_snr_dual(
            knowledge_gain=10.0,
            gate_score=0.9,
            pool_score=0.8,
            conditional_entropy=2.0,
            stress_tension=1.0,
            hallucination_rate=0.5,
        )
        expected_num = 10.0 * 0.9 * 0.8  # 7.2
        expected_den = 2.0 + 1.0 + 0.5  # 3.5
        expected_raw = expected_num / expected_den
        assert abs(result["snr_raw"] - expected_raw) < 1e-9

        expected_norm = 1.0 / (1.0 + math.exp(-expected_raw))
        assert abs(result["snr_normalized"] - expected_norm) < 1e-9

    def test_snr_dual_zero_noise_does_not_divide_by_zero(self) -> None:
        """Zero noise uses epsilon floor (1e-10) to prevent division by zero."""
        result = compute_snr_dual(
            knowledge_gain=5.0,
            gate_score=1.0,
            pool_score=1.0,
            conditional_entropy=0.0,
            stress_tension=0.0,
            hallucination_rate=0.0,
        )
        # Very large raw SNR, normalized should approach 1.0
        assert result["snr_raw"] > 1e6
        assert result["snr_normalized"] > 0.99

    def test_snr_dual_zero_knowledge_gain(self) -> None:
        """Zero knowledge gain produces snr_raw = 0 regardless of verification."""
        result = compute_snr_dual(
            knowledge_gain=0.0,
            gate_score=1.0,
            pool_score=1.0,
            conditional_entropy=1.0,
            stress_tension=0.5,
            hallucination_rate=0.5,
        )
        assert result["snr_raw"] == 0.0
        assert abs(result["snr_normalized"] - 0.5) < 1e-6
