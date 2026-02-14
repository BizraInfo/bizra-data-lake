"""
Unit Tests for IhsanVector - Constitutional 8-Dimension Excellence Enforcement

Tests cover:
1. Canonical constants integrity (weights, methods, thresholds)
2. Enum correctness (ExecutionContext, DimensionId)
3. IhsanDimension validation, scoring, verification, serialization
4. IhsanVector creation, scoring, threshold verification, immutability
5. ThresholdResult structure and serialization
6. IhsanReceipt integrity verification and serialization
7. Convenience functions (create_verifier, quick_ihsan, passes_production)
8. Serialization roundtrips and JSON stability
9. Immutability guarantees across all mutation operations

Standing on Giants: Al-Ghazali, Shannon, de Moura
"""

import hashlib
import json
import logging
import math
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

import pytest

from core.sovereign.ihsan_vector import (
    ANTI_CENTRALIZATION_GINI_THRESHOLD,
    CANONICAL_WEIGHTS,
    CONTEXT_THRESHOLDS,
    VERIFY_METHODS,
    DimensionId,
    ExecutionContext,
    IhsanDimension,
    IhsanReceipt,
    IhsanVector,
    ThresholdResult,
    create_verifier,
    passes_production,
    quick_ihsan,
)


# =============================================================================
# CANONICAL CONSTANTS
# =============================================================================


class TestCanonicalConstants:
    """Tests for module-level canonical constants."""

    def test_canonical_weights_sum_to_one(self) -> None:
        """CANONICAL_WEIGHTS must sum exactly to 1.0 (within floating-point epsilon)."""
        total = sum(CANONICAL_WEIGHTS.values())
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_canonical_weights_has_eight_dimensions(self) -> None:
        """There must be exactly 8 dimension weights."""
        assert len(CANONICAL_WEIGHTS) == 8

    def test_verify_methods_has_eight_dimensions(self) -> None:
        """There must be exactly 8 verification methods."""
        assert len(VERIFY_METHODS) == 8

    def test_canonical_weights_keys_match_verify_methods_keys(self) -> None:
        """Weight keys and verify method keys must be identical sets."""
        assert set(CANONICAL_WEIGHTS.keys()) == set(VERIFY_METHODS.keys())

    def test_gini_threshold_is_035(self) -> None:
        """Anti-centralization Gini threshold must be 0.35."""
        assert ANTI_CENTRALIZATION_GINI_THRESHOLD == 0.35

    def test_context_thresholds_has_all_four_contexts(self) -> None:
        """Context thresholds must exist for all 4 execution contexts."""
        assert set(CONTEXT_THRESHOLDS.keys()) == {
            ExecutionContext.DEVELOPMENT,
            ExecutionContext.STAGING,
            ExecutionContext.PRODUCTION,
            ExecutionContext.CRITICAL,
        }


# =============================================================================
# EXECUTION CONTEXT ENUM
# =============================================================================


class TestExecutionContext:
    """Tests for the ExecutionContext enum."""

    def test_four_members_exist(self) -> None:
        """ExecutionContext must have exactly 4 members."""
        assert len(ExecutionContext) == 4

    def test_string_values_correct(self) -> None:
        """Each member must have the expected string value."""
        assert ExecutionContext.DEVELOPMENT.value == "development"
        assert ExecutionContext.STAGING.value == "staging"
        assert ExecutionContext.PRODUCTION.value == "production"
        assert ExecutionContext.CRITICAL.value == "critical"

    def test_is_str_subclass(self) -> None:
        """ExecutionContext must be a str subclass (str, Enum)."""
        assert isinstance(ExecutionContext.PRODUCTION, str)

    def test_can_construct_from_value(self) -> None:
        """ExecutionContext can be constructed from its string value."""
        assert ExecutionContext("production") is ExecutionContext.PRODUCTION


# =============================================================================
# DIMENSION ID ENUM
# =============================================================================


class TestDimensionId:
    """Tests for the DimensionId IntEnum."""

    def test_eight_members(self) -> None:
        """DimensionId must have exactly 8 members."""
        assert len(DimensionId) == 8

    def test_correctness_is_zero(self) -> None:
        """CORRECTNESS must be ordinal 0."""
        assert DimensionId.CORRECTNESS == 0

    def test_fairness_is_seven(self) -> None:
        """FAIRNESS must be ordinal 7."""
        assert DimensionId.FAIRNESS == 7

    def test_contiguous_values(self) -> None:
        """All ordinal values 0-7 must be present."""
        values = sorted(d.value for d in DimensionId)
        assert values == list(range(8))

    def test_weight_property_returns_correct_value(self) -> None:
        """weight property must return the canonical weight for each dimension."""
        for dim_id in DimensionId:
            expected = CANONICAL_WEIGHTS[dim_id.name.lower()]
            assert dim_id.weight == expected, f"Weight mismatch for {dim_id.name}"

    def test_verify_method_property_returns_correct_value(self) -> None:
        """verify_method property must return the canonical method for each dimension."""
        for dim_id in DimensionId:
            expected = VERIFY_METHODS[dim_id.name.lower()]
            assert dim_id.verify_method == expected, (
                f"Verify method mismatch for {dim_id.name}"
            )


# =============================================================================
# IHSAN DIMENSION
# =============================================================================


class TestIhsanDimension:
    """Tests for the IhsanDimension dataclass."""

    def test_default_construction(self) -> None:
        """Default construction with required fields only."""
        dim = IhsanDimension(id=DimensionId.CORRECTNESS, weight=0.22)
        assert dim.score == 0.0
        assert dim.verified is False
        assert dim.verification_timestamp is None
        assert dim.verification_proof is None

    def test_score_below_zero_raises(self) -> None:
        """Score < 0 must raise ValueError."""
        with pytest.raises(ValueError, match="Score must be in"):
            IhsanDimension(id=DimensionId.CORRECTNESS, weight=0.22, score=-0.01)

    def test_score_above_one_raises(self) -> None:
        """Score > 1 must raise ValueError."""
        with pytest.raises(ValueError, match="Score must be in"):
            IhsanDimension(id=DimensionId.CORRECTNESS, weight=0.22, score=1.01)

    def test_score_boundary_zero_valid(self) -> None:
        """Score = 0.0 must be valid."""
        dim = IhsanDimension(id=DimensionId.CORRECTNESS, weight=0.22, score=0.0)
        assert dim.score == 0.0

    def test_score_boundary_one_valid(self) -> None:
        """Score = 1.0 must be valid."""
        dim = IhsanDimension(id=DimensionId.CORRECTNESS, weight=0.22, score=1.0)
        assert dim.score == 1.0

    def test_weight_zero_raises(self) -> None:
        """Weight <= 0 must raise ValueError."""
        with pytest.raises(ValueError, match="Weight must be in"):
            IhsanDimension(id=DimensionId.CORRECTNESS, weight=0.0)

    def test_weight_negative_raises(self) -> None:
        """Negative weight must raise ValueError."""
        with pytest.raises(ValueError, match="Weight must be in"):
            IhsanDimension(id=DimensionId.CORRECTNESS, weight=-0.1)

    def test_weight_above_one_raises(self) -> None:
        """Weight > 1 must raise ValueError."""
        with pytest.raises(ValueError, match="Weight must be in"):
            IhsanDimension(id=DimensionId.CORRECTNESS, weight=1.01)

    def test_default_verify_method_from_dimension_id(self) -> None:
        """When verify_method is empty, __post_init__ must set it from DimensionId."""
        dim = IhsanDimension(id=DimensionId.SAFETY, weight=0.22)
        assert dim.verify_method == "aegis_lambda_zero_trust"

    def test_custom_verify_method_preserved(self) -> None:
        """Explicitly provided verify_method must not be overwritten."""
        dim = IhsanDimension(
            id=DimensionId.SAFETY, weight=0.22, verify_method="custom_method"
        )
        assert dim.verify_method == "custom_method"

    def test_weighted_score(self) -> None:
        """weighted_score must equal weight * score."""
        dim = IhsanDimension(id=DimensionId.CORRECTNESS, weight=0.22, score=0.8)
        assert dim.weighted_score == pytest.approx(0.22 * 0.8)

    def test_weighted_score_zero(self) -> None:
        """weighted_score with score=0 must be 0."""
        dim = IhsanDimension(id=DimensionId.CORRECTNESS, weight=0.22, score=0.0)
        assert dim.weighted_score == 0.0

    def test_verified_weighted_score_when_verified(self) -> None:
        """verified_weighted_score must return weight*score when verified=True."""
        dim = IhsanDimension(
            id=DimensionId.CORRECTNESS, weight=0.22, score=0.9, verified=True
        )
        assert dim.verified_weighted_score == pytest.approx(0.22 * 0.9)

    def test_verified_weighted_score_when_unverified(self) -> None:
        """verified_weighted_score must return 0 when verified=False."""
        dim = IhsanDimension(
            id=DimensionId.CORRECTNESS, weight=0.22, score=0.9, verified=False
        )
        assert dim.verified_weighted_score == 0.0

    def test_mark_verified_returns_new_instance(self) -> None:
        """mark_verified must return a NEW IhsanDimension (immutability)."""
        original = IhsanDimension(id=DimensionId.CORRECTNESS, weight=0.22, score=0.9)
        verified = original.mark_verified(proof="test_proof")
        assert verified is not original

    def test_mark_verified_preserves_score_weight_id(self) -> None:
        """mark_verified must preserve score, weight, and id."""
        original = IhsanDimension(id=DimensionId.SAFETY, weight=0.22, score=0.85)
        verified = original.mark_verified()
        assert verified.id == DimensionId.SAFETY
        assert verified.weight == 0.22
        assert verified.score == 0.85

    def test_mark_verified_sets_verified_true(self) -> None:
        """mark_verified must set verified=True."""
        dim = IhsanDimension(id=DimensionId.CORRECTNESS, weight=0.22, score=0.9)
        verified = dim.mark_verified()
        assert verified.verified is True

    def test_mark_verified_with_proof_stores_proof(self) -> None:
        """mark_verified with proof must store the proof string."""
        dim = IhsanDimension(id=DimensionId.CORRECTNESS, weight=0.22, score=0.9)
        verified = dim.mark_verified(proof="z3_smt_passed_2024")
        assert verified.verification_proof == "z3_smt_passed_2024"

    def test_mark_verified_auto_generates_timestamp(self) -> None:
        """mark_verified without timestamp must auto-generate one."""
        dim = IhsanDimension(id=DimensionId.CORRECTNESS, weight=0.22, score=0.9)
        verified = dim.mark_verified()
        assert verified.verification_timestamp is not None
        assert "T" in verified.verification_timestamp  # ISO format

    def test_mark_verified_with_custom_timestamp(self) -> None:
        """mark_verified with custom timestamp must use provided value."""
        dim = IhsanDimension(id=DimensionId.CORRECTNESS, weight=0.22, score=0.9)
        ts = "2024-01-15T12:00:00Z"
        verified = dim.mark_verified(timestamp=ts)
        assert verified.verification_timestamp == ts

    def test_to_dict_has_all_keys(self) -> None:
        """to_dict must include all required keys."""
        dim = IhsanDimension(
            id=DimensionId.CORRECTNESS,
            weight=0.22,
            score=0.95,
            verified=True,
            verify_method="z3_smt_proof",
            verification_timestamp="2024-01-15T12:00:00Z",
            verification_proof="test_proof",
        )
        d = dim.to_dict()
        expected_keys = {
            "id",
            "weight",
            "score",
            "verified",
            "verify_method",
            "verification_timestamp",
            "verification_proof",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_from_dict_roundtrip(self) -> None:
        """to_dict -> from_dict must produce equivalent dimension."""
        original = IhsanDimension(
            id=DimensionId.EFFICIENCY,
            weight=0.12,
            score=0.88,
            verified=True,
            verify_method="green_ai_metrics",
            verification_timestamp="2024-06-01T00:00:00Z",
            verification_proof="efficiency_proof_123",
        )
        reconstructed = IhsanDimension.from_dict(original.to_dict())
        assert reconstructed.id == original.id
        assert reconstructed.weight == original.weight
        assert reconstructed.score == pytest.approx(original.score, abs=1e-6)
        assert reconstructed.verified == original.verified
        assert reconstructed.verify_method == original.verify_method
        assert reconstructed.verification_timestamp == original.verification_timestamp
        assert reconstructed.verification_proof == original.verification_proof

    def test_from_dict_with_defaults_for_missing_keys(self) -> None:
        """from_dict with minimal data must fill defaults from DimensionId."""
        data = {"id": "correctness"}
        dim = IhsanDimension.from_dict(data)
        assert dim.id == DimensionId.CORRECTNESS
        assert dim.weight == DimensionId.CORRECTNESS.weight
        assert dim.score == 0.0
        assert dim.verified is False
        assert dim.verify_method == DimensionId.CORRECTNESS.verify_method


# =============================================================================
# IHSAN VECTOR
# =============================================================================


class TestIhsanVector:
    """Tests for the IhsanVector dataclass."""

    def test_default_construction_creates_all_eight_dimensions(self) -> None:
        """Default IhsanVector must have all 8 dimensions."""
        vec = IhsanVector()
        assert len(vec.dimensions) == 8
        for dim_id in DimensionId:
            assert dim_id in vec.dimensions

    def test_default_construction_scores_zero(self) -> None:
        """Default construction must initialize all scores to 0."""
        vec = IhsanVector()
        for dim in vec.dimensions.values():
            assert dim.score == 0.0

    def test_from_scores_creates_correct_scores(self) -> None:
        """from_scores must assign the provided scores to correct dimensions."""
        vec = IhsanVector.from_scores(
            correctness=0.98,
            safety=0.97,
            user_benefit=0.92,
            efficiency=0.88,
            auditability=0.90,
            anti_centralization=0.85,
            robustness=0.82,
            fairness=0.80,
        )
        assert vec.dimensions[DimensionId.CORRECTNESS].score == 0.98
        assert vec.dimensions[DimensionId.SAFETY].score == 0.97
        assert vec.dimensions[DimensionId.USER_BENEFIT].score == 0.92
        assert vec.dimensions[DimensionId.EFFICIENCY].score == 0.88
        assert vec.dimensions[DimensionId.AUDITABILITY].score == 0.90
        assert vec.dimensions[DimensionId.ANTI_CENTRALIZATION].score == 0.85
        assert vec.dimensions[DimensionId.ROBUSTNESS].score == 0.82
        assert vec.dimensions[DimensionId.FAIRNESS].score == 0.80

    def test_from_scores_with_context(self) -> None:
        """from_scores with context must store the context."""
        vec = IhsanVector.from_scores(
            correctness=0.9,
            safety=0.9,
            user_benefit=0.9,
            efficiency=0.9,
            auditability=0.9,
            anti_centralization=0.9,
            robustness=0.9,
            fairness=0.9,
            context=ExecutionContext.PRODUCTION,
        )
        assert vec.context is ExecutionContext.PRODUCTION

    def test_neutral_all_05(self) -> None:
        """neutral() must create vector with all scores at 0.5."""
        vec = IhsanVector.neutral()
        for dim in vec.dimensions.values():
            assert dim.score == 0.5

    def test_neutral_score_is_05(self) -> None:
        """neutral() vector score must equal 0.5."""
        vec = IhsanVector.neutral()
        assert vec.calculate_score() == pytest.approx(0.5, abs=1e-9)

    def test_perfect_all_10(self) -> None:
        """perfect() must create vector with all scores at 1.0."""
        vec = IhsanVector.perfect()
        for dim in vec.dimensions.values():
            assert dim.score == 1.0

    def test_perfect_score_is_10(self) -> None:
        """perfect() vector score must equal 1.0."""
        vec = IhsanVector.perfect()
        assert vec.calculate_score() == pytest.approx(1.0, abs=1e-9)

    def test_calculate_score_with_known_values(self) -> None:
        """calculate_score must match hand-computed weighted sum."""
        vec = IhsanVector.from_scores(
            correctness=1.0,
            safety=1.0,
            user_benefit=1.0,
            efficiency=1.0,
            auditability=1.0,
            anti_centralization=0.0,
            robustness=0.0,
            fairness=0.0,
        )
        # Expected: 0.22 + 0.22 + 0.14 + 0.12 + 0.12 + 0 + 0 + 0 = 0.82
        assert vec.calculate_score() == pytest.approx(0.82, abs=1e-9)

    def test_calculate_score_for_neutral(self) -> None:
        """calculate_score for neutral vector must be 0.5."""
        vec = IhsanVector.neutral()
        assert vec.calculate_score() == pytest.approx(0.5, abs=1e-9)

    def test_calculate_score_for_perfect(self) -> None:
        """calculate_score for perfect vector must be 1.0."""
        vec = IhsanVector.perfect()
        assert vec.calculate_score() == pytest.approx(1.0, abs=1e-9)

    def test_calculate_score_for_zeros(self) -> None:
        """calculate_score for all-zero vector must be 0.0."""
        vec = IhsanVector()
        assert vec.calculate_score() == pytest.approx(0.0, abs=1e-9)

    def test_calculate_verified_score_all_unverified(self) -> None:
        """calculate_verified_score with no verified dims must be 0.0."""
        vec = IhsanVector.perfect()
        assert vec.calculate_verified_score() == pytest.approx(0.0, abs=1e-9)

    def test_calculate_verified_score_after_verify_dimension(self) -> None:
        """calculate_verified_score must include only verified dimension contributions."""
        vec = IhsanVector.perfect()
        vec2 = vec.verify_dimension(DimensionId.CORRECTNESS, proof="ok")
        # Only correctness verified: 0.22 * 1.0 = 0.22
        assert vec2.calculate_verified_score() == pytest.approx(0.22, abs=1e-9)

    def test_calculate_verified_score_after_verify_all_equals_score(self) -> None:
        """calculate_verified_score after verify_all must equal calculate_score."""
        vec = IhsanVector.perfect()
        vec_verified = vec.verify_all()
        assert vec_verified.calculate_verified_score() == pytest.approx(
            vec_verified.calculate_score(), abs=1e-9
        )

    def test_verified_count_default_zero(self) -> None:
        """Default verified_count must be 0."""
        vec = IhsanVector()
        assert vec.verified_count == 0

    def test_verified_count_after_verify_dimension(self) -> None:
        """verified_count after one verify_dimension must be 1."""
        vec = IhsanVector.perfect()
        vec2 = vec.verify_dimension(DimensionId.SAFETY)
        assert vec2.verified_count == 1

    def test_unverified_count_default_eight(self) -> None:
        """Default unverified_count must be 8."""
        vec = IhsanVector()
        assert vec.unverified_count == 8

    def test_all_verified_default_false(self) -> None:
        """Default all_verified must be False."""
        vec = IhsanVector()
        assert vec.all_verified is False

    def test_all_verified_after_verify_all_true(self) -> None:
        """all_verified after verify_all must be True."""
        vec = IhsanVector.perfect()
        vec_verified = vec.verify_all()
        assert vec_verified.all_verified is True

    def test_verify_thresholds_development_passes(self) -> None:
        """Development: score >= 0.85 with 4+ verified dims must pass."""
        vec = IhsanVector.from_scores(
            correctness=0.95,
            safety=0.95,
            user_benefit=0.90,
            efficiency=0.85,
            auditability=0.80,
            anti_centralization=0.70,
            robustness=0.65,
            fairness=0.60,
        )
        # Verify 4 dimensions
        vec = vec.verify_dimension(DimensionId.CORRECTNESS)
        vec = vec.verify_dimension(DimensionId.SAFETY)
        vec = vec.verify_dimension(DimensionId.USER_BENEFIT)
        vec = vec.verify_dimension(DimensionId.EFFICIENCY)
        result = vec.verify_thresholds(ExecutionContext.DEVELOPMENT)
        assert result.passed is True

    def test_verify_thresholds_development_fails_score(self) -> None:
        """Development: score < 0.85 must fail even with enough verified dims."""
        vec = IhsanVector.from_scores(
            correctness=0.5,
            safety=0.5,
            user_benefit=0.5,
            efficiency=0.5,
            auditability=0.5,
            anti_centralization=0.5,
            robustness=0.5,
            fairness=0.5,
        )
        # Verify 4 dimensions
        vec = vec.verify_dimension(DimensionId.CORRECTNESS)
        vec = vec.verify_dimension(DimensionId.SAFETY)
        vec = vec.verify_dimension(DimensionId.USER_BENEFIT)
        vec = vec.verify_dimension(DimensionId.EFFICIENCY)
        result = vec.verify_thresholds(ExecutionContext.DEVELOPMENT)
        assert result.passed is False

    def test_verify_thresholds_development_fails_dims(self) -> None:
        """Development: fewer than 4 verified dims must fail even with high score."""
        vec = IhsanVector.perfect()
        # Only 3 verified
        vec = vec.verify_dimension(DimensionId.CORRECTNESS)
        vec = vec.verify_dimension(DimensionId.SAFETY)
        vec = vec.verify_dimension(DimensionId.USER_BENEFIT)
        result = vec.verify_thresholds(ExecutionContext.DEVELOPMENT)
        assert result.passed is False

    def test_verify_thresholds_staging_passes(self) -> None:
        """Staging: score >= 0.90 with 6+ verified dims must pass."""
        vec = IhsanVector.from_scores(
            correctness=0.95,
            safety=0.95,
            user_benefit=0.92,
            efficiency=0.90,
            auditability=0.88,
            anti_centralization=0.85,
            robustness=0.80,
            fairness=0.75,
        )
        # Verify 6 dimensions
        for dim_id in list(DimensionId)[:6]:
            vec = vec.verify_dimension(dim_id)
        result = vec.verify_thresholds(ExecutionContext.STAGING)
        assert result.passed is True

    def test_verify_thresholds_production_passes(self) -> None:
        """Production: score >= 0.95 with all 8 verified must pass."""
        vec = IhsanVector.perfect()
        vec = vec.verify_all()
        result = vec.verify_thresholds(ExecutionContext.PRODUCTION)
        assert result.passed is True

    def test_verify_thresholds_production_fails_score(self) -> None:
        """Production: score < 0.95 must fail even with all verified."""
        vec = IhsanVector.from_scores(
            correctness=0.90,
            safety=0.90,
            user_benefit=0.90,
            efficiency=0.90,
            auditability=0.90,
            anti_centralization=0.90,
            robustness=0.90,
            fairness=0.90,
        )
        vec = vec.verify_all()
        result = vec.verify_thresholds(ExecutionContext.PRODUCTION)
        assert result.passed is False

    def test_verify_thresholds_production_fails_unverified(self) -> None:
        """Production: unverified dims must fail even with high score."""
        vec = IhsanVector.perfect()
        # Do not verify any
        result = vec.verify_thresholds(ExecutionContext.PRODUCTION)
        assert result.passed is False

    def test_verify_thresholds_critical_needs_manual_review(self) -> None:
        """Critical: without manual review must fail even with perfect scores."""
        vec = IhsanVector.perfect()
        vec = vec.verify_all()
        result = vec.verify_thresholds(ExecutionContext.CRITICAL)
        assert result.passed is False

    def test_verify_thresholds_critical_with_manual_review_passes(self) -> None:
        """Critical: with manual review and perfect scores must pass."""
        vec = IhsanVector.perfect()
        vec = vec.verify_all()
        result = vec.verify_thresholds(
            ExecutionContext.CRITICAL, manual_review_approved=True
        )
        assert result.passed is True

    def test_verify_thresholds_failures_list_populated(self) -> None:
        """Failed thresholds must populate failures list with descriptive messages."""
        vec = IhsanVector.neutral()  # Score 0.5, 0 verified
        result = vec.verify_thresholds(ExecutionContext.PRODUCTION)
        assert len(result.failures) >= 2  # At least score + dims failures
        assert any("Score" in f for f in result.failures)
        assert any("Verified dimensions" in f for f in result.failures)

    def test_passes_context_delegates_to_verify_thresholds(self) -> None:
        """passes_context must return same boolean as verify_thresholds.passed."""
        vec = IhsanVector.perfect()
        vec = vec.verify_all()
        assert vec.passes_context(ExecutionContext.PRODUCTION) is True
        assert vec.passes_context(ExecutionContext.CRITICAL) is False
        assert vec.passes_context(ExecutionContext.CRITICAL, manual_review_approved=True) is True

    def test_get_dimension_returns_correct_dimension(self) -> None:
        """get_dimension must return the dimension matching the given id."""
        vec = IhsanVector.from_scores(correctness=0.99, safety=0.88)
        dim = vec.get_dimension(DimensionId.CORRECTNESS)
        assert dim.id == DimensionId.CORRECTNESS
        assert dim.score == 0.99

    def test_set_score_returns_new_vector(self) -> None:
        """set_score must return a new IhsanVector (immutability)."""
        original = IhsanVector.perfect()
        updated = original.set_score(DimensionId.CORRECTNESS, 0.5)
        assert updated is not original

    def test_set_score_resets_verification(self) -> None:
        """set_score must reset verification state for the modified dimension."""
        vec = IhsanVector.perfect()
        vec = vec.verify_dimension(DimensionId.CORRECTNESS, proof="proof_1")
        assert vec.get_dimension(DimensionId.CORRECTNESS).verified is True

        updated = vec.set_score(DimensionId.CORRECTNESS, 0.8)
        assert updated.get_dimension(DimensionId.CORRECTNESS).verified is False

    def test_set_score_updates_value(self) -> None:
        """set_score must correctly update the dimension score."""
        vec = IhsanVector.perfect()
        updated = vec.set_score(DimensionId.FAIRNESS, 0.42)
        assert updated.get_dimension(DimensionId.FAIRNESS).score == 0.42

    def test_verify_dimension_returns_new_vector(self) -> None:
        """verify_dimension must return a new IhsanVector."""
        original = IhsanVector.perfect()
        verified = original.verify_dimension(DimensionId.SAFETY, proof="test")
        assert verified is not original

    def test_verify_all_with_proofs(self) -> None:
        """verify_all with proofs dict must store proof for each dimension."""
        vec = IhsanVector.perfect()
        proofs = {
            DimensionId.CORRECTNESS: "z3_proof",
            DimensionId.SAFETY: "aegis_proof",
        }
        verified = vec.verify_all(proofs=proofs)
        assert verified.get_dimension(DimensionId.CORRECTNESS).verification_proof == "z3_proof"
        assert verified.get_dimension(DimensionId.SAFETY).verification_proof == "aegis_proof"
        # Dimensions without proofs should still be verified but with no proof
        assert verified.get_dimension(DimensionId.FAIRNESS).verified is True
        assert verified.get_dimension(DimensionId.FAIRNESS).verification_proof is None

    def test_verify_all_without_proofs(self) -> None:
        """verify_all without proofs must verify all dimensions with None proofs."""
        vec = IhsanVector.perfect()
        verified = vec.verify_all()
        assert verified.all_verified is True
        for dim in verified.dimensions.values():
            assert dim.verified is True
            assert dim.verification_proof is None

    def test_to_dict_has_all_fields(self) -> None:
        """to_dict must include created_at, context, scores, and dimensions."""
        vec = IhsanVector.from_scores(
            correctness=0.9,
            safety=0.9,
            user_benefit=0.9,
            efficiency=0.9,
            auditability=0.9,
            anti_centralization=0.9,
            robustness=0.9,
            fairness=0.9,
            context=ExecutionContext.STAGING,
        )
        d = vec.to_dict()
        expected_keys = {
            "created_at",
            "context",
            "aggregate_score",
            "verified_score",
            "verified_count",
            "dimensions",
        }
        assert set(d.keys()) == expected_keys

    def test_from_dict_roundtrip_preserves_scores_and_verification(self) -> None:
        """to_dict -> from_dict must preserve all scores and verification state."""
        original = IhsanVector.from_scores(
            correctness=0.98,
            safety=0.97,
            user_benefit=0.92,
            efficiency=0.88,
            auditability=0.90,
            anti_centralization=0.85,
            robustness=0.82,
            fairness=0.80,
        )
        original = original.verify_dimension(DimensionId.CORRECTNESS, proof="proof_c")
        original = original.verify_dimension(DimensionId.SAFETY, proof="proof_s")

        reconstructed = IhsanVector.from_dict(original.to_dict())
        for dim_id in DimensionId:
            orig_dim = original.get_dimension(dim_id)
            recon_dim = reconstructed.get_dimension(dim_id)
            assert recon_dim.score == pytest.approx(orig_dim.score, abs=1e-6)
            assert recon_dim.verified == orig_dim.verified


# =============================================================================
# THRESHOLD RESULT
# =============================================================================


class TestThresholdResult:
    """Tests for the ThresholdResult dataclass."""

    def _make_result(
        self,
        passed: bool = True,
        failures: Optional[list] = None,
    ) -> ThresholdResult:
        """Helper to create a ThresholdResult for testing."""
        return ThresholdResult(
            passed=passed,
            context=ExecutionContext.PRODUCTION,
            aggregate_score=0.96,
            verified_count=8,
            required_score=0.95,
            required_verified=8,
            requires_manual_review=False,
            manual_review_approved=False,
            failures=failures or [],
            dimension_summary={
                "correctness": {"score": 0.98, "verified": True, "weighted": 0.2156}
            },
        )

    def test_to_dict_has_all_fields(self) -> None:
        """to_dict must include all ThresholdResult fields."""
        result = self._make_result()
        d = result.to_dict()
        expected_keys = {
            "passed",
            "context",
            "aggregate_score",
            "verified_count",
            "required_score",
            "required_verified",
            "requires_manual_review",
            "manual_review_approved",
            "failures",
            "dimension_summary",
        }
        assert set(d.keys()) == expected_keys

    def test_failures_list_preserved(self) -> None:
        """to_dict must preserve the failures list."""
        result = self._make_result(
            passed=False,
            failures=["Score too low", "Not enough verified dims"],
        )
        d = result.to_dict()
        assert d["failures"] == ["Score too low", "Not enough verified dims"]

    def test_dimension_summary_structure(self) -> None:
        """dimension_summary must contain per-dimension score/verified/weighted."""
        result = self._make_result()
        d = result.to_dict()
        summary = d["dimension_summary"]
        assert "correctness" in summary
        assert "score" in summary["correctness"]
        assert "verified" in summary["correctness"]
        assert "weighted" in summary["correctness"]

    def test_context_value_in_dict(self) -> None:
        """to_dict must serialize context as its string value."""
        result = self._make_result()
        d = result.to_dict()
        assert d["context"] == "production"

    def test_aggregate_score_rounded(self) -> None:
        """to_dict must round aggregate_score to 6 decimal places."""
        result = ThresholdResult(
            passed=True,
            context=ExecutionContext.PRODUCTION,
            aggregate_score=0.123456789,
            verified_count=8,
            required_score=0.1,
            required_verified=0,
            requires_manual_review=False,
            manual_review_approved=False,
            failures=[],
            dimension_summary={},
        )
        d = result.to_dict()
        assert d["aggregate_score"] == round(0.123456789, 6)


# =============================================================================
# IHSAN RECEIPT
# =============================================================================


class TestIhsanReceipt:
    """Tests for the IhsanReceipt dataclass."""

    def test_to_receipt_generates_valid_hash(self) -> None:
        """to_receipt must generate a BLAKE3 hash of the vector JSON (SEC-001)."""
        from core.proof_engine.canonical import hex_digest

        vec = IhsanVector.perfect()
        receipt = vec.to_receipt()
        # Recompute hash using BLAKE3
        expected_hash = hex_digest(receipt.receipt_json.encode())
        assert receipt.receipt_hash == expected_hash

    def test_verify_integrity_returns_true_for_valid(self) -> None:
        """verify_integrity must return True for an untampered receipt."""
        vec = IhsanVector.perfect()
        receipt = vec.to_receipt()
        assert receipt.verify_integrity() is True

    def test_verify_integrity_returns_false_for_tampered(self) -> None:
        """verify_integrity must return False if receipt_json is modified."""
        vec = IhsanVector.perfect()
        receipt = vec.to_receipt()
        # Tamper with the JSON
        receipt.receipt_json = receipt.receipt_json.replace("1.0", "0.5")
        assert receipt.verify_integrity() is False

    def test_verify_integrity_returns_false_for_tampered_hash(self) -> None:
        """verify_integrity must return False if receipt_hash is modified."""
        vec = IhsanVector.perfect()
        receipt = vec.to_receipt()
        receipt.receipt_hash = "0" * 64
        assert receipt.verify_integrity() is False

    def test_to_dict_includes_integrity_valid(self) -> None:
        """to_dict must include integrity_valid key."""
        vec = IhsanVector.perfect()
        receipt = vec.to_receipt()
        d = receipt.to_dict()
        assert "integrity_valid" in d
        assert d["integrity_valid"] is True

    def test_to_dict_includes_vector_and_metadata(self) -> None:
        """to_dict must include receipt_hash, timestamp, and vector."""
        vec = IhsanVector.perfect()
        receipt = vec.to_receipt()
        d = receipt.to_dict()
        assert "receipt_hash" in d
        assert "timestamp" in d
        assert "vector" in d

    def test_from_dict_roundtrip(self) -> None:
        """to_dict -> from_dict must produce a receipt with valid integrity."""
        vec = IhsanVector.from_scores(
            correctness=0.95,
            safety=0.95,
            user_benefit=0.90,
            efficiency=0.85,
            auditability=0.88,
            anti_centralization=0.80,
            robustness=0.75,
            fairness=0.70,
        )
        original_receipt = vec.to_receipt()
        d = original_receipt.to_dict()
        reconstructed = IhsanReceipt.from_dict(d)
        assert reconstructed.receipt_hash == original_receipt.receipt_hash
        assert reconstructed.timestamp == original_receipt.timestamp
        assert reconstructed.verify_integrity() is True

    def test_receipt_hash_is_sha256_hex(self) -> None:
        """receipt_hash must be a valid 64-character hex string (SHA-256)."""
        vec = IhsanVector.perfect()
        receipt = vec.to_receipt()
        assert len(receipt.receipt_hash) == 64
        # Verify it is valid hexadecimal
        int(receipt.receipt_hash, 16)

    def test_receipt_timestamp_is_utc_iso(self) -> None:
        """Receipt timestamp must be a UTC ISO 8601 string."""
        vec = IhsanVector.perfect()
        receipt = vec.to_receipt()
        assert receipt.timestamp.endswith("Z")
        assert "T" in receipt.timestamp

    def test_different_vectors_produce_different_hashes(self) -> None:
        """Different vectors must produce different receipt hashes."""
        vec1 = IhsanVector.perfect()
        vec2 = IhsanVector.neutral()
        receipt1 = vec1.to_receipt()
        receipt2 = vec2.to_receipt()
        assert receipt1.receipt_hash != receipt2.receipt_hash


# =============================================================================
# CREATE VERIFIER
# =============================================================================


class TestCreateVerifier:
    """Tests for the create_verifier function."""

    def test_verifier_passes_when_func_returns_true(self) -> None:
        """Verifier must verify dimension when verify_func returns (True, proof)."""

        def always_pass(dim: IhsanDimension) -> Tuple[bool, Optional[str]]:
            return True, "auto_proof"

        verifier = create_verifier(always_pass)
        vec = IhsanVector.perfect()
        result = verifier(vec, DimensionId.CORRECTNESS)
        assert result.get_dimension(DimensionId.CORRECTNESS).verified is True

    def test_verifier_no_ops_when_func_returns_false(self) -> None:
        """Verifier must not verify dimension when verify_func returns (False, None)."""

        def always_fail(dim: IhsanDimension) -> Tuple[bool, Optional[str]]:
            return False, None

        verifier = create_verifier(always_fail)
        vec = IhsanVector.perfect()
        result = verifier(vec, DimensionId.CORRECTNESS)
        assert result.get_dimension(DimensionId.CORRECTNESS).verified is False

    def test_verifier_marks_dimension_verified_with_proof(self) -> None:
        """Verifier must store the proof string from verify_func."""

        def pass_with_proof(dim: IhsanDimension) -> Tuple[bool, Optional[str]]:
            return True, f"proof_for_{dim.id.name.lower()}"

        verifier = create_verifier(pass_with_proof)
        vec = IhsanVector.perfect()
        result = verifier(vec, DimensionId.SAFETY)
        dim = result.get_dimension(DimensionId.SAFETY)
        assert dim.verified is True
        assert dim.verification_proof == "proof_for_safety"

    def test_verifier_preserves_other_dimensions(self) -> None:
        """Verifier must not modify dimensions other than the target."""

        def always_pass(dim: IhsanDimension) -> Tuple[bool, Optional[str]]:
            return True, "proof"

        verifier = create_verifier(always_pass)
        vec = IhsanVector.perfect()
        # Pre-verify one dimension
        vec = vec.verify_dimension(DimensionId.FAIRNESS, proof="existing")
        result = verifier(vec, DimensionId.CORRECTNESS)
        # Fairness should remain verified
        assert result.get_dimension(DimensionId.FAIRNESS).verified is True
        assert result.get_dimension(DimensionId.FAIRNESS).verification_proof == "existing"

    def test_verifier_returns_original_vector_on_failure(self) -> None:
        """When verify_func returns False, verifier must return the original vector."""

        def always_fail(dim: IhsanDimension) -> Tuple[bool, Optional[str]]:
            return False, None

        verifier = create_verifier(always_fail)
        vec = IhsanVector.perfect()
        result = verifier(vec, DimensionId.CORRECTNESS)
        # Should be the same object since no modification was made
        assert result is vec


# =============================================================================
# QUICK IHSAN
# =============================================================================


class TestQuickIhsan:
    """Tests for the quick_ihsan convenience function."""

    def test_all_zeros(self) -> None:
        """All zeros must produce 0.0."""
        result = quick_ihsan(0, 0, 0, 0, 0, 0, 0, 0)
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_all_ones(self) -> None:
        """All ones must produce 1.0."""
        result = quick_ihsan(1, 1, 1, 1, 1, 1, 1, 1)
        assert result == pytest.approx(1.0, abs=1e-9)

    def test_matches_canonical_weights_formula(self) -> None:
        """quick_ihsan must match the hand-computed weighted sum."""
        scores = (0.98, 0.97, 0.92, 0.88, 0.90, 0.85, 0.82, 0.80)
        weights = (0.22, 0.22, 0.14, 0.12, 0.12, 0.08, 0.06, 0.04)
        expected = sum(s * w for s, w in zip(scores, weights))
        result = quick_ihsan(*scores)
        assert result == pytest.approx(expected, abs=1e-9)

    def test_matches_ihsan_vector_calculate_score(self) -> None:
        """quick_ihsan must produce same result as IhsanVector.calculate_score."""
        scores = (0.95, 0.93, 0.88, 0.82, 0.79, 0.76, 0.71, 0.68)
        qi = quick_ihsan(*scores)
        vec = IhsanVector.from_scores(*scores)
        assert qi == pytest.approx(vec.calculate_score(), abs=1e-9)

    def test_partial_scores(self) -> None:
        """Quick ihsan with mixed scores must correctly weight them."""
        result = quick_ihsan(1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0)
        # 0.22*1 + 0.22*0 + 0.14*1 + 0.12*0 + 0.12*1 + 0.08*0 + 0.06*1 + 0.04*0
        expected = 0.22 + 0.14 + 0.12 + 0.06
        assert result == pytest.approx(expected, abs=1e-9)


# =============================================================================
# PASSES PRODUCTION
# =============================================================================


class TestPassesProduction:
    """Tests for the passes_production convenience function."""

    def test_all_ones_passes(self) -> None:
        """All 1.0 scores must pass production threshold."""
        assert passes_production(1, 1, 1, 1, 1, 1, 1, 1) is True

    def test_all_zeros_fails(self) -> None:
        """All 0.0 scores must fail production threshold."""
        assert passes_production(0, 0, 0, 0, 0, 0, 0, 0) is False

    def test_exactly_095_passes(self) -> None:
        """Score of exactly 0.95 must pass (>= threshold)."""
        # All 0.95 -> weighted sum = 0.95
        assert passes_production(0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95) is True

    def test_just_below_095_fails(self) -> None:
        """Score just below 0.95 must fail."""
        # All 0.949 -> weighted sum = 0.949
        assert passes_production(0.949, 0.949, 0.949, 0.949, 0.949, 0.949, 0.949, 0.949) is False

    def test_high_priority_dims_compensate(self) -> None:
        """High-weight dimensions at 1.0 can compensate for low-weight dims."""
        # correctness(0.22)*1 + safety(0.22)*1 + user_benefit(0.14)*1 +
        # efficiency(0.12)*1 + auditability(0.12)*1 = 0.82
        # Need 0.13 more from remaining 0.18 weight
        # anti_centralization(0.08)*1 + robustness(0.06)*0.833... + fairness(0.04)*1
        # 0.82 + 0.08 + 0.05 + 0.04 = 0.99 > 0.95
        result = passes_production(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.833, 1.0)
        # 0.22 + 0.22 + 0.14 + 0.12 + 0.12 + 0.08 + 0.04998 + 0.04 = 0.96998
        assert result is True


# =============================================================================
# SERIALIZATION ROUNDTRIPS
# =============================================================================


class TestSerialization:
    """Tests for serialization integrity across all classes."""

    def test_vector_roundtrip_preserves_all(self) -> None:
        """Vector to_dict -> from_dict must preserve all data."""
        original = IhsanVector.from_scores(
            correctness=0.98,
            safety=0.97,
            user_benefit=0.92,
            efficiency=0.88,
            auditability=0.90,
            anti_centralization=0.85,
            robustness=0.82,
            fairness=0.80,
            context=ExecutionContext.PRODUCTION,
        )
        original = original.verify_dimension(DimensionId.CORRECTNESS, proof="z3")
        original = original.verify_dimension(DimensionId.SAFETY, proof="aegis")

        d = original.to_dict()
        reconstructed = IhsanVector.from_dict(d)

        assert reconstructed.context is ExecutionContext.PRODUCTION
        assert reconstructed.calculate_score() == pytest.approx(
            original.calculate_score(), abs=1e-6
        )
        assert reconstructed.verified_count == 2

    def test_receipt_roundtrip(self) -> None:
        """Receipt to_dict -> from_dict must preserve integrity."""
        vec = IhsanVector.from_scores(
            correctness=0.95,
            safety=0.95,
            user_benefit=0.90,
            efficiency=0.85,
            auditability=0.88,
            anti_centralization=0.80,
            robustness=0.75,
            fairness=0.70,
        )
        receipt = vec.to_receipt()
        d = receipt.to_dict()
        reconstructed = IhsanReceipt.from_dict(d)
        assert reconstructed.verify_integrity() is True
        assert reconstructed.receipt_hash == receipt.receipt_hash

    def test_json_stability_same_vector_same_hash(self) -> None:
        """Same vector serialized twice must produce the same JSON hash."""
        vec = IhsanVector.from_scores(
            correctness=0.98,
            safety=0.97,
            user_benefit=0.92,
            efficiency=0.88,
            auditability=0.90,
            anti_centralization=0.85,
            robustness=0.82,
            fairness=0.80,
        )
        d1 = vec.to_dict()
        d2 = vec.to_dict()
        json1 = json.dumps(d1, sort_keys=True, separators=(",", ":"))
        json2 = json.dumps(d2, sort_keys=True, separators=(",", ":"))
        assert hashlib.sha256(json1.encode()).hexdigest() == hashlib.sha256(
            json2.encode()
        ).hexdigest()

    def test_from_dict_with_context(self) -> None:
        """from_dict must restore context when present."""
        d = {
            "context": "staging",
            "dimensions": {
                "correctness": {"id": "correctness", "score": 0.9},
            },
        }
        vec = IhsanVector.from_dict(d)
        assert vec.context is ExecutionContext.STAGING

    def test_from_dict_without_context(self) -> None:
        """from_dict must set context=None when not present."""
        d = {
            "dimensions": {
                "correctness": {"id": "correctness", "score": 0.9},
            },
        }
        vec = IhsanVector.from_dict(d)
        assert vec.context is None

    def test_ihsan_dimension_from_dict_with_minimal_data(self) -> None:
        """IhsanDimension.from_dict with only 'id' must fill all defaults."""
        dim = IhsanDimension.from_dict({"id": "fairness"})
        assert dim.id == DimensionId.FAIRNESS
        assert dim.weight == 0.04
        assert dim.score == 0.0
        assert dim.verified is False
        assert dim.verify_method == "kl_divergence_bias_check"

    def test_weight_sum_warning_logged_when_wrong(self, caplog: pytest.LogCaptureFixture) -> None:
        """IhsanVector __post_init__ must log warning when weights do not sum to 1.0."""
        bad_dims = {
            DimensionId.CORRECTNESS: IhsanDimension(
                id=DimensionId.CORRECTNESS, weight=0.5, score=0.9
            ),
            DimensionId.SAFETY: IhsanDimension(
                id=DimensionId.SAFETY, weight=0.5, score=0.9
            ),
        }
        with caplog.at_level(logging.WARNING, logger="core.sovereign.ihsan_vector"):
            # This will fill in 6 missing dims with canonical weights
            # Total = 0.5 + 0.5 + 0.14 + 0.12 + 0.12 + 0.08 + 0.06 + 0.04 = 1.56
            IhsanVector(dimensions=bad_dims)
        assert any("weights sum to" in record.message for record in caplog.records)

    def test_to_dict_aggregate_score_rounded(self) -> None:
        """to_dict aggregate_score must be rounded to 6 decimal places."""
        vec = IhsanVector.from_scores(
            correctness=0.333333333,
            safety=0.666666666,
            user_benefit=0.111111111,
            efficiency=0.222222222,
            auditability=0.444444444,
            anti_centralization=0.555555555,
            robustness=0.777777777,
            fairness=0.888888888,
        )
        d = vec.to_dict()
        score_str = str(d["aggregate_score"])
        # After the decimal, at most 6 digits
        if "." in score_str:
            decimal_part = score_str.split(".")[1]
            assert len(decimal_part) <= 6


# =============================================================================
# IMMUTABILITY
# =============================================================================


class TestImmutability:
    """Tests ensuring immutability guarantees across all mutation operations."""

    def test_set_score_does_not_modify_original(self) -> None:
        """set_score must not mutate the original vector."""
        original = IhsanVector.from_scores(
            correctness=0.9,
            safety=0.9,
            user_benefit=0.9,
            efficiency=0.9,
            auditability=0.9,
            anti_centralization=0.9,
            robustness=0.9,
            fairness=0.9,
        )
        original_score = original.calculate_score()
        _ = original.set_score(DimensionId.CORRECTNESS, 0.1)
        assert original.calculate_score() == pytest.approx(original_score, abs=1e-9)
        assert original.get_dimension(DimensionId.CORRECTNESS).score == 0.9

    def test_verify_dimension_does_not_modify_original(self) -> None:
        """verify_dimension must not mutate the original vector."""
        original = IhsanVector.perfect()
        assert original.verified_count == 0
        _ = original.verify_dimension(DimensionId.CORRECTNESS, proof="test")
        assert original.verified_count == 0
        assert original.get_dimension(DimensionId.CORRECTNESS).verified is False

    def test_verify_all_does_not_modify_original(self) -> None:
        """verify_all must not mutate the original vector."""
        original = IhsanVector.perfect()
        assert original.all_verified is False
        _ = original.verify_all()
        assert original.all_verified is False
        assert original.verified_count == 0

    def test_mark_verified_does_not_modify_original_dimension(self) -> None:
        """IhsanDimension.mark_verified must not mutate the original dimension."""
        original = IhsanDimension(
            id=DimensionId.CORRECTNESS, weight=0.22, score=0.95
        )
        assert original.verified is False
        _ = original.mark_verified(proof="test_proof")
        assert original.verified is False
        assert original.verification_proof is None
        assert original.verification_timestamp is None

    def test_chained_mutations_preserve_independence(self) -> None:
        """Chained mutations must produce independent vectors at each step."""
        v0 = IhsanVector.perfect()
        v1 = v0.set_score(DimensionId.CORRECTNESS, 0.5)
        v2 = v1.verify_dimension(DimensionId.SAFETY)
        v3 = v2.verify_all()

        # v0 unchanged
        assert v0.get_dimension(DimensionId.CORRECTNESS).score == 1.0
        assert v0.verified_count == 0

        # v1 has changed score but no verification
        assert v1.get_dimension(DimensionId.CORRECTNESS).score == 0.5
        assert v1.verified_count == 0

        # v2 has one verification
        assert v2.verified_count == 1
        assert v2.get_dimension(DimensionId.SAFETY).verified is True
        assert v2.get_dimension(DimensionId.CORRECTNESS).verified is False

        # v3 has all verified
        assert v3.all_verified is True


# =============================================================================
# EDGE CASES AND ADDITIONAL COVERAGE
# =============================================================================


class TestEdgeCases:
    """Tests for boundary conditions and edge cases."""

    def test_created_at_auto_populated(self) -> None:
        """created_at must be auto-populated with UTC ISO timestamp."""
        vec = IhsanVector()
        assert vec.created_at is not None
        assert "T" in vec.created_at

    def test_from_scores_default_all_zero(self) -> None:
        """from_scores with no arguments must create all-zero vector."""
        vec = IhsanVector.from_scores()
        assert vec.calculate_score() == pytest.approx(0.0, abs=1e-9)

    def test_verify_thresholds_critical_all_three_failures(self) -> None:
        """Critical context with low score, no verified, no manual must list 3 failures."""
        vec = IhsanVector.neutral()
        result = vec.verify_thresholds(ExecutionContext.CRITICAL)
        assert result.passed is False
        assert len(result.failures) == 3  # score + dims + manual

    def test_dimension_weight_boundary_small(self) -> None:
        """Weight at minimum valid value (just above 0) must be accepted."""
        dim = IhsanDimension(id=DimensionId.FAIRNESS, weight=0.001, score=0.5)
        assert dim.weight == 0.001

    def test_dimension_weight_boundary_one(self) -> None:
        """Weight at exactly 1.0 must be accepted."""
        dim = IhsanDimension(id=DimensionId.FAIRNESS, weight=1.0, score=0.5)
        assert dim.weight == 1.0

    def test_verify_thresholds_dimension_summary_has_all_dims(self) -> None:
        """verify_thresholds dimension_summary must contain all 8 dimensions."""
        vec = IhsanVector.perfect()
        result = vec.verify_thresholds(ExecutionContext.DEVELOPMENT)
        assert len(result.dimension_summary) == 8
        for dim_id in DimensionId:
            assert dim_id.name.lower() in result.dimension_summary

    def test_set_score_preserves_other_dimensions(self) -> None:
        """set_score must not alter any dimension except the target."""
        vec = IhsanVector.from_scores(
            correctness=0.9,
            safety=0.8,
            user_benefit=0.7,
            efficiency=0.6,
            auditability=0.5,
            anti_centralization=0.4,
            robustness=0.3,
            fairness=0.2,
        )
        updated = vec.set_score(DimensionId.CORRECTNESS, 0.1)
        assert updated.get_dimension(DimensionId.SAFETY).score == 0.8
        assert updated.get_dimension(DimensionId.USER_BENEFIT).score == 0.7
        assert updated.get_dimension(DimensionId.FAIRNESS).score == 0.2

    def test_multiple_verify_dimension_increments_count(self) -> None:
        """Calling verify_dimension multiple times must increment verified_count each time."""
        vec = IhsanVector.perfect()
        vec = vec.verify_dimension(DimensionId.CORRECTNESS)
        assert vec.verified_count == 1
        vec = vec.verify_dimension(DimensionId.SAFETY)
        assert vec.verified_count == 2
        vec = vec.verify_dimension(DimensionId.USER_BENEFIT)
        assert vec.verified_count == 3

    def test_from_dict_preserves_created_at(self) -> None:
        """from_dict must restore the created_at timestamp."""
        vec = IhsanVector.perfect()
        d = vec.to_dict()
        d["created_at"] = "2024-01-01T00:00:00Z"
        reconstructed = IhsanVector.from_dict(d)
        assert reconstructed.created_at == "2024-01-01T00:00:00Z"

    def test_to_dict_context_none_serializes_as_none(self) -> None:
        """to_dict with context=None must serialize context as None."""
        vec = IhsanVector.perfect()
        d = vec.to_dict()
        assert d["context"] is None

    def test_to_dict_context_present_serializes_as_string(self) -> None:
        """to_dict with context set must serialize context as string value."""
        vec = IhsanVector.from_scores(context=ExecutionContext.STAGING)
        d = vec.to_dict()
        assert d["context"] == "staging"
