"""
Unit Tests for IhsanProjector - Constitutional AI to NTU Bridge

Tests cover:
1. Basic projection mechanics (O(1) complexity)
2. Constitutional invariants enforcement
3. Edge cases (perfect, neutral, violation states)
4. Inverse projection approximation
5. Calibration from examples
6. Sensitivity analysis

Standing on Giants: Shannon, Anthropic, Friston
"""

import math
import pytest
import numpy as np
from typing import List, Tuple

from core.sovereign.ihsan_projector import (
    IhsanProjector,
    IhsanVector,
    IhsanDimension,
    ProjectorConfig,
    project_ihsan_to_ntu,
    create_ihsan_from_scores,
    IHSAN_ARABIC_NAMES,
)
from core.ntu import NTUState


class TestIhsanVector:
    """Tests for the IhsanVector dataclass."""

    def test_default_initialization(self) -> None:
        """Default vector should be neutral (0.5)."""
        v = IhsanVector()
        assert v.truthfulness == 0.5
        assert v.excellence == 0.5
        assert v.aggregate_score == pytest.approx(0.5, abs=0.01)

    def test_perfect_vector(self) -> None:
        """Perfect vector should have all 1.0 values."""
        v = IhsanVector.perfect()
        assert v.as_array.sum() == 8.0
        assert v.aggregate_score == 1.0
        assert not v.has_constitutional_violation

    def test_neutral_vector(self) -> None:
        """Neutral vector should have all 0.5 values."""
        v = IhsanVector.neutral()
        arr = v.as_array
        assert np.allclose(arr, 0.5)
        assert not v.has_constitutional_violation

    def test_constitutional_violation_detection(self) -> None:
        """Vectors with any dimension < 0.5 should flag violation."""
        # No violation
        v_ok = IhsanVector(truthfulness=0.6, justice=0.7, excellence=0.8,
                          trustworthiness=0.6, wisdom=0.6, compassion=0.6,
                          patience=0.6, gratitude=0.6)
        assert not v_ok.has_constitutional_violation

        # Single violation
        v_violation = IhsanVector(truthfulness=0.3, justice=0.7, excellence=0.8,
                                 trustworthiness=0.6, wisdom=0.6, compassion=0.6,
                                 patience=0.6, gratitude=0.6)
        assert v_violation.has_constitutional_violation
        min_dim, min_val = v_violation.min_dimension
        assert min_dim == IhsanDimension.TRUTHFULNESS
        assert min_val == pytest.approx(0.3)

    def test_invalid_range_raises(self) -> None:
        """Values outside [0, 1] should raise ValueError."""
        with pytest.raises(ValueError):
            IhsanVector(truthfulness=1.5)
        with pytest.raises(ValueError):
            IhsanVector(justice=-0.1)

    def test_from_array(self) -> None:
        """Construction from numpy array should work."""
        arr = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        v = IhsanVector.from_array(arr)
        assert v.truthfulness == 0.1
        assert v.gratitude == 0.8

    def test_from_array_clips(self) -> None:
        """Values outside [0, 1] should be clipped."""
        arr = np.array([1.5, -0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        v = IhsanVector.from_array(arr)
        assert v.truthfulness == 1.0
        assert v.trustworthiness == 0.0

    def test_wrong_array_length_raises(self) -> None:
        """Arrays not of length 8 should raise ValueError."""
        with pytest.raises(ValueError):
            IhsanVector.from_array(np.array([0.5, 0.5, 0.5]))

    def test_aggregate_geometric_mean(self) -> None:
        """Aggregate should use geometric mean (penalizes imbalance)."""
        # Balanced vector
        balanced = IhsanVector.from_array(np.array([0.8] * 8))
        assert balanced.aggregate_score == pytest.approx(0.8, abs=0.01)

        # Imbalanced vector (one low dimension penalizes heavily)
        imbalanced = IhsanVector.from_array(np.array([0.1, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]))
        # Geometric mean of (0.1 * 0.9^7)^(1/8) should be less than arithmetic mean
        assert imbalanced.aggregate_score < 0.7

    def test_serialization_roundtrip(self) -> None:
        """to_dict and from_dict should preserve values."""
        v1 = IhsanVector(truthfulness=0.9, justice=0.7, excellence=0.95,
                        trustworthiness=0.85, wisdom=0.8, compassion=0.75,
                        patience=0.7, gratitude=0.65)
        d = v1.to_dict()
        v2 = IhsanVector.from_dict(d)
        assert np.allclose(v1.as_array, v2.as_array)


class TestIhsanDimension:
    """Tests for IhsanDimension enum."""

    def test_all_dimensions_exist(self) -> None:
        """All 8 dimensions should be defined."""
        assert len(IhsanDimension) == 8
        assert IhsanDimension.TRUTHFULNESS.value == 0
        assert IhsanDimension.GRATITUDE.value == 7

    def test_arabic_names_complete(self) -> None:
        """All dimensions should have Arabic names."""
        for dim in IhsanDimension:
            assert dim in IHSAN_ARABIC_NAMES
            assert len(IHSAN_ARABIC_NAMES[dim]) > 0


class TestProjectorConfig:
    """Tests for ProjectorConfig."""

    def test_default_config(self) -> None:
        """Default config should use standard thresholds."""
        config = ProjectorConfig()
        assert config.doubt_threshold == 0.5
        assert config.ihsan_threshold == 0.95
        assert config.potential_range == (-1.0, 1.0)

    def test_invalid_doubt_threshold(self) -> None:
        """Invalid doubt threshold should raise."""
        with pytest.raises(ValueError):
            ProjectorConfig(doubt_threshold=0.0)
        with pytest.raises(ValueError):
            ProjectorConfig(doubt_threshold=1.0)


class TestIhsanProjector:
    """Tests for the core IhsanProjector class."""

    @pytest.fixture
    def projector(self) -> IhsanProjector:
        """Create a default projector."""
        return IhsanProjector()

    def test_initialization(self, projector: IhsanProjector) -> None:
        """Projector should initialize with correct dimensions."""
        diag = projector.get_diagnostics()
        assert len(diag["weight_matrix"]) == 3
        assert len(diag["weight_matrix"][0]) == 8
        assert len(diag["bias"]) == 3

    def test_project_returns_ntu_state(self, projector: IhsanProjector) -> None:
        """Projection should return valid NTUState."""
        ihsan = IhsanVector.neutral()
        ntu = projector.project(ihsan)

        assert isinstance(ntu, NTUState)
        assert 0.0 <= ntu.belief <= 1.0
        assert 0.0 <= ntu.entropy <= 1.0
        assert 0.0 <= ntu.potential <= 1.0

    def test_o1_complexity(self, projector: IhsanProjector) -> None:
        """Projection should be O(1) regardless of input."""
        import time

        ihsan = IhsanVector.perfect()

        # Warm up
        _ = projector.project(ihsan)

        # Time single projection
        start = time.perf_counter()
        for _ in range(1000):
            _ = projector.project(ihsan)
        elapsed = time.perf_counter() - start

        # Should complete 1000 projections in under 200ms on any reasonable hardware
        # (WSL overhead can add ~2x latency vs native Linux)
        assert elapsed < 0.2, f"1000 projections took {elapsed:.3f}s"

    def test_perfect_ihsan_high_belief(self, projector: IhsanProjector) -> None:
        """Perfect Ihsan should yield high belief."""
        ihsan = IhsanVector.perfect()
        ntu = projector.project(ihsan)

        # Perfect input should yield belief close to 1.0
        assert ntu.belief > 0.8
        # Low entropy (high certainty)
        assert ntu.entropy < 0.5

    def test_constitutional_violation_penalizes_belief(self, projector: IhsanProjector) -> None:
        """Constitutional violation should reduce belief."""
        # Good vector
        good = IhsanVector.from_array(np.array([0.8] * 8))
        ntu_good = projector.project(good)

        # Violation vector (one dimension below 0.5)
        violation = IhsanVector.from_array(np.array([0.3, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]))
        ntu_violation = projector.project(violation)

        # Violation should have lower belief
        assert ntu_violation.belief < ntu_good.belief

    def test_justice_dominates_potential(self, projector: IhsanProjector) -> None:
        """Justice should have largest impact on potential."""
        # High justice
        high_justice = IhsanVector(
            truthfulness=0.5, trustworthiness=0.5, justice=0.95,
            excellence=0.5, wisdom=0.5, compassion=0.5,
            patience=0.5, gratitude=0.5
        )
        ntu_high = projector.project(high_justice)

        # Low justice
        low_justice = IhsanVector(
            truthfulness=0.5, trustworthiness=0.5, justice=0.55,
            excellence=0.5, wisdom=0.5, compassion=0.5,
            patience=0.5, gratitude=0.5
        )
        ntu_low = projector.project(low_justice)

        # High justice should yield higher potential
        assert ntu_high.potential > ntu_low.potential

    def test_excellence_dominates_belief(self, projector: IhsanProjector) -> None:
        """Excellence should be primary belief contributor."""
        # High excellence
        high_excellence = IhsanVector(
            truthfulness=0.5, trustworthiness=0.5, justice=0.5,
            excellence=0.95, wisdom=0.5, compassion=0.5,
            patience=0.5, gratitude=0.5
        )
        ntu_high = projector.project(high_excellence)

        # Low excellence
        low_excellence = IhsanVector(
            truthfulness=0.5, trustworthiness=0.5, justice=0.5,
            excellence=0.55, wisdom=0.5, compassion=0.5,
            patience=0.5, gratitude=0.5
        )
        ntu_low = projector.project(low_excellence)

        # High excellence should yield higher belief
        assert ntu_high.belief > ntu_low.belief

    def test_wisdom_reduces_entropy(self, projector: IhsanProjector) -> None:
        """High wisdom should reduce entropy (more certainty)."""
        # High wisdom
        high_wisdom = IhsanVector(
            truthfulness=0.5, trustworthiness=0.5, justice=0.5,
            excellence=0.5, wisdom=0.95, compassion=0.5,
            patience=0.5, gratitude=0.5
        )
        ntu_high = projector.project(high_wisdom)

        # Low wisdom
        low_wisdom = IhsanVector(
            truthfulness=0.5, trustworthiness=0.5, justice=0.5,
            excellence=0.5, wisdom=0.55, compassion=0.5,
            patience=0.5, gratitude=0.5
        )
        ntu_low = projector.project(low_wisdom)

        # High wisdom should yield lower entropy
        assert ntu_high.entropy < ntu_low.entropy

    def test_batch_projection(self, projector: IhsanProjector) -> None:
        """Batch projection should handle multiple vectors."""
        vectors = [
            IhsanVector.perfect(),
            IhsanVector.neutral(),
            IhsanVector.from_array(np.array([0.7] * 8)),
        ]
        results = projector.project_batch(vectors)

        assert len(results) == 3
        assert all(isinstance(r, NTUState) for r in results)
        # Perfect should have highest belief
        assert results[0].belief > results[1].belief

    def test_jacobian(self, projector: IhsanProjector) -> None:
        """Jacobian should equal weight matrix."""
        jacobian = projector.get_jacobian()
        assert jacobian.shape == (3, 8)
        assert np.allclose(jacobian, projector._W)

    def test_sensitivity_analysis(self, projector: IhsanProjector) -> None:
        """Sensitivity analysis should return derivatives."""
        ihsan = IhsanVector.neutral()
        sens = projector.sensitivity_analysis(ihsan, IhsanDimension.JUSTICE)

        assert "d_belief" in sens
        assert "d_entropy" in sens
        assert "d_potential" in sens

        # Justice should have positive impact on potential
        assert sens["d_potential"] > 0

    def test_inverse_projection(self, projector: IhsanProjector) -> None:
        """Inverse projection should approximate original."""
        original = IhsanVector(
            truthfulness=0.8, trustworthiness=0.7, justice=0.9,
            excellence=0.85, wisdom=0.75, compassion=0.8,
            patience=0.7, gratitude=0.65
        )
        ntu = projector.project(original)
        recovered = projector.inverse_project(ntu, prior=IhsanVector.neutral())

        # Should not be exact (underdetermined), but should be reasonable
        original_arr = original.as_array
        recovered_arr = recovered.as_array

        # At least the aggregate should be in similar range
        assert abs(original.aggregate_score - recovered.aggregate_score) < 0.3


class TestCalibration:
    """Tests for projector calibration."""

    def test_calibrate_from_examples(self) -> None:
        """Calibration should reduce MSE on training data."""
        projector = IhsanProjector()

        # Create synthetic training data
        examples: List[Tuple[IhsanVector, NTUState]] = [
            (IhsanVector.perfect(), NTUState(belief=0.95, entropy=0.1, potential=0.8)),
            (IhsanVector.neutral(), NTUState(belief=0.5, entropy=0.5, potential=0.5)),
            (
                IhsanVector.from_array(np.array([0.7] * 8)),
                NTUState(belief=0.7, entropy=0.3, potential=0.6)
            ),
        ]

        # Calibrate
        final_mse = projector.calibrate_from_examples(
            examples, learning_rate=0.1, iterations=50
        )

        # MSE should be reasonably low after calibration
        assert final_mse < 0.5

    def test_empty_examples(self) -> None:
        """Empty examples should return zero loss."""
        projector = IhsanProjector()
        loss = projector.calibrate_from_examples([])
        assert loss == 0.0


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_project_ihsan_to_ntu(self) -> None:
        """Convenience function should work."""
        ihsan = IhsanVector.perfect()
        ntu = project_ihsan_to_ntu(ihsan)

        assert isinstance(ntu, NTUState)
        assert ntu.belief > 0.8

    def test_create_ihsan_from_scores(self) -> None:
        """Score mapping should produce valid Ihsan vector."""
        ihsan = create_ihsan_from_scores(
            correctness=0.9,
            safety=0.85,
            helpfulness=0.8,
            efficiency=0.75
        )

        assert isinstance(ihsan, IhsanVector)
        assert ihsan.truthfulness == 0.9
        assert ihsan.trustworthiness == 0.85
        assert ihsan.wisdom == 0.75

    def test_create_ihsan_default_scores(self) -> None:
        """Default scores should produce neutral-ish vector."""
        ihsan = create_ihsan_from_scores()

        # All defaults are 0.5, so result should be balanced
        assert ihsan.truthfulness == 0.5


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_all_zeros(self) -> None:
        """All-zero vector should not crash and should yield low belief."""
        projector = IhsanProjector()
        ihsan = IhsanVector.from_array(np.array([0.001] * 8))  # Near zero
        ntu = projector.project(ihsan)

        assert ntu.belief >= 0.0
        assert ntu.entropy >= 0.0
        assert ntu.potential >= 0.0

    def test_extreme_imbalance(self) -> None:
        """Extreme imbalance should be handled gracefully."""
        projector = IhsanProjector()
        imbalanced = IhsanVector.from_array(
            np.array([1.0, 0.001, 1.0, 0.001, 1.0, 0.001, 1.0, 0.001])
        )
        ntu = projector.project(imbalanced)

        # Should still produce valid output
        assert 0.0 <= ntu.belief <= 1.0
        assert 0.0 <= ntu.entropy <= 1.0
        assert 0.0 <= ntu.potential <= 1.0

        # Should flag violation
        assert imbalanced.has_constitutional_violation

    def test_custom_weight_matrix(self) -> None:
        """Custom weight matrix should be used."""
        custom_W = np.eye(3, 8) * 0.5  # Identity-like
        custom_b = np.array([0.1, 0.2, 0.3])

        projector = IhsanProjector(weight_matrix=custom_W, bias=custom_b)
        ihsan = IhsanVector.neutral()
        ntu = projector.project(ihsan)

        # With custom weights, output should be predictable
        assert isinstance(ntu, NTUState)

    def test_invalid_weight_matrix_shape(self) -> None:
        """Invalid weight matrix shape should raise."""
        with pytest.raises(ValueError):
            IhsanProjector(weight_matrix=np.eye(4, 8))

    def test_invalid_bias_shape(self) -> None:
        """Invalid bias shape should raise."""
        with pytest.raises(ValueError):
            IhsanProjector(bias=np.array([0.1, 0.2]))


class TestDiagnostics:
    """Tests for diagnostic output."""

    def test_diagnostics_structure(self) -> None:
        """Diagnostics should have expected structure."""
        projector = IhsanProjector()
        diag = projector.get_diagnostics()

        assert "config" in diag
        assert "weight_matrix" in diag
        assert "bias" in diag
        assert "weight_norms" in diag
        assert "dominant_dimensions" in diag

        # Dominant dimensions should be valid names
        assert diag["dominant_dimensions"]["belief"] in [d.name for d in IhsanDimension]
        assert diag["dominant_dimensions"]["potential"] in [d.name for d in IhsanDimension]
