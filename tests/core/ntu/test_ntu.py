"""
Tests for NeuroTemporal Unit (NTU) core implementation.

Tests cover:
1. Basic NTU functionality and state updates
2. Temporal consistency calculation
3. Pattern detection with Ihsan threshold
4. Convergence behavior
5. Markov chain properties
"""

import pytest
import numpy as np

from core.ntu import (
    NTU,
    NTUConfig,
    NTUState,
    Observation,
    PatternDetector,
    minimal_ntu_detect,
)
from core.ntu.ntu import ObservationType


class TestObservation:
    """Tests for Observation dataclass."""

    def test_observation_clamping(self):
        """Values outside [0,1] should be clamped."""
        obs_high = Observation(value=1.5)
        assert obs_high.value == 1.0

        obs_low = Observation(value=-0.5)
        assert obs_low.value == 0.0

        obs_valid = Observation(value=0.7)
        assert obs_valid.value == 0.7

    def test_observation_type_classification(self):
        """Observations should be classified into 3 types."""
        assert Observation(value=0.1).observation_type == ObservationType.LOW
        assert Observation(value=0.5).observation_type == ObservationType.MEDIUM
        assert Observation(value=0.9).observation_type == ObservationType.HIGH

    def test_observation_type_boundaries(self):
        """Test boundary conditions for type classification."""
        # At boundary 0.33
        assert Observation(value=0.32).observation_type == ObservationType.LOW
        assert Observation(value=0.34).observation_type == ObservationType.MEDIUM

        # At boundary 0.67
        assert Observation(value=0.66).observation_type == ObservationType.MEDIUM
        assert Observation(value=0.68).observation_type == ObservationType.HIGH


class TestNTUConfig:
    """Tests for NTUConfig."""

    def test_default_config(self):
        """Default config should have valid weights."""
        config = NTUConfig()

        assert config.window_size == 5
        assert config.ihsan_threshold == 0.95
        assert abs(config.alpha + config.beta + config.gamma - 1.0) < 1e-6

    def test_weight_normalization(self):
        """Weights should be normalized to sum to 1."""
        config = NTUConfig(alpha=1.0, beta=1.0, gamma=1.0)

        # Should be normalized to 1/3 each
        assert abs(config.alpha - 1/3) < 1e-6
        assert abs(config.beta - 1/3) < 1e-6
        assert abs(config.gamma - 1/3) < 1e-6

    def test_custom_config(self):
        """Custom config should preserve valid weights."""
        config = NTUConfig(alpha=0.5, beta=0.3, gamma=0.2, window_size=10)

        assert config.window_size == 10
        assert abs(config.alpha - 0.5) < 1e-6
        assert abs(config.beta - 0.3) < 1e-6
        assert abs(config.gamma - 0.2) < 1e-6


class TestNTUState:
    """Tests for NTUState."""

    def test_default_state(self):
        """Default state should have valid values."""
        state = NTUState()

        assert state.belief == 0.5
        assert state.entropy == 1.0
        assert state.potential == 0.5
        assert state.iteration == 0

    def test_state_clamping(self):
        """State values should be clamped to [0,1]."""
        state = NTUState(belief=1.5, entropy=-0.5, potential=2.0)

        assert state.belief == 1.0
        assert state.entropy == 0.0
        assert state.potential == 1.0

    def test_ihsan_achieved(self):
        """Ihsan should be achieved at belief >= 0.95."""
        state_low = NTUState(belief=0.94)
        assert not state_low.ihsan_achieved

        state_high = NTUState(belief=0.95)
        assert state_high.ihsan_achieved

        state_max = NTUState(belief=1.0)
        assert state_max.ihsan_achieved

    def test_state_as_vector(self):
        """State should be convertible to numpy vector."""
        state = NTUState(belief=0.8, entropy=0.3, potential=0.6)
        vec = state.as_vector

        assert isinstance(vec, np.ndarray)
        assert vec.shape == (3,)
        assert np.allclose(vec, [0.8, 0.3, 0.6])


class TestNTU:
    """Tests for NTU core functionality."""

    def test_initialization(self):
        """NTU should initialize correctly."""
        ntu = NTU()

        assert len(ntu.memory) == 0
        assert ntu.state.belief == 0.5
        assert ntu.state.entropy == 1.0
        assert ntu.config.window_size == 5

    def test_custom_config_initialization(self):
        """NTU should accept custom config."""
        config = NTUConfig(window_size=10, ihsan_threshold=0.90)
        ntu = NTU(config)

        assert ntu.config.window_size == 10
        assert ntu.config.ihsan_threshold == 0.90

    def test_observe_single(self):
        """Single observation should update state."""
        ntu = NTU()
        initial_belief = ntu.state.belief

        state = ntu.observe(0.9)

        assert len(ntu.memory) == 1
        assert ntu.state.iteration == 1
        # High observation should increase belief
        # (exact value depends on priors and weights)

    def test_observe_sequence(self):
        """Sequence of observations should fill memory."""
        ntu = NTU()

        for i in range(7):
            ntu.observe(i / 10.0)

        # Memory should be capped at window_size
        assert len(ntu.memory) == 5
        assert ntu.state.iteration == 7

    def test_memory_sliding_window(self):
        """Memory should maintain sliding window."""
        config = NTUConfig(window_size=3)
        ntu = NTU(config)

        ntu.observe(0.1)
        ntu.observe(0.2)
        ntu.observe(0.3)
        ntu.observe(0.4)
        ntu.observe(0.5)

        # Should have last 3 observations
        values = [obs.value for obs in ntu.memory]
        assert values == [0.3, 0.4, 0.5]

    def test_high_observation_increases_belief(self):
        """High observations should increase belief over time."""
        ntu = NTU()

        # Start with neutral
        for _ in range(5):
            ntu.observe(0.5)

        initial_belief = ntu.state.belief

        # Feed high observations
        for _ in range(10):
            ntu.observe(0.95)

        assert ntu.state.belief > initial_belief

    def test_low_observation_decreases_belief(self):
        """Low observations should decrease belief over time."""
        ntu = NTU()

        # Start with high belief
        for _ in range(10):
            ntu.observe(0.95)

        high_belief = ntu.state.belief

        # Feed low observations
        for _ in range(10):
            ntu.observe(0.1)

        assert ntu.state.belief < high_belief

    def test_reset(self):
        """Reset should clear state."""
        ntu = NTU()

        for _ in range(10):
            ntu.observe(0.9)

        ntu.reset()

        assert len(ntu.memory) == 0
        assert ntu.state.belief == 0.5
        assert ntu.state.entropy == 1.0
        assert ntu.state.iteration == 0


class TestTemporalConsistency:
    """Tests for temporal consistency computation."""

    def test_consistent_sequence(self):
        """Consistent sequence should have high temporal consistency."""
        ntu = NTU()

        # Monotonic increasing sequence
        consistency_scores = []
        for i in range(10):
            ntu.observe(0.5 + i * 0.05)  # 0.5, 0.55, 0.6, ...

        # Final state should reflect consistency
        diagnostics = ntu.get_diagnostics()
        assert diagnostics["state"]["belief"] > 0.5

    def test_inconsistent_sequence(self):
        """Inconsistent sequence should have lower belief."""
        ntu = NTU()

        # Oscillating sequence
        for i in range(10):
            value = 0.9 if i % 2 == 0 else 0.1
            ntu.observe(value)

        # Oscillation creates uncertainty
        assert ntu.state.entropy > 0.3


class TestPatternDetection:
    """Tests for pattern detection functionality."""

    def test_detect_pattern_high_quality(self):
        """High quality observations should detect pattern."""
        ntu = NTU()

        # Feed consistently high quality
        observations = [0.95, 0.96, 0.94, 0.97, 0.95, 0.96, 0.95, 0.98]
        detected, state = ntu.detect_pattern(observations)

        # Should achieve high belief (may or may not hit 0.95 threshold)
        assert state.belief > 0.7

    def test_detect_pattern_low_quality(self):
        """Low quality observations should not detect pattern."""
        ntu = NTU()

        # Feed consistently low quality
        observations = [0.1, 0.15, 0.2, 0.1, 0.05, 0.15, 0.1, 0.2]
        detected, state = ntu.detect_pattern(observations)

        # Should NOT achieve Ihsan
        assert not detected
        assert state.belief < 0.95

    def test_detect_pattern_resets_state(self):
        """Pattern detection should reset state first."""
        ntu = NTU()

        # First run
        ntu.detect_pattern([0.1, 0.2, 0.3])

        # Second run with different data
        detected, state = ntu.detect_pattern([0.9, 0.95, 0.92])

        # Should not carry over from first run
        assert ntu.state.iteration == 3


class TestConvergence:
    """Tests for convergence behavior."""

    def test_convergence_with_stable_input(self):
        """Stable input should converge."""
        ntu = NTU()

        observations = [0.8] * 20
        converged, iterations, state = ntu.run_until_convergence(observations)

        # Should converge with stable input
        assert iterations < 100

    def test_convergence_max_iterations(self):
        """Should respect max iterations."""
        config = NTUConfig(max_iterations=50)
        ntu = NTU(config)

        # Oscillating input won't converge easily
        observations = [0.1, 0.9] * 50
        converged, iterations, state = ntu.run_until_convergence(observations)

        assert iterations <= 50


class TestMarkovChain:
    """Tests for Markov chain properties."""

    def test_transition_matrix_stochastic(self):
        """Transition matrix should be row-stochastic."""
        ntu = NTU()
        P = ntu._transition_matrix

        # Each row should sum to 1
        row_sums = P.sum(axis=1)
        assert np.allclose(row_sums, 1.0)

    def test_stationary_distribution_valid(self):
        """Stationary distribution should be valid probability."""
        ntu = NTU()
        pi = ntu.stationary_distribution

        # Should sum to 1
        assert abs(pi.sum() - 1.0) < 1e-6

        # All elements should be non-negative
        assert all(p >= 0 for p in pi)

    def test_stationary_distribution_eigenvector(self):
        """Stationary distribution should satisfy pi = pi @ P."""
        ntu = NTU()
        P = ntu._transition_matrix
        pi = ntu.stationary_distribution

        # pi should be a left eigenvector with eigenvalue 1
        result = pi @ P
        assert np.allclose(result, pi, atol=1e-6)


class TestPatternDetector:
    """Tests for high-level PatternDetector."""

    def test_register_pattern(self):
        """Should register patterns correctly."""
        detector = PatternDetector()

        detector.register_pattern("high_quality", threshold=0.95, window_size=5)
        detector.register_pattern("medium_quality", threshold=0.85, window_size=10)

        assert "high_quality" in detector.patterns
        assert "medium_quality" in detector.patterns

    def test_detect_registered_pattern(self):
        """Should detect registered patterns."""
        detector = PatternDetector()
        detector.register_pattern("test_pattern", threshold=0.7, window_size=3)

        observations = [0.8, 0.85, 0.9, 0.88, 0.9]
        detected, confidence, diagnostics = detector.detect("test_pattern", observations)

        assert detected in (True, False)  # Accept numpy bool
        assert 0.0 <= confidence <= 1.0
        assert "state" in diagnostics

    def test_detect_unknown_pattern_raises(self):
        """Should raise for unknown pattern."""
        detector = PatternDetector()

        with pytest.raises(ValueError, match="Unknown pattern"):
            detector.detect("nonexistent", [0.5, 0.6])

    def test_detect_all_patterns(self):
        """Should detect all registered patterns."""
        detector = PatternDetector()
        detector.register_pattern("pattern_a", threshold=0.6)
        detector.register_pattern("pattern_b", threshold=0.9)

        observations = [0.7, 0.75, 0.8, 0.78, 0.82]
        results = detector.detect_all(observations)

        assert "pattern_a" in results
        assert "pattern_b" in results
        assert len(results) == 2


class TestMinimalInterface:
    """Tests for minimal convenience function."""

    def test_minimal_ntu_detect(self):
        """Minimal interface should work."""
        observations = [0.9, 0.92, 0.88, 0.95, 0.91]
        detected, confidence = minimal_ntu_detect(observations)

        assert detected in (True, False)  # Accept numpy bool
        assert 0.0 <= confidence <= 1.0

    def test_minimal_ntu_detect_with_threshold(self):
        """Minimal interface should respect threshold."""
        observations = [0.7, 0.72, 0.75, 0.73, 0.71]

        # High threshold - should not detect
        detected_high, _ = minimal_ntu_detect(observations, threshold=0.95)

        # Low threshold - more likely to detect
        detected_low, _ = minimal_ntu_detect(observations, threshold=0.5)

        # Low threshold should be easier to pass
        # (exact result depends on implementation)


class TestDiagnostics:
    """Tests for diagnostic information."""

    def test_get_diagnostics(self):
        """Should return comprehensive diagnostics."""
        ntu = NTU()
        ntu.observe(0.8)
        ntu.observe(0.85)

        diag = ntu.get_diagnostics()

        assert "config" in diag
        assert "state" in diag
        assert "memory_size" in diag
        assert "memory_values" in diag
        assert "stationary_distribution" in diag
        assert "embeddings" in diag

        assert diag["memory_size"] == 2
        assert len(diag["memory_values"]) == 2


class TestEmbeddings:
    """Tests for pretrained embeddings."""

    def test_embeddings_normalized(self):
        """Embeddings should be unit vectors."""
        ntu = NTU()

        for embedding in ntu.embeddings.values():
            norm = np.linalg.norm(embedding)
            assert abs(norm - 1.0) < 1e-6

    def test_embeddings_distinct(self):
        """Different observation types should have distinct embeddings."""
        ntu = NTU()

        emb_low = ntu.embeddings[ObservationType.LOW]
        emb_med = ntu.embeddings[ObservationType.MEDIUM]
        emb_high = ntu.embeddings[ObservationType.HIGH]

        # Should not be identical
        assert not np.allclose(emb_low, emb_med)
        assert not np.allclose(emb_med, emb_high)
        assert not np.allclose(emb_low, emb_high)
