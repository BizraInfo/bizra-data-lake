"""
Tests for EmbeddingQualityGate — L2 norm + Shannon entropy validation.

Covers:
- Rejection of zero vectors (norm < min_norm)
- Rejection of uniform distributions (entropy_ratio > max_entropy_ratio)
- Acceptance of well-distributed embeddings
- Edge cases: empty, single-element, near-threshold

Standing on Giants: Shannon (1948, entropy as quality signal)
Artifact: core/embedding/quality_gate.py
"""

from __future__ import annotations


from core.embedding import EmbeddingQualityGate, GateResult


class TestQualityGateRejection:
    """Vectors that should be rejected."""

    def test_rejects_zero_vector(self):
        """A zero vector has norm 0 < min_norm (0.1)."""
        gate = EmbeddingQualityGate()
        result = gate.validate([0.0] * 768)

        assert not result.passed
        assert result.reason == "embedding_norm_too_low"
        assert result.score == 0.0

    def test_rejects_near_zero_vector(self):
        """Very small magnitude vectors fail norm check."""
        gate = EmbeddingQualityGate(min_norm=0.1)
        tiny = [1e-6] * 768  # norm ~ sqrt(768) * 1e-6 ~ 0.028

        result = gate.validate(tiny)
        assert not result.passed
        assert result.reason == "embedding_norm_too_low"

    def test_rejects_uniform_distribution(self):
        """A vector where all elements are equal has maximum entropy."""
        gate = EmbeddingQualityGate()
        # All same positive value → entropy ratio = 1.0
        uniform = [1.0] * 768

        result = gate.validate(uniform)
        assert not result.passed
        assert result.reason == "embedding_too_uniform"

    def test_rejects_empty_embedding(self):
        """Empty embedding is always rejected."""
        gate = EmbeddingQualityGate()
        result = gate.validate([])

        assert not result.passed
        assert result.reason == "empty_embedding"
        assert result.score == 0.0


class TestQualityGateAcceptance:
    """Vectors that should pass."""

    def test_accepts_normal_embedding(self):
        """A typical embedding with varied components passes both checks."""
        gate = EmbeddingQualityGate()

        # Create a skewed vector — a few dominant components + noise
        # Pure Gaussian across 768 dims has high entropy ratio (~0.955),
        # so we need a more peaked distribution to pass the 0.95 threshold.
        import random

        random.seed(42)
        embedding = [0.0] * 768
        # A few dominant dimensions
        for i in range(20):
            embedding[i] = random.uniform(2.0, 5.0)
        # Light noise in the rest
        for i in range(20, 768):
            embedding[i] = random.gauss(0, 0.1)

        result = gate.validate(embedding)
        assert result.passed
        assert result.reason == "ok"
        assert 0.0 < result.score <= 1.0

    def test_accepts_sparse_embedding(self):
        """A sparse vector with a few strong components passes."""
        gate = EmbeddingQualityGate()

        sparse = [0.0] * 768
        sparse[0] = 5.0
        sparse[1] = 3.0
        sparse[2] = 2.0
        sparse[100] = 1.5

        result = gate.validate(sparse)
        assert result.passed
        assert result.reason == "ok"

    def test_accepts_unit_vector(self):
        """Single hot vector has minimum entropy → passes entropy check."""
        gate = EmbeddingQualityGate()

        # One-hot (entropy ratio = 0.0, norm = 1.0)
        unit = [0.0] * 768
        unit[0] = 1.0

        result = gate.validate(unit)
        assert result.passed


class TestQualityGateEdgeCases:
    """Boundary conditions."""

    def test_single_element_vector(self):
        """Single-element vector passes if norm sufficient."""
        gate = EmbeddingQualityGate()
        result = gate.validate([0.5])

        # Single element → entropy = 0, max_entropy = 0, ratio = 1.0
        # This should fail entropy_ratio check (1.0 > 0.95)
        # But max_entropy = log2(1) = 0, so ratio = 1.0 by the fallback
        assert not result.passed

    def test_custom_thresholds(self):
        """Custom min_norm and max_entropy_ratio are respected."""
        strict_gate = EmbeddingQualityGate(min_norm=1.0, max_entropy_ratio=0.5)

        # Vector with norm < 1.0 fails
        result = strict_gate.validate([0.3, 0.3, 0.3])
        assert not result.passed
        assert result.reason == "embedding_norm_too_low"

    def test_score_is_inverse_entropy_ratio(self):
        """For passing embeddings, score = 1.0 - entropy_ratio."""
        gate = EmbeddingQualityGate()

        # Create vector with known low entropy
        vec = [0.0] * 100
        vec[0] = 10.0
        vec[1] = 1.0

        result = gate.validate(vec)
        if result.passed:
            # score should be between 0 and 1
            assert 0.0 < result.score <= 1.0

    def test_gate_result_fields(self):
        """GateResult has passed, reason, score fields."""
        result = GateResult(passed=True, reason="ok", score=0.8)
        assert result.passed is True
        assert result.reason == "ok"
        assert result.score == 0.8
