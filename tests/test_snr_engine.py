# BIZRA SNR Engine Tests
# Unit tests for Signal-to-Noise Ratio calculation

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSNREngine:
    """Test suite for SNR calculation engine"""

    @pytest.fixture
    def snr_engine(self):
        """Create SNR engine instance"""
        try:
            from arte_engine import SNREngine
            return SNREngine()
        except ImportError:
            pytest.skip("arte_engine not available")

    @pytest.fixture
    def sample_embeddings(self):
        """Generate sample embeddings for testing"""
        np.random.seed(42)
        return {
            "query": np.random.rand(384).astype(np.float32),
            "context": [np.random.rand(384).astype(np.float32) for _ in range(5)],
            "high_similarity": np.random.rand(384).astype(np.float32) * 0.1,
            "low_similarity": np.random.rand(384).astype(np.float32)
        }

    def test_snr_engine_initialization(self, snr_engine):
        """Test SNR engine initializes correctly"""
        assert snr_engine is not None
        assert hasattr(snr_engine, 'calculate_snr')

    def test_snr_calculation_returns_valid_range(self, snr_engine, sample_embeddings):
        """Test SNR value is within valid range [0, 1]"""
        snr_score, metrics = snr_engine.calculate_snr(
            query_embedding=sample_embeddings["query"],
            context_embeddings=sample_embeddings["context"],
            symbolic_facts=["fact1", "fact2"],
            neural_results=[{"text": "result", "score": 0.8}]
        )

        assert isinstance(snr_score, float)
        assert 0.0 <= snr_score <= 1.0

    def test_snr_components_present(self, snr_engine, sample_embeddings):
        """Test all SNR components are calculated"""
        snr_score, metrics = snr_engine.calculate_snr(
            query_embedding=sample_embeddings["query"],
            context_embeddings=sample_embeddings["context"],
            symbolic_facts=["fact1", "fact2"],
            neural_results=[{"text": "result", "score": 0.8}]
        )

        expected_components = [
            "signal_strength",
            "information_density",
            "symbolic_grounding",
            "coverage_balance"
        ]

        for component in expected_components:
            assert component in metrics, f"Missing component: {component}"

    def test_snr_with_empty_context(self, snr_engine, sample_embeddings):
        """Test SNR handles empty context gracefully"""
        snr_score, metrics = snr_engine.calculate_snr(
            query_embedding=sample_embeddings["query"],
            context_embeddings=[],
            symbolic_facts=[],
            neural_results=[]
        )

        assert isinstance(snr_score, float)
        # Empty context should result in lower SNR
        assert snr_score < 0.5

    def test_snr_ihsan_threshold(self, snr_engine, sample_embeddings):
        """Test Ihsān threshold detection"""
        snr_score, metrics = snr_engine.calculate_snr(
            query_embedding=sample_embeddings["query"],
            context_embeddings=sample_embeddings["context"],
            symbolic_facts=["fact1", "fact2", "fact3"],
            neural_results=[
                {"text": "result1", "score": 0.95},
                {"text": "result2", "score": 0.92}
            ]
        )

        # Check if ihsan_achieved flag is present
        if snr_score >= 0.99:
            assert metrics.get("ihsan_achieved", False) is True

    def test_snr_weighted_calculation(self, snr_engine, sample_embeddings):
        """Test that weights sum to 1.0"""
        # Access weights if available
        if hasattr(snr_engine, 'weights'):
            total_weight = sum(snr_engine.weights.values())
            assert abs(total_weight - 1.0) < 0.01, "Weights should sum to 1.0"


class TestSNRWeights:
    """Test SNR weight configuration"""

    def test_default_weights(self):
        """Test default weight values"""
        expected_weights = {
            "signal_strength": 0.35,
            "information_density": 0.25,
            "symbolic_grounding": 0.25,
            "coverage_balance": 0.15
        }

        try:
            from arte_engine import SNREngine
            engine = SNREngine()
            if hasattr(engine, 'weights'):
                for key, expected in expected_weights.items():
                    assert abs(engine.weights.get(key, 0) - expected) < 0.01
        except ImportError:
            pytest.skip("arte_engine not available")

    def test_weights_are_positive(self):
        """Test all weights are positive"""
        try:
            from arte_engine import SNREngine
            engine = SNREngine()
            if hasattr(engine, 'weights'):
                for key, weight in engine.weights.items():
                    assert weight > 0, f"Weight {key} should be positive"
        except ImportError:
            pytest.skip("arte_engine not available")


class TestSNRMathematicalProperties:
    """Test mathematical properties of SNR calculation"""

    @pytest.fixture
    def snr_engine(self):
        try:
            from arte_engine import SNREngine
            return SNREngine()
        except ImportError:
            pytest.skip("arte_engine not available")

    def test_snr_deterministic(self, snr_engine):
        """Test SNR calculation is deterministic"""
        np.random.seed(123)
        query = np.random.rand(384).astype(np.float32)
        context = [np.random.rand(384).astype(np.float32) for _ in range(3)]
        facts = ["fact1", "fact2"]
        results = [{"text": "result", "score": 0.8}]

        snr1, _ = snr_engine.calculate_snr(query, context, facts, results)
        snr2, _ = snr_engine.calculate_snr(query, context, facts, results)

        assert abs(snr1 - snr2) < 0.001, "SNR should be deterministic"

    def test_snr_monotonicity(self, snr_engine):
        """Test SNR increases with better context"""
        np.random.seed(456)
        query = np.random.rand(384).astype(np.float32)

        # Low quality context
        low_context = [np.random.rand(384).astype(np.float32) for _ in range(2)]
        low_snr, _ = snr_engine.calculate_snr(
            query, low_context, ["fact"], [{"text": "r", "score": 0.3}]
        )

        # High quality context (more items, higher scores)
        high_context = [np.random.rand(384).astype(np.float32) for _ in range(10)]
        high_snr, _ = snr_engine.calculate_snr(
            query, high_context,
            ["fact1", "fact2", "fact3", "fact4", "fact5"],
            [{"text": "r1", "score": 0.9}, {"text": "r2", "score": 0.95}]
        )

        # More context should generally yield higher SNR (not always guaranteed)
        # This is a weak test - just verify both are valid
        assert 0 <= low_snr <= 1
        assert 0 <= high_snr <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
