"""
Tests for Scalar Quantization + HNSW Auto-Tuning.

Validates 4x memory reduction, search accuracy preservation,
calibration, persistence, and parameter auto-tuning.

Standing on Giants:
- Guo et al. (2020): Scalar quantization for ANN
- Malkov & Yashunin (2016): HNSW parameter guidance
"""

from pathlib import Path
from typing import List

import numpy as np
import pytest

from core.memory.config import HNSWConfig
from core.memory.hnsw_index import HNSWIndex, ScalarQuantizer

# ── Helpers ──────────────────────────────────────────────────────────


def _random_vectors(n: int, dim: int = 384, seed: int = 42) -> np.ndarray:
    """Generate reproducible random float32 vectors."""
    rng = np.random.RandomState(seed)
    return rng.randn(n, dim).astype(np.float32)


def _populate_index(
    index: HNSWIndex, vectors: np.ndarray, prefix: str = "rec"
) -> List[str]:
    """Add vectors to index, return record IDs."""
    ids = []
    for i, vec in enumerate(vectors):
        rid = f"{prefix}-{i:04d}"
        index.add(rid, vec)
        ids.append(rid)
    return ids


# ── ScalarQuantizer Unit Tests ───────────────────────────────────────


class TestScalarQuantizer:
    """Unit tests for the ScalarQuantizer class."""

    def test_calibration_from_vectors(self):
        sq = ScalarQuantizer(dimensions=4, calibration_size=5)
        assert not sq.calibrated
        vecs = np.array(
            [
                [0.0, 1.0, -1.0, 0.5],
                [0.5, 0.0, 0.0, 1.0],
                [1.0, -1.0, 1.0, 0.0],
                [-0.5, 0.5, 0.5, -0.5],
                [0.2, 0.3, -0.3, 0.8],
            ],
            dtype=np.float32,
        )
        for v in vecs[:-1]:
            assert not sq.add_calibration(v)
        assert sq.add_calibration(vecs[-1])
        assert sq.calibrated

    def test_quantize_dequantize_roundtrip(self):
        sq = ScalarQuantizer(dimensions=4, calibration_size=3)
        vecs = np.array(
            [[0.0, 1.0, -1.0, 0.5], [1.0, 0.0, 0.0, -0.5], [-1.0, 0.5, 1.0, 0.0]],
            dtype=np.float32,
        )
        for v in vecs:
            sq.add_calibration(v)
        assert sq.calibrated

        original = np.array([0.5, 0.3, -0.5, 0.2], dtype=np.float32)
        quantized = sq.quantize(original)
        assert quantized.dtype == np.uint8
        dequantized = sq.dequantize(quantized)
        # Roundtrip error should be small (< 1% per dimension for 8-bit)
        max_error = np.max(np.abs(original - dequantized))
        assert max_error < 0.02, f"Max roundtrip error {max_error} exceeds 0.02"

    def test_quantize_batch(self):
        sq = ScalarQuantizer(dimensions=4, calibration_size=2)
        sq.add_calibration(np.array([-1, -1, -1, -1], dtype=np.float32))
        sq.add_calibration(np.array([1, 1, 1, 1], dtype=np.float32))
        batch = np.array([[0, 0, 0, 0], [0.5, -0.5, 0.5, -0.5]], dtype=np.float32)
        q_batch = sq.quantize_batch(batch)
        assert q_batch.shape == batch.shape
        assert q_batch.dtype == np.uint8

    def test_memory_ratio(self):
        sq = ScalarQuantizer(dimensions=384, calibration_size=2)
        assert sq.memory_ratio == 1.0
        sq.add_calibration(np.zeros(384, dtype=np.float32))
        sq.add_calibration(np.ones(384, dtype=np.float32))
        assert sq.memory_ratio == 4.0

    def test_force_calibrate(self):
        sq = ScalarQuantizer(dimensions=4, calibration_size=100)
        sq.add_calibration(np.array([0, 0, 0, 0], dtype=np.float32))
        sq.add_calibration(np.array([1, 1, 1, 1], dtype=np.float32))
        assert not sq.calibrated
        sq.force_calibrate()
        assert sq.calibrated

    def test_state_dict_roundtrip(self):
        sq = ScalarQuantizer(dimensions=4, calibration_size=2)
        sq.add_calibration(np.array([-1, 0, 0.5, -0.5], dtype=np.float32))
        sq.add_calibration(np.array([1, 2, 1.5, 0.5], dtype=np.float32))
        state = sq.state_dict()

        sq2 = ScalarQuantizer.from_state_dict(state)
        assert sq2.calibrated
        vec = np.array([0.5, 1.0, 1.0, 0.0], dtype=np.float32)
        np.testing.assert_array_equal(sq.quantize(vec), sq2.quantize(vec))

    def test_clamping_out_of_range(self):
        sq = ScalarQuantizer(dimensions=2, calibration_size=2)
        sq.add_calibration(np.array([0.0, 0.0], dtype=np.float32))
        sq.add_calibration(np.array([1.0, 1.0], dtype=np.float32))
        # Values outside [0, 1] should be clamped
        q = sq.quantize(np.array([2.0, -1.0], dtype=np.float32))
        assert q[0] == 255  # clamped to max
        assert q[1] == 0  # clamped to min


# ── HNSWIndex Quantization Integration ──────────────────────────────


class TestQuantizedIndex:
    """Integration tests for quantized numpy fallback."""

    @pytest.fixture
    def q_config(self) -> HNSWConfig:
        return HNSWConfig(
            dimensions=32,
            space="cosine",
            quantize=True,
            quantize_calibration_size=10,
        )

    @pytest.fixture
    def q_index(self, q_config: HNSWConfig) -> HNSWIndex:
        idx = HNSWIndex(q_config)
        idx._use_hnswlib = False  # Force numpy fallback
        idx.initialize()
        return idx

    def test_quantizer_created_when_enabled(self, q_index: HNSWIndex):
        assert q_index._quantizer is not None
        assert not q_index._quantizer.calibrated

    def test_calibration_after_threshold_vectors(self, q_index: HNSWIndex):
        vecs = _random_vectors(15, dim=32)
        _populate_index(q_index, vecs)
        assert q_index._quantizer.calibrated
        assert len(q_index._quantized_vectors) == 15

    def test_search_accuracy_with_quantization(self, q_index: HNSWIndex):
        """Quantized search should return similar top-k as full precision."""
        vecs = _random_vectors(50, dim=32)
        _populate_index(q_index, vecs)

        # Also build a non-quantized index for comparison
        nq_config = HNSWConfig(dimensions=32, space="cosine", quantize=False)
        nq_index = HNSWIndex(nq_config)
        nq_index._use_hnswlib = False
        nq_index.initialize()
        _populate_index(nq_index, vecs)

        query = _random_vectors(1, dim=32, seed=99)[0]
        q_results = q_index.search(query, top_k=5)
        nq_results = nq_index.search(query, top_k=5)

        # At least 3/5 of top-5 should overlap (<2% accuracy loss at scale,
        # but 32-dim test vectors have higher quantization noise)
        q_ids = {r[0] for r in q_results}
        nq_ids = {r[0] for r in nq_results}
        overlap = len(q_ids & nq_ids)
        assert overlap >= 3, f"Only {overlap}/5 overlap between quantized and full"

    def test_search_before_calibration(self, q_index: HNSWIndex):
        """Search works before calibration (falls back to full precision)."""
        vecs = _random_vectors(5, dim=32)
        _populate_index(q_index, vecs[:5])
        assert not q_index._quantizer.calibrated
        results = q_index.search(vecs[0], top_k=3)
        assert len(results) > 0

    def test_remove_cleans_quantized(self, q_index: HNSWIndex):
        vecs = _random_vectors(15, dim=32)
        _populate_index(q_index, vecs)
        assert "rec-0005" in q_index._quantized_vectors
        q_index.remove("rec-0005")
        assert "rec-0005" not in q_index._quantized_vectors
        assert "rec-0005" not in q_index._fallback_vectors

    def test_quantization_stats(self, q_index: HNSWIndex):
        vecs = _random_vectors(15, dim=32)
        _populate_index(q_index, vecs)
        stats = q_index.quantization_stats
        assert stats["enabled"] is True
        assert stats["calibrated"] is True
        assert stats["quantized_count"] == 15
        assert stats["memory_ratio"] == 4.0
        assert stats["uint8_bytes"] < stats["float32_bytes"]

    def test_quantization_disabled_stats(self):
        config = HNSWConfig(dimensions=32, quantize=False)
        idx = HNSWIndex(config)
        idx._use_hnswlib = False
        stats = idx.quantization_stats
        assert stats["enabled"] is False

    def test_memory_savings(self, q_index: HNSWIndex):
        """Verify actual 4x memory reduction."""
        vecs = _random_vectors(100, dim=32)
        _populate_index(q_index, vecs)
        stats = q_index.quantization_stats
        float_bytes = stats["float32_bytes"]
        uint8_bytes = stats["uint8_bytes"]
        ratio = float_bytes / uint8_bytes
        assert ratio == pytest.approx(4.0, rel=0.01)

    def test_clear_resets_quantizer(self, q_index: HNSWIndex):
        vecs = _random_vectors(15, dim=32)
        _populate_index(q_index, vecs)
        assert q_index._quantizer.calibrated
        q_index.clear()
        assert not q_index._quantizer.calibrated
        assert len(q_index._quantized_vectors) == 0


# ── Persistence with Quantization ────────────────────────────────────


class TestQuantizedPersistence:
    """Test save/load with quantization state."""

    def test_save_load_roundtrip(self, tmp_path: Path):
        config = HNSWConfig(dimensions=16, quantize=True, quantize_calibration_size=5)
        idx = HNSWIndex(config)
        idx._use_hnswlib = False
        idx.initialize()

        vecs = _random_vectors(20, dim=16)
        _populate_index(idx, vecs)

        # Save
        save_path = tmp_path / "test_index.bin"
        idx.save(save_path)

        # Load into new index
        idx2 = HNSWIndex(config)
        idx2._use_hnswlib = False
        loaded = idx2.load(save_path)
        assert loaded

        # Verify quantizer restored
        assert idx2._quantizer is not None
        assert idx2._quantizer.calibrated
        assert len(idx2._quantized_vectors) == 20

        # Verify search returns same results
        query = vecs[0]
        r1 = idx.search(query, top_k=5)
        r2 = idx2.search(query, top_k=5)
        assert [r[0] for r in r1] == [r[0] for r in r2]

    def test_load_without_quantize_flag_ignores_qdata(self, tmp_path: Path):
        """If saved with quantization but loaded without, it still works."""
        config_q = HNSWConfig(dimensions=16, quantize=True, quantize_calibration_size=5)
        idx = HNSWIndex(config_q)
        idx._use_hnswlib = False
        idx.initialize()
        _populate_index(idx, _random_vectors(20, dim=16))
        idx.save(tmp_path / "test.bin")

        config_nq = HNSWConfig(dimensions=16, quantize=False)
        idx2 = HNSWIndex(config_nq)
        idx2._use_hnswlib = False
        assert idx2.load(tmp_path / "test.bin")
        assert idx2._quantizer is None
        assert idx2.count == 20


# ── Auto-Tuning ─────────────────────────────────────────────────────


class TestAutoTuning:
    """Test HNSW parameter auto-tuning."""

    def test_small_dataset_params(self):
        config = HNSWConfig(max_elements=5_000, auto_tune=True)
        idx = HNSWIndex(config)
        idx._use_hnswlib = False
        idx.initialize()
        assert config.m == 8
        assert config.ef_construction == 100
        assert config.ef_search == 50

    def test_medium_dataset_params(self):
        config = HNSWConfig(max_elements=50_000, auto_tune=True)
        idx = HNSWIndex(config)
        idx._use_hnswlib = False
        idx.initialize()
        assert config.m == 16
        assert config.ef_construction == 200
        assert config.ef_search == 100

    def test_large_dataset_params(self):
        config = HNSWConfig(max_elements=500_000, auto_tune=True)
        idx = HNSWIndex(config)
        idx._use_hnswlib = False
        idx.initialize()
        assert config.m == 32
        assert config.ef_construction == 300
        assert config.ef_search == 150

    def test_auto_tune_disabled_preserves_manual(self):
        config = HNSWConfig(m=24, ef_construction=250, auto_tune=False)
        idx = HNSWIndex(config)
        idx._use_hnswlib = False
        idx.initialize()
        assert config.m == 24
        assert config.ef_construction == 250


# ── Multi-Space Quantized Search ─────────────────────────────────────


class TestQuantizedMultiSpace:
    """Quantized search works correctly across different spaces."""

    @pytest.fixture(params=["cosine", "l2", "ip"])
    def space_index(self, request) -> HNSWIndex:
        config = HNSWConfig(
            dimensions=16,
            space=request.param,
            quantize=True,
            quantize_calibration_size=5,
        )
        idx = HNSWIndex(config)
        idx._use_hnswlib = False
        idx.initialize()
        vecs = _random_vectors(20, dim=16)
        _populate_index(idx, vecs)
        return idx

    def test_quantized_search_returns_results(self, space_index: HNSWIndex):
        query = _random_vectors(1, dim=16, seed=99)[0]
        results = space_index.search(query, top_k=5)
        assert len(results) == 5
        # Distances should be ordered (ascending)
        dists = [r[1] for r in results]
        assert dists == sorted(dists)
