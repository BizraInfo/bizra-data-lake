"""Tests for HNSW vector index."""

from __future__ import annotations

import numpy as np
import pytest

from core.memory.config import HNSWConfig
from core.memory.hnsw_index import HNSWIndex

from .conftest import random_embedding


@pytest.fixture
def hnsw(small_hnsw_config):
    idx = HNSWIndex(small_hnsw_config)
    idx.initialize()
    return idx


class TestHNSWBasic:
    def test_init_empty(self, hnsw):
        assert hnsw.count == 0

    def test_add_and_count(self, hnsw):
        hnsw.add("r1", random_embedding(8))
        assert hnsw.count == 1

    def test_add_multiple(self, hnsw):
        for i in range(10):
            hnsw.add(f"r{i}", random_embedding(8))
        assert hnsw.count == 10

    def test_add_duplicate_updates(self, hnsw):
        new_vec = random_embedding(8)
        hnsw.add("r1", random_embedding(8))
        hnsw.add("r1", new_vec)
        # After re-add, searching for the new vector should find r1
        # Add a second vector so search has enough valid entries
        hnsw.add("r2", random_embedding(8))
        results = hnsw.search(new_vec, top_k=2)
        result_ids = [r[0] for r in results]
        assert "r1" in result_ids

    def test_wrong_dimension_raises(self, hnsw):
        with pytest.raises(ValueError, match="dim"):
            hnsw.add("r1", [1.0, 2.0, 3.0])  # dim=3 != dim=8


class TestHNSWSearch:
    def test_search_finds_nearest(self, hnsw):
        target = random_embedding(8)
        hnsw.add("target", target)

        # Add dissimilar vectors
        for i in range(5):
            hnsw.add(f"noise_{i}", random_embedding(8))

        results = hnsw.search(target, top_k=1)
        assert len(results) == 1
        assert results[0][0] == "target"

    def test_search_returns_distances(self, hnsw):
        vec = random_embedding(8)
        hnsw.add("r1", vec)
        results = hnsw.search(vec, top_k=1)
        # Same vector should have distance ~0
        assert results[0][1] < 0.01

    def test_search_top_k(self, hnsw):
        for i in range(20):
            hnsw.add(f"r{i}", random_embedding(8))
        results = hnsw.search(random_embedding(8), top_k=5)
        assert len(results) == 5

    def test_search_empty_index(self, hnsw):
        results = hnsw.search(random_embedding(8), top_k=5)
        assert results == []

    def test_search_top_k_larger_than_count(self, hnsw):
        hnsw.add("r1", random_embedding(8))
        hnsw.add("r2", random_embedding(8))
        results = hnsw.search(random_embedding(8), top_k=10)
        assert len(results) == 2


class TestHNSWRemove:
    def test_remove_existing(self, hnsw):
        hnsw.add("r1", random_embedding(8))
        assert hnsw.remove("r1") is True

    def test_remove_nonexistent(self, hnsw):
        assert hnsw.remove("nonexistent") is False

    def test_removed_not_in_search(self, hnsw):
        vec = random_embedding(8)
        hnsw.add("r1", vec)
        hnsw.add("r2", random_embedding(8))
        hnsw.remove("r1")
        results = hnsw.search(vec, top_k=10)
        result_ids = [r[0] for r in results]
        assert "r1" not in result_ids


class TestHNSWSaveLoad:
    def test_save_and_load(self, small_hnsw_config, tmp_path):
        idx = HNSWIndex(small_hnsw_config)
        idx.initialize()

        vecs = {}
        for i in range(5):
            vec = random_embedding(8)
            idx.add(f"r{i}", vec)
            vecs[f"r{i}"] = vec

        path = tmp_path / "test.index"
        idx.save(path)

        # Load into new index
        idx2 = HNSWIndex(small_hnsw_config)
        assert idx2.load(path) is True
        assert idx2.count == 5

        # Verify search still works
        results = idx2.search(vecs["r0"], top_k=1)
        assert results[0][0] == "r0"

    def test_load_nonexistent(self, small_hnsw_config, tmp_path):
        idx = HNSWIndex(small_hnsw_config)
        assert idx.load(tmp_path / "nonexistent.index") is False


class TestHNSWClear:
    def test_clear_resets(self, hnsw):
        hnsw.add("r1", random_embedding(8))
        hnsw.clear()
        assert hnsw.count == 0
