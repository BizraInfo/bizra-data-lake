"""Tests for BIZRA Reflex Cache — O(1) HashMap with precipitation."""

import os
import sys
import json
import time
import tempfile
import threading
import pytest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reflex_cache import ReflexCache, ReflexEntry, PrecipitationCandidate, CacheStats


@pytest.fixture
def cache():
    return ReflexCache(max_entries=10)


@pytest.fixture
def cache_with_path(tmp_path):
    return ReflexCache(max_entries=10, persistence_path=tmp_path / "cache.json")


def _observe(cache, text="hello world", ihsan=0.92, n=1):
    """Helper: record n observations of the same input."""
    result = None
    for _ in range(n):
        result = cache.record_observation(
            input_text=text,
            output_text=f"Response to: {text}",
            ihsan_composite=ihsan,
            ihsan_tensor={
                "moral_clarity": 0.95, "epistemic_humility": 0.90,
                "structural_integrity": 0.92, "verifiability": 0.91,
                "intent_alignment": 0.93, "resilience": 0.88,
            },
        )
    return result


class TestLookup:
    def test_miss_on_empty_cache(self, cache):
        assert cache.lookup("anything") is None

    def test_miss_increments_counter(self, cache):
        cache.lookup("miss")
        assert cache.stats.cache_misses == 1
        assert cache.stats.total_lookups == 1

    def test_hit_after_precipitation(self, cache):
        _observe(cache, "test query", ihsan=0.95, n=3)
        entry = cache.lookup("test query")
        assert entry is not None
        assert entry.hit_count == 1

    def test_hit_increments_counter(self, cache):
        _observe(cache, "q", ihsan=0.95, n=3)
        cache.lookup("q")
        cache.lookup("q")
        assert cache.stats.cache_hits == 2

    def test_stale_entry_returns_none(self, cache):
        _observe(cache, "stale", ihsan=0.95, n=3)
        pattern_hash = cache._hash_input("stale")
        cache.invalidate(pattern_hash)
        assert cache.lookup("stale") is None

    def test_case_insensitive_matching(self, cache):
        _observe(cache, "Hello World", ihsan=0.95, n=3)
        entry = cache.lookup("hello world")
        assert entry is not None

    def test_whitespace_normalization(self, cache):
        _observe(cache, "  hello   world  ", ihsan=0.95, n=3)
        entry = cache.lookup("hello world")
        assert entry is not None


class TestPrecipitation:
    def test_no_precipitation_below_threshold(self, cache):
        result = _observe(cache, "low quality", ihsan=0.50, n=3)
        assert result is None
        assert cache.size == 0

    def test_no_precipitation_with_two_hits(self, cache):
        result = _observe(cache, "two hits", ihsan=0.95, n=2)
        assert result is None
        assert cache.size == 0

    def test_precipitation_at_three_hits(self, cache):
        result = _observe(cache, "three hits", ihsan=0.95, n=3)
        assert result is not None
        assert isinstance(result, ReflexEntry)
        assert cache.size == 1

    def test_precipitated_entry_has_correct_ihsan(self, cache):
        _observe(cache, "quality", ihsan=0.95, n=3)
        entry = cache.lookup("quality")
        assert entry.ihsan_composite == 0.95

    def test_best_output_selected(self, cache):
        cache.record_observation("pick best", "low output", 0.91, {"a": 0.91})
        cache.record_observation("pick best", "high output", 0.98, {"a": 0.98})
        result = cache.record_observation("pick best", "mid output", 0.95, {"a": 0.95})
        assert result is not None
        assert result.ihsan_composite == 0.98
        assert "high output" in result.output_template

    def test_precipitation_clears_candidate(self, cache):
        _observe(cache, "cleared", ihsan=0.95, n=3)
        assert "cleared" not in str(cache._candidates)

    def test_stats_track_precipitations(self, cache):
        _observe(cache, "stat track", ihsan=0.95, n=3)
        assert cache.stats.precipitations == 1


class TestInvalidation:
    def test_invalidate_existing_entry(self, cache):
        _observe(cache, "inv test", ihsan=0.95, n=3)
        ph = cache._hash_input("inv test")
        assert cache.invalidate(ph) is True
        assert cache.stats.invalidations == 1

    def test_invalidate_nonexistent_returns_false(self, cache):
        assert cache.invalidate("nonexistent_hash") is False

    def test_validate_entry_within_delta(self, cache):
        _observe(cache, "valid", ihsan=0.95, n=3)
        ph = cache._hash_input("valid")
        assert cache.validate_entry(ph, 0.94) is True  # delta=0.01 < 0.05

    def test_validate_entry_beyond_delta(self, cache):
        _observe(cache, "drift", ihsan=0.95, n=3)
        ph = cache._hash_input("drift")
        assert cache.validate_entry(ph, 0.80) is False  # delta=0.15 > 0.05

    def test_validation_resets_hit_counter(self, cache):
        _observe(cache, "reset", ihsan=0.95, n=3)
        ph = cache._hash_input("reset")
        entry = cache._cache[ph]
        entry.validation_hits_since = 50
        cache.validate_entry(ph, 0.94)
        assert entry.validation_hits_since == 0


class TestLRUEviction:
    def test_evicts_oldest_at_capacity(self, cache):
        for i in range(12):  # max_entries=10
            _observe(cache, f"entry-{i}", ihsan=0.95, n=3)
        assert cache.size == 10
        assert cache.stats.evictions == 2

    def test_lru_order_updated_on_lookup(self, cache):
        for i in range(10):
            _observe(cache, f"lru-{i}", ihsan=0.95, n=3)
        # Access oldest to move it to front
        cache.lookup("lru-0")
        # Add one more to trigger eviction
        _observe(cache, "lru-new", ihsan=0.95, n=3)
        # lru-0 should survive (was accessed), lru-1 should be evicted
        assert cache.lookup("lru-0") is not None
        assert cache.lookup("lru-1") is None


class TestPersistence:
    def test_save_and_load(self, tmp_path):
        path = tmp_path / "persist.json"
        c1 = ReflexCache(max_entries=10, persistence_path=path)
        _observe(c1, "persist me", ihsan=0.95, n=3)
        c1.save_to_disk()
        assert path.exists()

        c2 = ReflexCache(max_entries=10, persistence_path=path)
        assert c2.size == 1
        entry = c2.lookup("persist me")
        assert entry is not None

    def test_corrupt_file_handled(self, tmp_path):
        path = tmp_path / "corrupt.json"
        path.write_text("{invalid json")
        c = ReflexCache(max_entries=10, persistence_path=path)
        assert c.size == 0  # Should recover gracefully


class TestThreadSafety:
    def test_concurrent_lookups(self, cache):
        _observe(cache, "concurrent", ihsan=0.95, n=3)
        errors = []

        def worker():
            try:
                for _ in range(100):
                    cache.lookup("concurrent")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert cache.stats.cache_hits == 500

    def test_concurrent_observations(self, cache):
        errors = []

        def worker(idx):
            try:
                _observe(cache, f"thread-{idx}", ihsan=0.95, n=3)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert cache.size == 5


class TestIntrospection:
    def test_entries_needing_validation(self, cache):
        _observe(cache, "needs-val", ihsan=0.95, n=3)
        entry = cache._cache[cache._hash_input("needs-val")]
        entry.validation_hits_since = 200
        needing = cache.entries_needing_validation()
        assert len(needing) == 1

    def test_get_all_entries(self, cache):
        _observe(cache, "all-1", ihsan=0.95, n=3)
        _observe(cache, "all-2", ihsan=0.95, n=3)
        assert len(cache.get_all_entries()) == 2

    def test_clear(self, cache):
        _observe(cache, "clear me", ihsan=0.95, n=3)
        cache.clear()
        assert cache.size == 0

    def test_stats_as_dict(self, cache):
        d = cache.stats.as_dict()
        assert "hit_rate" in d
        assert "precipitations" in d
