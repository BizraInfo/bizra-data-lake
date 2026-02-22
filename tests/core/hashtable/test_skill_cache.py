"""
Tests for SkillCache — Phase 44

Standing on Giants: Kahneman (2011), Anderson (1982)
"""

import threading
import time
from unittest.mock import patch

import pytest

from core.hashtable.skill_cache import (
    CachedSkillResult,
    SkillCache,
    TemporalGranularityPolicy,
)


@pytest.fixture
def cache():
    """Small cache for testing."""
    return SkillCache(max_size=4, default_ttl=60, ihsan_floor=0.90)


class TestSkillCachePutGet:
    """Basic put/get semantics."""

    def test_put_and_get(self, cache):
        cache.put("key1", {"answer": 42}, snr_score=0.95)
        result = cache.get("key1")
        assert result is not None
        assert result.result == {"answer": 42}
        assert result.snr_score == 0.95

    def test_get_miss(self, cache):
        assert cache.get("nonexistent") is None

    def test_put_updates_existing(self, cache):
        cache.put("k", {"v": 1}, snr_score=0.95)
        cache.put("k", {"v": 2}, snr_score=0.96)
        result = cache.get("k")
        assert result is not None
        assert result.result == {"v": 2}
        assert result.snr_score == 0.96

    def test_query_pattern_stored(self, cache):
        cache.put("k", {"v": 1}, snr_score=0.95, query_pattern="test-pattern")
        result = cache.get("k")
        assert result is not None
        assert result.query_pattern == "test-pattern"


class TestSkillCacheLRUEviction:
    """LRU eviction when max_size exceeded."""

    def test_evicts_oldest(self, cache):
        # Fill to capacity (4)
        for i in range(4):
            cache.put(f"key-{i}", {"i": i}, snr_score=0.95)

        # Add 5th — should evict key-0
        cache.put("key-4", {"i": 4}, snr_score=0.95)
        assert cache.get("key-0") is None
        assert cache.get("key-4") is not None

    def test_access_refreshes_position(self, cache):
        for i in range(4):
            cache.put(f"key-{i}", {"i": i}, snr_score=0.95)

        # Access key-0 to move it to end
        cache.get("key-0")

        # Add 5th — should evict key-1 (now oldest)
        cache.put("key-4", {"i": 4}, snr_score=0.95)
        assert cache.get("key-0") is not None
        assert cache.get("key-1") is None


class TestSkillCacheTTL:
    """TTL expiry."""

    def test_expired_entry_returns_none(self):
        cache = SkillCache(max_size=10, default_ttl=1, ihsan_floor=0.90)
        cache.put("k", {"v": 1}, snr_score=0.95)

        # Manually expire by patching time.monotonic
        entry = cache._cache["k"]
        entry.created_at = time.monotonic() - 100  # 100 seconds ago

        assert cache.get("k") is None

    def test_custom_ttl(self):
        cache = SkillCache(max_size=10, default_ttl=3600, ihsan_floor=0.90)
        cache.put("k", {"v": 1}, snr_score=0.95, ttl=1)

        entry = cache._cache["k"]
        entry.created_at = time.monotonic() - 2

        assert cache.get("k") is None

    def test_non_expired_entry_returns(self):
        cache = SkillCache(max_size=10, default_ttl=3600, ihsan_floor=0.90)
        cache.put("k", {"v": 1}, snr_score=0.95)
        assert cache.get("k") is not None

    def test_hhmm_lower_layer_gets_shorter_ttl_than_upper(self):
        cache = SkillCache(max_size=10, default_ttl=100, ihsan_floor=0.90)

        cache.put("low", {"v": 1}, snr_score=0.95, hhmm_layer=0)
        cache.put("high", {"v": 2}, snr_score=0.95, hhmm_layer=4)

        low_ttl = cache._cache["low"].ttl_seconds
        high_ttl = cache._cache["high"].ttl_seconds

        assert low_ttl < high_ttl

    def test_explicit_ttl_overrides_hhmm_layer(self):
        cache = SkillCache(max_size=10, default_ttl=100, ihsan_floor=0.90)
        cache.put("k", {"v": 1}, snr_score=0.95, ttl=7, hhmm_layer=4)
        assert cache._cache["k"].ttl_seconds == 7


class TestSkillCacheIhsanFloor:
    """Ihsan quality gate."""

    def test_below_floor_rejected_on_put(self, cache):
        # Floor is 0.90 in fixture
        cache.put("k", {"v": 1}, snr_score=0.80)
        assert cache.get("k") is None
        assert len(cache) == 0

    def test_at_floor_accepted(self, cache):
        cache.put("k", {"v": 1}, snr_score=0.90)
        assert cache.get("k") is not None

    def test_below_floor_evicted_on_get(self):
        cache = SkillCache(max_size=10, default_ttl=3600, ihsan_floor=0.95)
        # Bypass floor check by setting a passing score initially
        cache.put("k", {"v": 1}, snr_score=0.96)

        # Lower the score in the internal entry
        cache._cache["k"].snr_score = 0.50

        # Get should evict it
        assert cache.get("k") is None


class TestSkillCacheStructuralHash:
    """Deterministic structural hashing of thought chains."""

    def test_deterministic(self):
        cache = SkillCache()
        chain = [{"type": "observe", "data": "x"}, {"type": "infer", "data": "y"}]
        h1 = cache.structural_hash(chain)
        h2 = cache.structural_hash(chain)
        assert h1 == h2

    def test_different_chains_different_hashes(self):
        cache = SkillCache()
        chain_a = [{"type": "observe"}]
        chain_b = [{"type": "infer"}]
        assert cache.structural_hash(chain_a) != cache.structural_hash(chain_b)

    def test_hash_is_16_hex_chars(self):
        cache = SkillCache()
        h = cache.structural_hash([{"type": "test"}])
        assert len(h) == 16
        # Valid hex
        int(h, 16)

    def test_structural_hash_preserves_order(self):
        """Fix 1 from plan: list order matters for thought chains."""
        cache = SkillCache()
        chain_a = [{"type": "observe"}, {"type": "hypothesize"}, {"type": "test"}]
        chain_b = [{"type": "test"}, {"type": "hypothesize"}, {"type": "observe"}]
        assert cache.structural_hash(chain_a) != cache.structural_hash(chain_b)

    def test_empty_chain(self):
        cache = SkillCache()
        h = cache.structural_hash([])
        assert len(h) == 16


class TestSkillCacheStats:
    """Statistics tracking."""

    def test_initial_stats(self, cache):
        stats = cache.stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["evictions"] == 0
        assert stats["size"] == 0
        assert stats["fill_ratio"] == 0.0

    def test_hit_tracking(self, cache):
        cache.put("k", {"v": 1}, snr_score=0.95)
        cache.get("k")
        cache.get("k")
        stats = cache.stats()
        assert stats["hits"] == 2

    def test_miss_tracking(self, cache):
        cache.get("nonexistent")
        stats = cache.stats()
        assert stats["misses"] == 1

    def test_eviction_tracking(self, cache):
        for i in range(5):
            cache.put(f"key-{i}", {"i": i}, snr_score=0.95)
        stats = cache.stats()
        assert stats["evictions"] >= 1

    def test_fill_ratio(self, cache):
        cache.put("k1", {"v": 1}, snr_score=0.95)
        cache.put("k2", {"v": 2}, snr_score=0.95)
        stats = cache.stats()
        assert stats["fill_ratio"] == 0.5  # 2 of 4

    def test_hit_rate(self, cache):
        cache.put("k", {"v": 1}, snr_score=0.95)
        cache.get("k")  # hit
        cache.get("miss")  # miss
        stats = cache.stats()
        assert stats["hit_rate"] == 0.5

    def test_hit_count_on_result(self, cache):
        cache.put("k", {"v": 1}, snr_score=0.95)
        r1 = cache.get("k")
        assert r1 is not None
        assert r1.hit_count == 1
        r2 = cache.get("k")
        assert r2 is not None
        assert r2.hit_count == 2

    def test_temporal_policy_in_stats(self, cache):
        stats = cache.stats()
        assert "temporal_policy" in stats
        policy = stats["temporal_policy"]
        assert policy["hierarchy_levels"] >= 2


class TestSkillCacheInvalidation:
    """Manual invalidation."""

    def test_invalidate_existing(self, cache):
        cache.put("k", {"v": 1}, snr_score=0.95)
        assert cache.invalidate("k") is True
        assert cache.get("k") is None

    def test_invalidate_nonexistent(self, cache):
        assert cache.invalidate("nope") is False

    def test_clear(self, cache):
        for i in range(3):
            cache.put(f"k{i}", {"i": i}, snr_score=0.95)
        cache.clear()
        assert len(cache) == 0


class TestSkillCacheThreadSafety:
    """Concurrent access must not corrupt state."""

    def test_concurrent_puts(self):
        cache = SkillCache(max_size=1000, default_ttl=3600, ihsan_floor=0.90)
        errors: list[Exception] = []

        def writer(thread_id: int):
            try:
                for i in range(100):
                    key = f"t{thread_id}-{i}"
                    cache.put(key, {"t": thread_id, "i": i}, snr_score=0.95)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
        assert len(cache) <= 1000

    def test_concurrent_read_write(self):
        cache = SkillCache(max_size=100, default_ttl=3600, ihsan_floor=0.90)
        errors: list[Exception] = []

        # Pre-populate
        for i in range(50):
            cache.put(f"key-{i}", {"i": i}, snr_score=0.95)

        def reader():
            try:
                for i in range(200):
                    cache.get(f"key-{i % 50}")
            except Exception as e:
                errors.append(e)

        def writer():
            try:
                for i in range(200):
                    cache.put(f"new-{i}", {"i": i}, snr_score=0.95)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=reader),
            threading.Thread(target=reader),
            threading.Thread(target=writer),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors


class TestSkillCacheEmptyState:
    """Behavior on empty cache."""

    def test_empty_len(self, cache):
        assert len(cache) == 0

    def test_empty_stats(self, cache):
        stats = cache.stats()
        assert stats["size"] == 0
        assert stats["hit_rate"] == 0.0


class TestCachedSkillResult:
    """Frozen dataclass properties."""

    def test_frozen(self):
        result = CachedSkillResult(
            structural_hash="abc123",
            query_pattern="test",
            result={"v": 1},
            snr_score=0.95,
            created_at=1.0,
            ttl_seconds=3600,
            hit_count=0,
            last_hit=0.0,
        )
        with pytest.raises(AttributeError):
            result.snr_score = 0.5  # type: ignore[misc]


class TestSkillCacheRepr:

    def test_repr(self, cache):
        r = repr(cache)
        assert "SkillCache" in r
        assert "0/4" in r


class TestTemporalGranularityPolicy:
    def test_ttl_for_layer_monotonic(self):
        policy = TemporalGranularityPolicy(
            min_ttl_seconds=10,
            max_ttl_seconds=100,
            hierarchy_levels=5,
        )
        values = [policy.ttl_for_layer(i) for i in range(5)]
        assert values[0] == 10
        assert values[-1] == 100
        assert values == sorted(values)

    def test_layer_clamped_to_bounds(self):
        policy = TemporalGranularityPolicy(
            min_ttl_seconds=5,
            max_ttl_seconds=50,
            hierarchy_levels=5,
        )
        assert policy.ttl_for_layer(-9) == 5
        assert policy.ttl_for_layer(99) == 50
