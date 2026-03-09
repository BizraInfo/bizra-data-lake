"""
Tests for ReflexCompiler — System-1 O(1) cache with precipitation.

Covers: lookup, record_observation, precipitation, invalidation,
validation, LRU eviction, persistence, SDPO compilation, stats.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from core.integration.constants import (
    REFLEX_INVALIDATION_DELTA,
    REFLEX_INVALIDATION_INTERVAL,
    REFLEX_MAX_ENTRIES,
    REFLEX_PRECIPITATION_HITS,
    REFLEX_PRECIPITATION_IHSAN,
    REFLEX_STALENESS_DAYS,
)
from core.sovereign.reflex_compiler import (
    CacheStats,
    PrecipitationCandidate,
    ReflexCompiler,
    ReflexEntry,
)


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def compiler(tmp_path: Path) -> ReflexCompiler:
    """Fresh ReflexCompiler with persistence."""
    return ReflexCompiler(persistence_path=tmp_path / "reflexes.json")


@pytest.fixture
def compiler_no_persist() -> ReflexCompiler:
    """ReflexCompiler without persistence."""
    return ReflexCompiler()


# ═══════════════════════════════════════════════════════════════════════════════
# REFLEX ENTRY
# ═══════════════════════════════════════════════════════════════════════════════


class TestReflexEntry:
    def test_age_days(self):
        entry = ReflexEntry(
            pattern_hash="abc",
            input_template="test",
            output_template="result",
            ihsan_composite=0.95,
            created_at=time.time() - 86400 * 7,
        )
        assert 6.9 < entry.age_days() < 7.1

    def test_needs_validation_when_stale(self):
        entry = ReflexEntry(
            pattern_hash="abc",
            input_template="test",
            output_template="result",
            ihsan_composite=0.95,
            stale=True,
        )
        assert entry.needs_validation() is True

    def test_needs_validation_after_interval(self):
        entry = ReflexEntry(
            pattern_hash="abc",
            input_template="test",
            output_template="result",
            ihsan_composite=0.95,
            validation_hits_since=REFLEX_INVALIDATION_INTERVAL + 1,
        )
        assert entry.needs_validation() is True

    def test_needs_validation_after_staleness_days(self):
        entry = ReflexEntry(
            pattern_hash="abc",
            input_template="test",
            output_template="result",
            ihsan_composite=0.95,
            created_at=time.time() - 86400 * (REFLEX_STALENESS_DAYS + 1),
        )
        assert entry.needs_validation() is True

    def test_fresh_entry_no_validation_needed(self):
        entry = ReflexEntry(
            pattern_hash="abc",
            input_template="test",
            output_template="result",
            ihsan_composite=0.95,
            created_at=time.time(),
            validation_hits_since=0,
        )
        assert entry.needs_validation() is False


# ═══════════════════════════════════════════════════════════════════════════════
# PRECIPITATION CANDIDATE
# ═══════════════════════════════════════════════════════════════════════════════


class TestPrecipitationCandidate:
    def test_consecutive_high_quality_all_high(self):
        candidate = PrecipitationCandidate(
            pattern_hash="abc",
            input_template="test",
            observations=[
                {"ihsan_composite": 0.95, "output": "r1"},
                {"ihsan_composite": 0.92, "output": "r2"},
                {"ihsan_composite": 0.91, "output": "r3"},
            ],
        )
        assert candidate.consecutive_high_quality() == 3

    def test_consecutive_high_quality_broken_streak(self):
        candidate = PrecipitationCandidate(
            pattern_hash="abc",
            input_template="test",
            observations=[
                {"ihsan_composite": 0.95, "output": "r1"},
                {"ihsan_composite": 0.50, "output": "r2"},  # breaks streak
                {"ihsan_composite": 0.95, "output": "r3"},
            ],
        )
        assert candidate.consecutive_high_quality() == 1

    def test_ready_to_precipitate(self):
        observations = [
            {"ihsan_composite": 0.95, "output": f"r{i}"}
            for i in range(REFLEX_PRECIPITATION_HITS)
        ]
        candidate = PrecipitationCandidate(
            pattern_hash="abc",
            input_template="test",
            observations=observations,
        )
        assert candidate.ready_to_precipitate() is True

    def test_not_ready_insufficient_hits(self):
        candidate = PrecipitationCandidate(
            pattern_hash="abc",
            input_template="test",
            observations=[
                {"ihsan_composite": 0.95, "output": "r1"},
            ],
        )
        assert candidate.ready_to_precipitate() is False

    def test_best_output(self):
        candidate = PrecipitationCandidate(
            pattern_hash="abc",
            input_template="test",
            observations=[
                {"ihsan_composite": 0.91, "output": "r1"},
                {"ihsan_composite": 0.98, "output": "r2"},
                {"ihsan_composite": 0.93, "output": "r3"},
            ],
        )
        best = candidate.best_output()
        assert best is not None
        assert best["output"] == "r2"
        assert best["ihsan_composite"] == 0.98

    def test_best_output_empty(self):
        candidate = PrecipitationCandidate(
            pattern_hash="abc",
            input_template="test",
        )
        assert candidate.best_output() is None


# ═══════════════════════════════════════════════════════════════════════════════
# CACHE STATS
# ═══════════════════════════════════════════════════════════════════════════════


class TestCacheStats:
    def test_hit_rate_zero_lookups(self):
        stats = CacheStats()
        assert stats.hit_rate == 0.0

    def test_hit_rate_calculation(self):
        stats = CacheStats(total_lookups=100, cache_hits=75)
        assert stats.hit_rate == 0.75

    def test_as_dict(self):
        stats = CacheStats(total_lookups=10, cache_hits=7, precipitations=2)
        d = stats.as_dict()
        assert d["total_lookups"] == 10
        assert d["cache_hits"] == 7
        assert d["hit_rate"] == 0.7
        assert d["precipitations"] == 2


# ═══════════════════════════════════════════════════════════════════════════════
# REFLEX COMPILER — LOOKUP
# ═══════════════════════════════════════════════════════════════════════════════


class TestReflexCompilerLookup:
    def test_miss_on_empty_cache(self, compiler: ReflexCompiler):
        assert compiler.lookup("anything") is None
        assert compiler.stats.cache_misses == 1

    def test_hit_after_precipitation(self, compiler: ReflexCompiler):
        # Precipitate a pattern
        for i in range(REFLEX_PRECIPITATION_HITS):
            compiler.record_observation("hello world", f"response {i}", 0.95)

        # Now lookup should hit
        entry = compiler.lookup("hello world")
        assert entry is not None
        assert entry.ihsan_composite == 0.95
        assert compiler.stats.cache_hits == 1

    def test_normalization_case_insensitive(self, compiler: ReflexCompiler):
        for i in range(REFLEX_PRECIPITATION_HITS):
            compiler.record_observation("Hello World", f"r{i}", 0.95)

        # Different casing should still hit
        entry = compiler.lookup("hello world")
        assert entry is not None

    def test_normalization_whitespace(self, compiler: ReflexCompiler):
        for i in range(REFLEX_PRECIPITATION_HITS):
            compiler.record_observation("  hello   world  ", f"r{i}", 0.95)

        entry = compiler.lookup("hello world")
        assert entry is not None

    def test_stale_entry_returns_none(self, compiler: ReflexCompiler):
        for i in range(REFLEX_PRECIPITATION_HITS):
            compiler.record_observation("test query", f"r{i}", 0.95)

        entry = compiler.lookup("test query")
        assert entry is not None

        # Invalidate it
        compiler.invalidate(entry.pattern_hash)

        # Should now miss
        assert compiler.lookup("test query") is None

    def test_hit_updates_stats(self, compiler: ReflexCompiler):
        for i in range(REFLEX_PRECIPITATION_HITS):
            compiler.record_observation("count me", f"r{i}", 0.95)

        compiler.lookup("count me")
        compiler.lookup("count me")
        compiler.lookup("count me")

        assert compiler.stats.cache_hits == 3
        assert compiler.stats.total_lookups == 3


# ═══════════════════════════════════════════════════════════════════════════════
# REFLEX COMPILER — PRECIPITATION
# ═══════════════════════════════════════════════════════════════════════════════


class TestReflexCompilerPrecipitation:
    def test_no_precipitation_below_threshold(self, compiler: ReflexCompiler):
        # Low Ihsan should not precipitate
        for i in range(REFLEX_PRECIPITATION_HITS + 2):
            result = compiler.record_observation("low quality", f"r{i}", 0.50)
        assert result is None
        assert compiler.size == 0

    def test_precipitation_at_threshold(self, compiler: ReflexCompiler):
        result = None
        for i in range(REFLEX_PRECIPITATION_HITS):
            result = compiler.record_observation(
                "high quality", f"response {i}", REFLEX_PRECIPITATION_IHSAN
            )
        assert result is not None
        assert compiler.size == 1
        assert compiler.stats.precipitations == 1

    def test_precipitation_selects_best_output(self, compiler: ReflexCompiler):
        scores = [0.91, 0.98, 0.93]
        result = None
        for i, score in enumerate(scores):
            result = compiler.record_observation("best test", f"response {i}", score)

        assert result is not None
        assert result.output_template == "response 1"  # highest Ihsan
        assert result.ihsan_composite == 0.98

    def test_observation_memory_bounded(self, compiler: ReflexCompiler):
        # Record 15 low-quality observations (won't precipitate)
        for i in range(15):
            compiler.record_observation("bounded", f"r{i}", 0.50)

        # Internal candidate should have at most 10 observations
        h = ReflexCompiler._hash_input("bounded")
        assert len(compiler._candidates[h].observations) <= 10


# ═══════════════════════════════════════════════════════════════════════════════
# REFLEX COMPILER — INVALIDATION
# ═══════════════════════════════════════════════════════════════════════════════


class TestReflexCompilerInvalidation:
    def test_invalidate_existing(self, compiler: ReflexCompiler):
        for i in range(REFLEX_PRECIPITATION_HITS):
            compiler.record_observation("to invalidate", f"r{i}", 0.95)

        entry = compiler.lookup("to invalidate")
        assert entry is not None

        assert compiler.invalidate(entry.pattern_hash) is True
        assert compiler.stats.invalidations == 1

    def test_invalidate_nonexistent(self, compiler: ReflexCompiler):
        assert compiler.invalidate("nonexistent_hash") is False

    def test_validate_entry_stays_valid(self, compiler: ReflexCompiler):
        for i in range(REFLEX_PRECIPITATION_HITS):
            compiler.record_observation("validate me", f"r{i}", 0.95)

        entry = compiler.lookup("validate me")
        assert entry is not None

        # Fresh score within delta
        assert compiler.validate_entry(entry.pattern_hash, 0.96) is True

    def test_validate_entry_invalidates_on_drift(self, compiler: ReflexCompiler):
        for i in range(REFLEX_PRECIPITATION_HITS):
            compiler.record_observation("drift test", f"r{i}", 0.95)

        entry = compiler.lookup("drift test")
        assert entry is not None

        # Fresh score with large drift
        drifted = entry.ihsan_composite - REFLEX_INVALIDATION_DELTA - 0.01
        assert compiler.validate_entry(entry.pattern_hash, drifted) is False
        assert compiler.stats.invalidations == 1


# ═══════════════════════════════════════════════════════════════════════════════
# REFLEX COMPILER — LRU EVICTION
# ═══════════════════════════════════════════════════════════════════════════════


class TestReflexCompilerEviction:
    def test_eviction_at_capacity(self, tmp_path: Path):
        compiler = ReflexCompiler(max_entries=3)

        # Fill to capacity
        for i in range(3):
            for j in range(REFLEX_PRECIPITATION_HITS):
                compiler.record_observation(f"pattern {i}", f"r{j}", 0.95)

        assert compiler.size == 3

        # One more should evict the oldest
        for j in range(REFLEX_PRECIPITATION_HITS):
            compiler.record_observation("pattern overflow", f"r{j}", 0.95)

        assert compiler.size == 3
        assert compiler.stats.evictions == 1

        # First pattern should be evicted
        assert compiler.lookup("pattern 0") is None
        # New pattern should exist
        assert compiler.lookup("pattern overflow") is not None


# ═══════════════════════════════════════════════════════════════════════════════
# REFLEX COMPILER — PERSISTENCE
# ═══════════════════════════════════════════════════════════════════════════════


class TestReflexCompilerPersistence:
    def test_save_and_load(self, tmp_path: Path):
        path = tmp_path / "reflexes.json"
        compiler1 = ReflexCompiler(persistence_path=path)

        # Precipitate a pattern
        for i in range(REFLEX_PRECIPITATION_HITS):
            compiler1.record_observation("persistent", f"r{i}", 0.95)

        assert compiler1.size == 1
        compiler1.save_to_disk()

        # Load into new compiler
        compiler2 = ReflexCompiler(persistence_path=path)
        assert compiler2.size == 1

        entry = compiler2.lookup("persistent")
        assert entry is not None
        assert entry.ihsan_composite == 0.95

    def test_save_without_path(self, compiler_no_persist: ReflexCompiler):
        # Should not raise
        compiler_no_persist.save_to_disk()

    def test_load_corrupt_file(self, tmp_path: Path):
        path = tmp_path / "corrupt.json"
        path.write_text("not valid json{{{")

        # Should not raise, just log warning
        compiler = ReflexCompiler(persistence_path=path)
        assert compiler.size == 0


# ═══════════════════════════════════════════════════════════════════════════════
# REFLEX COMPILER — SDPO COMPILATION
# ═══════════════════════════════════════════════════════════════════════════════


class TestReflexCompilerFromSDPO:
    def test_compile_from_candidate(self, compiler: ReflexCompiler):
        entry = compiler.compile_from_candidate(
            pattern_id="sdpo_pattern_001",
            input_template="what is autopoiesis?",
            output_template="Autopoiesis is self-creation...",
            ihsan_score=0.98,
            observation_count=7,
        )

        assert entry.pattern_hash == "sdpo_pattern_001"
        assert entry.ihsan_composite == 0.98
        assert entry.precipitation_count == 7
        assert compiler.size == 1
        assert compiler.stats.precipitations == 1

    def test_compile_evicts_at_capacity(self):
        compiler = ReflexCompiler(max_entries=2)

        compiler.compile_from_candidate("p1", "q1", "a1", 0.98, 5)
        compiler.compile_from_candidate("p2", "q2", "a2", 0.98, 5)
        compiler.compile_from_candidate("p3", "q3", "a3", 0.98, 5)

        assert compiler.size == 2
        assert compiler.stats.evictions == 1


# ═══════════════════════════════════════════════════════════════════════════════
# REFLEX COMPILER — STATUS
# ═══════════════════════════════════════════════════════════════════════════════


class TestReflexCompilerStatus:
    def test_get_status(self, compiler: ReflexCompiler):
        status = compiler.get_status()
        assert status["size"] == 0
        assert status["candidates"] == 0
        assert status["max_entries"] == REFLEX_MAX_ENTRIES
        assert status["hit_rate"] == 0.0

    def test_status_after_activity(self, compiler: ReflexCompiler):
        for i in range(REFLEX_PRECIPITATION_HITS):
            compiler.record_observation("status test", f"r{i}", 0.95)

        compiler.lookup("status test")
        compiler.lookup("miss")

        status = compiler.get_status()
        assert status["size"] == 1
        assert status["cache_hits"] == 1
        assert status["cache_misses"] == 1
        assert status["precipitations"] == 1

    def test_clear(self, compiler: ReflexCompiler):
        for i in range(REFLEX_PRECIPITATION_HITS):
            compiler.record_observation("to clear", f"r{i}", 0.95)

        assert compiler.size == 1
        compiler.clear()
        assert compiler.size == 0


# ═══════════════════════════════════════════════════════════════════════════════
# REFLEX COMPILER — CONSTANTS VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════


class TestReflexCompilerConstants:
    """Verify the compiler uses constitutional constants from SSOT."""

    def test_max_entries_from_constants(self):
        assert REFLEX_MAX_ENTRIES == 500

    def test_precipitation_hits_from_constants(self):
        assert REFLEX_PRECIPITATION_HITS == 3

    def test_precipitation_ihsan_from_constants(self):
        assert REFLEX_PRECIPITATION_IHSAN == 0.90

    def test_invalidation_interval_from_constants(self):
        assert REFLEX_INVALIDATION_INTERVAL == 100

    def test_invalidation_delta_from_constants(self):
        assert REFLEX_INVALIDATION_DELTA == 0.05

    def test_staleness_days_from_constants(self):
        assert REFLEX_STALENESS_DAYS == 30
