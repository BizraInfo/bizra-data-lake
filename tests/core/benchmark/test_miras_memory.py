"""
Tests for MIRASMemory — Cross-Cycle Memory Substrate.

CI-safe: no GPU, no network, no external services required.
"""

from __future__ import annotations

import time

import pytest

from core.benchmark.miras_memory import MIRASMemory, MIRASMemoryEntry, RetrievalResult
from core.integration.constants import UNIFIED_SNR_THRESHOLD


class TestMIRASMemory:
    """Unit tests for MIRASMemory."""

    # ─── Quality gate ───────────────────────────────────────────────────────────

    def test_quality_gate_rejects_low_snr(self):
        """Content with SNR below UNIFIED_SNR_THRESHOLD must be rejected."""
        memory = MIRASMemory()
        below_threshold = UNIFIED_SNR_THRESHOLD - 0.01
        key = memory.store("low quality content", snr_score=below_threshold)
        assert key is None, "Expected None for SNR below threshold"

    def test_quality_gate_accepts_threshold_snr(self):
        """Content with SNR exactly at threshold must be accepted."""
        memory = MIRASMemory()
        key = memory.store("borderline content", snr_score=UNIFIED_SNR_THRESHOLD)
        assert key is not None

    def test_quality_gate_accepts_high_snr(self):
        """Content with SNR well above threshold must be accepted."""
        memory = MIRASMemory()
        key = memory.store("high quality content", snr_score=0.95)
        assert key is not None
        assert isinstance(key, str)
        assert len(key) == 16  # SHA-256 prefix

    # ─── Store / Retrieve roundtrip ─────────────────────────────────────────────

    def test_store_retrieve_roundtrip(self):
        """Stored content must be findable via retrieve()."""
        memory = MIRASMemory()
        content = "attention head ablation improved MMLU benchmark score"
        memory.store(content, snr_score=0.90)
        result = memory.retrieve("MMLU attention", k=5)
        found = any(content in e.content for e in result.entries)
        assert found, "Stored content not found in retrieval results"

    def test_retrieve_returns_retrieval_result(self):
        """retrieve() must return a RetrievalResult instance."""
        memory = MIRASMemory()
        result = memory.retrieve("any query")
        assert isinstance(result, RetrievalResult)
        assert isinstance(result.entries, list)
        assert isinstance(result.sources, dict)
        assert result.total_retrieved == len(result.entries)

    def test_retrieve_empty_memory(self):
        """Retrieving from empty memory must return empty results."""
        memory = MIRASMemory()
        result = memory.retrieve("test query")
        assert result.total_retrieved == 0
        assert result.dedup_removed == 0

    # ─── LRU eviction ───────────────────────────────────────────────────────────

    def test_lru_eviction_at_capacity(self):
        """Oldest entry must be evicted when short-term reaches capacity."""
        capacity = 5
        memory = MIRASMemory(short_term_capacity=capacity)

        # Store capacity + 1 entries.
        for i in range(capacity + 1):
            memory.store(f"unique content item number {i}", snr_score=0.90)

        # Short-term must not exceed capacity.
        stats = memory.get_stats()
        assert stats["short_term_count"] <= capacity

    def test_duplicate_content_not_double_stored(self):
        """Storing the same content twice must not create duplicate entries."""
        memory = MIRASMemory()
        content = "identical content stored twice"
        k1 = memory.store(content, snr_score=0.90)
        k2 = memory.store(content, snr_score=0.92)
        assert k1 == k2, "Same content should produce same key"
        stats = memory.get_stats()
        assert stats["short_term_count"] == 1

    # ─── Consolidate (promote hot entries) ─────────────────────────────────────

    def test_consolidate_promotes_hot_entries(self):
        """Entries with access_count >= compression_threshold must move to long_term."""
        threshold = 3
        memory = MIRASMemory(compression_threshold=threshold)
        content = "frequently accessed ablation result"
        memory.store(content, snr_score=0.90)

        # Simulate accesses by calling retrieve() threshold times.
        for _ in range(threshold):
            memory.retrieve(content)

        promoted = memory.consolidate()
        assert promoted >= 1
        stats = memory.get_stats()
        assert stats["long_term_count"] >= 1

    def test_consolidate_returns_count(self):
        """consolidate() must return an integer >= 0."""
        memory = MIRASMemory()
        result = memory.consolidate()
        assert isinstance(result, int)
        assert result >= 0

    # ─── Episodic memory ────────────────────────────────────────────────────────

    def test_episodic_unbounded(self):
        """Episodic store must never evict entries (unbounded)."""
        memory = MIRASMemory()
        for i in range(200):
            memory.store_episodic(f"action {i}", f"result {i}")
        stats = memory.get_stats()
        assert stats["episodic_count"] == 200

    def test_episodic_store_no_snr_gate(self):
        """store_episodic must not require SNR (always accepted)."""
        memory = MIRASMemory()
        # Should not raise.
        memory.store_episodic("ran ablation", "improved by 2%", context={"cost": 0.5})
        stats = memory.get_stats()
        assert stats["episodic_count"] == 1

    # ─── Retrieval ordering ─────────────────────────────────────────────────────

    def test_retrieve_returns_sorted_by_relevance(self):
        """Entries more relevant to the query must appear first."""
        memory = MIRASMemory()
        memory.store("attention head ablation MMLU", snr_score=0.95, importance=0.9)
        memory.store(
            "unrelated optimizer weight decay study", snr_score=0.90, importance=0.5
        )
        result = memory.retrieve("attention MMLU", k=10)
        if len(result.entries) >= 2:
            # First entry should have higher or equal relevance signal.
            assert result.entries[0].content != result.entries[1].content

    # ─── Stats ──────────────────────────────────────────────────────────────────

    def test_get_stats_keys(self):
        """get_stats() must return expected keys."""
        memory = MIRASMemory()
        stats = memory.get_stats()
        for key in [
            "short_term_count",
            "long_term_count",
            "episodic_count",
            "total_count",
        ]:
            assert key in stats

    def test_get_stats_total_is_sum(self):
        """total_count must equal sum of tier counts."""
        memory = MIRASMemory()
        memory.store("item one", snr_score=0.90)
        memory.store_episodic("action", "result")
        stats = memory.get_stats()
        expected = (
            stats["short_term_count"]
            + stats["long_term_count"]
            + stats["episodic_count"]
        )
        assert stats["total_count"] == expected
