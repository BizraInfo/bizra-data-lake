"""Tests for core.memory_coder -- Memory Auto-Coder package.

Covers: SynthesizedPattern, PatternCodebook, MemorySynthesizer,
clustering, centroid extraction, and linear-scan fallback.
"""
from __future__ import annotations

import math
from typing import Any, List, Optional

import pytest

from core.integration.constants import SNR_THRESHOLD_T1_HIGH, UNIFIED_SNR_THRESHOLD
from core.memory_coder.memory_synthesizer import MemoryRecord, MemorySynthesizer
from core.memory_coder.pattern_codebook import (
    PatternCodebook,
    SynthesizedPattern,
    cosine_similarity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pattern(
    *,
    pid: str = "test-pattern",
    embedding: Optional[List[float]] = None,
    keywords: Optional[List[str]] = None,
    snr: float = 0.90,
    source_count: int = 5,
    access_count: int = 1,
) -> SynthesizedPattern:
    """Factory for test patterns with sensible defaults."""
    return SynthesizedPattern(
        pattern_id=pid,
        embedding=embedding or [1.0, 0.0, 0.0],
        keywords=keywords or ["alpha", "beta"],
        snr=snr,
        source_count=source_count,
        access_count=access_count,
    )


def _make_record(
    *,
    rid: str = "rec-1",
    embedding: Optional[List[float]] = None,
    keywords: Optional[List[str]] = None,
    snr: float = 0.90,
    access_count: int = 5,
) -> MemoryRecord:
    """Factory for test memory records."""
    return MemoryRecord(
        record_id=rid,
        content=f"content for {rid}",
        embedding=embedding or [1.0, 0.0, 0.0],
        keywords=keywords or ["kw"],
        snr=snr,
        access_count=access_count,
    )


# ═══════════════════════════════════════════════════════════════════════════
# SynthesizedPattern tests
# ═══════════════════════════════════════════════════════════════════════════


class TestSynthesizedPattern:
    """Tests for the SynthesizedPattern dataclass."""

    def test_synthesized_pattern_is_strong(self) -> None:
        """Pattern with snr >= T1_HIGH and source_count >= 10 is strong."""
        pattern = _make_pattern(snr=0.96, source_count=15)
        assert pattern.is_strong is True
        # Verify the threshold comes from constants
        assert pattern.snr >= SNR_THRESHOLD_T1_HIGH
        assert pattern.source_count >= 10

    def test_synthesized_pattern_not_strong_low_snr(self) -> None:
        """Pattern with low SNR is not strong even with many sources."""
        pattern = _make_pattern(snr=0.80, source_count=15)
        assert pattern.is_strong is False

    def test_synthesized_pattern_not_strong_low_sources(self) -> None:
        """Pattern with few sources is not strong even with high SNR."""
        pattern = _make_pattern(snr=0.96, source_count=3)
        assert pattern.is_strong is False


# ═══════════════════════════════════════════════════════════════════════════
# PatternCodebook tests
# ═══════════════════════════════════════════════════════════════════════════


class TestPatternCodebook:
    """Tests for the PatternCodebook collection."""

    def test_pattern_codebook_add_and_lookup(self) -> None:
        """Add a pattern, then retrieve it via embedding lookup."""
        codebook = PatternCodebook()
        pattern = _make_pattern(embedding=[1.0, 0.0, 0.0])
        codebook.add(pattern)

        results = codebook.lookup(query_embedding=[1.0, 0.0, 0.0], top_k=1)
        assert len(results) == 1
        assert results[0].pattern_id == pattern.pattern_id

    def test_pattern_codebook_contains_similar(self) -> None:
        """Duplicate detection at 0.90 similarity threshold."""
        codebook = PatternCodebook()
        existing = _make_pattern(pid="existing", embedding=[1.0, 0.0, 0.0])
        codebook.add(existing)

        # Nearly identical embedding should be detected as similar
        duplicate = _make_pattern(pid="dup", embedding=[0.99, 0.01, 0.0])
        assert codebook.contains_similar(duplicate, threshold=0.90) is True

        # Orthogonal embedding should NOT be similar
        different = _make_pattern(pid="diff", embedding=[0.0, 1.0, 0.0])
        assert codebook.contains_similar(different, threshold=0.90) is False

    def test_pattern_codebook_size_property(self) -> None:
        """Size reflects number of stored patterns."""
        codebook = PatternCodebook()
        assert codebook.size == 0

        codebook.add(_make_pattern(pid="a"))
        assert codebook.size == 1

        codebook.add(_make_pattern(pid="b"))
        assert codebook.size == 2

    def test_pattern_codebook_strong_patterns(self) -> None:
        """strong_patterns filters to only patterns meeting is_strong."""
        codebook = PatternCodebook()
        strong = _make_pattern(pid="s", snr=0.96, source_count=15)
        weak = _make_pattern(pid="w", snr=0.80, source_count=2)
        codebook.add(strong)
        codebook.add(weak)

        strongs = codebook.strong_patterns
        assert len(strongs) == 1
        assert strongs[0].pattern_id == "s"

    def test_codebook_without_agent_db_uses_linear_scan(self) -> None:
        """Without agent_db, lookup falls back to linear scan."""
        codebook = PatternCodebook(agent_db=None)
        p1 = _make_pattern(pid="p1", embedding=[1.0, 0.0, 0.0])
        p2 = _make_pattern(pid="p2", embedding=[0.0, 1.0, 0.0])
        p3 = _make_pattern(pid="p3", embedding=[0.0, 0.0, 1.0])
        codebook.add(p1)
        codebook.add(p2)
        codebook.add(p3)

        results = codebook.lookup(query_embedding=[1.0, 0.0, 0.0], top_k=2)
        assert len(results) == 2
        # p1 should rank first (cosine = 1.0 with query)
        assert results[0].pattern_id == "p1"


# ═══════════════════════════════════════════════════════════════════════════
# MemorySynthesizer tests
# ═══════════════════════════════════════════════════════════════════════════


class TestMemorySynthesizer:
    """Tests for the MemorySynthesizer PDCA engine."""

    def test_memory_synthesizer_no_agent_db(self) -> None:
        """Without agent_db, synthesize_cycle returns an empty list."""
        synthesizer = MemorySynthesizer(agent_db=None)
        patterns = synthesizer.synthesize_cycle()
        assert patterns == []

    def test_memory_synthesizer_with_records(self) -> None:
        """Manually invoke clustering and extraction on known records."""
        synthesizer = MemorySynthesizer(agent_db=None)

        # 4 similar records (cluster) + 1 outlier
        records = [
            _make_record(rid="r1", embedding=[1.0, 0.0, 0.0], keywords=["ai"]),
            _make_record(rid="r2", embedding=[0.98, 0.02, 0.0], keywords=["ai"]),
            _make_record(rid="r3", embedding=[0.99, 0.01, 0.0], keywords=["ml"]),
            _make_record(rid="r4", embedding=[0.97, 0.03, 0.0], keywords=["ai"]),
            _make_record(rid="r5", embedding=[0.0, 1.0, 0.0], keywords=["other"]),
        ]

        clusters = synthesizer._cluster_by_embedding(
            records, min_cluster_size=3, threshold=UNIFIED_SNR_THRESHOLD
        )
        assert len(clusters) >= 1
        # The similar records should form at least one cluster
        assert any(len(c) >= 3 for c in clusters)

        # Extract pattern from the first cluster
        pattern = synthesizer._extract_pattern(clusters[0])
        assert isinstance(pattern, SynthesizedPattern)
        assert pattern.source_count >= 3

    def test_cluster_by_embedding_groups_similar(self) -> None:
        """Clustering produces expected groupings of similar vectors."""
        synthesizer = MemorySynthesizer(agent_db=None)

        # Group A: near [1,0,0]
        # Group B: near [0,1,0]
        records = [
            _make_record(rid="a1", embedding=[1.0, 0.0, 0.0]),
            _make_record(rid="a2", embedding=[0.99, 0.01, 0.0]),
            _make_record(rid="a3", embedding=[0.98, 0.02, 0.0]),
            _make_record(rid="b1", embedding=[0.0, 1.0, 0.0]),
            _make_record(rid="b2", embedding=[0.01, 0.99, 0.0]),
            _make_record(rid="b3", embedding=[0.02, 0.98, 0.0]),
        ]

        clusters = synthesizer._cluster_by_embedding(
            records, min_cluster_size=3, threshold=UNIFIED_SNR_THRESHOLD
        )
        assert len(clusters) == 2

        # Verify each cluster is internally consistent
        for cluster in clusters:
            ids = {r.record_id for r in cluster}
            # Either all a-records or all b-records
            assert ids <= {"a1", "a2", "a3"} or ids <= {"b1", "b2", "b3"}

    def test_extract_pattern_centroid(self) -> None:
        """Centroid is the element-wise mean of cluster embeddings."""
        synthesizer = MemorySynthesizer(agent_db=None)

        records = [
            _make_record(rid="r1", embedding=[1.0, 0.0, 0.0], snr=0.90),
            _make_record(rid="r2", embedding=[0.0, 1.0, 0.0], snr=0.80),
            _make_record(rid="r3", embedding=[0.0, 0.0, 1.0], snr=0.85),
        ]

        pattern = synthesizer._extract_pattern(records)

        # Centroid should be [1/3, 1/3, 1/3]
        expected = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
        for actual, exp in zip(pattern.embedding, expected):
            assert abs(actual - exp) < 1e-9, f"{actual} != {exp}"

        # Mean SNR
        expected_snr = (0.90 + 0.80 + 0.85) / 3.0
        assert abs(pattern.snr - expected_snr) < 1e-9

        # Source count
        assert pattern.source_count == 3
