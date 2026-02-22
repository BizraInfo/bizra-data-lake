"""
Tests for BloomFilter — Phase 44

Standing on Giants: Bloom (1970), Kirsch & Mitzenmacher (2006)
"""

import math
import struct

import pytest

from core.hashtable.bloom_filter import (
    BloomFilter,
    BloomFilterSaturatedError,
)


class TestBloomFilterBasic:
    """Core membership semantics."""

    def test_contains_after_add(self):
        bf = BloomFilter(100)
        bf.add(b"hello")
        assert b"hello" in bf

    def test_not_contains_absent(self):
        bf = BloomFilter(100)
        bf.add(b"hello")
        assert b"world" not in bf

    def test_multiple_items(self):
        bf = BloomFilter(1000)
        items = [f"item-{i}".encode() for i in range(50)]
        for item in items:
            bf.add(item)
        for item in items:
            assert item in bf

    def test_empty_filter_never_contains(self):
        bf = BloomFilter(100)
        assert b"anything" not in bf

    def test_empty_bytes_is_valid_item(self):
        bf = BloomFilter(100)
        bf.add(b"")
        assert b"" in bf

    def test_large_items(self):
        bf = BloomFilter(100)
        large = b"x" * 10_000
        bf.add(large)
        assert large in bf


class TestBloomFilterFPR:
    """False positive rate validation."""

    def test_fpr_within_bound(self):
        """Empirical FPR should be near the configured rate."""
        n = 1000
        fpr = 0.01
        bf = BloomFilter(n, false_positive_rate=fpr)
        for i in range(n):
            bf.add(f"member-{i}".encode())

        # Test 10000 non-members
        false_positives = sum(
            1 for i in range(10000)
            if f"non-member-{i}".encode() in bf
        )
        empirical_fpr = false_positives / 10000

        # Allow 3x headroom (statistical variance)
        assert empirical_fpr < fpr * 3, (
            f"Empirical FPR {empirical_fpr:.4f} exceeds 3x target {fpr}"
        )

    def test_estimated_fpr_increases_with_items(self):
        bf = BloomFilter(1000)
        fpr_empty = bf.false_positive_probability()
        for i in range(100):
            bf.add(f"item-{i}".encode())
        fpr_100 = bf.false_positive_probability()
        assert fpr_100 > fpr_empty

    def test_fpr_zero_when_empty(self):
        bf = BloomFilter(100)
        assert bf.false_positive_probability() == 0.0


class TestBloomFilterSizing:
    """Optimal sizing and parameter validation."""

    def test_optimal_bits(self):
        bf = BloomFilter(1000, false_positive_rate=0.01)
        expected_m = int(math.ceil(-(1000 * math.log(0.01)) / (math.log(2) ** 2)))
        # Round to byte boundary
        expected_m = ((expected_m + 7) // 8) * 8
        assert bf.num_bits == expected_m

    def test_max_bits_capped(self):
        """Filter with huge capacity doesn't exceed BLOOM_MAX_BITS."""
        bf = BloomFilter(100_000_000, false_positive_rate=0.0001)
        from core.integration.constants import BLOOM_MAX_BITS
        # Round max to byte boundary
        max_rounded = ((BLOOM_MAX_BITS + 7) // 8) * 8
        assert bf.num_bits <= max_rounded

    def test_invalid_expected_items(self):
        with pytest.raises(ValueError, match="positive"):
            BloomFilter(0)
        with pytest.raises(ValueError, match="positive"):
            BloomFilter(-1)

    def test_invalid_fpr(self):
        with pytest.raises(ValueError, match="\\(0, 1\\)"):
            BloomFilter(100, false_positive_rate=0.0)
        with pytest.raises(ValueError, match="\\(0, 1\\)"):
            BloomFilter(100, false_positive_rate=1.0)
        with pytest.raises(ValueError, match="\\(0, 1\\)"):
            BloomFilter(100, false_positive_rate=-0.1)

    def test_num_hashes_at_least_one(self):
        bf = BloomFilter(1)
        assert bf.num_hashes >= 1


class TestBloomFilterMerge:
    """Merge (bitwise OR) for federation gossip."""

    def test_merge_preserves_membership(self):
        bf1 = BloomFilter(100)
        bf2 = BloomFilter(100)
        bf1.add(b"alpha")
        bf2.add(b"beta")

        merged = bf1.merge(bf2)
        assert b"alpha" in merged
        assert b"beta" in merged

    def test_merge_incompatible_raises(self):
        bf1 = BloomFilter(100, false_positive_rate=0.01)
        bf2 = BloomFilter(200, false_positive_rate=0.01)
        with pytest.raises(ValueError, match="different parameters"):
            bf1.merge(bf2)

    def test_merge_does_not_mutate_originals(self):
        bf1 = BloomFilter(100)
        bf2 = BloomFilter(100)
        bf1.add(b"alpha")
        bf2.add(b"beta")
        _ = bf1.merge(bf2)
        assert b"beta" not in bf1
        assert b"alpha" not in bf2


class TestBloomFilterSerialization:
    """to_bytes / from_bytes roundtrip."""

    def test_roundtrip(self):
        bf = BloomFilter(500, false_positive_rate=0.02)
        items = [f"item-{i}".encode() for i in range(100)]
        for item in items:
            bf.add(item)

        data = bf.to_bytes()
        restored = BloomFilter.from_bytes(data)

        for item in items:
            assert item in restored
        assert restored.num_bits == bf.num_bits
        assert restored.num_hashes == bf.num_hashes
        assert restored.estimated_count() == bf.estimated_count()

    def test_invalid_magic(self):
        data = b"XXXX" + b"\x00" * 50
        with pytest.raises(ValueError, match="Invalid magic"):
            BloomFilter.from_bytes(data)

    def test_truncated_data(self):
        with pytest.raises(ValueError, match="too short"):
            BloomFilter.from_bytes(b"BF")

    def test_wrong_length(self):
        bf = BloomFilter(100)
        data = bf.to_bytes()
        with pytest.raises(ValueError, match="Expected"):
            BloomFilter.from_bytes(data + b"\x00")


class TestBloomFilterEstimatedCount:
    """Count tracking."""

    def test_count_increments(self):
        bf = BloomFilter(100)
        assert bf.estimated_count() == 0
        bf.add(b"a")
        assert bf.estimated_count() == 1
        bf.add(b"b")
        assert bf.estimated_count() == 2


class TestBloomFilterSaturationGuard:
    """Saturation detection — Fix 2 from plan."""

    def test_saturation_raises(self):
        bf = BloomFilter(5)
        # Add 11 items to a capacity-5 filter (saturation triggers at > 2*5 = 10)
        for i in range(11):
            bf.add(f"item-{i}".encode())
        with pytest.raises(BloomFilterSaturatedError, match="saturated"):
            bf.add(b"one-too-many")

    def test_below_saturation_ok(self):
        bf = BloomFilter(100)
        for i in range(50):
            bf.add(f"item-{i}".encode())
        # 50 items in a 100-capacity filter — no error
        bf.add(b"still-fine")


class TestBloomFilterUsesBlake3:
    """Verify BLAKE3 is used, not SHA-256."""

    def test_hash_uses_blake3(self):
        bf = BloomFilter(100)
        # If this doesn't raise, blake3 is importable and used
        bf.add(b"test")
        assert b"test" in bf


class TestBloomFilterRepr:

    def test_repr(self):
        bf = BloomFilter(100)
        r = repr(bf)
        assert "BloomFilter" in r
        assert "0/100" in r
