"""
Bloom Filter — Probabilistic Set Membership

Standing on Giants:
  Bloom (1970) — "Space/Time Trade-offs in Hash Coding with Allowable Errors"
  Kirsch & Mitzenmacher (2006) — Double hashing: k hashes from two

Uses BLAKE3 (via core.proof_engine.canonical) for cross-language
interoperability with the Rust bizra-omega crates.

Key properties:
- False positives possible (bounded by FPR parameter)
- False negatives impossible
- O(k) add/query, O(m) space where m = bit array size
- Merge via bitwise OR for federation gossip sharing
"""

from __future__ import annotations

import math
import struct
from typing import Final

from core.integration.constants import BLOOM_DEFAULT_FPR, BLOOM_MAX_BITS
from core.proof_engine.canonical import blake3_digest

# Wire format magic bytes for serialization
_MAGIC: Final[bytes] = b"BFLT"
_VERSION: Final[int] = 1


class BloomFilterSaturatedError(RuntimeError):
    """Raised when a Bloom filter exceeds 2x its expected capacity."""


class BloomFilter:
    """
    Probabilistic set membership filter using BLAKE3 double hashing.

    >>> bf = BloomFilter(1000)
    >>> bf.add(b"hello")
    >>> b"hello" in bf
    True
    >>> b"world" in bf  # probably False
    False
    """

    __slots__ = ("_bits", "_num_bits", "_num_hashes", "_expected_items", "_count")

    def __init__(
        self,
        expected_items: int,
        false_positive_rate: float = BLOOM_DEFAULT_FPR,
    ) -> None:
        if expected_items <= 0:
            raise ValueError("expected_items must be positive")
        if not (0.0 < false_positive_rate < 1.0):
            raise ValueError("false_positive_rate must be in (0, 1)")

        self._expected_items = expected_items

        # Optimal sizing: m = -(n * ln(p)) / (ln(2))^2
        m = int(
            math.ceil(
                -(expected_items * math.log(false_positive_rate)) / (math.log(2) ** 2)
            )
        )
        m = min(m, BLOOM_MAX_BITS)
        # Round up to multiple of 8 for byte alignment
        m = ((m + 7) // 8) * 8
        self._num_bits = max(m, 8)

        # Optimal hash count: k = (m/n) * ln(2)
        k = max(1, int(round((self._num_bits / expected_items) * math.log(2))))
        self._num_hashes = k

        self._bits = bytearray(self._num_bits // 8)
        self._count = 0

    @property
    def num_bits(self) -> int:
        return self._num_bits

    @property
    def num_hashes(self) -> int:
        return self._num_hashes

    @property
    def expected_items(self) -> int:
        return self._expected_items

    def _hash_indices(self, item: bytes) -> list[int]:
        """
        Kirsch-Mitzenmacher double hashing.

        Extract two uint64 from 32-byte BLAKE3 digest, then derive
        k hash positions as h_i = (h1 + i * h2) % m.
        """
        digest = blake3_digest(item)
        h1, h2 = struct.unpack_from("<QQ", digest)
        return [(h1 + i * h2) % self._num_bits for i in range(self._num_hashes)]

    def add(self, item: bytes) -> None:
        """Add an item to the filter."""
        if self._count > self._expected_items * 2:
            raise BloomFilterSaturatedError(
                f"Bloom filter saturated: {self._count} items "
                f"(capacity {self._expected_items}). "
                "A saturated filter defeats its purpose — all queries return True."
            )
        for idx in self._hash_indices(item):
            byte_idx, bit_idx = divmod(idx, 8)
            self._bits[byte_idx] |= 1 << bit_idx
        self._count += 1

    def __contains__(self, item: bytes) -> bool:
        """Test probable membership (may return false positive, never false negative)."""
        for idx in self._hash_indices(item):
            byte_idx, bit_idx = divmod(idx, 8)
            if not (self._bits[byte_idx] & (1 << bit_idx)):
                return False
        return True

    def estimated_count(self) -> int:
        """Return the number of items added."""
        return self._count

    def false_positive_probability(self) -> float:
        """
        Estimated FPR given current fill.

        p ≈ (1 - e^(-kn/m))^k
        """
        if self._count == 0:
            return 0.0
        exponent = -(self._num_hashes * self._count) / self._num_bits
        return (1.0 - math.exp(exponent)) ** self._num_hashes

    def merge(self, other: BloomFilter) -> BloomFilter:
        """
        Merge two Bloom filters via bitwise OR.

        Both must have identical parameters (same m and k).
        Used in federation gossip: node A shares its filter with node B,
        B merges to know the union of both membership sets.
        """
        if self._num_bits != other._num_bits or self._num_hashes != other._num_hashes:
            raise ValueError(
                f"Cannot merge filters with different parameters: "
                f"({self._num_bits}, {self._num_hashes}) vs "
                f"({other._num_bits}, {other._num_hashes})"
            )
        result = BloomFilter.__new__(BloomFilter)
        result._num_bits = self._num_bits
        result._num_hashes = self._num_hashes
        result._expected_items = self._expected_items
        result._bits = bytearray(len(self._bits))
        for i in range(len(self._bits)):
            result._bits[i] = self._bits[i] | other._bits[i]
        result._count = self._count + other._count
        return result

    def to_bytes(self) -> bytes:
        """
        Serialize to wire format.

        Format: MAGIC(4) | VERSION(1) | num_bits(4) | num_hashes(1) |
                expected_items(4) | count(4) | bits(m/8)
        """
        header = struct.pack(
            "<4sBIBII",
            _MAGIC,
            _VERSION,
            self._num_bits,
            self._num_hashes,
            self._expected_items,
            self._count,
        )
        return header + bytes(self._bits)

    @classmethod
    def from_bytes(cls, data: bytes) -> BloomFilter:
        """Deserialize from wire format."""
        header_size = struct.calcsize("<4sBIBII")
        if len(data) < header_size:
            raise ValueError("Data too short for Bloom filter header")

        magic, version, num_bits, num_hashes, expected_items, count = (
            struct.unpack_from("<4sBIBII", data)
        )

        if magic != _MAGIC:
            raise ValueError(f"Invalid magic bytes: {magic!r}")
        if version != _VERSION:
            raise ValueError(f"Unsupported version: {version}")

        expected_byte_len = header_size + (num_bits // 8)
        if len(data) != expected_byte_len:
            raise ValueError(f"Expected {expected_byte_len} bytes, got {len(data)}")

        bf = cls.__new__(cls)
        bf._num_bits = num_bits
        bf._num_hashes = num_hashes
        bf._expected_items = expected_items
        bf._count = count
        bf._bits = bytearray(data[header_size:])
        return bf

    def __repr__(self) -> str:
        return (
            f"BloomFilter(items={self._count}/{self._expected_items}, "
            f"bits={self._num_bits}, hashes={self._num_hashes}, "
            f"fpr={self.false_positive_probability():.4f})"
        )
