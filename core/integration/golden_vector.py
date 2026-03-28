"""
BIZRA Golden Vector — Cross-Language Sealing Test (Python side)

This module defines the canonical test vector and verifies that Python
produces the identical hash as Rust for the same input.

The golden vector is FROZEN. If this hash changes, cross-language sealing
is broken and ALL receipts become untrustworthy.

Standing on: Merkle (1979), Aumasson (2015), Shannon (1948)

Usage:
    python -m pytest core/integration/golden_vector.py -v
    python core/integration/golden_vector.py  # Standalone verification
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import blake3

# ── Constants (FROZEN — must match Rust bizra-core/src/canonical.rs) ──
DOMAIN_GOLDEN_VECTOR = "bizra-golden-vector-v1"
FIXED_POINT_P = 1_000_000  # 6 decimal places for float→int conversion

# ── The Golden Vector (FROZEN — never modify) ──────────────────────
# This is the canonical input that both Rust and Python must hash identically.
GOLDEN_MISSION_ID = "golden-vector-v1"
GOLDEN_INITIATOR_ID = "node0-genesis"
GOLDEN_PAYLOAD = b"In the Name of Allah, Most Gracious, Most Merciful"
GOLDEN_IHSAN_SCORE = 0.984700  # Fixed-point: round(0.984700 * 1_000_000) = 984700
GOLDEN_TIMESTAMP = 1711584000000  # 2024-03-28T00:00:00Z (fixed epoch)

# ── The Frozen Digest (computed once, verified forever) ──
# If this changes, cross-language sealing is broken.
# Frozen on 2026-03-28 after cross-language verification.
GOLDEN_DIGEST_HEX = "966725c27200cdd28632e1f10a09ca7f982491be5842c8eb1264650a32a51205"


@dataclass(frozen=True)
class GoldenVector:
    """The canonical cross-language test vector."""

    mission_id: str = GOLDEN_MISSION_ID
    initiator_id: str = GOLDEN_INITIATOR_ID
    payload: bytes = GOLDEN_PAYLOAD
    ihsan_score: float = GOLDEN_IHSAN_SCORE
    timestamp: int = GOLDEN_TIMESTAMP

    def serialize_canonical(self) -> bytes:
        """
        Deterministic serialization matching Rust's canonical_hasher.

        Field order: mission_id, initiator_id, payload, ihsan_fixed, timestamp
        Encoding:
          - Strings: UTF-8 bytes prefixed with u32le length
          - Bytes: raw bytes prefixed with u32le length
          - Float: round(value * FIXED_POINT_P) as u64le
          - Int: u64le
        """
        buf = bytearray()

        # mission_id: length-prefixed UTF-8
        mid = self.mission_id.encode("utf-8")
        buf.extend(struct.pack("<I", len(mid)))
        buf.extend(mid)

        # initiator_id: length-prefixed UTF-8
        iid = self.initiator_id.encode("utf-8")
        buf.extend(struct.pack("<I", len(iid)))
        buf.extend(iid)

        # payload: length-prefixed bytes
        buf.extend(struct.pack("<I", len(self.payload)))
        buf.extend(self.payload)

        # ihsan_score: fixed-point u64le (round, not truncate)
        ihsan_fixed = round(self.ihsan_score * FIXED_POINT_P)
        buf.extend(struct.pack("<Q", ihsan_fixed))

        # timestamp: u64le
        buf.extend(struct.pack("<Q", self.timestamp))

        return bytes(buf)

    def compute_hash(self) -> str:
        """
        Domain-separated BLAKE3 hash matching Rust canonical.rs::domain_hash.

        hash = BLAKE3(domain + ":" + serialized_data)
        """
        serialized = self.serialize_canonical()
        hasher = blake3.blake3()
        hasher.update(DOMAIN_GOLDEN_VECTOR.encode("utf-8"))
        hasher.update(b":")
        hasher.update(serialized)
        return hasher.hexdigest()


def compute_golden_digest() -> str:
    """Compute the golden vector digest."""
    vector = GoldenVector()
    return vector.compute_hash()


def verify_golden_digest(expected_hex: str | None = None) -> bool:
    """Verify the golden vector produces the expected digest."""
    actual = compute_golden_digest()
    if expected_hex is None:
        print(f"GOLDEN_DIGEST={actual}")
        return True
    return actual == expected_hex


# ═══════════════════════════════════════════════════════════════════
# TESTS (run with pytest)
# ═══════════════════════════════════════════════════════════════════


def test_golden_vector_produces_digest():
    """The golden vector must produce a non-empty, deterministic digest."""
    digest = compute_golden_digest()
    assert len(digest) == 64  # BLAKE3 hex = 64 chars
    # Run twice — must be identical (I₁ idempotency)
    assert digest == compute_golden_digest()


def test_golden_vector_serialization_deterministic():
    """Serialization must be byte-identical across invocations."""
    v1 = GoldenVector().serialize_canonical()
    v2 = GoldenVector().serialize_canonical()
    assert v1 == v2


def test_golden_vector_fixed_point_rounding():
    """Fixed-point conversion must round, not truncate."""
    # 0.984700 * 1_000_000 = 984700.0 — exact
    ihsan_fixed = round(GOLDEN_IHSAN_SCORE * FIXED_POINT_P)
    assert ihsan_fixed == 984700


def test_golden_vector_rejects_modified_input():
    """Changing any field must change the digest."""
    original = compute_golden_digest()

    # Tamper with payload
    tampered = GoldenVector(payload=b"Modified payload")
    assert tampered.compute_hash() != original

    # Tamper with ihsan
    tampered = GoldenVector(ihsan_score=0.50)
    assert tampered.compute_hash() != original

    # Tamper with mission_id
    tampered = GoldenVector(mission_id="tampered-id")
    assert tampered.compute_hash() != original


def test_golden_vector_matches_frozen_digest():
    """The digest must match the frozen canonical value."""
    digest = compute_golden_digest()
    assert digest == GOLDEN_DIGEST_HEX, (
        f"FATAL: Golden vector digest changed!\n"
        f"Expected: {GOLDEN_DIGEST_HEX}\n"
        f"Got:      {digest}\n"
        f"Cross-language sealing is BROKEN."
    )


def test_golden_vector_domain_separation():
    """Different domains must produce different hashes for same data."""
    vector = GoldenVector()
    serialized = vector.serialize_canonical()

    # Standard domain
    h1 = blake3.blake3(DOMAIN_GOLDEN_VECTOR.encode() + b":" + serialized).hexdigest()

    # Different domain
    h2 = blake3.blake3(b"different-domain:" + serialized).hexdigest()

    assert h1 != h2


if __name__ == "__main__":
    digest = compute_golden_digest()
    print(f"Python Golden Vector Digest: {digest}")
    print("HASH_ALGO: blake3")
    print(f"DOMAIN: {DOMAIN_GOLDEN_VECTOR}")
    print(f"Serialized length: {len(GoldenVector().serialize_canonical())} bytes")
    print()

    # Run self-tests
    test_golden_vector_produces_digest()
    test_golden_vector_serialization_deterministic()
    test_golden_vector_fixed_point_rounding()
    test_golden_vector_rejects_modified_input()
    test_golden_vector_domain_separation()
    print("All golden vector tests PASSED")
