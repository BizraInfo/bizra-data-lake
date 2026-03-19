"""
Walking Skeleton — Python Constitutional Liveness Proof
========================================================

The thinnest possible end-to-end path proving the BIZRA architecture
works at runtime from the Python side.

Path: Genesis → Autopoietic Cycle → Constitutional Gate → Canonical Checkpoint → Evidence Receipt

Standing on Giants: Cockburn (Walking Skeleton), Shannon (SNR), Al-Ghazali (Ihsan),
                    Maturana & Varela (Autopoiesis), Merkle (Hash Chains)
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

# Constitutional constants — must match Rust (bizra-core/src/lib.rs)
IHSAN_THRESHOLD: float = 0.95
SNR_THRESHOLD: float = 0.85
SKELETON_DOMAIN: bytes = b"bizra-walking-skeleton-v1:"
AUTOPOIESIS_DOMAIN: bytes = b"bizra-autopoiesis-v1:"
CANONICAL_DOMAIN: bytes = b"bizra-canonical-v1:"
FIXED_POINT_P: float = 1_000_000.0

# Use BLAKE3 if available, fall back to hashlib sha3_256 with domain tag
try:
    import blake3 as _blake3

    def _blake3_hash(data: bytes) -> bytes:
        return _blake3.blake3(data).digest()

    HASH_ALGO = "blake3"
except ImportError:
    # Fallback: domain-tagged SHA3-256 (still cryptographically sound)
    def _blake3_hash(data: bytes) -> bytes:
        return hashlib.sha3_256(data).digest()

    HASH_ALGO = "sha3-256-fallback"


def blake3_domain_hash(domain: bytes, data: bytes) -> bytes:
    """Compute a domain-separated hash (BLAKE3 or SHA3-256 fallback)."""
    return _blake3_hash(domain + data)


@dataclass
class AutopoieticState:
    """Mirrors Rust AutopoieticState — minimal fields for the skeleton."""

    cycle_count: int = 0
    ihsan_ema: float = IHSAN_THRESHOLD
    quality_estimate: float = IHSAN_THRESHOLD
    learning_rate: float = 0.01
    total_seed: float = 0.0
    improvement_streak: int = 0
    halt_count: int = 0
    total_cycles: int = 0

    _alpha: float = field(default=0.1, repr=False)

    def execute_cycle(self, actual_quality: float, snr: float) -> dict:
        """Execute one autopoietic cycle. Returns outcome dict."""
        self.total_cycles += 1

        # Predict
        predicted = self.quality_estimate

        # Prediction error
        prediction_error = abs(actual_quality - predicted)

        # Score Ihsan (EMA update)
        self.ihsan_ema = (
            self._alpha * actual_quality + (1.0 - self._alpha) * self.ihsan_ema
        )

        # Constitutional gate: Ihsan
        if self.ihsan_ema < IHSAN_THRESHOLD:
            self.improvement_streak = 0
            self.halt_count += 1
            return {
                "outcome": "halted",
                "reason": f"Ihsan EMA {self.ihsan_ema:.4f} below threshold {IHSAN_THRESHOLD}",
                "ihsan_score": self.ihsan_ema,
            }

        # Constitutional gate: SNR
        if snr < SNR_THRESHOLD:
            self.improvement_streak = 0
            self.halt_count += 1
            return {
                "outcome": "halted",
                "reason": f"SNR {snr:.4f} below threshold {SNR_THRESHOLD}",
                "ihsan_score": self.ihsan_ema,
            }

        # Approved
        self.cycle_count += 1
        self.improvement_streak += 1
        self.total_seed += 1.0

        # Learn
        error = self.ihsan_ema - self.quality_estimate
        self.quality_estimate += self.learning_rate * error
        self.quality_estimate = max(0.0, min(1.0, self.quality_estimate))

        return {
            "outcome": "approved",
            "ihsan_score": self.ihsan_ema,
            "snr_score": snr,
            "prediction_error": prediction_error,
            "seed_earned": 1.0,
        }

    def to_canonical_hash(self) -> bytes:
        """Compute canonical hash of invariant fields."""
        data = CANONICAL_DOMAIN
        data += int(self.total_seed * FIXED_POINT_P).to_bytes(8, "little")
        data += int(self.ihsan_ema * FIXED_POINT_P).to_bytes(8, "little")
        data += int(self.quality_estimate * FIXED_POINT_P).to_bytes(8, "little")
        data += self.cycle_count.to_bytes(8, "little")
        return _blake3_hash(data)


@dataclass
class SkeletonReceipt:
    """The atomic proof that the Python walking skeleton is alive."""

    genesis_hash: str
    cycle_count: int
    ihsan_score: float
    snr_score: float
    constitutional_pass: bool
    state_root: str
    evidence_hash: str
    timestamp: str
    era_version: int
    elapsed_us: int
    hash_algorithm: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, data: str) -> "SkeletonReceipt":
        return cls(**json.loads(data))


def run_skeleton() -> SkeletonReceipt:
    """
    Run the complete walking skeleton path (Python implementation).

    Mirrors the Rust implementation:
    1. Genesis: BLAKE3 identity
    2. Create AutopoieticState
    3. Run one cycle
    4. Check constitutional gates
    5. Create canonical checkpoint
    6. Generate evidence receipt

    Returns a SkeletonReceipt if the system is constitutionally alive.
    Raises RuntimeError if any step fails.
    """
    start = time.monotonic()

    # Step 1: Genesis — BLAKE3 identity
    genesis_payload = b"bizra-node0-walking-skeleton-genesis"
    genesis_hash = blake3_domain_hash(SKELETON_DOMAIN, genesis_payload)
    if genesis_hash == b"\x00" * 32:
        raise RuntimeError("Genesis hash is zero")

    # Step 2: Create autopoietic state
    state = AutopoieticState()
    if state.cycle_count != 0:
        raise RuntimeError(f"Initial cycle_count should be 0, got {state.cycle_count}")

    # Step 3: One autopoietic cycle
    actual_quality = 0.97
    snr = 0.90
    outcome = state.execute_cycle(actual_quality, snr)

    if outcome["outcome"] != "approved":
        raise RuntimeError(f"Autopoietic cycle halted: {outcome['reason']}")

    # Step 4: Constitutional gate
    ihsan_score = outcome["ihsan_score"]
    snr_score = outcome["snr_score"]
    constitutional_pass = ihsan_score >= IHSAN_THRESHOLD and snr_score >= SNR_THRESHOLD

    if not constitutional_pass:
        raise RuntimeError(
            f"Constitutional gate failed: ihsan={ihsan_score:.4f}, snr={snr_score:.4f}"
        )

    # Step 5: Canonical checkpoint
    state_root = state.to_canonical_hash()
    if state_root == b"\x00" * 32:
        raise RuntimeError("State root is zero")

    # Step 6: Evidence receipt
    timestamp = datetime.now(timezone.utc).isoformat()
    elapsed_us = int((time.monotonic() - start) * 1_000_000)

    evidence_data = (
        SKELETON_DOMAIN
        + genesis_hash
        + state.cycle_count.to_bytes(8, "little")
        + _float_to_le_bytes(ihsan_score)
        + _float_to_le_bytes(snr_score)
        + state_root
        + timestamp.encode()
    )
    evidence_hash = _blake3_hash(evidence_data)

    if evidence_hash == b"\x00" * 32:
        raise RuntimeError("Evidence hash is zero")

    return SkeletonReceipt(
        genesis_hash=genesis_hash.hex(),
        cycle_count=state.cycle_count,
        ihsan_score=ihsan_score,
        snr_score=snr_score,
        constitutional_pass=constitutional_pass,
        state_root=state_root.hex(),
        evidence_hash=evidence_hash.hex(),
        timestamp=timestamp,
        era_version=1,
        elapsed_us=elapsed_us,
        hash_algorithm=HASH_ALGO,
    )


def _float_to_le_bytes(value: float) -> bytes:
    """Convert float to little-endian 8-byte representation (IEEE 754)."""
    import struct

    return struct.pack("<d", value)


if __name__ == "__main__":
    receipt = run_skeleton()
    print(receipt.to_json())
    print(
        f"\nWALKING SKELETON PASSED in {receipt.elapsed_us}us "
        f"— system is constitutionally alive ({receipt.hash_algorithm})",
        flush=True,
    )
