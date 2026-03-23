"""
Walking Skeleton — Python Integration Test
============================================

THE constitutional liveness proof for the Python side.
If this test passes, the Python path is alive.
"""

import json
import time


from core.walking_skeleton import (
    IHSAN_THRESHOLD,
    SNR_THRESHOLD,
    AutopoieticState,
    SkeletonReceipt,
    blake3_domain_hash,
    run_skeleton,
)


class TestWalkingSkeletonLiveness:
    """Core liveness tests — if these fail, something fundamental is broken."""

    def test_walking_skeleton_proves_constitutional_liveness(self):
        receipt = run_skeleton()

        # Constitutional gates must pass
        assert receipt.constitutional_pass, "Constitutional gate must pass"
        assert (
            receipt.ihsan_score >= IHSAN_THRESHOLD
        ), f"Ihsan {receipt.ihsan_score:.4f} must meet threshold {IHSAN_THRESHOLD}"
        assert (
            receipt.snr_score >= SNR_THRESHOLD
        ), f"SNR {receipt.snr_score:.4f} must meet threshold {SNR_THRESHOLD}"

        # Cryptographic artifacts must be non-trivial
        assert receipt.genesis_hash != "00" * 32, "Genesis hash must be non-zero"
        assert receipt.state_root != "00" * 32, "State root must be non-zero"
        assert receipt.evidence_hash != "00" * 32, "Evidence hash must be non-zero"

        # Exactly one cycle completed
        assert receipt.cycle_count == 1, "Exactly one cycle must complete"

        # Era version valid
        assert receipt.era_version >= 1, "Era version must be >= 1"

        # Timestamp present
        assert receipt.timestamp, "Timestamp must be present"

    def test_walking_skeleton_fast(self):
        start = time.monotonic()
        receipt = run_skeleton()
        elapsed_ms = (time.monotonic() - start) * 1000

        assert elapsed_ms < 1000, f"Skeleton took {elapsed_ms:.1f}ms, must be <1000ms"
        assert (
            receipt.elapsed_us < 1_000_000
        ), f"Receipt reports {receipt.elapsed_us}us, must be <1s"

    def test_walking_skeleton_deterministic(self):
        r1 = run_skeleton()
        r2 = run_skeleton()

        # Core fields must match
        assert r1.genesis_hash == r2.genesis_hash
        assert r1.cycle_count == r2.cycle_count
        assert r1.state_root == r2.state_root
        assert r1.constitutional_pass == r2.constitutional_pass
        assert r1.era_version == r2.era_version
        assert abs(r1.ihsan_score - r2.ihsan_score) < 1e-10
        assert abs(r1.snr_score - r2.snr_score) < 1e-10


class TestSkeletonReceiptSerialization:
    """Test that the receipt round-trips through JSON."""

    def test_receipt_to_json(self):
        receipt = run_skeleton()
        json_str = receipt.to_json()
        data = json.loads(json_str)

        assert data["constitutional_pass"] is True
        assert data["cycle_count"] == 1
        assert isinstance(data["genesis_hash"], str)
        assert isinstance(data["state_root"], str)

    def test_receipt_round_trip(self):
        receipt = run_skeleton()
        json_str = receipt.to_json()
        restored = SkeletonReceipt.from_json(json_str)

        assert restored.genesis_hash == receipt.genesis_hash
        assert restored.state_root == receipt.state_root
        assert restored.evidence_hash == receipt.evidence_hash
        assert restored.cycle_count == receipt.cycle_count
        assert restored.constitutional_pass == receipt.constitutional_pass


class TestAutopoieticCycle:
    """Test the Python autopoietic state directly."""

    def test_cycle_approved_with_good_inputs(self):
        state = AutopoieticState()
        outcome = state.execute_cycle(0.97, 0.90)

        assert outcome["outcome"] == "approved"
        assert outcome["ihsan_score"] >= IHSAN_THRESHOLD
        assert outcome["snr_score"] >= SNR_THRESHOLD
        assert state.cycle_count == 1
        assert state.total_seed > 0

    def test_cycle_halted_on_low_snr(self):
        state = AutopoieticState()
        outcome = state.execute_cycle(0.99, 0.50)

        assert outcome["outcome"] == "halted"
        assert "SNR" in outcome["reason"]

    def test_canonical_hash_deterministic(self):
        s1 = AutopoieticState()
        s1.execute_cycle(0.97, 0.90)
        s2 = AutopoieticState()
        s2.execute_cycle(0.97, 0.90)

        assert s1.to_canonical_hash() == s2.to_canonical_hash()

    def test_canonical_hash_varies(self):
        s1 = AutopoieticState()
        s1.execute_cycle(0.97, 0.90)
        s2 = AutopoieticState()
        s2.execute_cycle(0.99, 0.92)

        assert s1.to_canonical_hash() != s2.to_canonical_hash()


class TestBlake3DomainHash:
    """Test the hashing primitive."""

    def test_domain_separation(self):
        h1 = blake3_domain_hash(b"domain-a:", b"data")
        h2 = blake3_domain_hash(b"domain-b:", b"data")
        assert h1 != h2, "Different domains must produce different hashes"

    def test_deterministic(self):
        h1 = blake3_domain_hash(b"test:", b"hello")
        h2 = blake3_domain_hash(b"test:", b"hello")
        assert h1 == h2, "Same inputs must produce same hash"

    def test_non_zero(self):
        h = blake3_domain_hash(b"test:", b"data")
        assert h != b"\x00" * 32, "Hash must be non-zero"
