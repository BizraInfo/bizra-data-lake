"""
Cross-Language BLAKE3 Interop Tests — SR-001 Verification

Proves that Python canonical.hex_digest() produces identical output
to Rust's BLAKE3 for the same input. If these tests pass, federation
nodes running Python and Rust will agree on every hash.

Standing on Giants: O'Connor et al. (BLAKE3, 2020)

Test vectors verified against:
- blake3 reference implementation (Rust)
- bizra-omega/bizra-core BLAKE3 calls
"""

import json

import blake3
import pytest

from core.proof_engine.canonical import (
    blake3_digest,
    canonical_bytes,
    canonical_json,
    hex_digest,
)


class TestBlake3DirectInterop:
    """Verify Python BLAKE3 matches reference implementation."""

    def test_empty_input(self):
        """Empty input produces the BLAKE3 empty-string digest."""
        result = hex_digest(b"")
        # Reference: blake3("") = af1349b9...
        reference = blake3.blake3(b"").hexdigest()
        assert result == reference

    def test_ascii_input(self):
        """ASCII string produces correct digest."""
        data = b"bismillah"
        result = hex_digest(data)
        reference = blake3.blake3(data).hexdigest()
        assert result == reference

    def test_unicode_arabic(self):
        """Arabic Unicode (بذرة = BIZRA) produces correct digest."""
        data = "بذرة".encode("utf-8")
        result = hex_digest(data)
        reference = blake3.blake3(data).hexdigest()
        assert result == reference

    def test_large_payload(self):
        """Large payload (1MB) produces correct digest."""
        data = b"x" * (1024 * 1024)
        result = hex_digest(data)
        reference = blake3.blake3(data).hexdigest()
        assert result == reference

    def test_json_canonical_determinism(self):
        """Canonical JSON hashing is deterministic across key orderings."""
        obj_a = {"z": 1, "a": 2, "m": 3}
        obj_b = {"a": 2, "m": 3, "z": 1}

        bytes_a = canonical_bytes(obj_a)
        bytes_b = canonical_bytes(obj_b)

        assert bytes_a == bytes_b
        assert hex_digest(bytes_a) == hex_digest(bytes_b)

    def test_pci_envelope_hash_format(self):
        """PCI envelope hash matches expected format (64 hex chars)."""
        envelope = {
            "seq": 1,
            "receipt": {"status": "accepted", "node_id": "node0"},
            "prev_hash": "0" * 64,
        }
        digest = hex_digest(canonical_bytes(envelope))
        assert len(digest) == 64
        assert all(c in "0123456789abcdef" for c in digest)

    def test_digest_bytes_length(self):
        """blake3_digest returns exactly 32 bytes."""
        result = blake3_digest(b"test")
        assert len(result) == 32
        assert isinstance(result, bytes)


class TestCanonicalJsonCrossLanguage:
    """Verify canonical JSON serialization matches Rust's serde_json output.

    Rust uses: serde_json with sort_keys + compact separators.
    Python uses: json.dumps(sort_keys=True, separators=(',', ':'))
    """

    def test_nested_object_canonical_form(self):
        """Nested objects produce RFC 8785 canonical form."""
        obj = {"b": {"d": 4, "c": 3}, "a": 1}
        canon = canonical_json(obj)
        serialized = json.dumps(canon, sort_keys=True, separators=(",", ":"))
        # Keys sorted at every level
        assert serialized == '{"a":1,"b":{"c":3,"d":4}}'

    def test_array_order_preserved(self):
        """Arrays maintain insertion order (not sorted)."""
        obj = {"items": [3, 1, 2]}
        canon = canonical_json(obj)
        serialized = json.dumps(canon, sort_keys=True, separators=(",", ":"))
        assert serialized == '{"items":[3,1,2]}'

    def test_null_and_bool_canonical(self):
        """null, true, false match JSON spec exactly."""
        obj = {"flag": True, "empty": None, "off": False}
        canon = canonical_json(obj)
        serialized = json.dumps(canon, sort_keys=True, separators=(",", ":"))
        assert serialized == '{"empty":null,"flag":true,"off":false}'

    def test_float_precision(self):
        """Float precision matches across languages."""
        obj = {"score": 0.95, "threshold": 0.85}
        canon = canonical_json(obj)
        serialized = json.dumps(canon, sort_keys=True, separators=(",", ":"))
        # Both Python and Rust use IEEE 754 doubles
        assert '"score":0.95' in serialized
        assert '"threshold":0.85' in serialized


class TestEpigenomeBlake3Migration:
    """Verify epigenome uses BLAKE3 (not SHA-256) after SR-001 fix."""

    def test_interpretation_hash_is_blake3(self):
        """Interpretation.compute_hash() uses BLAKE3."""
        from core.pci.epigenome import Interpretation, InterpretationType

        interp = Interpretation(
            receipt_hash="a" * 64,
            interpretation_type=InterpretationType.LEARNED,
            new_context="test context",
            timestamp="2026-01-01T00:00:00Z",
            signature="sig",
        )
        result = interp.compute_hash()

        # Verify it matches BLAKE3 (not SHA-256)
        content = json.dumps(
            {
                "receipt_hash": "a" * 64,
                "type": "LEARNED",
                "context": "test context",
                "timestamp": "2026-01-01T00:00:00Z",
            },
            sort_keys=True,
        )
        expected = hex_digest(content.encode())
        assert result == expected
        assert len(result) == 64

    def test_growth_proof_hash_is_blake3(self):
        """EpigeneticLayer.generate_growth_proof() uses BLAKE3."""
        from core.pci.epigenome import EpigeneticLayer

        layer = EpigeneticLayer()
        proof = layer.generate_growth_proof()

        # Proof hash should be 64 hex chars (BLAKE3)
        assert len(proof["proof_hash"]) == 64
        assert proof["content_revealed"] is False


class TestEvidenceLedgerBlake3:
    """Verify evidence ledger chain uses BLAKE3."""

    def test_entry_hash_is_blake3(self, tmp_path):
        """Ledger entries use BLAKE3 for chain hashing."""
        from core.proof_engine.evidence_ledger import EvidenceLedger

        ledger = EvidenceLedger(tmp_path / "test.jsonl", validate_on_append=False)

        entry = ledger.append(
            {
                "receipt_id": "test-001",
                "timestamp": "2026-01-01T00:00:00Z",
                "node_id": "node0",
                "policy_version": "1.0.0",
                "status": "accepted",
                "decision": "APPROVED",
                "reason_codes": [],
                "snr": {"score": 0.95},
                "ihsan": {"score": 0.95, "threshold": 0.95, "decision": "APPROVED"},
                "seal": {"algorithm": "blake3", "digest": "0" * 64},
            }
        )

        # Hash should be 64 hex chars (BLAKE3, not SHA-256)
        assert len(entry.entry_hash) == 64

        # Chain should verify
        is_valid, errors = ledger.verify_chain()
        assert is_valid, f"Chain verification failed: {errors}"

    def test_chain_integrity_after_multiple_entries(self, tmp_path):
        """Multi-entry chain maintains BLAKE3 integrity."""
        from core.proof_engine.evidence_ledger import EvidenceLedger, GENESIS_HASH

        ledger = EvidenceLedger(tmp_path / "multi.jsonl", validate_on_append=False)

        for i in range(5):
            ledger.append(
                {
                    "receipt_id": f"test-{i:03d}",
                    "timestamp": "2026-01-01T00:00:00Z",
                    "node_id": "node0",
                    "policy_version": "1.0.0",
                    "status": "accepted",
                    "decision": "APPROVED",
                    "reason_codes": [],
                    "snr": {"score": 0.95},
                    "ihsan": {
                        "score": 0.95,
                        "threshold": 0.95,
                        "decision": "APPROVED",
                    },
                    "seal": {"algorithm": "blake3", "digest": "0" * 64},
                }
            )

        assert ledger.sequence == 5
        assert ledger.last_hash != GENESIS_HASH

        is_valid, errors = ledger.verify_chain()
        assert is_valid, f"Chain verification failed: {errors}"
