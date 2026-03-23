"""Tests for core.bounty.impact_proof — ImpactProof, EntropyMeasurement,
DomainEvent, ImpactProofBuilder, ImpactProofVerifier.

Covers:
- Severity and VulnCategory enums
- DomainEvent construction and to_dict
- EntropyMeasurement: total_entropy, average_entropy, to_dict
- ImpactProof: delta_e, multiplier, body_bytes, sign/verify, digest, to_dict
- ImpactProofBuilder: build, from_scan_result
- ImpactProofVerifier: verify pass/fail (SNR, Ihsān, signature, delta_e, exploit_hash)

Blueprint Reference: P3 Coverage Ratchet — bounty module (0.25 → higher)
"""

from datetime import datetime, timezone


from core.bounty.impact_proof import (
    DomainEvent,
    EntropyMeasurement,
    ImpactProof,
    ImpactProofBuilder,
    ImpactProofVerifier,
    Severity,
    VulnCategory,
)
from core.proof_engine.receipt import Ed25519Signer

# ═══════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════


class TestSeverity:
    def test_values(self):
        assert Severity.CRITICAL.value == "critical"
        assert Severity.HIGH.value == "high"
        assert Severity.MEDIUM.value == "medium"
        assert Severity.LOW.value == "low"
        assert Severity.INFORMATIONAL.value == "informational"

    def test_members(self):
        assert len(Severity) == 5


class TestVulnCategory:
    def test_has_reentrancy(self):
        assert VulnCategory.REENTRANCY.value == "reentrancy"

    def test_members(self):
        assert len(VulnCategory) == 10


# ═══════════════════════════════════════════════════════════════════════════
# DomainEvent
# ═══════════════════════════════════════════════════════════════════════════


class TestDomainEvent:
    def test_construction(self):
        now = datetime.now(timezone.utc)
        event = DomainEvent(
            event_type="transaction",
            timestamp=now,
            data={"from": "0x1", "to": "0x2"},
            gas_used=21000,
        )
        assert event.event_type == "transaction"
        assert event.gas_used == 21000

    def test_to_dict(self):
        now = datetime.now(timezone.utc)
        event = DomainEvent(event_type="call", timestamp=now, data={"fn": "transfer"})
        d = event.to_dict()
        assert d["event_type"] == "call"
        assert isinstance(d["timestamp"], str)
        assert d["gas_used"] == 0
        assert d["state_change"] is None


# ═══════════════════════════════════════════════════════════════════════════
# EntropyMeasurement
# ═══════════════════════════════════════════════════════════════════════════


class TestEntropyMeasurement:
    def test_total_entropy(self):
        em = EntropyMeasurement(
            surface_entropy=0.1,
            structural_entropy=0.2,
            behavioral_entropy=0.3,
            hypothetical_entropy=0.15,
            contextual_entropy=0.25,
        )
        assert abs(em.total_entropy - 1.0) < 1e-9

    def test_average_entropy(self):
        em = EntropyMeasurement(
            surface_entropy=0.5,
            structural_entropy=0.5,
            behavioral_entropy=0.5,
            hypothetical_entropy=0.5,
            contextual_entropy=0.5,
        )
        assert abs(em.average_entropy - 0.5) < 1e-9

    def test_default_zeros(self):
        em = EntropyMeasurement()
        assert em.total_entropy == 0.0
        assert em.average_entropy == 0.0

    def test_to_dict(self):
        em = EntropyMeasurement(surface_entropy=0.3)
        d = em.to_dict()
        assert d["surface"] == 0.3
        assert "total" in d
        assert "average" in d


# ═══════════════════════════════════════════════════════════════════════════
# ImpactProof
# ═══════════════════════════════════════════════════════════════════════════


class TestImpactProof:
    def _make_proof(self, **kwargs):
        defaults = {
            "proof_id": "test-001",
            "target_address": "0xDEAD",
            "severity": Severity.HIGH,
            "snr_score": 0.96,
            "ihsan_score": 0.97,
            "entropy_before": EntropyMeasurement(
                surface_entropy=0.8,
                structural_entropy=0.6,
                behavioral_entropy=0.7,
                hypothetical_entropy=0.5,
                contextual_entropy=0.4,
            ),
            "entropy_after": EntropyMeasurement(
                surface_entropy=0.2,
                structural_entropy=0.1,
                behavioral_entropy=0.15,
                hypothetical_entropy=0.1,
                contextual_entropy=0.05,
            ),
            "exploit_hash": b"\x01" * 32,
            "funds_at_risk": 50000.0,
        }
        defaults.update(kwargs)
        return ImpactProof(**defaults)

    def test_delta_e_positive(self):
        proof = self._make_proof()
        # before total = 3.0, after total = 0.6, delta = 2.4
        assert proof.delta_e > 0

    def test_delta_e_negative(self):
        proof = self._make_proof(
            entropy_before=EntropyMeasurement(),
            entropy_after=EntropyMeasurement(surface_entropy=1.0),
        )
        assert proof.delta_e < 0

    def test_body_bytes_deterministic(self):
        proof = self._make_proof()
        assert proof.body_bytes() == proof.body_bytes()

    def test_sign_and_verify(self):
        signer = Ed25519Signer.generate()
        proof = self._make_proof()
        proof.sign_with(signer)
        assert len(proof.signature) > 0
        assert len(proof.hunter_pubkey) > 0
        assert proof.verify_signature(signer)

    def test_digest_not_empty(self):
        proof = self._make_proof()
        assert len(proof.digest()) > 0

    def test_hex_digest(self):
        proof = self._make_proof()
        hd = proof.hex_digest()
        assert isinstance(hd, str)
        assert len(hd) > 0

    def test_to_dict(self):
        proof = self._make_proof()
        d = proof.to_dict()
        assert d["proof_id"] == "test-001"
        assert d["severity"] == "high"
        assert d["delta_e"] > 0
        assert d["funds_at_risk"] == 50000.0
        assert "proof_digest" in d


# ═══════════════════════════════════════════════════════════════════════════
# ImpactProofBuilder
# ═══════════════════════════════════════════════════════════════════════════


class TestImpactProofBuilder:
    def test_build(self):
        signer = Ed25519Signer.generate()
        builder = ImpactProofBuilder(signer)
        proof = builder.build(
            target_address="0xABC",
            vuln_category=VulnCategory.REENTRANCY,
            severity=Severity.CRITICAL,
            title="Reentrancy in withdraw",
            description="Double-spend via reentrant call",
            exploit_code=b"exploit_bytecode",
            entropy_before=EntropyMeasurement(
                surface_entropy=0.9, structural_entropy=0.8
            ),
            entropy_after=EntropyMeasurement(
                surface_entropy=0.1, structural_entropy=0.1
            ),
            reproduction_steps=[],
            funds_at_risk=1_000_000.0,
            snr_score=0.98,
            ihsan_score=0.99,
        )
        assert proof.proof_id.startswith("imp_")
        assert proof.severity == Severity.CRITICAL
        assert len(proof.signature) > 0
        assert proof.verify_signature(signer)

    def test_from_scan_result(self):
        signer = Ed25519Signer.generate()
        builder = ImpactProofBuilder(signer)
        scan = {
            "target": "0xFFF",
            "category": "flash_loan",
            "severity": "high",
            "title": "Flash loan attack",
            "description": "Price oracle manipulation",
            "funds_at_risk": 500000,
            "chain": "polygon",
            "protocol": "TestDeFi",
            "entropy": {"surface": 0.7, "structural": 0.6},
            "reproduction": [
                {"type": "flashloan", "data": {"amount": 1000000}},
            ],
        }
        proof = builder.from_scan_result(scan, b"exploit_code")
        assert proof.target_address == "0xFFF"
        assert proof.vuln_category == VulnCategory.FLASH_LOAN
        assert proof.target_chain == "polygon"
        assert len(proof.reproduction_steps) == 1

    def test_incremental_ids(self):
        signer = Ed25519Signer.generate()
        builder = ImpactProofBuilder(signer)
        p1 = builder.build(
            "0x1",
            VulnCategory.LOGIC_ERROR,
            Severity.LOW,
            "T",
            "D",
            b"code",
            EntropyMeasurement(surface_entropy=0.5),
            EntropyMeasurement(),
            [],
            snr_score=0.95,
            ihsan_score=0.95,
        )
        p2 = builder.build(
            "0x2",
            VulnCategory.LOGIC_ERROR,
            Severity.LOW,
            "T",
            "D",
            b"code",
            EntropyMeasurement(surface_entropy=0.5),
            EntropyMeasurement(),
            [],
            snr_score=0.95,
            ihsan_score=0.95,
        )
        assert p1.proof_id != p2.proof_id


# ═══════════════════════════════════════════════════════════════════════════
# ImpactProofVerifier
# ═══════════════════════════════════════════════════════════════════════════


class TestImpactProofVerifier:
    def _signed_proof(self, signer, **overrides):
        defaults = {
            "proof_id": "vfy-001",
            "target_address": "0xABC",
            "severity": Severity.HIGH,
            "snr_score": 0.96,
            "ihsan_score": 0.97,
            "entropy_before": EntropyMeasurement(
                surface_entropy=0.8, structural_entropy=0.6
            ),
            "entropy_after": EntropyMeasurement(
                surface_entropy=0.1, structural_entropy=0.1
            ),
            "exploit_hash": b"\x01" * 32,
        }
        defaults.update(overrides)
        proof = ImpactProof(**defaults)
        proof.sign_with(signer)
        return proof

    def test_verify_valid(self):
        signer = Ed25519Signer.generate()
        proof = self._signed_proof(signer)
        verifier = ImpactProofVerifier(signer)
        valid, error = verifier.verify(proof)
        assert valid is True
        assert error is None

    def test_verify_low_snr(self):
        signer = Ed25519Signer.generate()
        proof = self._signed_proof(signer, snr_score=0.5)
        verifier = ImpactProofVerifier(signer)
        valid, error = verifier.verify(proof)
        assert valid is False
        assert "SNR" in error

    def test_verify_low_ihsan(self):
        signer = Ed25519Signer.generate()
        proof = self._signed_proof(signer, ihsan_score=0.5)
        verifier = ImpactProofVerifier(signer)
        valid, error = verifier.verify(proof)
        assert valid is False
        assert "Ihsān" in error

    def test_verify_negative_delta_e(self):
        signer = Ed25519Signer.generate()
        proof = self._signed_proof(
            signer,
            entropy_before=EntropyMeasurement(),
            entropy_after=EntropyMeasurement(surface_entropy=1.0),
        )
        verifier = ImpactProofVerifier(signer)
        valid, error = verifier.verify(proof)
        assert valid is False
        assert "entropy" in error.lower()

    def test_verify_missing_exploit_hash(self):
        signer = Ed25519Signer.generate()
        proof = self._signed_proof(signer, exploit_hash=b"")
        verifier = ImpactProofVerifier(signer)
        valid, error = verifier.verify(proof)
        assert valid is False
        assert "exploit" in error.lower()

    def test_stats(self):
        signer = Ed25519Signer.generate()
        verifier = ImpactProofVerifier(signer)
        # Verify one valid
        proof_ok = self._signed_proof(signer)
        verifier.verify(proof_ok)
        # Verify one bad
        proof_bad = self._signed_proof(signer, snr_score=0.1)
        verifier.verify(proof_bad)
        stats = verifier.get_stats()
        assert stats["total_verified"] == 1
        assert stats["total_rejected"] == 1
        assert abs(stats["acceptance_rate"] - 0.5) < 1e-9
