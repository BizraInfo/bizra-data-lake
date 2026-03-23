"""
Tests for Hardware Fingerprint Rotation — DID Handoff
"""

import secrets


from core.pat.hardware_rotation import (
    MAX_ROTATION_CHAIN_LENGTH,
    HardwareRotationCeremony,
    RotationCertificate,
    compute_fingerprint_hash,
)


class TestFingerprintHash:
    """Test hardware fingerprint hashing."""

    def test_deterministic_hash(self):
        components = {"cpu": "Intel i9", "gpu": "RTX 4090", "node": "12345"}
        h1 = compute_fingerprint_hash(components)
        h2 = compute_fingerprint_hash(components)
        assert h1 == h2

    def test_different_hardware_different_hash(self):
        h1 = compute_fingerprint_hash({"cpu": "Intel i9"})
        h2 = compute_fingerprint_hash({"cpu": "AMD Ryzen"})
        assert h1 != h2

    def test_auto_collection(self):
        h = compute_fingerprint_hash()
        assert len(h) == 64  # SHA-256 hex digest


class TestRotationCertificate:
    """Test rotation certificate creation."""

    def test_certificate_hash(self):
        cert = RotationCertificate(
            old_node_id="BIZRA-AAAAAAAA",
            new_node_id="BIZRA-BBBBBBBB",
            old_public_key="a" * 64,
            new_public_key="b" * 64,
            old_fingerprint_hash="c" * 64,
            new_fingerprint_hash="d" * 64,
            rotation_reason="hardware_upgrade",
        )
        assert len(cert.certificate_hash) == 64
        assert cert.created_at != ""

    def test_certificate_to_dict(self):
        cert = RotationCertificate(
            old_node_id="BIZRA-AAAAAAAA",
            new_node_id="BIZRA-BBBBBBBB",
            old_public_key="a" * 64,
            new_public_key="b" * 64,
            old_fingerprint_hash="c" * 64,
            new_fingerprint_hash="d" * 64,
            rotation_reason="hardware_upgrade",
        )
        d = cert.to_dict()
        assert d["old_node_id"] == "BIZRA-AAAAAAAA"
        assert d["new_node_id"] == "BIZRA-BBBBBBBB"


class TestHardwareRotationCeremony:
    """Test the rotation ceremony."""

    def test_detect_hardware_change(self):
        ceremony = HardwareRotationCeremony()
        stored = compute_fingerprint_hash({"cpu": "Intel i9"})

        assert not ceremony.detect_hardware_change(stored, {"cpu": "Intel i9"})
        assert ceremony.detect_hardware_change(stored, {"cpu": "AMD Ryzen"})

    def test_initiate_rotation(self):
        ceremony = HardwareRotationCeremony()
        old_key = secrets.token_hex(32)
        new_key = secrets.token_hex(32)

        handoff = ceremony.initiate_rotation(
            old_node_id="BIZRA-AAAAAAAA",
            old_public_key=old_key,
            old_private_key=secrets.token_hex(32),
            new_public_key=new_key,
            old_fingerprint_hash="a" * 64,
            new_fingerprint_hash="b" * 64,
            reason="hardware_upgrade",
            poi_score=0.85,
        )

        assert handoff.handoff_complete
        assert handoff.poi_history_transferred
        assert handoff.reputation_preserved
        assert handoff.poi_score_at_rotation == 0.85
        assert handoff.certificate.old_node_id == "BIZRA-AAAAAAAA"
        assert handoff.certificate.new_node_id.startswith("BIZRA-")
        assert handoff.certificate.old_identity_signature != ""

    def test_rotation_chain(self):
        ceremony = HardwareRotationCeremony()

        # First rotation
        h1 = ceremony.initiate_rotation(
            old_node_id="BIZRA-AAAAAAAA",
            old_public_key=secrets.token_hex(32),
            old_private_key=secrets.token_hex(32),
            new_public_key=secrets.token_hex(32),
            old_fingerprint_hash="a" * 64,
            new_fingerprint_hash="b" * 64,
        )

        new_id = h1.certificate.new_node_id

        # Second rotation from new identity
        h2 = ceremony.initiate_rotation(
            old_node_id=new_id,
            old_public_key=secrets.token_hex(32),
            old_private_key=secrets.token_hex(32),
            new_public_key=secrets.token_hex(32),
            old_fingerprint_hash="b" * 64,
            new_fingerprint_hash="c" * 64,
        )

        assert h2.handoff_complete

        # Verify chain
        chain = ceremony.verify_rotation_chain(new_id)
        assert chain["has_rotation_history"]
        assert chain["chain_length"] >= 1

    def test_poi_history_preserved(self):
        """SAPE spec: generate identity on fingerprint A, change to B,
        verify PoI history preserved."""
        ceremony = HardwareRotationCeremony()

        handoff = ceremony.initiate_rotation(
            old_node_id="BIZRA-AAAAAAAA",
            old_public_key=secrets.token_hex(32),
            old_private_key=secrets.token_hex(32),
            new_public_key=secrets.token_hex(32),
            old_fingerprint_hash=compute_fingerprint_hash({"hw": "A"}),
            new_fingerprint_hash=compute_fingerprint_hash({"hw": "B"}),
            poi_score=0.92,
        )

        assert handoff.handoff_complete
        assert handoff.poi_score_at_rotation == 0.92
        assert handoff.poi_history_transferred
        assert handoff.reputation_preserved

    def test_chain_length_limit(self):
        ceremony = HardwareRotationCeremony()
        node_id = "BIZRA-AAAAAAAA"

        # Exhaust the chain
        for i in range(MAX_ROTATION_CHAIN_LENGTH):
            new_pub = secrets.token_hex(32)
            h = ceremony.initiate_rotation(
                old_node_id=node_id,
                old_public_key=secrets.token_hex(32),
                old_private_key=secrets.token_hex(32),
                new_public_key=new_pub,
                old_fingerprint_hash=f"{i:064x}",
                new_fingerprint_hash=f"{i+1:064x}",
            )
            node_id = h.certificate.new_node_id

        # Next rotation should fail
        h = ceremony.initiate_rotation(
            old_node_id=node_id,
            old_public_key=secrets.token_hex(32),
            old_private_key=secrets.token_hex(32),
            new_public_key=secrets.token_hex(32),
            old_fingerprint_hash="x" * 64,
            new_fingerprint_hash="y" * 64,
        )
        assert h.error is not None
        assert "Re-attestation" in h.error

    def test_verify_no_history(self):
        ceremony = HardwareRotationCeremony()
        chain = ceremony.verify_rotation_chain("BIZRA-NONEXIST")
        assert not chain["has_rotation_history"]
        assert chain["chain_length"] == 0
