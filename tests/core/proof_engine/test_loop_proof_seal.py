"""Tests for Loop Proof Seal — signature verification gate."""

import json
import pytest
from pathlib import Path

from core.pci.crypto import generate_keypair, sign_message
from core.proof_engine.loop_proof import LoopProof
from core.proof_engine.loop_proof_seal import (
    SealStatus,
    canonicalize_proof,
    sidecar_path_for,
    verify_seal,
    write_sidecar,
)


@pytest.fixture
def proof_file(tmp_path) -> Path:
    """Create a minimal loop proof artifact."""
    proof = LoopProof(mission="test mission", node_id="test-node")
    proof.add_step("a", "x", "ok")
    proof.add_step("b", "y", "ok")
    proof.compute_manifest_hash()
    path = tmp_path / "proof.json"
    path.write_text(proof.to_json())
    return path


@pytest.fixture
def keypair():
    """Generate a fresh Ed25519 keypair for testing."""
    private_hex, public_hex = generate_keypair()
    return private_hex, public_hex


class TestVerifySeal:
    def test_unsigned_proof(self, proof_file):
        status = verify_seal(proof_file)
        assert status.state == "unsigned"
        assert not status.is_canonical
        assert status.manifest_hash

    def test_missing_proof(self, tmp_path):
        status = verify_seal(tmp_path / "nonexistent.json")
        assert status.state == "missing_proof"

    def test_sidecar_path(self, proof_file):
        assert sidecar_path_for(proof_file) == proof_file.with_suffix(".sig.json")


class TestWriteSidecar:
    def test_write_and_verify_valid(self, proof_file, keypair):
        private_hex, public_hex = keypair
        proof = json.loads(proof_file.read_text())
        manifest_hash = proof["manifest_hash"]

        signature_hex = sign_message(manifest_hash, private_hex)
        write_sidecar(proof_file, signature_hex, public_hex)

        status = verify_seal(proof_file)
        assert status.state == "signed_valid"
        assert status.is_canonical

    def test_write_invalid_signature(self, proof_file, keypair):
        _, public_hex = keypair
        write_sidecar(proof_file, "deadbeef" * 16, public_hex)

        status = verify_seal(proof_file)
        assert status.state == "signed_invalid"
        assert not status.is_canonical

    def test_wrong_key_fails(self, proof_file, keypair):
        private_hex, _ = keypair
        _, wrong_public = generate_keypair()

        proof = json.loads(proof_file.read_text())
        manifest_hash = proof["manifest_hash"]
        signature_hex = sign_message(manifest_hash, private_hex)

        write_sidecar(proof_file, signature_hex, wrong_public)
        status = verify_seal(proof_file)
        assert status.state == "signed_invalid"


class TestCanonicalize:
    def test_canonicalize_after_valid_sign(self, proof_file, keypair):
        private_hex, public_hex = keypair
        proof = json.loads(proof_file.read_text())
        manifest_hash = proof["manifest_hash"]

        signature_hex = sign_message(manifest_hash, private_hex)
        write_sidecar(proof_file, signature_hex, public_hex)

        ok = canonicalize_proof(proof_file)
        assert ok

        proof_after = json.loads(proof_file.read_text())
        assert proof_after["canonical"] is True
        assert proof_after["signature"] == signature_hex

    def test_cannot_canonicalize_unsigned(self, proof_file):
        ok = canonicalize_proof(proof_file)
        assert not ok

        proof_after = json.loads(proof_file.read_text())
        assert proof_after["canonical"] is False

    def test_cannot_canonicalize_invalid(self, proof_file, keypair):
        _, public_hex = keypair
        write_sidecar(proof_file, "bad" * 32, public_hex)

        ok = canonicalize_proof(proof_file)
        assert not ok


class TestSealStatusShape:
    def test_to_dict(self, proof_file):
        status = verify_seal(proof_file)
        d = status.to_dict()
        assert "state" in d
        assert "is_canonical" in d
        assert "manifest_hash" in d
