"""
Constitutional Gate Tests -- Comprehensive Coverage for Admission Controller
=============================================================================
Tests for the ConstitutionalGate class, verifying all four admission pillars:
- PILLAR 1 (Runtime): Z3-proven agents admitted with score=1.0
- PILLAR 2 (Museum): High-SNR candidates queued for proofing
- PILLAR 3 (Sandbox): Medium-SNR candidates in restricted sandbox
- PILLAR 4 (Cutoff): Rejection below quality threshold

Standing on Giants:
- Z3 SMT Solver (de Moura & Bjorner, 2008)
- Shannon (1948) — SNR as quality metric
- Saltzer & Schroeder (1975) — Fail-safe defaults

Created: 2026-02-11
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import blake3
import pytest

from core.pci.crypto import (
    PrivateKeyWrapper,
    canonical_json,
    generate_keypair,
)
from core.sovereign.constitutional_gate import (
    Z3_CERT_DOMAIN_PREFIX,
    ConstitutionalGate,
)
from core.sovereign.integration_types import (
    AdmissionResult,
    AdmissionStatus,
    Z3Certificate,
)


# =============================================================================
# HELPERS
# =============================================================================


def _clean_env():
    """Return a dict suitable for patching os.environ to remove BIZRA keys."""
    return {
        "BIZRA_Z3_CERT_PUBKEY": "",
        "BIZRA_Z3_CERT_ALLOW_UNSIGNED": "0",
        "BIZRA_Z3_CERT_ALLOW_SELF_SIGNED": "0",
        "BIZRA_ENV": "",
        "BIZRA_PRODUCTION_ALLOW_UNSIGNED_OVERRIDE": "0",
        "BIZRA_KEYPAIR_PATH": "/nonexistent/path/keypair.json",
    }


def _make_z3_cert_data(
    candidate_id: str = "test_candidate",
    valid: bool = True,
    proof_type: str = "z3-smt2",
    verified_at: str = "2026-02-11T00:00:00Z",
    signature: str = "",
    public_key: str = "",
) -> Dict[str, Any]:
    """Build a Z3 certificate data dict for testing."""
    data: Dict[str, Any] = {
        "hash": candidate_id,
        "valid": valid,
        "proof_type": proof_type,
        "verified_at": verified_at,
    }
    if signature:
        data["signature"] = signature
    if public_key:
        data["public_key"] = public_key
    return data


def _sign_z3_cert(
    data: Dict[str, Any],
    candidate_id: str,
    private_key: PrivateKeyWrapper,
) -> str:
    """Sign a Z3 certificate using the real crypto pipeline and return hex signature."""
    signable = {
        "hash": data.get("hash", candidate_id),
        "valid": bool(data.get("valid", False)),
        "proof_type": data.get("proof_type", "z3-smt2"),
        "verified_at": data.get("verified_at", ""),
    }
    canonical = canonical_json(signable)
    hasher = blake3.blake3()
    hasher.update(Z3_CERT_DOMAIN_PREFIX.encode("utf-8"))
    hasher.update(canonical)
    digest_hex = hasher.hexdigest()
    return private_key.sign(digest_hex)


def _write_cert_file(
    cert_dir: Path,
    candidate_id: str,
    cert_data: Dict[str, Any],
) -> Path:
    """Write a Z3 certificate JSON to the given directory."""
    cert_dir.mkdir(parents=True, exist_ok=True)
    path = cert_dir / f"{candidate_id}_z3.cert"
    with open(path, "w") as f:
        json.dump(cert_data, f)
    return path


def _make_gate(tmp_path: Path, env_overrides: Dict[str, str] | None = None) -> ConstitutionalGate:
    """Create a ConstitutionalGate with clean env and tmp paths."""
    env = _clean_env()
    if env_overrides:
        env.update(env_overrides)
    with patch.dict(os.environ, env, clear=False):
        gate = ConstitutionalGate(
            z3_certificates_path=tmp_path / "proofs",
            museum_path=tmp_path / "museum",
        )
    return gate


# =============================================================================
# 1. INITIALIZATION TESTS
# =============================================================================


class TestConstitutionalGateInit:
    """Tests for ConstitutionalGate.__init__."""

    def test_default_paths(self) -> None:
        """Default paths should use ~/.bizra/proofs and ~/.bizra/museum."""
        with patch.dict(os.environ, _clean_env(), clear=False):
            gate = ConstitutionalGate()
        assert gate.z3_certificates_path == Path.home() / ".bizra" / "proofs"
        assert gate.museum_path == Path.home() / ".bizra" / "museum"

    def test_custom_paths(self, tmp_path: Path) -> None:
        """Custom paths should override defaults."""
        proofs = tmp_path / "custom_proofs"
        museum = tmp_path / "custom_museum"
        with patch.dict(os.environ, _clean_env(), clear=False):
            gate = ConstitutionalGate(
                z3_certificates_path=proofs,
                museum_path=museum,
            )
        assert gate.z3_certificates_path == proofs
        assert gate.museum_path == museum

    def test_empty_initial_state(self, tmp_path: Path) -> None:
        """Gate should start with empty queues and caches."""
        gate = _make_gate(tmp_path)
        assert gate.museum_queue == []
        assert gate.runtime_agents == []
        assert gate._z3_cache == {}

    def test_unsigned_allowed_flag_off_by_default(self, tmp_path: Path) -> None:
        """_allow_unsigned_z3 should be False by default."""
        gate = _make_gate(tmp_path)
        assert gate._allow_unsigned_z3 is False

    def test_self_signed_allowed_flag_off_by_default(self, tmp_path: Path) -> None:
        """_allow_self_signed_z3 should be False by default."""
        gate = _make_gate(tmp_path)
        assert gate._allow_self_signed_z3 is False

    def test_unsigned_allowed_when_env_set(self, tmp_path: Path) -> None:
        """_allow_unsigned_z3 should be True when env var is '1'."""
        gate = _make_gate(tmp_path, {"BIZRA_Z3_CERT_ALLOW_UNSIGNED": "1"})
        assert gate._allow_unsigned_z3 is True

    def test_self_signed_allowed_when_env_set(self, tmp_path: Path) -> None:
        """_allow_self_signed_z3 should be True when env var is '1'."""
        gate = _make_gate(tmp_path, {"BIZRA_Z3_CERT_ALLOW_SELF_SIGNED": "1"})
        assert gate._allow_self_signed_z3 is True

    def test_production_rejects_unsigned(self, tmp_path: Path) -> None:
        """Production env must raise RuntimeError if unsigned certs allowed."""
        with pytest.raises(RuntimeError, match="SECURITY HALT"):
            _make_gate(tmp_path, {
                "BIZRA_ENV": "production",
                "BIZRA_Z3_CERT_ALLOW_UNSIGNED": "1",
            })

    def test_production_rejects_self_signed(self, tmp_path: Path) -> None:
        """Production env must raise RuntimeError if self-signed certs allowed."""
        with pytest.raises(RuntimeError, match="SECURITY HALT"):
            _make_gate(tmp_path, {
                "BIZRA_ENV": "production",
                "BIZRA_Z3_CERT_ALLOW_SELF_SIGNED": "1",
            })

    def test_production_override_allows_unsigned(self, tmp_path: Path) -> None:
        """Production with explicit override should not raise."""
        gate = _make_gate(tmp_path, {
            "BIZRA_ENV": "production",
            "BIZRA_Z3_CERT_ALLOW_UNSIGNED": "1",
            "BIZRA_PRODUCTION_ALLOW_UNSIGNED_OVERRIDE": "1",
        })
        assert gate._allow_unsigned_z3 is True

    def test_production_override_allows_self_signed(self, tmp_path: Path) -> None:
        """Production with explicit override should not raise for self-signed."""
        gate = _make_gate(tmp_path, {
            "BIZRA_ENV": "production",
            "BIZRA_Z3_CERT_ALLOW_SELF_SIGNED": "1",
            "BIZRA_PRODUCTION_ALLOW_UNSIGNED_OVERRIDE": "1",
        })
        assert gate._allow_self_signed_z3 is True

    def test_non_production_allows_unsigned(self, tmp_path: Path) -> None:
        """Non-production env should not raise even if unsigned allowed."""
        gate = _make_gate(tmp_path, {
            "BIZRA_ENV": "development",
            "BIZRA_Z3_CERT_ALLOW_UNSIGNED": "1",
        })
        assert gate._allow_unsigned_z3 is True

    def test_loads_trusted_pubkey_from_env(self, tmp_path: Path) -> None:
        """Should load trusted pubkey from BIZRA_Z3_CERT_PUBKEY env var."""
        fake_key = "abcdef1234567890" * 4
        gate = _make_gate(tmp_path, {"BIZRA_Z3_CERT_PUBKEY": fake_key})
        assert gate._trusted_z3_pubkey == fake_key

    def test_loads_trusted_pubkey_from_file(self, tmp_path: Path) -> None:
        """Should load trusted pubkey from keypair.json file."""
        keypair_path = tmp_path / "keypair.json"
        pubkey = "deadbeef" * 8
        with open(keypair_path, "w") as f:
            json.dump({"public_key": pubkey}, f)
        gate = _make_gate(tmp_path, {"BIZRA_KEYPAIR_PATH": str(keypair_path)})
        assert gate._trusted_z3_pubkey == pubkey

    def test_empty_pubkey_when_no_source(self, tmp_path: Path) -> None:
        """Should have empty pubkey when no env var and no file."""
        gate = _make_gate(tmp_path)
        assert gate._trusted_z3_pubkey == ""


# =============================================================================
# 2. COMPUTE HASH TESTS
# =============================================================================


class TestComputeHash:
    """Tests for ConstitutionalGate._compute_hash."""

    def test_deterministic(self, tmp_path: Path) -> None:
        """Same content should always produce the same hash."""
        gate = _make_gate(tmp_path)
        h1 = gate._compute_hash("hello world")
        h2 = gate._compute_hash("hello world")
        assert h1 == h2

    def test_blake3_prefix_16(self, tmp_path: Path) -> None:
        """Hash should be first 16 chars of BLAKE3 hex (SEC-001)."""
        gate = _make_gate(tmp_path)
        content = "test content"
        from core.proof_engine.canonical import hex_digest
        expected = hex_digest(content.encode())[:16]
        assert gate._compute_hash(content) == expected

    def test_different_content_different_hash(self, tmp_path: Path) -> None:
        """Different content should produce different hashes."""
        gate = _make_gate(tmp_path)
        assert gate._compute_hash("alpha") != gate._compute_hash("beta")

    def test_empty_string(self, tmp_path: Path) -> None:
        """Empty string should produce a valid 16-char hex hash."""
        gate = _make_gate(tmp_path)
        result = gate._compute_hash("")
        assert len(result) == 16
        # Verify it is valid hex
        int(result, 16)


# =============================================================================
# 3. FALLBACK SNR TESTS
# =============================================================================


class TestFallbackSNR:
    """Tests for ConstitutionalGate._fallback_snr."""

    def test_empty_text_returns_zero(self, tmp_path: Path) -> None:
        """Empty text should return 0.0."""
        gate = _make_gate(tmp_path)
        assert gate._fallback_snr("") == 0.0

    def test_whitespace_only_returns_zero(self, tmp_path: Path) -> None:
        """Whitespace-only text should return 0.0 (split produces empty list)."""
        gate = _make_gate(tmp_path)
        assert gate._fallback_snr("   ") == 0.0

    def test_single_word(self, tmp_path: Path) -> None:
        """Single word: unique_ratio=1.0, length_factor=1/100, result = (1.0*0.6 + 0.01*0.4)*0.9."""
        gate = _make_gate(tmp_path)
        result = gate._fallback_snr("hello")
        expected = (1.0 * 0.6 + (1 / 100) * 0.4) * 0.9
        assert abs(result - expected) < 1e-9

    def test_all_same_words(self, tmp_path: Path) -> None:
        """All same words should have unique_ratio = 1/n."""
        gate = _make_gate(tmp_path)
        text = " ".join(["repeat"] * 50)
        result = gate._fallback_snr(text)
        unique_ratio = 1 / 50
        length_factor = min(1.0, 50 / 100)
        expected = (unique_ratio * 0.6 + length_factor * 0.4) * 0.9
        assert abs(result - expected) < 1e-9

    def test_all_unique_words_100(self, tmp_path: Path) -> None:
        """100 unique words: unique_ratio=1.0, length_factor=1.0, result=0.9."""
        gate = _make_gate(tmp_path)
        text = " ".join(f"word{i}" for i in range(100))
        result = gate._fallback_snr(text)
        expected = (1.0 * 0.6 + 1.0 * 0.4) * 0.9
        assert abs(result - expected) < 1e-9

    def test_long_text_caps_length_factor(self, tmp_path: Path) -> None:
        """Length factor should cap at 1.0 for texts longer than 100 words."""
        gate = _make_gate(tmp_path)
        text = " ".join(f"word{i}" for i in range(200))
        result = gate._fallback_snr(text)
        # unique_ratio = 200/200 = 1.0, length_factor = min(1.0, 200/100) = 1.0
        expected = (1.0 * 0.6 + 1.0 * 0.4) * 0.9
        assert abs(result - expected) < 1e-9

    def test_result_always_capped_at_0_9(self, tmp_path: Path) -> None:
        """Maximum possible fallback SNR is 0.9 (the 0.9 multiplier)."""
        gate = _make_gate(tmp_path)
        # Best case: all unique, 100+ words
        text = " ".join(f"unique{i}" for i in range(150))
        result = gate._fallback_snr(text)
        assert result <= 0.9 + 1e-9


# =============================================================================
# 4. ADMISSION TESTS (ASYNC)
# =============================================================================


class TestAdmission:
    """Tests for ConstitutionalGate.admit — all four admission paths."""

    @pytest.mark.asyncio
    async def test_runtime_admission_with_valid_z3_cert(self, tmp_path: Path) -> None:
        """Candidate with valid Z3 certificate should get RUNTIME status and score=1.0."""
        gate = _make_gate(tmp_path, {"BIZRA_Z3_CERT_ALLOW_UNSIGNED": "1"})

        # Write a valid unsigned certificate file
        candidate_id = "proven_agent"
        cert_data = _make_z3_cert_data(candidate_id=candidate_id, valid=True)
        cert_dir = tmp_path / "proofs"
        _write_cert_file(cert_dir, candidate_id, cert_data)

        result = await gate.admit("some candidate", "some query", candidate_id=candidate_id)

        assert result.status == AdmissionStatus.RUNTIME
        assert result.score == 1.0
        assert result.evidence["verification"] == "Z3_PROVEN"
        assert result.evidence["ihsan"] == 1.0
        assert result.evidence["proof_type"] == "z3-smt2"

    @pytest.mark.asyncio
    async def test_museum_admission_high_snr(self, tmp_path: Path) -> None:
        """Candidate without Z3 cert but with SNR >= 0.85 should get MUSEUM status."""
        gate = _make_gate(tmp_path)
        # Force calculator to None so fallback is used
        gate.calculator = None

        # Build text that produces SNR >= 0.85 via fallback
        # Need: (unique_ratio * 0.6 + length_factor * 0.4) * 0.9 >= 0.85
        # Best we can get is 0.9 with all unique, 100+ words
        text = " ".join(f"word{i}" for i in range(120))

        result = await gate.admit(text, "query for museum")

        assert result.status == AdmissionStatus.MUSEUM
        assert result.score >= 0.85
        assert result.evidence["verification"] == "SNR_V2_PENDING"
        assert result.promotion_path == "background_z3_synthesis"

    @pytest.mark.asyncio
    async def test_sandbox_admission_medium_snr(self, tmp_path: Path) -> None:
        """Candidate with SNR between 0.70 and 0.85 should get SANDBOX status."""
        gate = _make_gate(tmp_path)
        gate.calculator = None

        # Mock fallback to return 0.78 (between 0.70 and 0.85)
        with patch.object(gate, "_fallback_snr", return_value=0.78):
            result = await gate.admit("medium quality text", "query")

        assert result.status == AdmissionStatus.SANDBOX
        assert 0.70 <= result.score < 0.85
        assert result.evidence["verification"] == "SANDBOX_ONLY"
        restrictions = result.evidence["restrictions"]
        assert "read_only_data_lake" in restrictions
        assert "no_pci_signing" in restrictions
        assert "no_consensus_votes" in restrictions

    @pytest.mark.asyncio
    async def test_rejected_admission_low_snr(self, tmp_path: Path) -> None:
        """Candidate with SNR < 0.70 should be REJECTED."""
        gate = _make_gate(tmp_path)
        gate.calculator = None

        with patch.object(gate, "_fallback_snr", return_value=0.50):
            result = await gate.admit("low quality text", "query")

        assert result.status == AdmissionStatus.REJECTED
        assert result.score < 0.70
        assert result.evidence["verification"] == "SUBSTANDARD"
        assert "0.70" in result.evidence["reason"]

    @pytest.mark.asyncio
    async def test_admit_uses_compute_hash_when_no_candidate_id(self, tmp_path: Path) -> None:
        """When candidate_id is None, admit should compute hash from content."""
        gate = _make_gate(tmp_path)
        gate.calculator = None

        with patch.object(gate, "_fallback_snr", return_value=0.50):
            result = await gate.admit("some content", "query")

        assert result.status == AdmissionStatus.REJECTED

    @pytest.mark.asyncio
    async def test_admit_uses_provided_candidate_id(self, tmp_path: Path) -> None:
        """When candidate_id is provided, admit should use it for cert lookup."""
        gate = _make_gate(tmp_path, {"BIZRA_Z3_CERT_ALLOW_UNSIGNED": "1"})

        cid = "explicit_id"
        cert_data = _make_z3_cert_data(candidate_id=cid, valid=True)
        _write_cert_file(tmp_path / "proofs", cid, cert_data)

        result = await gate.admit("content", "query", candidate_id=cid)
        assert result.status == AdmissionStatus.RUNTIME

    @pytest.mark.asyncio
    async def test_admit_with_invalid_z3_cert_falls_through(self, tmp_path: Path) -> None:
        """Invalid Z3 cert (valid=False) should fall through to SNR check."""
        gate = _make_gate(tmp_path, {"BIZRA_Z3_CERT_ALLOW_UNSIGNED": "1"})
        gate.calculator = None

        cid = "invalid_cert_agent"
        cert_data = _make_z3_cert_data(candidate_id=cid, valid=False)
        _write_cert_file(tmp_path / "proofs", cid, cert_data)

        with patch.object(gate, "_fallback_snr", return_value=0.50):
            result = await gate.admit("content", "query", candidate_id=cid)

        assert result.status == AdmissionStatus.REJECTED

    @pytest.mark.asyncio
    async def test_admit_with_snr_calculator(self, tmp_path: Path) -> None:
        """When SNR v2 calculator is available, it should be used instead of fallback."""
        gate = _make_gate(tmp_path)

        mock_calc = MagicMock()
        mock_result = MagicMock()
        mock_result.snr = 0.92
        mock_result.to_dict.return_value = {"method": "snr_v2", "score": 0.92}
        mock_calc.calculate_simple.return_value = mock_result
        gate.calculator = mock_calc

        result = await gate.admit("candidate text", "query text")

        assert result.status == AdmissionStatus.MUSEUM
        assert result.score == 0.92
        mock_calc.calculate_simple.assert_called_once()


# =============================================================================
# 5. Z3 CERTIFICATE VERIFICATION TESTS
# =============================================================================


class TestZ3CertificateVerification:
    """Tests for _verify_z3_certificate_signature."""

    def test_unsigned_rejected_by_default(self, tmp_path: Path) -> None:
        """Unsigned certificate should be rejected when allow_unsigned is off."""
        gate = _make_gate(tmp_path)
        data = _make_z3_cert_data(signature="", public_key="")
        result = gate._verify_z3_certificate_signature(data, "cid")
        assert result is False

    def test_unsigned_accepted_with_env_flag(self, tmp_path: Path) -> None:
        """Unsigned certificate should be accepted when allow_unsigned is on."""
        gate = _make_gate(tmp_path, {"BIZRA_Z3_CERT_ALLOW_UNSIGNED": "1"})
        data = _make_z3_cert_data(signature="", public_key="")
        result = gate._verify_z3_certificate_signature(data, "cid")
        assert result is True

    def test_self_signed_rejected_without_trusted_key(self, tmp_path: Path) -> None:
        """Self-signed cert should be rejected when no trusted pubkey and self-sign not allowed."""
        gate = _make_gate(tmp_path)
        # Has a signature and pubkey, but no trusted key and self-signed not allowed
        data = _make_z3_cert_data(signature="aabbccdd" * 16, public_key="1122334455" * 6)
        result = gate._verify_z3_certificate_signature(data, "cid")
        assert result is False

    def test_self_signed_accepted_with_env_flag(self, tmp_path: Path) -> None:
        """Self-signed cert should be accepted when BIZRA_Z3_CERT_ALLOW_SELF_SIGNED=1."""
        key = PrivateKeyWrapper.generate()
        pubkey_hex = key.public_key_hex

        gate = _make_gate(tmp_path, {"BIZRA_Z3_CERT_ALLOW_SELF_SIGNED": "1"})

        data = _make_z3_cert_data(
            candidate_id="self_signed_id",
            valid=True,
            public_key=pubkey_hex,
        )
        sig = _sign_z3_cert(data, "self_signed_id", key)
        data["signature"] = sig

        result = gate._verify_z3_certificate_signature(data, "self_signed_id")
        assert result is True

    def test_self_signed_rejected_if_missing_pubkey(self, tmp_path: Path) -> None:
        """Self-signed cert must include public_key field; reject if missing."""
        gate = _make_gate(tmp_path, {"BIZRA_Z3_CERT_ALLOW_SELF_SIGNED": "1"})
        data = _make_z3_cert_data(signature="aabbccdd" * 16, public_key="")
        result = gate._verify_z3_certificate_signature(data, "cid")
        assert result is False

    def test_valid_signature_with_trusted_key(self, tmp_path: Path) -> None:
        """Certificate signed with trusted key should verify."""
        key = PrivateKeyWrapper.generate()
        pubkey_hex = key.public_key_hex

        gate = _make_gate(tmp_path, {"BIZRA_Z3_CERT_PUBKEY": pubkey_hex})

        cid = "trusted_cert_id"
        data = _make_z3_cert_data(
            candidate_id=cid,
            valid=True,
            public_key=pubkey_hex,
        )
        sig = _sign_z3_cert(data, cid, key)
        data["signature"] = sig

        result = gate._verify_z3_certificate_signature(data, cid)
        assert result is True

    def test_pubkey_mismatch_rejected(self, tmp_path: Path) -> None:
        """Cert with different pubkey than trusted key should be rejected."""
        trusted_key = PrivateKeyWrapper.generate()
        other_key = PrivateKeyWrapper.generate()

        gate = _make_gate(tmp_path, {"BIZRA_Z3_CERT_PUBKEY": trusted_key.public_key_hex})

        cid = "mismatch_cert_id"
        data = _make_z3_cert_data(
            candidate_id=cid,
            valid=True,
            public_key=other_key.public_key_hex,
        )
        sig = _sign_z3_cert(data, cid, other_key)
        data["signature"] = sig

        result = gate._verify_z3_certificate_signature(data, cid)
        assert result is False

    def test_invalid_signature_rejected(self, tmp_path: Path) -> None:
        """Certificate with wrong signature should fail verification."""
        key = PrivateKeyWrapper.generate()
        gate = _make_gate(tmp_path, {"BIZRA_Z3_CERT_PUBKEY": key.public_key_hex})

        cid = "bad_sig_id"
        data = _make_z3_cert_data(candidate_id=cid, valid=True, public_key=key.public_key_hex)
        data["signature"] = "ff" * 64  # Garbage signature

        result = gate._verify_z3_certificate_signature(data, cid)
        assert result is False


# =============================================================================
# 6. Z3 DIGEST TESTS
# =============================================================================


class TestZ3Digest:
    """Tests for _z3_digest and _z3_signable_dict."""

    def test_digest_is_hex_string(self, tmp_path: Path) -> None:
        """Digest should be a valid hex string."""
        gate = _make_gate(tmp_path)
        signable = {"hash": "abc", "valid": True, "proof_type": "z3-smt2", "verified_at": ""}
        digest = gate._z3_digest(signable)
        # Should be valid hex
        int(digest, 16)
        # BLAKE3 produces 64-char hex by default
        assert len(digest) == 64

    def test_digest_deterministic(self, tmp_path: Path) -> None:
        """Same signable should produce the same digest."""
        gate = _make_gate(tmp_path)
        signable = {"hash": "xyz", "valid": False, "proof_type": "z3-smt2", "verified_at": "t"}
        d1 = gate._z3_digest(signable)
        d2 = gate._z3_digest(signable)
        assert d1 == d2

    def test_digest_includes_domain_prefix(self, tmp_path: Path) -> None:
        """Digest should differ from one computed without domain prefix."""
        gate = _make_gate(tmp_path)
        signable = {"hash": "test", "valid": True, "proof_type": "z3-smt2", "verified_at": ""}
        domain_digest = gate._z3_digest(signable)

        # Compute without domain prefix
        canonical = canonical_json(signable)
        hasher = blake3.blake3()
        hasher.update(canonical)
        no_prefix_digest = hasher.hexdigest()

        assert domain_digest != no_prefix_digest

    def test_signable_dict_fields(self, tmp_path: Path) -> None:
        """_z3_signable_dict should extract canonical fields."""
        gate = _make_gate(tmp_path)
        data = {
            "hash": "my_hash",
            "valid": True,
            "proof_type": "z3-smt2",
            "verified_at": "2026-01-01",
            "extra_field": "ignored",
        }
        signable = gate._z3_signable_dict(data, "fallback_id")
        assert signable == {
            "hash": "my_hash",
            "valid": True,
            "proof_type": "z3-smt2",
            "verified_at": "2026-01-01",
        }

    def test_signable_dict_uses_candidate_id_fallback(self, tmp_path: Path) -> None:
        """_z3_signable_dict should use candidate_id when hash is missing."""
        gate = _make_gate(tmp_path)
        data = {"valid": True, "proof_type": "z3-smt2", "verified_at": ""}
        signable = gate._z3_signable_dict(data, "fallback_cid")
        assert signable["hash"] == "fallback_cid"


# =============================================================================
# 7. GET Z3 CERTIFICATE TESTS
# =============================================================================


class TestGetZ3Certificate:
    """Tests for _get_z3_certificate — cache, file, FFI fallback."""

    def test_returns_none_when_no_cert(self, tmp_path: Path) -> None:
        """Should return None when no certificate exists."""
        gate = _make_gate(tmp_path)
        cert = gate._get_z3_certificate("nonexistent")
        assert cert is None

    def test_loads_from_file_unsigned_allowed(self, tmp_path: Path) -> None:
        """Should load cert from file when unsigned certs allowed."""
        gate = _make_gate(tmp_path, {"BIZRA_Z3_CERT_ALLOW_UNSIGNED": "1"})
        cid = "file_cert_id"
        cert_data = _make_z3_cert_data(candidate_id=cid, valid=True)
        _write_cert_file(tmp_path / "proofs", cid, cert_data)

        cert = gate._get_z3_certificate(cid)
        assert cert is not None
        assert cert.valid is True
        assert cert.hash == cid
        assert cert.proof_type == "z3-smt2"

    def test_caches_loaded_certificate(self, tmp_path: Path) -> None:
        """Second call for same ID should return from cache, not re-read file."""
        gate = _make_gate(tmp_path, {"BIZRA_Z3_CERT_ALLOW_UNSIGNED": "1"})
        cid = "cached_cert"
        cert_data = _make_z3_cert_data(candidate_id=cid, valid=True)
        _write_cert_file(tmp_path / "proofs", cid, cert_data)

        cert1 = gate._get_z3_certificate(cid)
        assert cert1 is not None

        # Delete the file; second call should still return from cache
        (tmp_path / "proofs" / f"{cid}_z3.cert").unlink()
        cert2 = gate._get_z3_certificate(cid)
        assert cert2 is not None
        assert cert2.hash == cid

    def test_returns_none_when_signature_verification_fails(self, tmp_path: Path) -> None:
        """Should return None if cert file exists but signature fails."""
        gate = _make_gate(tmp_path)  # unsigned not allowed
        cid = "unsigned_cert"
        cert_data = _make_z3_cert_data(candidate_id=cid, valid=True)
        _write_cert_file(tmp_path / "proofs", cid, cert_data)

        cert = gate._get_z3_certificate(cid)
        assert cert is None

    def test_signed_cert_loads_from_file(self, tmp_path: Path) -> None:
        """Properly signed cert should load successfully from file."""
        key = PrivateKeyWrapper.generate()
        cid = "signed_file_cert"

        gate = _make_gate(tmp_path, {"BIZRA_Z3_CERT_PUBKEY": key.public_key_hex})

        cert_data = _make_z3_cert_data(
            candidate_id=cid,
            valid=True,
            public_key=key.public_key_hex,
        )
        sig = _sign_z3_cert(cert_data, cid, key)
        cert_data["signature"] = sig
        cert_data["public_key"] = key.public_key_hex

        _write_cert_file(tmp_path / "proofs", cid, cert_data)

        cert = gate._get_z3_certificate(cid)
        assert cert is not None
        assert cert.valid is True
        assert cert.hash == cid

    def test_malformed_json_returns_none(self, tmp_path: Path) -> None:
        """Should return None and not crash when cert file has invalid JSON."""
        gate = _make_gate(tmp_path, {"BIZRA_Z3_CERT_ALLOW_UNSIGNED": "1"})
        cid = "malformed"
        cert_dir = tmp_path / "proofs"
        cert_dir.mkdir(parents=True, exist_ok=True)
        cert_path = cert_dir / f"{cid}_z3.cert"
        cert_path.write_text("{ this is not valid json }")

        cert = gate._get_z3_certificate(cid)
        assert cert is None


# =============================================================================
# 8. MUSEUM QUEUE TESTS
# =============================================================================


class TestMuseumQueue:
    """Tests for museum queue tracking."""

    @pytest.mark.asyncio
    async def test_museum_admission_adds_to_queue(self, tmp_path: Path) -> None:
        """Museum admission should add (candidate_id, snr_score) to museum_queue."""
        gate = _make_gate(tmp_path)
        gate.calculator = None

        text = " ".join(f"word{i}" for i in range(120))
        await gate.admit(text, "query")

        assert gate.get_museum_count() == 1
        cid, score = gate.museum_queue[0]
        assert len(cid) == 16  # sha256[:16]
        assert score >= 0.85

    @pytest.mark.asyncio
    async def test_museum_count_increments(self, tmp_path: Path) -> None:
        """Multiple museum admissions should increment the count."""
        gate = _make_gate(tmp_path)
        gate.calculator = None

        for i in range(3):
            text = " ".join(f"word{i}_{j}" for j in range(120))
            await gate.admit(text, "query")

        assert gate.get_museum_count() == 3

    @pytest.mark.asyncio
    async def test_museum_position_in_evidence(self, tmp_path: Path) -> None:
        """Museum result should include museum_position in evidence."""
        gate = _make_gate(tmp_path)
        gate.calculator = None

        text = " ".join(f"unique{i}" for i in range(120))
        result = await gate.admit(text, "query")

        assert result.evidence["museum_position"] == 1


# =============================================================================
# 9. RUNTIME AGENTS TESTS
# =============================================================================


class TestRuntimeAgents:
    """Tests for runtime agent tracking."""

    @pytest.mark.asyncio
    async def test_runtime_admission_adds_to_agents(self, tmp_path: Path) -> None:
        """Runtime admission should add candidate to runtime_agents."""
        gate = _make_gate(tmp_path, {"BIZRA_Z3_CERT_ALLOW_UNSIGNED": "1"})

        cid = "runtime_agent_1"
        cert_data = _make_z3_cert_data(candidate_id=cid, valid=True)
        _write_cert_file(tmp_path / "proofs", cid, cert_data)

        await gate.admit("content", "query", candidate_id=cid)

        assert gate.get_runtime_count() == 1
        assert cid in gate.runtime_agents

    @pytest.mark.asyncio
    async def test_runtime_count_increments(self, tmp_path: Path) -> None:
        """Multiple runtime admissions should increment the count."""
        gate = _make_gate(tmp_path, {"BIZRA_Z3_CERT_ALLOW_UNSIGNED": "1"})

        for i in range(3):
            cid = f"agent_{i}"
            cert_data = _make_z3_cert_data(candidate_id=cid, valid=True)
            _write_cert_file(tmp_path / "proofs", cid, cert_data)
            await gate.admit(f"content_{i}", "query", candidate_id=cid)

        assert gate.get_runtime_count() == 3


# =============================================================================
# 10. PROMOTE TO RUNTIME TESTS
# =============================================================================


class TestPromoteToRuntime:
    """Tests for ConstitutionalGate.promote_to_runtime."""

    @pytest.mark.asyncio
    async def test_promote_with_valid_cert_succeeds(self, tmp_path: Path) -> None:
        """Promoting a museum candidate with a valid cert should succeed."""
        gate = _make_gate(tmp_path, {"BIZRA_Z3_CERT_ALLOW_UNSIGNED": "1"})
        gate.calculator = None

        # First, admit to museum
        text = " ".join(f"word{i}" for i in range(120))
        result = await gate.admit(text, "query", candidate_id="promo_candidate")

        assert result.status == AdmissionStatus.MUSEUM
        assert gate.get_museum_count() == 1

        # Now create a valid cert for the candidate
        cert_data = _make_z3_cert_data(candidate_id="promo_candidate", valid=True)
        _write_cert_file(tmp_path / "proofs", "promo_candidate", cert_data)

        # Promote
        success = gate.promote_to_runtime("promo_candidate")
        assert success is True
        assert gate.get_museum_count() == 0
        assert gate.get_runtime_count() == 1
        assert "promo_candidate" in gate.runtime_agents

    @pytest.mark.asyncio
    async def test_promote_without_cert_fails(self, tmp_path: Path) -> None:
        """Promoting a museum candidate without a valid cert should fail."""
        gate = _make_gate(tmp_path)
        gate.calculator = None

        text = " ".join(f"word{i}" for i in range(120))
        await gate.admit(text, "query", candidate_id="no_cert_candidate")

        success = gate.promote_to_runtime("no_cert_candidate")
        assert success is False
        assert gate.get_museum_count() == 1

    def test_promote_nonexistent_candidate(self, tmp_path: Path) -> None:
        """Promoting a candidate not in the museum should return False."""
        gate = _make_gate(tmp_path)
        success = gate.promote_to_runtime("does_not_exist")
        assert success is False

    @pytest.mark.asyncio
    async def test_promote_with_invalid_cert_fails(self, tmp_path: Path) -> None:
        """Promoting with cert where valid=False should fail."""
        gate = _make_gate(tmp_path, {"BIZRA_Z3_CERT_ALLOW_UNSIGNED": "1"})
        gate.calculator = None

        text = " ".join(f"word{i}" for i in range(120))
        await gate.admit(text, "query", candidate_id="invalid_cert_cand")

        cert_data = _make_z3_cert_data(candidate_id="invalid_cert_cand", valid=False)
        _write_cert_file(tmp_path / "proofs", "invalid_cert_cand", cert_data)

        success = gate.promote_to_runtime("invalid_cert_cand")
        assert success is False
        assert gate.get_museum_count() == 1


# =============================================================================
# 11. BACKGROUND PROOFING TESTS
# =============================================================================


class TestScheduleBackgroundProofing:
    """Tests for _schedule_background_proofing."""

    def _call_proofing(self, gate, candidate_id, content):
        """Call _schedule_background_proofing with a mock event loop to avoid cross-test pollution."""
        mock_loop = MagicMock()
        mock_loop.is_running.return_value = False
        with patch("asyncio.get_event_loop", return_value=mock_loop):
            gate._schedule_background_proofing(candidate_id, content)

    def test_writes_json_to_museum_path(self, tmp_path: Path) -> None:
        """Should write a JSON file to the museum directory."""
        gate = _make_gate(tmp_path)
        self._call_proofing(gate, "bg_candidate", "candidate content")

        entry_path = tmp_path / "museum" / "bg_candidate.json"
        assert entry_path.exists()

        with open(entry_path) as f:
            data = json.load(f)
        assert data["id"] == "bg_candidate"
        assert data["status"] == "pending_proof"
        assert "content_hash" in data

    def test_creates_museum_directory(self, tmp_path: Path) -> None:
        """Should create museum directory if it does not exist."""
        gate = _make_gate(tmp_path)
        museum = tmp_path / "museum"
        assert not museum.exists()

        self._call_proofing(gate, "test_id", "content")
        assert museum.exists()

    def test_content_hash_matches_compute_hash(self, tmp_path: Path) -> None:
        """Content hash in the entry should match _compute_hash output."""
        gate = _make_gate(tmp_path)
        content = "the candidate content"
        self._call_proofing(gate, "hash_check_id", content)

        with open(tmp_path / "museum" / "hash_check_id.json") as f:
            data = json.load(f)
        assert data["content_hash"] == gate._compute_hash(content)


# =============================================================================
# 12. CALCULATE SNR TESTS
# =============================================================================


class TestCalculateSNR:
    """Tests for _calculate_snr routing."""

    def test_uses_calculator_when_available(self, tmp_path: Path) -> None:
        """Should use SNR v2 calculator when present."""
        gate = _make_gate(tmp_path)

        mock_calc = MagicMock()
        mock_result = MagicMock()
        mock_result.snr = 0.91
        mock_result.to_dict.return_value = {"method": "v2", "score": 0.91}
        mock_calc.calculate_simple.return_value = mock_result
        gate.calculator = mock_calc

        score, components = gate._calculate_snr("query", "candidate")
        assert score == 0.91
        assert components["method"] == "v2"
        mock_calc.calculate_simple.assert_called_once_with(
            query="query", texts=["candidate"], iaas_score=0.9
        )

    def test_falls_back_when_no_calculator(self, tmp_path: Path) -> None:
        """Should use fallback SNR when calculator is None."""
        gate = _make_gate(tmp_path)
        gate.calculator = None

        score, components = gate._calculate_snr("query", "candidate text")
        assert isinstance(score, float)
        assert components == {"method": "fallback"}


# =============================================================================
# 13. LOAD TRUSTED Z3 PUBKEY TESTS
# =============================================================================


class TestLoadTrustedZ3Pubkey:
    """Tests for _load_trusted_z3_pubkey."""

    def test_prefers_env_var_over_file(self, tmp_path: Path) -> None:
        """Env var should take precedence over keypair.json file."""
        keypair_path = tmp_path / "keypair.json"
        with open(keypair_path, "w") as f:
            json.dump({"public_key": "file_key"}, f)

        gate = _make_gate(tmp_path, {
            "BIZRA_Z3_CERT_PUBKEY": "env_key",
            "BIZRA_KEYPAIR_PATH": str(keypair_path),
        })
        assert gate._trusted_z3_pubkey == "env_key"

    def test_returns_empty_for_corrupt_keypair_file(self, tmp_path: Path) -> None:
        """Should return empty string when keypair.json is corrupt."""
        keypair_path = tmp_path / "keypair.json"
        keypair_path.write_text("not valid json")

        gate = _make_gate(tmp_path, {"BIZRA_KEYPAIR_PATH": str(keypair_path)})
        assert gate._trusted_z3_pubkey == ""

    def test_returns_empty_for_missing_public_key_field(self, tmp_path: Path) -> None:
        """Should return empty string when keypair.json has no public_key field."""
        keypair_path = tmp_path / "keypair.json"
        with open(keypair_path, "w") as f:
            json.dump({"private_key": "secret"}, f)

        gate = _make_gate(tmp_path, {"BIZRA_KEYPAIR_PATH": str(keypair_path)})
        assert gate._trusted_z3_pubkey == ""


# =============================================================================
# 14. INTEGRATION SCENARIO TESTS
# =============================================================================


class TestIntegrationScenarios:
    """End-to-end scenario tests combining multiple behaviors."""

    @pytest.mark.asyncio
    async def test_full_lifecycle_museum_to_runtime(self, tmp_path: Path) -> None:
        """Full lifecycle: admit to museum, create cert, promote to runtime."""
        key = PrivateKeyWrapper.generate()
        gate = _make_gate(tmp_path, {"BIZRA_Z3_CERT_PUBKEY": key.public_key_hex})
        gate.calculator = None

        cid = "lifecycle_candidate"

        # Step 1: Admit to museum (high SNR text)
        text = " ".join(f"word{i}" for i in range(120))
        result = await gate.admit(text, "query", candidate_id=cid)
        assert result.status == AdmissionStatus.MUSEUM
        assert gate.get_museum_count() == 1
        assert gate.get_runtime_count() == 0

        # Step 2: Create signed Z3 certificate
        cert_data = _make_z3_cert_data(
            candidate_id=cid,
            valid=True,
            public_key=key.public_key_hex,
        )
        sig = _sign_z3_cert(cert_data, cid, key)
        cert_data["signature"] = sig
        cert_data["public_key"] = key.public_key_hex
        _write_cert_file(tmp_path / "proofs", cid, cert_data)

        # Step 3: Promote to runtime
        success = gate.promote_to_runtime(cid)
        assert success is True
        assert gate.get_museum_count() == 0
        assert gate.get_runtime_count() == 1

    @pytest.mark.asyncio
    async def test_multiple_admissions_mixed_statuses(self, tmp_path: Path) -> None:
        """Multiple candidates should receive appropriate statuses."""
        gate = _make_gate(tmp_path, {"BIZRA_Z3_CERT_ALLOW_UNSIGNED": "1"})
        gate.calculator = None

        # Runtime candidate (has valid cert)
        _write_cert_file(
            tmp_path / "proofs", "runtime_cand",
            _make_z3_cert_data(candidate_id="runtime_cand", valid=True),
        )
        r1 = await gate.admit("c1", "q", candidate_id="runtime_cand")
        assert r1.status == AdmissionStatus.RUNTIME

        # Museum candidate (high SNR)
        text = " ".join(f"w{i}" for i in range(120))
        r2 = await gate.admit(text, "q", candidate_id="museum_cand")
        assert r2.status == AdmissionStatus.MUSEUM

        # Sandbox candidate
        with patch.object(gate, "_fallback_snr", return_value=0.75):
            r3 = await gate.admit("sandbox text", "q", candidate_id="sandbox_cand")
        assert r3.status == AdmissionStatus.SANDBOX

        # Rejected candidate
        with patch.object(gate, "_fallback_snr", return_value=0.40):
            r4 = await gate.admit("bad text", "q", candidate_id="rejected_cand")
        assert r4.status == AdmissionStatus.REJECTED

        assert gate.get_runtime_count() == 1
        assert gate.get_museum_count() == 1
