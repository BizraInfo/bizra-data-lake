"""
BIZRA Node0 Sovereignty Acceptance Tests
=========================================
These tests verify the cryptographic hardening of Node0 identity.

Run with: pytest tests/test_node0_sovereignty.py -v

Test Categories:
1. Genesis signature verification
2. Tier-1 mismatch enforcement (restricted mode)
3. Tier-2 mismatch requires attestation
4. Genesis sync tamper behavior
"""

import json
import os
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 1: GENESIS SIGNATURE VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestGenesisSignature:
    """Tests for genesis cryptographic signature."""

    def test_signed_genesis_verifies(self):
        """A properly signed genesis should verify successfully."""
        pytest.importorskip("cryptography")

        from bizra_kernel.node0_identity import Node0Identity, CRYPTO_AVAILABLE

        if not CRYPTO_AVAILABLE:
            pytest.skip("cryptography library not available")

        # Create test identity with temp keys using dependency injection
        with tempfile.TemporaryDirectory() as tmpdir:
            key_dir = Path(tmpdir) / "keys"
            att_dir = Path(tmpdir) / "attestations"

            identity = Node0Identity.load_or_create(
                force_create=True,
                key_dir=key_dir,
                attestation_dir=att_dir,
            )

            # Create test genesis
            genesis = {
                "version": "1.0",
                "name": "Test Genesis",
                "timestamp": "2026-01-22T00:00:00Z",
                "genesis_block": {"block_number": 0},
            }

            # Sign it
            signature = identity.sign_genesis(genesis)
            genesis["signature"] = signature

            # Verify it
            result = identity.verify_genesis(genesis)

            assert result["verified"] is True
            assert "pubkey_fingerprint" in result

    def test_modified_genesis_fails_verification(self):
        """Modifying one byte in genesis should fail verification."""
        pytest.importorskip("cryptography")

        from bizra_kernel.node0_identity import Node0Identity, CRYPTO_AVAILABLE

        if not CRYPTO_AVAILABLE:
            pytest.skip("cryptography library not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            key_dir = Path(tmpdir) / "keys"
            att_dir = Path(tmpdir) / "attestations"

            identity = Node0Identity.load_or_create(
                force_create=True,
                key_dir=key_dir,
                attestation_dir=att_dir,
            )

            # Create and sign genesis
            genesis = {
                "version": "1.0",
                "name": "Test Genesis",
                "timestamp": "2026-01-22T00:00:00Z",
            }
            signature = identity.sign_genesis(genesis)
            genesis["signature"] = signature

            # Modify one byte
            genesis["name"] = "Test Genesis!"  # Added !

            # Verify should fail
            result = identity.verify_genesis(genesis)

            assert result["verified"] is False
            assert "mismatch" in result.get("error", "").lower() or "modified" in result.get("error", "").lower()

    def test_unsigned_genesis_detected(self):
        """Genesis without signature should be detected."""
        pytest.importorskip("cryptography")

        from bizra_kernel.node0_identity import Node0Identity, CRYPTO_AVAILABLE

        if not CRYPTO_AVAILABLE:
            pytest.skip("cryptography library not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            key_dir = Path(tmpdir) / "keys"
            att_dir = Path(tmpdir) / "attestations"

            identity = Node0Identity.load_or_create(
                force_create=True,
                key_dir=key_dir,
                attestation_dir=att_dir,
            )

            # Genesis without signature
            genesis = {
                "version": "1.0",
                "name": "Unsigned Genesis",
            }

            result = identity.verify_genesis(genesis)

            assert result["verified"] is False
            assert "No signature" in result.get("error", "")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 2: TIER-1 MISMATCH ENFORCEMENT
# ═══════════════════════════════════════════════════════════════════════════════

class TestTier1Enforcement:
    """Tests for Tier-1 hardware mismatch enforcement."""

    def test_tier1_mismatch_enters_restricted_mode(self):
        """Tier-1 mismatch should activate restricted mode."""
        pytest.importorskip("cryptography")

        from bizra_kernel.node0_identity import Node0Identity, CRYPTO_AVAILABLE

        if not CRYPTO_AVAILABLE:
            pytest.skip("cryptography library not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            key_dir = Path(tmpdir) / "keys"
            att_dir = Path(tmpdir) / "attestations"

            with patch("bizra_kernel.node0_identity.RESTRICTED_MODE_FLAG", Path(tmpdir) / "restricted"):
                identity = Node0Identity.load_or_create(
                    force_create=True,
                    key_dir=key_dir,
                    attestation_dir=att_dir,
                )

                # Enter restricted mode
                identity.enter_restricted_mode("Tier-1 mismatch: wrong CPU")

                assert identity.is_restricted is True
                assert "signing" in identity.restricted_state.disabled_capabilities

    def test_restricted_mode_blocks_signing(self):
        """Signing should be blocked in restricted mode."""
        pytest.importorskip("cryptography")

        from bizra_kernel.node0_identity import Node0Identity, CRYPTO_AVAILABLE

        if not CRYPTO_AVAILABLE:
            pytest.skip("cryptography library not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            key_dir = Path(tmpdir) / "keys"
            att_dir = Path(tmpdir) / "attestations"

            with patch("bizra_kernel.node0_identity.RESTRICTED_MODE_FLAG", Path(tmpdir) / "restricted"):
                identity = Node0Identity.load_or_create(
                    force_create=True,
                    key_dir=key_dir,
                    attestation_dir=att_dir,
                )
                identity.enter_restricted_mode("Test restriction")

                genesis = {"version": "1.0"}

                with pytest.raises(RuntimeError, match="restricted mode"):
                    identity.sign_genesis(genesis)

    def test_tier1_mismatch_detection(self):
        """Tier-1 mismatch should be correctly detected."""
        from bizra_kernel.hardware_fingerprint import verify_fingerprint

        # Expected fingerprint is NODE0's
        expected = "f63681b9230613cc8d3e081ac4a4e6e9840db17beef6bb21aad07729a075acf8"

        # Mock a different fingerprint
        with patch("bizra_kernel.hardware_fingerprint.generate_fingerprint") as mock_fp:
            mock_fp.return_value = {
                "fingerprint": "wrong_fingerprint_not_node0",
                "tiered_covenant": {
                    "tier_1_root": {
                        "hash": "wrong_fingerprint_not_node0",
                    },
                    "tier_2_mutable": {"hash": "some_hash"},
                    "tier_3_contextual": {"hash": "some_hash"},
                },
            }

            result = verify_fingerprint(expected, None)

            assert result["verified"] is False
            assert result["tier_results"]["tier_1_root"]["action"] == "HARD_FAIL"


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 3: TIER-2 ATTESTATION REQUIREMENT
# ═══════════════════════════════════════════════════════════════════════════════

class TestTier2Attestation:
    """Tests for Tier-2 hardware change attestation."""

    def test_tier2_mismatch_without_attestation_fails(self):
        """Tier-2 mismatch without attestation should fail."""
        pytest.importorskip("cryptography")

        from bizra_kernel.node0_identity import Node0Identity, CRYPTO_AVAILABLE

        if not CRYPTO_AVAILABLE:
            pytest.skip("cryptography library not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            key_dir = Path(tmpdir) / "keys"
            att_dir = Path(tmpdir) / "attestations"

            identity = Node0Identity.load_or_create(
                force_create=True,
                key_dir=key_dir,
                attestation_dir=att_dir,
            )

            # Check Tier-2 mismatch without attestation
            result = identity.verify_tier2_attestation(
                current_tier2_hash="new_ram_hash",
                expected_tier2_hash="old_ram_hash",
            )

            assert result["verified"] is False
            assert result["attestation_required"] is True

    def test_tier2_mismatch_with_valid_attestation_passes(self):
        """Tier-2 mismatch with valid attestation should pass."""
        pytest.importorskip("cryptography")

        from bizra_kernel.node0_identity import Node0Identity, CRYPTO_AVAILABLE

        if not CRYPTO_AVAILABLE:
            pytest.skip("cryptography library not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            key_dir = Path(tmpdir) / "keys"
            att_dir = Path(tmpdir) / "attestations"

            identity = Node0Identity.load_or_create(
                force_create=True,
                key_dir=key_dir,
                attestation_dir=att_dir,
            )

            # Create attestation
            attestation = identity.create_tier2_attestation(
                previous_hash="old_ram_hash",
                new_hash="new_ram_hash",
                reason="RAM upgrade from 64GB to 128GB",
                changed_components={"ram": "128GB DDR5"},
            )

            # Now verify
            result = identity.verify_tier2_attestation(
                current_tier2_hash="new_ram_hash",
                expected_tier2_hash="old_ram_hash",
            )

            assert result["verified"] is True
            assert result["attestation_valid"] is True


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 4: GENESIS SYNC TAMPER BEHAVIOR
# ═══════════════════════════════════════════════════════════════════════════════

class TestGenesisSyncTamper:
    """Tests for genesis sync tamper detection."""

    def test_different_genesis_in_data_lake_raises_error(self):
        """Different genesis in Data Lake should raise GenesisTamperError."""
        from bizra_kernel.genesis_sync import (
            sync_genesis, GenesisLocations, GenesisTamperError
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "source" / "genesis.json"
            target_path = Path(tmpdir) / "datalake" / "genesis.json"
            local_path = Path(tmpdir) / "local" / "genesis.json"
            backup_path = Path(tmpdir) / "backups"
            archive_path = Path(tmpdir) / "archive"

            source_path.parent.mkdir(parents=True)
            target_path.parent.mkdir(parents=True)
            local_path.parent.mkdir(parents=True)

            # Create source genesis (with all required fields)
            source_genesis = {
                "version": "1.0",
                "name": "Source Genesis",
                "timestamp": "2026-01-22T00:00:00Z",
                "node_id": "NODE0-TEST",
                "genesis_block": {"block_number": 0, "parent_hash": "0x000"},
                "policy": {"riba_forbidden": True, "gharar_forbidden": True},
            }
            source_path.write_text(json.dumps(source_genesis))

            # Create DIFFERENT target genesis (tampered)
            tampered_genesis = {
                "version": "1.0",
                "name": "TAMPERED Genesis",  # Different!
                "timestamp": "2026-01-22T00:00:00Z",
                "node_id": "NODE0-TEST",
                "genesis_block": {"block_number": 0, "parent_hash": "0x000"},
                "policy": {"riba_forbidden": True, "gharar_forbidden": True},
            }
            target_path.write_text(json.dumps(tampered_genesis))

            # Create test locations using dependency injection
            test_locations = GenesisLocations(
                taskmaster=source_path,
                data_lake=target_path,
                local=local_path,
                backup_dir=backup_path,
                local_archive=archive_path,
            )

            # Should raise GenesisTamperError
            with pytest.raises(GenesisTamperError):
                sync_genesis(locations=test_locations)

    def test_same_genesis_does_not_overwrite(self):
        """Identical genesis should not be overwritten."""
        from bizra_kernel.genesis_sync import sync_genesis, GenesisLocations

        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "source" / "genesis.json"
            target_path = Path(tmpdir) / "datalake" / "genesis.json"
            local_path = Path(tmpdir) / "local" / "genesis.json"
            backup_path = Path(tmpdir) / "backups"
            archive_path = Path(tmpdir) / "archive"

            source_path.parent.mkdir(parents=True)
            target_path.parent.mkdir(parents=True)
            local_path.parent.mkdir(parents=True)

            # Create identical genesis (with all required fields)
            genesis = {
                "version": "1.0",
                "name": "Same Genesis",
                "timestamp": "2026-01-22T00:00:00Z",
                "node_id": "NODE0-TEST",
                "genesis_block": {"block_number": 0, "parent_hash": "0x000"},
                "policy": {"riba_forbidden": True, "gharar_forbidden": True},
            }
            source_path.write_text(json.dumps(genesis))
            target_path.write_text(json.dumps(genesis))

            # Create test locations using dependency injection
            test_locations = GenesisLocations(
                taskmaster=source_path,
                data_lake=target_path,
                local=local_path,
                backup_dir=backup_path,
                local_archive=archive_path,
            )

            result = sync_genesis(locations=test_locations)

            assert result["success"] is True
            assert any(
                loc["status"] == "already_synced"
                for loc in result["locations_synced"]
            )

    def test_local_mismatch_creates_archive(self):
        """Local genesis mismatch should create archive, not overwrite."""
        from bizra_kernel.genesis_sync import sync_genesis, GenesisLocations

        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "source" / "genesis.json"
            datalake_path = Path(tmpdir) / "datalake" / "genesis.json"
            local_path = Path(tmpdir) / "local" / "genesis.json"
            backup_path = Path(tmpdir) / "backups"
            archive_path = Path(tmpdir) / "archive"

            source_path.parent.mkdir(parents=True)
            datalake_path.parent.mkdir(parents=True)
            local_path.parent.mkdir(parents=True)

            # Create source genesis (with all required fields)
            source_genesis = {
                "version": "1.0",
                "name": "Source Genesis",
                "timestamp": "2026-01-22T00:00:00Z",
                "node_id": "NODE0-TEST",
                "genesis_block": {"block_number": 0, "parent_hash": "0x000"},
                "policy": {"riba_forbidden": True, "gharar_forbidden": True},
            }
            source_path.write_text(json.dumps(source_genesis))

            # Create different local genesis (will be archived)
            local_genesis = {
                "version": "1.0",
                "name": "Old Local Genesis",
                "timestamp": "2026-01-22T00:00:00Z",
                "node_id": "NODE0-TEST",
                "genesis_block": {"block_number": 0, "parent_hash": "0x000"},
                "policy": {"riba_forbidden": True, "gharar_forbidden": True},
            }
            local_path.write_text(json.dumps(local_genesis))

            # Create test locations using dependency injection
            test_locations = GenesisLocations(
                taskmaster=source_path,
                data_lake=datalake_path,  # Doesn't exist, will be created
                local=local_path,
                backup_dir=backup_path,
                local_archive=archive_path,
            )

            result = sync_genesis(locations=test_locations)

            # Check archive was created
            assert archive_path.exists(), f"Archive path should exist: {archive_path}"
            archive_files = list(archive_path.glob("genesis_block_*.json"))
            assert len(archive_files) >= 1, "Should have archived old genesis"

            # Check tamper receipt was created
            tamper_receipts = list(archive_path.glob("tamper_receipt_*.json"))
            assert len(tamper_receipts) >= 1, "Should have created tamper receipt"


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 5: WSL/WINDOWS DUAL-CONTEXT (TIER-1 STABILITY)
# ═══════════════════════════════════════════════════════════════════════════════

class TestPlatformStability:
    """Tests for cross-platform fingerprint stability."""

    def test_tier1_excludes_volatile_components(self):
        """Tier-1 should only include stable hardware identifiers."""
        from bizra_kernel.hardware_fingerprint import generate_fingerprint

        fp = generate_fingerprint()
        tier_1 = fp["tiered_covenant"]["tier_1_root"]["components"]

        # Tier-1 should have CPU, GPU, platform
        assert "cpu_fingerprint" in tier_1
        assert "gpu_fingerprint" in tier_1
        assert "platform_signature" in tier_1

        # Tier-1 should NOT have volatile components
        assert "mac_address" not in tier_1
        assert "hostname" not in tier_1
        assert "ram_signature" not in tier_1

    def test_tier2_contains_mutable_components(self):
        """Tier-2 should contain mutable but trackable components."""
        from bizra_kernel.hardware_fingerprint import generate_fingerprint

        fp = generate_fingerprint()
        tier_2 = fp["tiered_covenant"]["tier_2_mutable"]["components"]

        # Tier-2 should have mutable components
        assert "ram_signature" in tier_2
        assert "mac_address" in tier_2
        assert "hostname" in tier_2

    def test_tier3_contains_contextual_components(self):
        """Tier-3 should contain environmental context."""
        from bizra_kernel.hardware_fingerprint import generate_fingerprint

        fp = generate_fingerprint()
        tier_3 = fp["tiered_covenant"]["tier_3_contextual"]["components"]

        # Tier-3 should have contextual components
        assert "os_fingerprint" in tier_3
        assert "wsl_context" in tier_3


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TEST
# ═══════════════════════════════════════════════════════════════════════════════

class TestFullCoronationFlow:
    """Integration test for the complete coronation flow."""

    def test_complete_coronation_flow(self):
        """Test the complete Node0 coronation flow."""
        pytest.importorskip("cryptography")

        from bizra_kernel.node0_identity import Node0Identity, CRYPTO_AVAILABLE
        from bizra_kernel.hardware_fingerprint import generate_fingerprint

        if not CRYPTO_AVAILABLE:
            pytest.skip("cryptography library not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            key_dir = Path(tmpdir) / "keys"
            att_dir = Path(tmpdir) / "attestations"

            with patch("bizra_kernel.node0_identity.RESTRICTED_MODE_FLAG", Path(tmpdir) / "restricted"):
                # 1. Create identity using dependency injection
                identity = Node0Identity.load_or_create(
                    force_create=True,
                    key_dir=key_dir,
                    attestation_dir=att_dir,
                )
                assert identity.has_private_key

                # 2. Generate hardware fingerprint
                fp = generate_fingerprint()
                assert "fingerprint" in fp
                assert "tiered_covenant" in fp

                # 3. Create genesis
                genesis = {
                    "version": "1.0",
                    "name": "BIZRA Genesis",
                    "genesis_block": {"block_number": 0},
                    "hardware_covenant": {
                        "fingerprint": fp["fingerprint"],
                        "tiered_covenant": fp["tiered_covenant"],
                    },
                }

                # 4. Sign genesis
                signature = identity.sign_genesis(genesis)
                genesis["signature"] = signature
                assert "signature" in signature

                # 5. Verify genesis
                result = identity.verify_genesis(genesis)
                assert result["verified"] is True

                # 6. Identity is NOT restricted
                assert identity.is_restricted is False

                print("\n✅ Complete coronation flow passed!")
                print(f"   Public Key: {identity.public_key_fingerprint[:32]}...")
                print(f"   Hardware:   {fp['fingerprint'][:32]}...")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 6: LEGACY KEY DETECTION (prevents silent key rotation)
# ═══════════════════════════════════════════════════════════════════════════════

class TestLegacyKeyDetection:
    """Tests for legacy key fallback to prevent silent Node0 identity rotation."""

    def test_legacy_keys_detected_over_new_default(self):
        """If legacy keys exist, they should be used instead of creating new ones."""
        pytest.importorskip("cryptography")

        from bizra_kernel.node0_identity import (
            Node0Identity, CRYPTO_AVAILABLE,
            _get_vault_dir, _dir_has_keys, DEFAULT_KEY_DIR, LEGACY_KEY_DIRS
        )

        if not CRYPTO_AVAILABLE:
            pytest.skip("cryptography library not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Simulate legacy path structure
            legacy_dir = Path(tmpdir) / "legacy_keys"
            new_default = Path(tmpdir) / "new_default"

            # Create keys in legacy location
            identity_legacy = Node0Identity.load_or_create(
                force_create=True,
                key_dir=legacy_dir,
            )
            legacy_fingerprint = identity_legacy.public_key_fingerprint

            # Verify keys exist in legacy
            assert _dir_has_keys(legacy_dir)

            # Now simulate _get_vault_dir with patched paths
            with patch("bizra_kernel.node0_identity.DEFAULT_KEY_DIR", new_default):
                with patch("bizra_kernel.node0_identity.LEGACY_KEY_DIRS", [legacy_dir]):
                    # Should find and use legacy keys
                    resolved_dir = _get_vault_dir()
                    assert resolved_dir == legacy_dir

            # Load identity from the resolved legacy path
            identity_loaded = Node0Identity.load_or_create(key_dir=legacy_dir)

            # CRITICAL: Must be the SAME identity (same public key)
            assert identity_loaded.public_key_fingerprint == legacy_fingerprint

    def test_new_default_used_when_no_legacy_exists(self):
        """Fresh installation should use new secure default path."""
        from bizra_kernel.node0_identity import _get_vault_dir, DEFAULT_KEY_DIR

        with tempfile.TemporaryDirectory() as tmpdir:
            new_default = Path(tmpdir) / "new_default"
            empty_legacy = Path(tmpdir) / "empty_legacy"

            with patch("bizra_kernel.node0_identity.DEFAULT_KEY_DIR", new_default):
                with patch("bizra_kernel.node0_identity.LEGACY_KEY_DIRS", [empty_legacy]):
                    with patch.dict(os.environ, {}, clear=False):
                        # Remove env override if present
                        os.environ.pop("BIZRA_VAULT_DIR", None)

                        resolved_dir = _get_vault_dir()
                        assert resolved_dir == new_default

    def test_env_override_takes_priority(self):
        """BIZRA_VAULT_DIR env var should take priority over all defaults."""
        from bizra_kernel.node0_identity import _get_vault_dir

        with tempfile.TemporaryDirectory() as tmpdir:
            env_override = Path(tmpdir) / "env_override"
            new_default = Path(tmpdir) / "new_default"
            legacy_dir = Path(tmpdir) / "legacy"

            # Create keys in legacy (should be ignored due to env override)
            legacy_dir.mkdir(parents=True)
            (legacy_dir / "node0_signing.key").write_text("dummy")
            (legacy_dir / "node0_signing.pub").write_text("dummy")

            with patch("bizra_kernel.node0_identity.DEFAULT_KEY_DIR", new_default):
                with patch("bizra_kernel.node0_identity.LEGACY_KEY_DIRS", [legacy_dir]):
                    with patch.dict(os.environ, {"BIZRA_VAULT_DIR": str(env_override)}):
                        resolved_dir = _get_vault_dir()
                        assert resolved_dir == env_override

    def test_windows_mount_detection(self):
        """Should detect Windows mounts (any /mnt/<letter>/)."""
        from bizra_kernel.node0_identity import _is_windows_mount

        # Should detect as Windows mount
        assert _is_windows_mount(Path("/mnt/c/some/path")) is True
        assert _is_windows_mount(Path("/mnt/d/other/path")) is True
        assert _is_windows_mount(Path("/mnt/z/another/path")) is True

        # Should NOT detect as Windows mount
        assert _is_windows_mount(Path("/home/user/.bizra")) is False
        assert _is_windows_mount(Path("/var/lib/bizra")) is False
        assert _is_windows_mount(Path("/mnt/wsl/some/path")) is False  # WSL mount, not Windows
        assert _is_windows_mount(Path("/mnt/cc/fake")) is False  # Not single letter


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
