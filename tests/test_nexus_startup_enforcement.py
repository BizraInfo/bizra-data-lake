"""
BIZRA Sovereign Nexus Startup Enforcement Tests
================================================
Tests that Nexus boot is FAIL-CLOSED on hardware mismatch.

These tests verify:
1. Tier-1 hardware mismatch raises SovereignNexusStartupError
2. Permissive mode bypasses enforcement (for testing only)
3. Restricted mode is entered on failure
4. Missing genesis/covenant blocks startup

This is critical security infrastructure - failure IS NOT an option.
"""

import json
import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

# Import the exception and helper
from bizra_kernel.sovereign_nexus.nexus import (
    SovereignNexusStartupError,
    _is_permissive_startup,
)


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def valid_genesis_with_covenant():
    """Create a valid genesis with hardware covenant."""
    return {
        "version": "1.0",
        "name": "BIZRA Genesis",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "node_id": "NODE0-TEST",
        "genesis_block": {
            "block_number": 0,
            "parent_hash": "0x0000000000000000000000000000000000000000",
        },
        "policy": {
            "riba_forbidden": True,
            "gharar_forbidden": True,
        },
        "hardware_covenant": {
            "fingerprint": "abc123def456",
            "tiered_covenant": {
                "tier1_root": {"cpu_model": "Intel i9", "cpu_features": ["avx2"]},
                "tier2_mutable": {"total_ram_gb": 128, "mac_address": "00:11:22:33:44:55"},
                "tier3_environmental": {"os_platform": "linux"},
            },
            "hardware_class": "MSI Titan GT77 HX",
        },
        "signature": {
            "algorithm": "Ed25519",
            "pubkey": "fake-pubkey-pem",
            "pubkey_fingerprint": "96d32bc33a67034251d3d7fe03111a1afb81b9863e41b177de7ce09a6bbc9760",
            "genesis_hash": "fake-hash",
            "signature": "fake-signature",
            "signed_at": datetime.now(timezone.utc).isoformat(),
        },
    }


@pytest.fixture
def genesis_without_covenant():
    """Create a genesis without hardware covenant."""
    return {
        "version": "1.0",
        "name": "BIZRA Genesis",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "node_id": "NODE0-TEST",
        "genesis_block": {
            "block_number": 0,
            "parent_hash": "0x0000000000000000000000000000000000000000",
        },
        "policy": {
            "riba_forbidden": True,
            "gharar_forbidden": True,
        },
    }


@pytest.fixture
def mock_verified_result():
    """Mock result for successful hardware verification."""
    return {
        "verified": True,
        "expected_fingerprint": "abc123def456",
        "current_fingerprint": "abc123def456",
        "tier_results": {
            "tier1_root": {"action": "PASS"},
            "tier2_mutable": {"action": "PASS"},
            "tier3_environmental": {"action": "PASS"},
        },
        "warnings": [],
    }


@pytest.fixture
def mock_tier1_failure_result():
    """Mock result for Tier-1 hardware verification failure."""
    return {
        "verified": False,
        "expected_fingerprint": "abc123def456",
        "current_fingerprint": "zzz999wronghw",
        "tier_results": {
            "tier1_root": {
                "action": "HARD_FAIL",
                "differences": ["CPU model mismatch: expected Intel i9, got AMD Ryzen"],
            },
            "tier2_mutable": {"action": "SKIP"},
            "tier3_environmental": {"action": "SKIP"},
        },
        "warnings": [],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS: _is_permissive_startup()
# ═══════════════════════════════════════════════════════════════════════════════


class TestPermissiveStartupDetection:
    """Test the permissive startup flag detection."""

    def test_not_set_is_not_permissive(self):
        """When env var is not set, should not be permissive."""
        with patch.dict(os.environ, {}, clear=True):
            # Ensure the var is definitely not set
            os.environ.pop("BIZRA_PERMISSIVE_STARTUP", None)
            assert _is_permissive_startup() is False

    def test_true_is_permissive(self):
        """BIZRA_PERMISSIVE_STARTUP=true should be permissive."""
        with patch.dict(os.environ, {"BIZRA_PERMISSIVE_STARTUP": "true"}):
            assert _is_permissive_startup() is True

    def test_one_is_permissive(self):
        """BIZRA_PERMISSIVE_STARTUP=1 should be permissive."""
        with patch.dict(os.environ, {"BIZRA_PERMISSIVE_STARTUP": "1"}):
            assert _is_permissive_startup() is True

    def test_yes_is_permissive(self):
        """BIZRA_PERMISSIVE_STARTUP=yes should be permissive."""
        with patch.dict(os.environ, {"BIZRA_PERMISSIVE_STARTUP": "yes"}):
            assert _is_permissive_startup() is True

    def test_false_is_not_permissive(self):
        """BIZRA_PERMISSIVE_STARTUP=false should NOT be permissive."""
        with patch.dict(os.environ, {"BIZRA_PERMISSIVE_STARTUP": "false"}):
            assert _is_permissive_startup() is False

    def test_random_string_is_not_permissive(self):
        """BIZRA_PERMISSIVE_STARTUP=xyz should NOT be permissive."""
        with patch.dict(os.environ, {"BIZRA_PERMISSIVE_STARTUP": "xyz"}):
            assert _is_permissive_startup() is False

    def test_case_insensitive(self):
        """Should be case insensitive."""
        with patch.dict(os.environ, {"BIZRA_PERMISSIVE_STARTUP": "TRUE"}):
            assert _is_permissive_startup() is True

        with patch.dict(os.environ, {"BIZRA_PERMISSIVE_STARTUP": "Yes"}):
            assert _is_permissive_startup() is True


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS: SovereignNexusStartupError
# ═══════════════════════════════════════════════════════════════════════════════


class TestSovereignNexusStartupError:
    """Test the startup error exception."""

    def test_exception_has_message(self):
        """Exception should have the message."""
        err = SovereignNexusStartupError("Test message")
        assert str(err) == "Test message"

    def test_exception_has_tier1_failed_flag(self):
        """Exception should track tier1_failed."""
        err = SovereignNexusStartupError("Test", tier1_failed=True)
        assert err.tier1_failed is True

        err2 = SovereignNexusStartupError("Test", tier1_failed=False)
        assert err2.tier1_failed is False

    def test_exception_has_restricted_mode_flag(self):
        """Exception should track restricted_mode."""
        err = SovereignNexusStartupError("Test", restricted_mode=True)
        assert err.restricted_mode is True


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS: _verify_genesis_hardware()
# ═══════════════════════════════════════════════════════════════════════════════


class TestNexusStartupEnforcement:
    """Test that Nexus startup enforces fail-closed on hardware mismatch."""

    def test_tier1_mismatch_raises_without_permissive(
        self, valid_genesis_with_covenant, mock_tier1_failure_result
    ):
        """Tier-1 mismatch must raise SovereignNexusStartupError when NOT permissive."""
        with patch.dict(os.environ, {"BIZRA_PERMISSIVE_STARTUP": "false"}):
            # Patch at source modules (where they're imported from inside the method)
            with patch("bizra_kernel.genesis_sync.load_genesis") as mock_load:
                with patch("bizra_kernel.genesis_sync.verify_hardware_covenant") as mock_verify:
                    with patch("bizra_kernel.node0_identity.Node0Identity") as mock_identity_cls:
                        # Setup mocks
                        mock_load.return_value = valid_genesis_with_covenant
                        mock_verify.return_value = mock_tier1_failure_result

                        # Mock identity
                        mock_identity = MagicMock()
                        mock_identity.verify_genesis.return_value = {"verified": True}
                        mock_identity.enter_restricted_mode = MagicMock()
                        mock_identity_cls.load_or_create.return_value = mock_identity

                        # Import and call
                        from bizra_kernel.sovereign_nexus.nexus import SovereignNexus

                        with pytest.raises(SovereignNexusStartupError) as exc_info:
                            # Create nexus - should fail on hardware verification
                            SovereignNexus(nexus_id="test-nexus")

                        assert exc_info.value.tier1_failed is True
                        assert exc_info.value.restricted_mode is True
                        assert "Tier-1 ROOT mismatch" in str(exc_info.value)

                        # Verify restricted mode was entered
                        mock_identity.enter_restricted_mode.assert_called_once()

    def test_tier1_mismatch_continues_with_permissive(
        self, valid_genesis_with_covenant, mock_tier1_failure_result
    ):
        """Tier-1 mismatch should continue when PERMISSIVE mode is enabled."""
        from bizra_kernel.sovereign_nexus.nexus import SovereignNexus

        with patch.dict(os.environ, {"BIZRA_PERMISSIVE_STARTUP": "true"}):
            with patch("bizra_kernel.genesis_sync.load_genesis") as mock_load:
                with patch("bizra_kernel.genesis_sync.verify_hardware_covenant") as mock_verify:
                    with patch("bizra_kernel.node0_identity.Node0Identity") as mock_identity_cls:
                        # Mock subsystem/adapter initialization to avoid real dependencies
                        with patch.object(SovereignNexus, "_initialize_subsystems"):
                            with patch.object(SovereignNexus, "_initialize_adapters"):
                                with patch.object(SovereignNexus, "_initialize_core_components"):
                                    with patch.object(SovereignNexus, "_wire_components"):
                                        # Setup mocks
                                        mock_load.return_value = valid_genesis_with_covenant
                                        mock_verify.return_value = mock_tier1_failure_result

                                        # Mock identity
                                        mock_identity = MagicMock()
                                        mock_identity.verify_genesis.return_value = {"verified": True}
                                        mock_identity_cls.load_or_create.return_value = mock_identity

                                        # Should NOT raise - permissive mode
                                        nexus = SovereignNexus(nexus_id="test-nexus")

                                        # Hardware verified should be False but didn't raise
                                        assert nexus._hardware_verified is False

    def test_missing_covenant_raises_without_permissive(self, genesis_without_covenant):
        """Missing hardware covenant must raise when NOT permissive."""
        with patch.dict(os.environ, {"BIZRA_PERMISSIVE_STARTUP": "false"}):
            with patch("bizra_kernel.genesis_sync.load_genesis") as mock_load:
                mock_load.return_value = genesis_without_covenant

                from bizra_kernel.sovereign_nexus.nexus import SovereignNexus

                with pytest.raises(SovereignNexusStartupError) as exc_info:
                    SovereignNexus(nexus_id="test-nexus")

                assert exc_info.value.tier1_failed is True
                assert "no hardware covenant" in str(exc_info.value)

    def test_missing_covenant_continues_with_permissive(self, genesis_without_covenant):
        """Missing hardware covenant should continue when PERMISSIVE."""
        from bizra_kernel.sovereign_nexus.nexus import SovereignNexus

        with patch.dict(os.environ, {"BIZRA_PERMISSIVE_STARTUP": "true"}):
            with patch("bizra_kernel.genesis_sync.load_genesis") as mock_load:
                with patch.object(SovereignNexus, "_initialize_subsystems"):
                    with patch.object(SovereignNexus, "_initialize_adapters"):
                        with patch.object(SovereignNexus, "_initialize_core_components"):
                            with patch.object(SovereignNexus, "_wire_components"):
                                mock_load.return_value = genesis_without_covenant

                                # Should NOT raise - permissive mode
                                nexus = SovereignNexus(nexus_id="test-nexus")
                                assert nexus._hardware_verified is True  # Permissive returns True

    def test_genesis_not_found_raises_without_permissive(self):
        """FileNotFoundError must raise when NOT permissive."""
        with patch.dict(os.environ, {"BIZRA_PERMISSIVE_STARTUP": "false"}):
            with patch("bizra_kernel.genesis_sync.load_genesis") as mock_load:
                mock_load.side_effect = FileNotFoundError("genesis.json not found")

                from bizra_kernel.sovereign_nexus.nexus import SovereignNexus

                with pytest.raises(SovereignNexusStartupError) as exc_info:
                    SovereignNexus(nexus_id="test-nexus")

                assert exc_info.value.tier1_failed is True
                assert "not found" in str(exc_info.value).lower()

    def test_successful_verification_passes(
        self, valid_genesis_with_covenant, mock_verified_result
    ):
        """Successful hardware verification should pass."""
        from bizra_kernel.sovereign_nexus.nexus import SovereignNexus

        with patch.dict(os.environ, {"BIZRA_PERMISSIVE_STARTUP": "false"}):
            with patch("bizra_kernel.genesis_sync.load_genesis") as mock_load:
                with patch("bizra_kernel.genesis_sync.verify_hardware_covenant") as mock_verify:
                    with patch("bizra_kernel.node0_identity.Node0Identity") as mock_identity_cls:
                        with patch.object(SovereignNexus, "_initialize_subsystems"):
                            with patch.object(SovereignNexus, "_initialize_adapters"):
                                with patch.object(SovereignNexus, "_initialize_core_components"):
                                    with patch.object(SovereignNexus, "_wire_components"):
                                        mock_load.return_value = valid_genesis_with_covenant
                                        mock_verify.return_value = mock_verified_result

                                        mock_identity = MagicMock()
                                        mock_identity.verify_genesis.return_value = {"verified": True}
                                        mock_identity_cls.load_or_create.return_value = mock_identity

                                        # Should NOT raise
                                        nexus = SovereignNexus(nexus_id="test-nexus")
                                        assert nexus._hardware_verified is True


class TestSignatureVerificationEnforcement:
    """Test that genesis signature verification is enforced."""

    def test_invalid_signature_raises_without_permissive(self, valid_genesis_with_covenant):
        """Invalid genesis signature must raise when NOT permissive."""
        with patch.dict(os.environ, {"BIZRA_PERMISSIVE_STARTUP": "false"}):
            with patch("bizra_kernel.genesis_sync.load_genesis") as mock_load:
                with patch("bizra_kernel.node0_identity.Node0Identity") as mock_identity_cls:
                    mock_load.return_value = valid_genesis_with_covenant

                    # Mock identity with FAILED signature verification
                    mock_identity = MagicMock()
                    mock_identity.verify_genesis.return_value = {
                        "verified": False,
                        "error": "Invalid signature - genesis may be tampered",
                    }
                    mock_identity.enter_restricted_mode = MagicMock()
                    mock_identity_cls.load_or_create.return_value = mock_identity

                    from bizra_kernel.sovereign_nexus.nexus import SovereignNexus

                    with pytest.raises(SovereignNexusStartupError) as exc_info:
                        SovereignNexus(nexus_id="test-nexus")

                    assert exc_info.value.tier1_failed is True
                    assert exc_info.value.restricted_mode is True
                    assert "signature" in str(exc_info.value).lower()

                    # Verify restricted mode was entered
                    mock_identity.enter_restricted_mode.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════════
# IMPORT GUARD
# ═══════════════════════════════════════════════════════════════════════════════


# Need to import here to avoid circular imports during patching
from bizra_kernel.sovereign_nexus.nexus import SovereignNexus


# ═══════════════════════════════════════════════════════════════════════════════
# RUN TESTS
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
