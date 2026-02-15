"""
Tests for GenesisCeremony — the birth event of a sovereign node.

Verifies:
    1. Full ceremony produces valid CeremonyResult
    2. Genesis Block₀ hash is deterministic
    3. Ceremony is resumable (idempotent on existing identity)
    4. Graceful degradation when subsystems fail
    5. JSON output mode works
    6. CLI argument parsing includes genesis command

Standing on: pytest (Krekel, 2004) + fixture-based isolation.
"""

import json
import hashlib
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.sovereign.genesis_ceremony import (
    CeremonyResult,
    GenesisCeremony,
    GenesisBlock,
    GENESIS_VERSION,
    GENESIS_CODENAME,
)


class TestGenesisBlock:
    """Block₀ creation and hash computation."""

    def test_01_default_block(self):
        block = GenesisBlock()
        assert block.version == GENESIS_VERSION
        assert block.codename == GENESIS_CODENAME
        assert block.parent_hash == "0" * 64

    def test_02_hash_deterministic(self):
        block = GenesisBlock(
            node_id="BIZRA-TEST001",
            public_key="abc123",
            timestamp_utc="2026-02-15T12:00:00+00:00",
        )
        h1 = block.compute_hash()
        h2 = block.compute_hash()
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_03_different_inputs_different_hash(self):
        block_a = GenesisBlock(node_id="NODE-A")
        block_b = GenesisBlock(node_id="NODE-B")
        assert block_a.compute_hash() != block_b.compute_hash()

    def test_04_hash_matches_manual(self):
        block = GenesisBlock(node_id="TEST")
        canonical = json.dumps(asdict(block), sort_keys=True, separators=(",", ":"))
        expected = hashlib.sha256(canonical.encode()).hexdigest()
        assert block.compute_hash() == expected


class TestCeremonyResult:
    """CeremonyResult dataclass."""

    def test_05_default_result(self):
        r = CeremonyResult()
        assert not r.success
        assert r.node_id == ""
        assert r.errors == []

    def test_06_populated_result(self):
        r = CeremonyResult(
            success=True,
            node_id="BIZRA-GENESIS01",
            pat_count=7,
            sat_count=5,
            genesis_hash="abc123",
        )
        assert r.success
        assert r.pat_count == 7


class TestGenesisCeremony:
    """Full ceremony integration (mocked subsystems)."""

    @pytest.fixture
    def tmp_node_dir(self, tmp_path):
        return tmp_path / "bizra-test-node"

    @pytest.fixture
    def mock_credentials(self):
        """Fake NodeCredentials for testing."""
        creds = MagicMock()
        creds.node_id = "BIZRA-TEST0001"
        creds.public_key = "ed25519_fake_public_key_0123456789abcdef"
        creds.private_key = "ed25519_fake_private_key"
        creds.sovereignty_tier = "SEED"
        creds.sovereignty_score = 0.0
        creds.pat_agent_ids = [
            "BIZRA-TEST0001-WRK-001",
            "BIZRA-TEST0001-RSC-002",
            "BIZRA-TEST0001-GRD-003",
        ]
        creds.sat_agent_ids = [
            "BIZRA-TEST0001-VAL-001",
            "BIZRA-TEST0001-CRD-002",
        ]
        return creds

    def test_07_ceremony_new_identity(self, tmp_node_dir, mock_credentials):
        """Full ceremony with mocked onboarding."""
        with patch("core.pat.onboarding.OnboardingWizard") as MockWizard:
            instance = MockWizard.return_value
            instance.load_existing_credentials.return_value = None
            instance.onboard.return_value = mock_credentials

            ceremony = GenesisCeremony(
                name="test",
                node_dir=tmp_node_dir,
                json_output=True,
            )
            result = ceremony.run()

            assert result.success
            assert result.node_id == "BIZRA-TEST0001"
            assert len(result.genesis_hash) == 64
            assert result.pat_count >= 0

    def test_08_ceremony_existing_identity(self, tmp_node_dir, mock_credentials):
        """Ceremony resumes from existing credentials."""
        with patch("core.pat.onboarding.OnboardingWizard") as MockWizard:
            instance = MockWizard.return_value
            instance.load_existing_credentials.return_value = mock_credentials

            ceremony = GenesisCeremony(
                node_dir=tmp_node_dir,
                json_output=True,
            )
            result = ceremony.run()

            assert result.success
            assert result.node_id == "BIZRA-TEST0001"
            # onboard should NOT have been called
            instance.onboard.assert_not_called()

    def test_09_ceremony_persists_block0(self, tmp_node_dir, mock_credentials):
        """Block₀ is persisted to genesis/block_0.json."""
        with patch("core.pat.onboarding.OnboardingWizard") as MockWizard:
            instance = MockWizard.return_value
            instance.load_existing_credentials.return_value = mock_credentials

            ceremony = GenesisCeremony(
                node_dir=tmp_node_dir,
                json_output=True,
            )
            result = ceremony.run()

            block_file = tmp_node_dir / "genesis" / "block_0.json"
            assert block_file.exists()
            block_data = json.loads(block_file.read_text())
            assert block_data["node_id"] == "BIZRA-TEST0001"
            assert "hash" in block_data

    def test_10_ceremony_graceful_degradation(self, tmp_node_dir, mock_credentials):
        """Ceremony succeeds even if guild/quest fail."""
        with patch("core.pat.onboarding.OnboardingWizard") as MockWizard:
            instance = MockWizard.return_value
            instance.load_existing_credentials.return_value = mock_credentials

            ceremony = GenesisCeremony(
                node_dir=tmp_node_dir,
                json_output=True,
            )
            # Deliberately don't mock guild/quest — they may fail
            result = ceremony.run()

            # Identity should succeed regardless
            assert result.success
            assert result.node_id == "BIZRA-TEST0001"

    def test_11_ceremony_json_output(self, tmp_node_dir, mock_credentials, capsys):
        """JSON output produces valid JSON."""
        with patch("core.pat.onboarding.OnboardingWizard") as MockWizard:
            instance = MockWizard.return_value
            instance.load_existing_credentials.return_value = mock_credentials

            from core.sovereign.genesis_ceremony import run_genesis_ceremony
            run_genesis_ceremony(
                node_dir=str(tmp_node_dir),
                json_output=True,
            )

            captured = capsys.readouterr()
            data = json.loads(captured.out)
            assert data["success"] is True
            assert "node_id" in data


class TestCLIIntegration:
    """Verify genesis command is wired into __main__.py."""

    def test_12_genesis_in_argparse(self):
        """The genesis subparser exists."""
        import sys
        from unittest.mock import patch as mock_patch

        with mock_patch.object(sys, "argv", ["prog", "genesis", "--help"]):
            from core.sovereign.__main__ import main
            with pytest.raises(SystemExit) as exc_info:
                main()
            # --help exits with 0
            assert exc_info.value.code == 0
