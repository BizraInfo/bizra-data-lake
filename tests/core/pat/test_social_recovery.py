"""
Tests for Shamir Secret Sharing — Social Recovery
"""

import secrets

import pytest

from core.pat.social_recovery import (
    DEFAULT_THRESHOLD,
    DEFAULT_TOTAL_SHARES,
    GuardianRegistry,
    RecoveryCeremony,
    ShamirReconstructor,
    ShamirSplitter,
    Share,
    generate_recovery_shares,
)


class TestShamirSplitter:
    """Test the ShamirSplitter."""

    def test_split_creates_correct_number_of_shares(self):
        splitter = ShamirSplitter(threshold=3, total_shares=5)
        secret = secrets.token_bytes(32)
        shares = splitter.split(secret)
        assert len(shares) == 5

    def test_split_shares_have_correct_metadata(self):
        splitter = ShamirSplitter(threshold=3, total_shares=5)
        secret = secrets.token_bytes(32)
        shares = splitter.split(secret)
        for i, share in enumerate(shares, 1):
            assert share.index == i
            assert share.threshold == 3
            assert share.total_shares == 5

    def test_split_shares_are_unique(self):
        splitter = ShamirSplitter(threshold=3, total_shares=5)
        secret = secrets.token_bytes(32)
        shares = splitter.split(secret)
        values = [s.value for s in shares]
        assert len(set(values)) == len(values)

    def test_split_invalid_threshold(self):
        with pytest.raises(ValueError, match="at least 2"):
            ShamirSplitter(threshold=1, total_shares=5)

    def test_split_threshold_exceeds_total(self):
        with pytest.raises(ValueError, match="Total shares must be >= threshold"):
            ShamirSplitter(threshold=5, total_shares=3)


class TestShamirReconstructor:
    """Test the ShamirReconstructor."""

    def test_reconstruct_with_exact_threshold(self):
        secret = secrets.token_bytes(32)
        splitter = ShamirSplitter(threshold=3, total_shares=5)
        shares = splitter.split(secret)

        recovered = ShamirReconstructor.reconstruct(shares[:3])
        assert recovered == secret

    def test_reconstruct_with_all_shares(self):
        secret = secrets.token_bytes(32)
        splitter = ShamirSplitter(threshold=3, total_shares=5)
        shares = splitter.split(secret)

        recovered = ShamirReconstructor.reconstruct(shares)
        assert recovered == secret

    def test_reconstruct_with_different_share_subsets(self):
        secret = secrets.token_bytes(32)
        splitter = ShamirSplitter(threshold=3, total_shares=5)
        shares = splitter.split(secret)

        # Any 3 shares should work
        subsets = [
            [shares[0], shares[1], shares[2]],
            [shares[0], shares[2], shares[4]],
            [shares[1], shares[3], shares[4]],
            [shares[2], shares[3], shares[4]],
        ]

        for subset in subsets:
            recovered = ShamirReconstructor.reconstruct(subset)
            assert recovered == secret

    def test_reconstruct_insufficient_shares(self):
        secret = secrets.token_bytes(32)
        splitter = ShamirSplitter(threshold=3, total_shares=5)
        shares = splitter.split(secret)

        with pytest.raises(ValueError, match="Need at least"):
            ShamirReconstructor.reconstruct(shares[:2])

    def test_reconstruct_no_shares(self):
        with pytest.raises(ValueError, match="No shares"):
            ShamirReconstructor.reconstruct([])

    def test_determinism_100_iterations(self):
        """Verify reconstruction is deterministic across 100 iterations."""
        secret = secrets.token_bytes(32)
        splitter = ShamirSplitter(threshold=3, total_shares=5)
        shares = splitter.split(secret)

        for _ in range(100):
            recovered = ShamirReconstructor.reconstruct(shares[:3])
            assert recovered == secret


class TestShareSerialization:
    """Test share serialization."""

    def test_share_to_hex_roundtrip(self):
        share = Share(index=1, value=12345, threshold=3, total_shares=5)
        hex_val = share.to_hex()
        recovered = Share.from_hex(1, hex_val, 3, 5)
        assert recovered.value == share.value

    def test_share_to_dict(self):
        share = Share(index=1, value=12345, threshold=3, total_shares=5)
        d = share.to_dict()
        assert d["index"] == 1
        assert d["threshold"] == 3
        assert d["total_shares"] == 5
        assert "value_hex" in d
        assert "share_hash" in d


class TestGuardianRegistry:
    """Test the GuardianRegistry."""

    def test_register_guardian(self):
        registry = GuardianRegistry(node_id="BIZRA-12345678")
        guardian = registry.register_guardian(
            display_name="Alice",
            share_index=1,
            contact_info="alice@example.com",
        )
        assert guardian.display_name == "Alice"
        assert guardian.share_index == 1
        assert guardian.contact_hash != ""
        assert len(registry.guardians) == 1

    def test_get_guardian_by_index(self):
        registry = GuardianRegistry(node_id="BIZRA-12345678")
        registry.register_guardian(display_name="Alice", share_index=1)
        registry.register_guardian(display_name="Bob", share_index=2)

        found = registry.get_guardian_by_index(2)
        assert found is not None
        assert found.display_name == "Bob"

    def test_get_guardian_missing_index(self):
        registry = GuardianRegistry(node_id="BIZRA-12345678")
        assert registry.get_guardian_by_index(99) is None


class TestRecoveryCeremony:
    """Test the RecoveryCeremony."""

    def test_ceremony_flow(self):
        secret = secrets.token_bytes(32)
        splitter = ShamirSplitter(threshold=3, total_shares=5)
        shares = splitter.split(secret)

        ceremony = RecoveryCeremony(node_id="BIZRA-12345678", threshold=3)
        assert not ceremony.is_ready
        assert ceremony.shares_needed == 3

        ceremony.submit_share(shares[0])
        assert ceremony.shares_collected == 1
        assert ceremony.shares_needed == 2

        ceremony.submit_share(shares[2])
        ceremony.submit_share(shares[4])
        assert ceremony.is_ready

        result = ceremony.reconstruct()
        assert result.success
        assert result.recovered_key == secret
        assert result.shares_used == 3

    def test_ceremony_rejects_duplicate_shares(self):
        secret = secrets.token_bytes(32)
        splitter = ShamirSplitter(threshold=3, total_shares=5)
        shares = splitter.split(secret)

        ceremony = RecoveryCeremony(node_id="BIZRA-12345678", threshold=3)
        assert ceremony.submit_share(shares[0])
        assert not ceremony.submit_share(shares[0])  # Duplicate rejected

    def test_ceremony_reconstruct_insufficient(self):
        secret = secrets.token_bytes(32)
        splitter = ShamirSplitter(threshold=3, total_shares=5)
        shares = splitter.split(secret)

        ceremony = RecoveryCeremony(node_id="BIZRA-12345678", threshold=3)
        ceremony.submit_share(shares[0])

        result = ceremony.reconstruct()
        assert not result.success
        assert "Not enough shares" in result.error


class TestGenerateRecoveryShares:
    """Test the convenience function."""

    def test_generate_and_recover(self):
        private_key = secrets.token_hex(32)
        shares = generate_recovery_shares(private_key, threshold=3, total_shares=5)

        assert len(shares) == 5
        recovered = ShamirReconstructor.reconstruct(shares[:3])
        assert recovered.hex() == private_key

    def test_default_parameters(self):
        private_key = secrets.token_hex(32)
        shares = generate_recovery_shares(private_key)
        assert len(shares) == DEFAULT_TOTAL_SHARES
        assert shares[0].threshold == DEFAULT_THRESHOLD
