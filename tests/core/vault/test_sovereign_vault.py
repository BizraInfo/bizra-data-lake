"""Tests for core.vault — SovereignVault security-critical module.

Phase 17: Security Test Scaffolding
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest

from core.vault.vault import SovereignVault, VaultEntry, derive_key, generate_salt, get_vault


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_vault_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for vault storage."""
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    return vault_dir


@pytest.fixture
def vault(tmp_vault_dir: Path) -> SovereignVault:
    """Create a SovereignVault with a test master secret."""
    return SovereignVault(vault_path=str(tmp_vault_dir), master_secret="test-secret-key-1234")


# ─────────────────────────────────────────────────────────────────────────────
# KEY DERIVATION
# ─────────────────────────────────────────────────────────────────────────────


class TestKeyDerivation:
    """Test cryptographic key derivation primitives."""

    def test_derive_key_returns_bytes(self) -> None:
        salt = generate_salt()
        key = derive_key("master-secret", salt)
        assert isinstance(key, bytes)
        assert len(key) > 0

    def test_derive_key_deterministic(self) -> None:
        """Same secret + salt must produce same key."""
        salt = generate_salt()
        k1 = derive_key("secret", salt)
        k2 = derive_key("secret", salt)
        assert k1 == k2

    def test_derive_key_different_secrets_differ(self) -> None:
        salt = generate_salt()
        k1 = derive_key("secret-a", salt)
        k2 = derive_key("secret-b", salt)
        assert k1 != k2

    def test_derive_key_different_salts_differ(self) -> None:
        s1 = generate_salt()
        s2 = generate_salt()
        k1 = derive_key("secret", s1)
        k2 = derive_key("secret", s2)
        assert k1 != k2

    def test_generate_salt_unique(self) -> None:
        """Each salt generation must be unique."""
        salts = {generate_salt() for _ in range(100)}
        assert len(salts) == 100


# ─────────────────────────────────────────────────────────────────────────────
# VAULT ENTRY
# ─────────────────────────────────────────────────────────────────────────────


class TestVaultEntry:
    """Test VaultEntry serialization round-trip."""

    def test_round_trip(self) -> None:
        now = "2025-01-01T00:00:00"
        entry = VaultEntry(
            key="my-key",
            ciphertext=b"cipher",
            salt=b"salt",
            created_at=now,
            updated_at=now,
            metadata={"role": "admin"},
        )
        d = entry.to_dict()
        restored = VaultEntry.from_dict(d)
        assert restored.key == "my-key"
        assert restored.ciphertext == b"cipher"
        assert restored.metadata["role"] == "admin"


# ─────────────────────────────────────────────────────────────────────────────
# SOVEREIGN VAULT CRUD
# ─────────────────────────────────────────────────────────────────────────────


class TestSovereignVault:
    """Test core CRUD operations of SovereignVault."""

    def test_put_and_get(self, vault: SovereignVault) -> None:
        assert vault.put("api-key", "sk-12345")
        result = vault.get("api-key")
        assert result == "sk-12345"

    def test_put_complex_value(self, vault: SovereignVault) -> None:
        data = {"tokens": [1, 2, 3], "nested": {"a": True}}
        vault.put("config", data)
        assert vault.get("config") == data

    def test_get_nonexistent_returns_none(self, vault: SovereignVault) -> None:
        assert vault.get("nonexistent") is None

    def test_delete(self, vault: SovereignVault) -> None:
        vault.put("temp", "value")
        assert vault.exists("temp")
        assert vault.delete("temp")
        assert not vault.exists("temp")

    def test_list_keys(self, vault: SovereignVault) -> None:
        vault.put("k1", "v1")
        vault.put("k2", "v2")
        keys = vault.list_keys()
        assert "k1" in keys
        assert "k2" in keys

    def test_exists(self, vault: SovereignVault) -> None:
        assert not vault.exists("ghost")
        vault.put("ghost", "boo")
        assert vault.exists("ghost")

    def test_metadata(self, vault: SovereignVault) -> None:
        vault.put("key", "val", metadata={"env": "prod"})
        meta = vault.get_metadata("key")
        assert meta is not None
        assert meta["env"] == "prod"

    def test_overwrite_value(self, vault: SovereignVault) -> None:
        vault.put("key", "v1")
        vault.put("key", "v2")
        assert vault.get("key") == "v2"

    def test_stats(self, vault: SovereignVault) -> None:
        vault.put("a", 1)
        vault.put("b", 2)
        stats = vault.get_stats()
        assert stats["entry_count"] >= 2


# ─────────────────────────────────────────────────────────────────────────────
# SECURITY PROPERTIES
# ─────────────────────────────────────────────────────────────────────────────


class TestVaultSecurity:
    """Security-focused tests for SovereignVault."""

    def test_wrong_secret_cannot_decrypt(self, tmp_vault_dir: Path) -> None:
        """Decryption with wrong master secret must fail."""
        v1 = SovereignVault(vault_path=str(tmp_vault_dir), master_secret="correct-key")
        v1.put("secret", "classified-data")

        v2 = SovereignVault(vault_path=str(tmp_vault_dir), master_secret="wrong-key")
        # Must raise ValueError — plaintext must never be returned
        with pytest.raises(ValueError, match="Decryption failed"):
            v2.get("secret")

    def test_no_secret_raises(self, tmp_vault_dir: Path) -> None:
        """Operations without master secret must raise."""
        v = SovereignVault(vault_path=str(tmp_vault_dir))
        with pytest.raises(Exception):
            v.put("key", "value")

    def test_rotate_master_secret(self, vault: SovereignVault) -> None:
        """Secret rotation must preserve all values."""
        vault.put("k1", "v1")
        vault.put("k2", "v2")
        rotated = vault.rotate_master_secret("test-secret-key-1234", "new-secret-key-5678")
        assert rotated >= 2
        # Values accessible with new secret
        vault.set_master_secret("new-secret-key-5678")
        assert vault.get("k1") == "v1"
        assert vault.get("k2") == "v2"

    def test_encrypted_on_disk(self, vault: SovereignVault, tmp_vault_dir: Path) -> None:
        """Plaintext values must NOT appear in vault files."""
        vault.put("api-key", "sk-super-secret-12345")
        # Read all files in vault dir — plaintext must not appear
        for fpath in tmp_vault_dir.rglob("*"):
            if fpath.is_file():
                content = fpath.read_bytes()
                assert b"sk-super-secret-12345" not in content


# ─────────────────────────────────────────────────────────────────────────────
# FACTORY FUNCTION
# ─────────────────────────────────────────────────────────────────────────────


class TestGetVault:
    """Test the get_vault factory function."""

    def test_get_vault_returns_instance(self, tmp_vault_dir: Path) -> None:
        v = get_vault(vault_path=str(tmp_vault_dir), master_secret="s3cret")
        assert isinstance(v, SovereignVault)
