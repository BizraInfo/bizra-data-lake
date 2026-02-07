"""
Tests for encrypted keypair storage (S-5).

Verifies:
- Keypairs stored in vault, not plaintext files
- Vault index has 0o600 permissions (owner-only)
- Plaintext migration: old file → vault, old file deleted
- Wrong secret raises clear RuntimeError
- Derived secret is deterministic (same path → same secret → reloadable)
"""

import hashlib
import json
import os
import shutil
import stat
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Skip all tests if cryptography is not available
try:
    from cryptography.fernet import Fernet

    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False

pytestmark = pytest.mark.skipif(not HAS_CRYPTO, reason="cryptography package not installed")


@pytest.fixture
def tmp_config(tmp_path):
    """Create a temporary SovereignConfig-like object."""
    keypair_path = tmp_path / ".bizra" / "keypair.json"
    keypair_path.parent.mkdir(parents=True, exist_ok=True)

    config = MagicMock()
    config.keypair_path = keypair_path
    config.network_mode = MagicMock()
    config.network_mode.value = "offline"
    config.model_store_path = tmp_path / "models"
    config.model_store_path.mkdir(parents=True, exist_ok=True)
    config.bootstrap_nodes = []
    config.default_model = None

    return config


@pytest.fixture
def runtime(tmp_config):
    """Create a SovereignRuntime with temporary config."""
    with patch("core.sovereign.integration_runtime.SovereignConfig.from_env", return_value=tmp_config):
        from core.sovereign.integration_runtime import SovereignRuntime

        rt = SovereignRuntime(config=tmp_config)
        return rt


class TestKeypairSecurity:
    """S-5: Encrypted keypair storage with vault."""

    def test_keypair_stored_in_vault(self, runtime, tmp_config):
        """Keypair must be stored in vault, not as plaintext file."""
        vault_secret = hashlib.sha256(
            f"bizra-vault-{tmp_config.keypair_path.resolve()}".encode()
        ).hexdigest()

        with patch.dict(os.environ, {"BIZRA_VAULT_SECRET": vault_secret}):
            private_key, public_key = runtime._load_or_generate_keypair()

        # Vault directory must exist
        vault_dir = tmp_config.keypair_path.parent / ".vault"
        assert vault_dir.exists(), "Vault directory should be created"

        # Vault index must exist
        vault_idx = vault_dir / "vault.idx"
        assert vault_idx.exists(), "Vault index should be created"

        # Plaintext file must NOT exist
        assert not tmp_config.keypair_path.exists(), "Plaintext keypair.json must not exist"

        # Keys must be valid
        assert len(private_key) >= 64
        assert len(public_key) >= 64

    @pytest.mark.skipif(sys.platform == "win32", reason="chmod not effective on Windows")
    def test_vault_permissions_hardened(self, runtime, tmp_config):
        """Vault index file must have 0o600 permissions."""
        vault_secret = "test-vault-secret-for-perms"

        with patch.dict(os.environ, {"BIZRA_VAULT_SECRET": vault_secret}):
            runtime._load_or_generate_keypair()

        vault_idx = tmp_config.keypair_path.parent / ".vault" / "vault.idx"
        assert vault_idx.exists()

        mode = stat.S_IMODE(os.stat(vault_idx).st_mode)
        assert mode == 0o600, f"Expected 0o600, got {oct(mode)}"

    def test_plaintext_migration(self, runtime, tmp_config):
        """Old plaintext keypair.json must be migrated into vault and deleted."""
        from core.pci import generate_keypair

        # Create old-style plaintext keypair
        private_key, public_key = generate_keypair()
        with open(tmp_config.keypair_path, "w") as f:
            json.dump({"private_key": private_key, "public_key": public_key}, f)

        assert tmp_config.keypair_path.exists(), "Pre-condition: plaintext file exists"

        vault_secret = "migration-test-secret"
        with patch.dict(os.environ, {"BIZRA_VAULT_SECRET": vault_secret}):
            loaded_private, loaded_public = runtime._load_or_generate_keypair()

        # Keys must match the originals
        assert loaded_private == private_key
        assert loaded_public == public_key

        # Plaintext file must be deleted
        assert not tmp_config.keypair_path.exists(), "Plaintext file should be deleted after migration"

        # Vault must exist
        vault_dir = tmp_config.keypair_path.parent / ".vault"
        assert vault_dir.exists()

    def test_wrong_secret_fails(self, runtime, tmp_config):
        """Wrong vault secret must raise clear RuntimeError."""
        # Generate keypair with one secret
        with patch.dict(os.environ, {"BIZRA_VAULT_SECRET": "correct-secret"}):
            runtime._load_or_generate_keypair()

        # Try to load with wrong secret
        with patch.dict(os.environ, {"BIZRA_VAULT_SECRET": "wrong-secret"}):
            with pytest.raises(RuntimeError, match="Cannot decrypt keypair vault"):
                runtime._load_or_generate_keypair()

    def test_derived_secret_deterministic(self, runtime, tmp_config):
        """Same path must derive same secret, allowing keypair reload."""
        # Clear any env override so derived secret is used
        env = {k: v for k, v in os.environ.items() if k != "BIZRA_VAULT_SECRET"}

        with patch.dict(os.environ, env, clear=True):
            pk1, pub1 = runtime._load_or_generate_keypair()

        # Load again — must get same keys
        with patch.dict(os.environ, env, clear=True):
            pk2, pub2 = runtime._load_or_generate_keypair()

        assert pk1 == pk2, "Private key must be deterministically reloadable"
        assert pub1 == pub2, "Public key must be deterministically reloadable"
