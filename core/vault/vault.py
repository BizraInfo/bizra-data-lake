"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   BIZRA SOVEREIGNTY — VAULT (Encryption at Rest)                             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║   Provides encryption for sensitive data zones.                              ║
║                                                                              ║
║   Sovereignty Pillar: DATA SOVEREIGNTY                                       ║
║   - All vault data encrypted with Fernet (AES-128-CBC + HMAC-SHA256)        ║
║   - Keys derived from master secret via PBKDF2                               ║
║   - No plaintext secrets on disk                                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import base64
import json
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

try:
    from cryptography.fernet import Fernet, InvalidToken
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    Fernet = None
    InvalidToken = Exception

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

VAULT_VERSION = "1.0.0"
DEFAULT_VAULT_DIR = Path("vault")
SALT_LENGTH = 32
ITERATIONS = 600_000  # OWASP 2023 recommendation for PBKDF2-SHA256

# ═══════════════════════════════════════════════════════════════════════════════
# KEY DERIVATION
# ═══════════════════════════════════════════════════════════════════════════════


def derive_key(master_secret: str, salt: bytes) -> bytes:
    """
    Derive encryption key from master secret using PBKDF2.

    Args:
        master_secret: The user's master password/secret
        salt: Random salt (must be stored alongside encrypted data)

    Returns:
        32-byte key suitable for Fernet
    """
    if not CRYPTO_AVAILABLE:
        raise RuntimeError("cryptography package not installed")

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=ITERATIONS,
    )
    key = base64.urlsafe_b64encode(kdf.derive(master_secret.encode()))
    return key


def generate_salt() -> bytes:
    """Generate a cryptographically secure random salt."""
    return secrets.token_bytes(SALT_LENGTH)


# ═══════════════════════════════════════════════════════════════════════════════
# VAULT ENTRY
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class VaultEntry:
    """A single encrypted entry in the vault."""

    key: str
    ciphertext: bytes
    salt: bytes
    created_at: str
    updated_at: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict:
        return {
            "key": self.key,
            "ciphertext": base64.b64encode(self.ciphertext).decode(),
            "salt": base64.b64encode(self.salt).decode(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "VaultEntry":
        return cls(
            key=d["key"],
            ciphertext=base64.b64decode(d["ciphertext"]),
            salt=base64.b64decode(d["salt"]),
            created_at=d["created_at"],
            updated_at=d["updated_at"],
            metadata=d.get("metadata", {}),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SOVEREIGN VAULT
# ═══════════════════════════════════════════════════════════════════════════════


class SovereignVault:
    """
    Encrypted key-value store for sensitive data.

    Sovereignty: All data encrypted at rest with user-controlled keys.
    """

    def __init__(self, vault_path: Union[str, Path] = None, master_secret: str = None):
        self.vault_path = Path(vault_path or DEFAULT_VAULT_DIR)
        self.vault_path.mkdir(parents=True, exist_ok=True)

        self._master_secret = master_secret
        self._entries: Dict[str, VaultEntry] = {}
        self._index_path = self.vault_path / "vault.idx"

        # Load existing entries
        self._load_index()

    def _load_index(self):
        """Load vault index (encrypted entry metadata)."""
        if self._index_path.exists():
            try:
                with open(self._index_path, "r") as f:
                    data = json.load(f)
                for entry_data in data.get("entries", []):
                    entry = VaultEntry.from_dict(entry_data)
                    self._entries[entry.key] = entry
            except Exception as e:
                print(f"⚠️ Failed to load vault index: {e}")

    def _save_index(self):
        """Save vault index."""
        data = {
            "version": VAULT_VERSION,
            "entries": [e.to_dict() for e in self._entries.values()],
        }
        with open(self._index_path, "w") as f:
            json.dump(data, f, indent=2)

    def set_master_secret(self, secret: str):
        """Set the master secret for encryption/decryption."""
        self._master_secret = secret

    def _require_secret(self):
        """Ensure master secret is set."""
        if not self._master_secret:
            raise ValueError("Master secret not set. Call set_master_secret() first.")

    def put(self, key: str, value: Any, metadata: Dict = None) -> bool:
        """
        Encrypt and store a value.

        Args:
            key: Unique identifier for this entry
            value: Any JSON-serializable value
            metadata: Optional unencrypted metadata

        Returns:
            True if successful
        """
        self._require_secret()

        if not CRYPTO_AVAILABLE:
            raise RuntimeError("cryptography package not installed")

        # Serialize value
        plaintext = json.dumps(value).encode("utf-8")

        # Generate unique salt for this entry
        salt = generate_salt()

        # Derive key and encrypt
        derived_key = derive_key(self._master_secret, salt)
        fernet = Fernet(derived_key)
        ciphertext = fernet.encrypt(plaintext)

        # Create entry
        now = datetime.now(timezone.utc).isoformat()
        existing = self._entries.get(key)

        entry = VaultEntry(
            key=key,
            ciphertext=ciphertext,
            salt=salt,
            created_at=existing.created_at if existing else now,
            updated_at=now,
            metadata=metadata or {},
        )

        self._entries[key] = entry
        self._save_index()

        return True

    def get(self, key: str) -> Optional[Any]:
        """
        Decrypt and retrieve a value.

        Args:
            key: Entry identifier

        Returns:
            Decrypted value or None if not found
        """
        self._require_secret()

        if not CRYPTO_AVAILABLE:
            raise RuntimeError("cryptography package not installed")

        entry = self._entries.get(key)
        if not entry:
            return None

        try:
            # Derive key and decrypt
            derived_key = derive_key(self._master_secret, entry.salt)
            fernet = Fernet(derived_key)
            plaintext = fernet.decrypt(entry.ciphertext)

            return json.loads(plaintext.decode("utf-8"))
        except InvalidToken:
            raise ValueError("Decryption failed - incorrect master secret")
        except Exception as e:
            raise ValueError(f"Decryption failed: {e}")

    def delete(self, key: str) -> bool:
        """Remove an entry from the vault."""
        if key in self._entries:
            del self._entries[key]
            self._save_index()
            return True
        return False

    def list_keys(self) -> list:
        """List all entry keys (does not require master secret)."""
        return list(self._entries.keys())

    def exists(self, key: str) -> bool:
        """Check if a key exists."""
        return key in self._entries

    def get_metadata(self, key: str) -> Optional[Dict]:
        """Get entry metadata (unencrypted)."""
        entry = self._entries.get(key)
        return entry.metadata if entry else None

    def rotate_master_secret(self, old_secret: str, new_secret: str) -> int:
        """
        Re-encrypt all entries with a new master secret.

        Returns:
            Number of entries rotated
        """
        if not CRYPTO_AVAILABLE:
            raise RuntimeError("cryptography package not installed")

        self._master_secret = old_secret
        count = 0

        for key in list(self._entries.keys()):
            try:
                # Decrypt with old secret
                value = self.get(key)
                metadata = self._entries[key].metadata

                # Switch to new secret
                self._master_secret = new_secret

                # Re-encrypt with new secret
                self.put(key, value, metadata)
                count += 1

                # Reset for next iteration
                self._master_secret = old_secret
            except Exception as e:
                print(f"⚠️ Failed to rotate {key}: {e}")

        # Final switch to new secret
        self._master_secret = new_secret
        return count

    def get_stats(self) -> Dict:
        """Get vault statistics."""
        return {
            "version": VAULT_VERSION,
            "entry_count": len(self._entries),
            "total_size_bytes": sum(len(e.ciphertext) for e in self._entries.values()),
            "keys": self.list_keys(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

_default_vault: Optional[SovereignVault] = None


def get_vault(vault_path: str = None, master_secret: str = None) -> SovereignVault:
    """Get or create the default vault instance."""
    global _default_vault

    if _default_vault is None:
        _default_vault = SovereignVault(vault_path, master_secret)
    elif master_secret:
        _default_vault.set_master_secret(master_secret)

    return _default_vault


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("BIZRA SOVEREIGN VAULT — Demo")
    print("=" * 70)

    if not CRYPTO_AVAILABLE:
        print("❌ cryptography package not installed")
        exit(1)

    # Create vault with test secret
    vault = SovereignVault(vault_path="./test_vault", master_secret="test_secret_123")

    # Store some data
    print("\n[1] Storing encrypted data...")
    vault.put("api_key", {"service": "openai", "key": "sk-test-12345"})
    vault.put(
        "node_identity",
        {"node_id": "node0", "private_key": "ed25519_private_key_here"},
        metadata={"type": "identity"},
    )

    print(f"  Stored {len(vault.list_keys())} entries")

    # Retrieve data
    print("\n[2] Retrieving encrypted data...")
    api_key = vault.get("api_key")
    print(f"  API Key: {api_key}")

    # List keys
    print("\n[3] Vault contents:")
    for key in vault.list_keys():
        meta = vault.get_metadata(key)
        print(f"  - {key}: {meta}")

    # Stats
    print("\n[4] Vault stats:")
    stats = vault.get_stats()
    print(f"  Entries: {stats['entry_count']}")
    print(f"  Size: {stats['total_size_bytes']} bytes")

    # Test wrong password
    print("\n[5] Testing wrong password...")
    vault.set_master_secret("wrong_password")
    try:
        vault.get("api_key")
        print("  ❌ Should have failed!")
    except ValueError as e:
        print(f"  ✅ Correctly rejected: {e}")

    # Cleanup
    import shutil

    shutil.rmtree("./test_vault", ignore_errors=True)

    print("\n" + "=" * 70)
    print("✅ Sovereign Vault Demo Complete")
    print("=" * 70)
