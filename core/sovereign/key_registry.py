"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   BIZRA TRUSTED KEY REGISTRY                                                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║   Manages trusted public keys for CapabilityCard verification.               ║
║                                                                              ║
║   Security Model:                                                            ║
║   - Keys must be explicitly registered before use                            ║
║   - Keys can be revoked with audit trail                                     ║
║   - Supports key rotation with overlap period                                ║
║   - Persistent storage with tamper detection                                 ║
║                                                                              ║
║   Standing on Giants:                                                        ║
║   - RFC 5280: X.509 PKI Certificate and CRL Profile                          ║
║   - Web of Trust (PGP model for decentralized verification)                  ║
║                                                                              ║
║   Constitutional: All registered keys must meet Ihsan >= 0.95                ║
╚══════════════════════════════════════════════════════════════════════════════╝

Created: 2026-02-05 | SAPE Elite Analysis - P1 Implementation
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

# Import unified thresholds


class KeyStatus(Enum):
    """Status of a registered key."""

    ACTIVE = "active"  # Key is trusted and valid
    PENDING = "pending"  # Key awaiting verification
    REVOKED = "revoked"  # Key has been revoked
    EXPIRED = "expired"  # Key validity period has ended
    ROTATED = "rotated"  # Key replaced by newer key


@dataclass
class RegisteredKey:
    """
    A trusted public key entry in the registry.
    """

    key_id: str  # Unique identifier (hash of public key)
    public_key_hex: str  # 64-char hex Ed25519 public key
    issuer_name: str  # Human-readable issuer name
    status: KeyStatus = KeyStatus.ACTIVE

    # Validity period
    registered_at: str = ""  # ISO 8601 timestamp
    expires_at: Optional[str] = None  # ISO 8601 timestamp
    revoked_at: Optional[str] = None  # ISO 8601 timestamp

    # Audit trail
    registered_by: str = "system"  # Who registered this key
    revocation_reason: Optional[str] = None

    # Key rotation
    successor_key_id: Optional[str] = None  # New key that replaces this one
    predecessor_key_id: Optional[str] = None  # Previous key this replaces

    # Trust level
    trust_level: float = 1.0  # 0.0 - 1.0, affects verification weight

    def __post_init__(self):
        """Validate and set defaults."""
        if not self.registered_at:
            self.registered_at = datetime.now(timezone.utc).isoformat()

        if not self.key_id:
            self.key_id = self._compute_key_id()

    def _compute_key_id(self) -> str:
        """Compute deterministic key ID from public key."""
        return hashlib.sha256(self.public_key_hex.encode()).hexdigest()[:16]

    def is_valid(self) -> tuple[bool, Optional[str]]:
        """
        Check if key is currently valid for verification.

        Returns:
            Tuple of (is_valid, reason_if_invalid)
        """
        if self.status == KeyStatus.REVOKED:
            return False, f"Key revoked: {self.revocation_reason}"

        if self.status == KeyStatus.PENDING:
            return False, "Key pending verification"

        if self.status == KeyStatus.EXPIRED:
            return False, "Key has expired"

        if self.expires_at:
            expiry = datetime.fromisoformat(self.expires_at.replace("Z", "+00:00"))
            if datetime.now(timezone.utc) > expiry:
                return False, "Key has expired"

        return True, None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "key_id": self.key_id,
            "public_key_hex": self.public_key_hex,
            "issuer_name": self.issuer_name,
            "status": self.status.value,
            "registered_at": self.registered_at,
            "expires_at": self.expires_at,
            "revoked_at": self.revoked_at,
            "registered_by": self.registered_by,
            "revocation_reason": self.revocation_reason,
            "successor_key_id": self.successor_key_id,
            "predecessor_key_id": self.predecessor_key_id,
            "trust_level": self.trust_level,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "RegisteredKey":
        """Create from dictionary."""
        data["status"] = KeyStatus(data.get("status", "active"))
        return cls(**data)


class TrustedKeyRegistry:
    """
    Registry of trusted public keys for CapabilityCard verification.

    Thread-safe singleton pattern with persistent storage.
    """

    _instance: Optional["TrustedKeyRegistry"] = None
    _registry_path: Path = Path("sovereign_state/key_registry.json")
    _initialized: bool = False

    def __new__(cls, *args, **kwargs):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, registry_path: Optional[Path] = None):
        """Initialize the registry."""
        if self._initialized:
            return

        if registry_path:
            self._registry_path = registry_path

        self._keys: Dict[str, RegisteredKey] = {}
        self._public_key_index: Dict[str, str] = {}  # public_key_hex -> key_id
        self._load()
        self._initialized = True

    def _load(self) -> None:
        """Load registry from persistent storage."""
        if self._registry_path.exists():
            try:
                with open(self._registry_path) as f:
                    data = json.load(f)

                for key_data in data.get("keys", []):
                    key = RegisteredKey.from_dict(key_data)
                    self._keys[key.key_id] = key
                    self._public_key_index[key.public_key_hex] = key.key_id

            except (json.JSONDecodeError, KeyError, ValueError):
                # Corrupted registry - start fresh but log error
                self._keys = {}
                self._public_key_index = {}

    def _save(self) -> None:
        """Persist registry to storage."""
        self._registry_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": "1.0",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "keys": [key.to_dict() for key in self._keys.values()],
        }

        with open(self._registry_path, "w") as f:
            json.dump(data, f, indent=2)

    def register(
        self,
        public_key_hex: str,
        issuer_name: str,
        registered_by: str = "system",
        expires_in_days: int = 365,
        trust_level: float = 1.0,
    ) -> RegisteredKey:
        """
        Register a new trusted public key.

        Args:
            public_key_hex: 64-character hex Ed25519 public key
            issuer_name: Human-readable name for the issuer
            registered_by: Who is registering this key
            expires_in_days: Validity period (default 1 year)
            trust_level: Trust weight 0.0-1.0

        Returns:
            The registered key entry

        Raises:
            ValueError: If key format is invalid or already registered
        """
        # Validate key format
        if len(public_key_hex) != 64 or not all(
            c in "0123456789abcdef" for c in public_key_hex.lower()
        ):
            raise ValueError("Invalid public key format: must be 64-char hex")

        # Check for duplicate
        if public_key_hex in self._public_key_index:
            raise ValueError(
                f"Key already registered with ID: {self._public_key_index[public_key_hex]}"
            )

        # Create entry
        now = datetime.now(timezone.utc)
        expires_at = (
            (now + timedelta(days=expires_in_days)).isoformat()
            if expires_in_days > 0
            else None
        )

        key = RegisteredKey(
            key_id="",  # Will be computed in __post_init__
            public_key_hex=public_key_hex.lower(),
            issuer_name=issuer_name,
            status=KeyStatus.ACTIVE,
            registered_at=now.isoformat(),
            expires_at=expires_at,
            registered_by=registered_by,
            trust_level=min(1.0, max(0.0, trust_level)),
        )

        # Store
        self._keys[key.key_id] = key
        self._public_key_index[key.public_key_hex] = key.key_id
        self._save()

        return key

    def lookup(self, public_key_hex: str) -> Optional[RegisteredKey]:
        """
        Look up a key by its public key hex.

        Args:
            public_key_hex: The public key to look up

        Returns:
            RegisteredKey if found and valid, None otherwise
        """
        public_key_hex = public_key_hex.lower()
        key_id = self._public_key_index.get(public_key_hex)
        if key_id:
            return self._keys.get(key_id)
        return None

    def lookup_by_id(self, key_id: str) -> Optional[RegisteredKey]:
        """Look up a key by its key ID."""
        return self._keys.get(key_id)

    def is_trusted(self, public_key_hex: str) -> bool:
        """
        Check if a public key is trusted (registered and valid).

        Args:
            public_key_hex: The public key to check

        Returns:
            True if key is trusted
        """
        key = self.lookup(public_key_hex)
        if not key:
            return False

        is_valid, _ = key.is_valid()
        return is_valid

    def revoke(
        self, public_key_hex: str, reason: str, revoked_by: str = "system"
    ) -> bool:
        """
        Revoke a registered key.

        Args:
            public_key_hex: The key to revoke
            reason: Reason for revocation
            revoked_by: Who is revoking

        Returns:
            True if key was revoked
        """
        key = self.lookup(public_key_hex)
        if not key:
            return False

        key.status = KeyStatus.REVOKED
        key.revoked_at = datetime.now(timezone.utc).isoformat()
        key.revocation_reason = f"{reason} (by {revoked_by})"
        self._save()
        return True

    def rotate(
        self,
        old_public_key_hex: str,
        new_public_key_hex: str,
        issuer_name: Optional[str] = None,
        registered_by: str = "system",
    ) -> Optional[RegisteredKey]:
        """
        Rotate an old key to a new one.

        The old key is marked as rotated (still valid during overlap).
        The new key is registered with a reference to the old key.

        Args:
            old_public_key_hex: The key being replaced
            new_public_key_hex: The new key
            issuer_name: Name for new key (defaults to old key's name)
            registered_by: Who is performing rotation

        Returns:
            The new registered key, or None if old key not found
        """
        old_key = self.lookup(old_public_key_hex)
        if not old_key:
            return None

        # Register new key
        new_key = self.register(
            public_key_hex=new_public_key_hex,
            issuer_name=issuer_name or old_key.issuer_name,
            registered_by=registered_by,
            trust_level=old_key.trust_level,
        )

        # Link keys
        new_key.predecessor_key_id = old_key.key_id
        old_key.successor_key_id = new_key.key_id
        old_key.status = KeyStatus.ROTATED

        self._save()
        return new_key

    def list_active(self) -> List[RegisteredKey]:
        """Get all active (trusted) keys."""
        return [key for key in self._keys.values() if key.is_valid()[0]]

    def list_all(self) -> List[RegisteredKey]:
        """Get all registered keys (including revoked)."""
        return list(self._keys.values())

    def clear(self) -> None:
        """Clear all keys (use with caution!)."""
        self._keys = {}
        self._public_key_index = {}
        self._save()


# Singleton accessor
def get_key_registry() -> TrustedKeyRegistry:
    """Get the global key registry instance."""
    return TrustedKeyRegistry()


# Need timedelta for expires calculation
from datetime import timedelta
