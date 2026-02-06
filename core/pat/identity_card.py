"""
BIZRA Identity Card — Sovereign Node Identity

Each human node in the BIZRA network receives a unique Identity Card
that cryptographically proves their membership and sovereignty.

Format: BIZRA-XXXXXXXX (8 hex characters = 4 billion unique nodes)

Standing on Giants: Ed25519 (Bernstein) + BLAKE3 (O'Connor)

Architecture:
    - Each card is signed by the SYSTEM_MINTER_KEY at creation
    - Cards can be self-signed once user generates their keypair
    - Sovereignty score tracks the user's network contribution

Security Model:
    - No hardcoded secrets - keys generated at runtime
    - Deterministic node_id from public_key hash
    - All timestamps in ISO 8601 UTC
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional, Tuple

# Import crypto primitives from PCI module
from core.pci.crypto import (
    canonical_json,
    domain_separated_digest,
    generate_keypair,
    sign_message,
    verify_signature,
)


class IdentityStatus(str, Enum):
    """Identity card status states."""

    PENDING = "pending"  # Awaiting first signature
    ACTIVE = "active"  # Fully activated
    SUSPENDED = "suspended"  # Temporarily disabled
    REVOKED = "revoked"  # Permanently invalidated


class SovereigntyTier(str, Enum):
    """Sovereignty tier based on contribution score."""

    SEED = "seed"  # 0.0 - 0.25: New node
    SPROUT = "sprout"  # 0.25 - 0.50: Growing contributor
    TREE = "tree"  # 0.50 - 0.75: Established node
    FOREST = "forest"  # 0.75 - 1.0: Network pillar


# Domain prefix for identity card signatures
IDENTITY_DOMAIN_PREFIX = "bizra-identity-v1:"


def _generate_node_id(public_key_hex: str) -> str:
    """
    Generate deterministic node ID from public key.

    Format: BIZRA-XXXXXXXX

    The node ID is derived from the first 8 hex characters of the
    BLAKE3 hash of the public key, prefixed with "BIZRA-".

    Args:
        public_key_hex: The 64-character hex public key

    Returns:
        Node ID in format BIZRA-XXXXXXXX
    """
    # Use BLAKE3 for consistent hashing with rest of system
    import blake3

    hasher = blake3.blake3()
    hasher.update(IDENTITY_DOMAIN_PREFIX.encode("utf-8"))
    hasher.update(bytes.fromhex(public_key_hex))
    digest = hasher.hexdigest()

    # Take first 8 characters (uppercase for readability)
    return f"BIZRA-{digest[:8].upper()}"


def _datetime_now_iso() -> str:
    """Get current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class IdentityCard:
    """
    BIZRA Identity Card — Sovereign Node Credential

    Represents a human node's identity in the BIZRA network.
    Cryptographically signed and verifiable.

    Attributes:
        node_id: Unique identifier (BIZRA-XXXXXXXX)
        public_key: Ed25519 public key (hex)
        creation_timestamp: ISO 8601 UTC timestamp
        sovereignty_score: Contribution score (0.0 - 1.0)
        status: Current card status
        minter_signature: Signature from system minter
        self_signature: Optional self-signature from user
        metadata: Additional card metadata
    """

    node_id: str
    public_key: str
    creation_timestamp: str
    sovereignty_score: float = 0.0
    status: IdentityStatus = IdentityStatus.PENDING
    minter_signature: Optional[str] = None
    minter_public_key: Optional[str] = None
    self_signature: Optional[str] = None
    version: str = "1.0.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate card fields after initialization."""
        # Validate node_id format
        if not self.node_id.startswith("BIZRA-") or len(self.node_id) != 14:
            raise ValueError(f"Invalid node_id format: {self.node_id}")

        # Validate public_key length (Ed25519 = 32 bytes = 64 hex chars)
        if len(self.public_key) != 64:
            raise ValueError(f"Invalid public_key length: {len(self.public_key)}")

        # Validate sovereignty_score range
        if not (0.0 <= self.sovereignty_score <= 1.0):
            raise ValueError(
                f"sovereignty_score must be 0.0-1.0: {self.sovereignty_score}"
            )

    @property
    def sovereignty_tier(self) -> SovereigntyTier:
        """Calculate sovereignty tier from score."""
        if self.sovereignty_score < 0.25:
            return SovereigntyTier.SEED
        elif self.sovereignty_score < 0.50:
            return SovereigntyTier.SPROUT
        elif self.sovereignty_score < 0.75:
            return SovereigntyTier.TREE
        else:
            return SovereigntyTier.FOREST

    def compute_digest(self) -> str:
        """
        Compute canonical digest of the identity card.

        Excludes signatures from the digest computation.
        """
        signable_data = {
            "version": self.version,
            "node_id": self.node_id,
            "public_key": self.public_key,
            "creation_timestamp": self.creation_timestamp,
            "sovereignty_score": self.sovereignty_score,
            "status": (
                self.status.value
                if isinstance(self.status, IdentityStatus)
                else self.status
            ),
            "metadata": self.metadata,
        }
        return domain_separated_digest(canonical_json(signable_data))

    def sign_as_minter(
        self, minter_private_key: str, minter_public_key: str
    ) -> "IdentityCard":
        """
        Sign the card as the system minter.

        Args:
            minter_private_key: Minter's Ed25519 private key (hex)
            minter_public_key: Minter's Ed25519 public key (hex)

        Returns:
            Self with minter_signature attached
        """
        # Set status to ACTIVE before computing digest so that both minter
        # and owner signatures are computed over the same canonical data.
        self.status = IdentityStatus.ACTIVE
        digest = self.compute_digest()
        self.minter_signature = sign_message(digest, minter_private_key)
        self.minter_public_key = minter_public_key
        return self

    def sign_as_owner(self, owner_private_key: str) -> "IdentityCard":
        """
        Self-sign the card as the owner.

        Args:
            owner_private_key: Owner's Ed25519 private key (hex)

        Returns:
            Self with self_signature attached
        """
        # Set status before computing digest so verification uses the same data
        self.status = IdentityStatus.ACTIVE
        digest = self.compute_digest()
        self.self_signature = sign_message(digest, owner_private_key)
        return self

    def verify_minter_signature(self) -> bool:
        """Verify the minter's signature on this card."""
        if not self.minter_signature or not self.minter_public_key:
            return False
        digest = self.compute_digest()
        return verify_signature(digest, self.minter_signature, self.minter_public_key)

    def verify_self_signature(self) -> bool:
        """Verify the owner's self-signature on this card."""
        if not self.self_signature:
            return False
        digest = self.compute_digest()
        return verify_signature(digest, self.self_signature, self.public_key)

    def is_fully_verified(self) -> bool:
        """Check if both signatures are valid."""
        return self.verify_minter_signature() and self.verify_self_signature()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = asdict(self)
        d["status"] = (
            self.status.value
            if isinstance(self.status, IdentityStatus)
            else self.status
        )
        d["sovereignty_tier"] = self.sovereignty_tier.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IdentityCard":
        """Reconstruct from dictionary."""
        data = data.copy()

        # Handle enum conversion
        if "status" in data:
            data["status"] = IdentityStatus(data["status"])

        # Remove computed fields
        data.pop("sovereignty_tier", None)

        return cls(**data)

    @classmethod
    def create(
        cls, public_key: str, metadata: Optional[Dict[str, Any]] = None
    ) -> "IdentityCard":
        """
        Factory method to create a new identity card.

        Args:
            public_key: User's Ed25519 public key (hex)
            metadata: Optional additional metadata

        Returns:
            New unsigned IdentityCard
        """
        node_id = _generate_node_id(public_key)
        return cls(
            node_id=node_id,
            public_key=public_key,
            creation_timestamp=_datetime_now_iso(),
            sovereignty_score=0.0,
            status=IdentityStatus.PENDING,
            metadata=metadata or {},
        )


def generate_identity_keypair() -> Tuple[str, str, str]:
    """
    Generate a new identity keypair and derived node_id.

    Returns:
        Tuple of (private_key_hex, public_key_hex, node_id)
    """
    private_key, public_key = generate_keypair()
    node_id = _generate_node_id(public_key)
    return private_key, public_key, node_id
