"""
BIZRA Capability Card - PCI-Signed Model Credentials

Every model accepted into BIZRA receives a CapabilityCard that
certifies its validated capabilities. Cards are Ed25519 signed
and include expiration dates.

"We do not assume. We verify with formal proofs."
"""

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

# Try to import cryptography for Ed25519 signatures
try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )

    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    Ed25519PrivateKey = None
    Ed25519PublicKey = None

# Import unified thresholds from authoritative source
from core.integration.constants import (
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_SNR_THRESHOLD,
)

# Use unified constants
IHSAN_THRESHOLD = UNIFIED_IHSAN_THRESHOLD
SNR_THRESHOLD = UNIFIED_SNR_THRESHOLD
CARD_VALIDITY_DAYS = 90


class ModelTier(Enum):
    """Model capability tiers."""

    EDGE = "EDGE"  # 0.5B-1.5B, CPU-capable
    LOCAL = "LOCAL"  # 7B-13B, GPU-recommended
    POOL = "POOL"  # 70B+, federation-capable


class TaskType(Enum):
    """Supported task types."""

    REASONING = "reasoning"
    CHAT = "chat"
    SUMMARIZATION = "summarization"
    CODE_GENERATION = "code_generation"
    TRANSLATION = "translation"
    CLASSIFICATION = "classification"
    EMBEDDING = "embedding"


@dataclass
class ModelCapabilities:
    """Validated capabilities from challenge."""

    max_context: int
    ihsan_score: float
    snr_score: float
    latency_ms: int
    tasks_supported: List[TaskType]


@dataclass
class CapabilityCard:
    """
    CapabilityCard - A signed credential for validated models.

    Every model that passes the Constitution Challenge receives a
    CapabilityCard certifying its capabilities.
    """

    model_id: str
    model_name: str
    parameter_count: Optional[int]
    quantization: str  # f16, q8, q4, q2, unknown
    tier: ModelTier
    capabilities: ModelCapabilities
    signature: str = ""
    issuer_public_key: str = ""
    issued_at: str = ""
    expires_at: str = ""
    revoked: bool = False

    def __post_init__(self):
        """Set timestamps if not provided."""
        if not self.issued_at:
            now = datetime.utcnow()
            self.issued_at = now.isoformat() + "Z"
        if not self.expires_at:
            now = datetime.fromisoformat(self.issued_at.rstrip("Z"))
            expires = now + timedelta(days=CARD_VALIDITY_DAYS)
            self.expires_at = expires.isoformat() + "Z"

    def canonical_bytes(self) -> bytes:
        """Get canonical bytes for signing."""
        data = "|".join(
            [
                self.model_id,
                self.tier.value,
                str(self.capabilities.ihsan_score),
                str(self.capabilities.snr_score),
                self.issued_at,
                self.expires_at,
            ]
        )
        return data.encode("utf-8")

    def is_valid(self) -> tuple[bool, Optional[str]]:
        """
        Check if the card is currently valid.

        Returns:
            Tuple of (is_valid, reason_if_invalid)
        """
        # Check revocation
        if self.revoked:
            return False, "Card has been revoked"

        # Check expiration
        now = datetime.utcnow()
        try:
            expires = datetime.fromisoformat(self.expires_at.rstrip("Z"))
            if now >= expires:
                return False, "Card has expired"
        except ValueError:
            return False, "Invalid expiration date format"

        # Check scores still meet thresholds
        if self.capabilities.ihsan_score < IHSAN_THRESHOLD:
            return False, f"Ihsān score {self.capabilities.ihsan_score} below threshold"

        if self.capabilities.snr_score < SNR_THRESHOLD:
            return False, f"SNR score {self.capabilities.snr_score} below threshold"

        return True, None

    def remaining_days(self) -> int:
        """Get remaining validity days."""
        now = datetime.utcnow()
        try:
            expires = datetime.fromisoformat(self.expires_at.rstrip("Z"))
            diff = expires - now
            return max(0, diff.days)
        except ValueError:
            return 0

    def fingerprint(self) -> str:
        """Calculate card fingerprint for identification."""
        h = hashlib.sha256()
        h.update(self.canonical_bytes())
        h.update(self.signature.encode("utf-8"))
        return h.hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "parameter_count": self.parameter_count,
            "quantization": self.quantization,
            "tier": self.tier.value,
            "capabilities": {
                "max_context": self.capabilities.max_context,
                "ihsan_score": self.capabilities.ihsan_score,
                "snr_score": self.capabilities.snr_score,
                "latency_ms": self.capabilities.latency_ms,
                "tasks_supported": [t.value for t in self.capabilities.tasks_supported],
            },
            "signature": self.signature,
            "issuer_public_key": self.issuer_public_key,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "revoked": self.revoked,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CapabilityCard":
        """Create from dictionary."""
        caps = data["capabilities"]
        capabilities = ModelCapabilities(
            max_context=caps["max_context"],
            ihsan_score=caps["ihsan_score"],
            snr_score=caps["snr_score"],
            latency_ms=caps["latency_ms"],
            tasks_supported=[TaskType(t) for t in caps["tasks_supported"]],
        )
        return cls(
            model_id=data["model_id"],
            model_name=data["model_name"],
            parameter_count=data.get("parameter_count"),
            quantization=data.get("quantization", "unknown"),
            tier=ModelTier(data["tier"]),
            capabilities=capabilities,
            signature=data.get("signature", ""),
            issuer_public_key=data.get("issuer_public_key", ""),
            issued_at=data.get("issued_at", ""),
            expires_at=data.get("expires_at", ""),
            revoked=data.get("revoked", False),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "CapabilityCard":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


def create_capability_card(
    model_id: str,
    tier: ModelTier,
    ihsan_score: float,
    snr_score: float,
    tasks_supported: List[TaskType],
    model_name: Optional[str] = None,
    parameter_count: Optional[int] = None,
    quantization: str = "unknown",
    max_context: int = 2048,
    latency_ms: int = 0,
) -> CapabilityCard:
    """
    Create a new CapabilityCard.

    Validates that scores meet constitutional thresholds.

    Args:
        model_id: Unique model identifier
        tier: Model capability tier
        ihsan_score: Validated Ihsān score (must be >= 0.95)
        snr_score: Validated SNR score (must be >= 0.85)
        tasks_supported: List of supported task types
        model_name: Human-readable name (defaults to model_id)
        parameter_count: Approximate parameter count
        quantization: Quantization level (f16, q8, q4, q2, unknown)
        max_context: Maximum context length
        latency_ms: Measured latency

    Returns:
        Unsigned CapabilityCard

    Raises:
        ValueError: If scores don't meet thresholds
    """
    if ihsan_score < IHSAN_THRESHOLD:
        raise ValueError(f"Ihsān score {ihsan_score} < threshold {IHSAN_THRESHOLD}")
    if snr_score < SNR_THRESHOLD:
        raise ValueError(f"SNR score {snr_score} < threshold {SNR_THRESHOLD}")

    capabilities = ModelCapabilities(
        max_context=max_context,
        ihsan_score=ihsan_score,
        snr_score=snr_score,
        latency_ms=latency_ms,
        tasks_supported=tasks_supported,
    )

    return CapabilityCard(
        model_id=model_id,
        model_name=model_name or model_id,
        parameter_count=parameter_count,
        quantization=quantization,
        tier=tier,
        capabilities=capabilities,
    )


class CardIssuer:
    """
    Card issuer for signing CapabilityCards.

    Uses Ed25519 signatures when cryptography is available,
    otherwise provides a simulation mode for development.
    """

    def __init__(self, private_key_bytes: Optional[bytes] = None):
        """
        Initialize the card issuer.

        Args:
            private_key_bytes: Ed25519 private key bytes (32 bytes).
                             If None, generates a new keypair.
        """
        self._private_key = None
        self._public_key = None
        self._simulation_mode = not CRYPTO_AVAILABLE

        if CRYPTO_AVAILABLE:
            if private_key_bytes:
                self._private_key = Ed25519PrivateKey.from_private_bytes(
                    private_key_bytes
                )
            else:
                self._private_key = Ed25519PrivateKey.generate()
            self._public_key = self._private_key.public_key()
        else:
            # Simulation mode - use hash-based pseudo-signatures
            self._sim_secret = (
                private_key_bytes or hashlib.sha256(str(time.time()).encode()).digest()
            )

    def public_key_hex(self) -> str:
        """Get the issuer's public key as hex string."""
        if CRYPTO_AVAILABLE and self._public_key:
            pk_bytes = self._public_key.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
            return pk_bytes.hex()
        else:
            # Simulation mode
            return hashlib.sha256(self._sim_secret).hexdigest()[:64]

    def issue(self, card: CapabilityCard) -> CapabilityCard:
        """
        Issue (sign) a CapabilityCard.

        Args:
            card: The card to sign

        Returns:
            Signed CapabilityCard
        """
        canonical = card.canonical_bytes()

        if CRYPTO_AVAILABLE and self._private_key:
            signature = self._private_key.sign(canonical)
            card.signature = signature.hex()
        else:
            # Simulation mode - HMAC-like signature
            h = hashlib.sha256()
            h.update(self._sim_secret)
            h.update(canonical)
            card.signature = "sim:" + h.hexdigest()

        card.issuer_public_key = self.public_key_hex()
        return card

    def verify(self, card: CapabilityCard) -> bool:
        """
        Verify a CapabilityCard signature.

        Args:
            card: The card to verify

        Returns:
            True if signature is valid
        """
        if not card.signature or not card.issuer_public_key:
            return False

        canonical = card.canonical_bytes()

        if card.signature.startswith("sim:"):
            # SECURITY: Simulation mode ONLY allowed in development/testing
            # Production environments MUST set BIZRA_ENV=production to disable
            import os

            if os.environ.get("BIZRA_ENV", "development").lower() == "production":
                # Reject simulation signatures in production
                return False

            # Development mode - allow simulation verification
            expected_sig = card.signature[4:]
            h = hashlib.sha256()
            h.update(self._sim_secret)
            h.update(canonical)
            return h.hexdigest() == expected_sig

        if CRYPTO_AVAILABLE:
            try:
                pk_bytes = bytes.fromhex(card.issuer_public_key)
                public_key = Ed25519PublicKey.from_public_bytes(pk_bytes)
                signature = bytes.fromhex(card.signature)
                public_key.verify(signature, canonical)
                return True
            except Exception:
                return False

        return False


def verify_capability_card(card: CapabilityCard) -> Dict[str, Any]:
    """
    Verify a CapabilityCard (standalone function).

    Args:
        card: The card to verify

    Returns:
        Verification result dictionary
    """
    is_valid, reason = card.is_valid()

    # Signature verification using trusted key registry
    from core.sovereign.key_registry import get_key_registry

    signature_valid = False
    if card.signature and card.issuer_public_key:
        registry = get_key_registry()
        # Check if issuer's key is in trusted registry
        if registry.is_trusted(card.issuer_public_key):
            # Key is trusted - verify signature with issuer public key
            if card.signature.startswith("sim:"):
                # Allow simulation signatures only outside production
                import os

                if os.environ.get("BIZRA_ENV", "development").lower() != "production":
                    signature_valid = True
            else:
                try:
                    if CRYPTO_AVAILABLE:
                        pk_bytes = bytes.fromhex(card.issuer_public_key)
                        public_key = Ed25519PublicKey.from_public_bytes(pk_bytes)
                        signature = bytes.fromhex(card.signature)
                        public_key.verify(signature, card.canonical_bytes())
                        signature_valid = True
                except Exception:
                    signature_valid = False
        else:
            # Key not in registry - signature not trusted
            signature_valid = False

    ihsan_valid = card.capabilities.ihsan_score >= IHSAN_THRESHOLD
    snr_valid = card.capabilities.snr_score >= SNR_THRESHOLD

    return {
        "is_valid": is_valid and signature_valid,
        "signature_valid": signature_valid,
        "is_expired": "expired" in (reason or "").lower(),
        "is_revoked": card.revoked,
        "ihsan_valid": ihsan_valid,
        "snr_valid": snr_valid,
        "model_id": card.model_id,
        "tier": card.tier.value,
        "reason": reason,
    }
