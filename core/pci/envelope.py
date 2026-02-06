"""
BIZRA PROOF-CARRYING INFERENCE (PCI) PROTOCOL
Envelope Definitions

Standing on Giants:
- Lamport (1982): Timestamp ordering in distributed systems
- Merkle (1988): Cryptographic hash integrity
- Bernstein (2011): Ed25519 signatures
"""

import secrets
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any,
    Dict,
    Final,
    List,
    Optional,
    Set,
    Tuple,
    TypedDict,
)

from .crypto import (
    canonical_json,
    domain_separated_digest,
    sign_message,
    timing_safe_compare,
)

# =============================================================================
# REPLAY PROTECTION CONSTANTS (Security Hardening S-1)
# =============================================================================

# Maximum age of a message before it's considered expired (5 minutes)
MAX_MESSAGE_AGE_SECONDS: Final[int] = 300

# Maximum future timestamp tolerance (prevents time-travel attacks)
MAX_FUTURE_TIMESTAMP_SECONDS: Final[int] = 30

# Nonce tracking for replay prevention
_seen_nonces: Set[str] = set()
_nonce_max_size: Final[int] = 100000


# =============================================================================
# TYPE DEFINITIONS (mypy --strict compliance)
# =============================================================================


class SenderDict(TypedDict, total=False):
    """Type definition for serialized EnvelopeSender."""

    agent_type: str
    agent_id: str
    public_key: str


class PayloadDict(TypedDict, total=False):
    """Type definition for serialized EnvelopePayload."""

    action: str
    data: Dict[str, Any]
    policy_hash: str
    state_hash: str


class MetadataDict(TypedDict, total=False):
    """Type definition for serialized EnvelopeMetadata."""

    ihsan_score: float
    snr_score: float
    urgency: str


class SignatureDict(TypedDict, total=False):
    """Type definition for serialized EnvelopeSignature."""

    algorithm: str
    value: str
    signed_fields: List[str]


class EnvelopeDict(TypedDict, total=False):
    """Type definition for serialized PCIEnvelope."""

    version: str
    envelope_id: str
    timestamp: str
    nonce: str
    sender: SenderDict
    payload: PayloadDict
    metadata: MetadataDict
    signature: Optional[SignatureDict]


# Type alias for freshness validation result
FreshnessResult = Tuple[bool, str]


def _timing_safe_nonce_lookup(nonce: str, seen_nonces: Set[str]) -> bool:
    """
    Timing-safe lookup of a nonce in the seen set.

    SECURITY: Standard `in` operator returns early on match, which
    creates timing variations that could leak information about stored
    nonces. This function iterates through ALL nonces to ensure
    constant-time behavior regardless of whether the nonce exists.

    Standing on Giants: Kocher (1996) - Timing Attacks on Implementations

    Note: For very large nonce sets, this has O(n) performance.
    The _nonce_max_size limit (100,000) bounds the worst case.

    Args:
        nonce: The nonce to look up
        seen_nonces: Set of previously seen nonces

    Returns:
        True if nonce was found, False otherwise (constant time for set size)
    """
    found: bool = False
    # Iterate through ALL nonces to ensure constant time
    for seen in seen_nonces:
        # Use timing-safe comparison for each check
        if timing_safe_compare(nonce, seen):
            found = True
            # Continue iterating to maintain constant time
    return found


class AgentType(str, Enum):
    PAT = "PAT"
    SAT = "SAT"


@dataclass
class EnvelopeSender:
    agent_type: AgentType
    agent_id: str
    public_key: str


@dataclass
class EnvelopePayload:
    action: str
    data: Dict[str, Any]
    policy_hash: str
    state_hash: str


@dataclass
class EnvelopeMetadata:
    ihsan_score: float
    snr_score: float
    urgency: str = "REAL_TIME"

    def __post_init__(self) -> None:
        if not (0.0 <= self.ihsan_score <= 1.0):
            raise ValueError("Ihsan score must be between 0.0 and 1.0")


@dataclass
class EnvelopeSignature:
    algorithm: str
    value: str
    signed_fields: List[str]


@dataclass
class PCIEnvelope:
    version: str
    envelope_id: str
    timestamp: str
    nonce: str
    sender: EnvelopeSender
    payload: EnvelopePayload
    metadata: EnvelopeMetadata
    signature: Optional[EnvelopeSignature] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def compute_digest(self) -> str:
        """Compute the canonical digest of the envelope (excluding signature)."""
        # Create a dict without signature
        d = asdict(self)
        if "signature" in d:
            del d["signature"]

        # Canonicalize and hash
        return domain_separated_digest(canonical_json(d))

    # =========================================================================
    # REPLAY PROTECTION (Security Hardening S-1)
    # Standing on Giants: Lamport (1982) - "Time, Clocks, and Ordering"
    # =========================================================================

    def is_expired(self) -> bool:
        """
        Check if message timestamp is outside acceptable window.

        SECURITY: Prevents replay attacks by rejecting old messages
        and time-travel attacks by rejecting future-dated messages.

        Returns:
            True if message should be rejected (expired or future-dated)
        """
        try:
            msg_time = datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            age_seconds = (now - msg_time).total_seconds()

            # Reject messages too old
            if age_seconds > MAX_MESSAGE_AGE_SECONDS:
                return True

            # Reject messages too far in future (time-travel attack)
            if age_seconds < -MAX_FUTURE_TIMESTAMP_SECONDS:
                return True

            return False
        except (ValueError, AttributeError):
            # Invalid timestamp format - reject
            return True

    def is_replay(self) -> bool:
        """
        Check if this nonce has been seen before.

        SECURITY: Prevents replay attacks by tracking seen nonces.
        Uses timing-safe comparison to prevent timing attacks that
        could leak information about stored nonces.

        Standing on Giants: Kocher (1996) - Timing Attacks

        Returns:
            True if this is a replayed message
        """
        global _seen_nonces

        # SECURITY: Use timing-safe comparison to prevent timing attacks
        # that could reveal information about stored nonces
        is_seen = _timing_safe_nonce_lookup(self.nonce, _seen_nonces)

        if is_seen:
            return True

        # Add to seen nonces
        _seen_nonces.add(self.nonce)

        # Evict oldest if over limit (simple size-based eviction)
        if len(_seen_nonces) > _nonce_max_size:
            # Remove ~10% of oldest entries
            to_remove = list(_seen_nonces)[: _nonce_max_size // 10]
            for nonce in to_remove:
                _seen_nonces.discard(nonce)

        return False

    def validate_freshness(self) -> FreshnessResult:
        """
        Validate message freshness (not expired, not replayed).

        Returns:
            FreshnessResult tuple of (valid: bool, error_message: str)
        """
        if self.is_expired():
            return False, f"Message expired or future-dated: {self.timestamp}"

        if self.is_replay():
            return False, f"Replay attack detected: nonce {self.nonce[:16]}..."

        return True, ""

    def sign(self, private_key_hex: str) -> "PCIEnvelope":
        """Sign the envelope and attach signature."""
        digest = self.compute_digest()
        sig_hex = sign_message(digest, private_key_hex)

        self.signature = EnvelopeSignature(
            algorithm="ed25519",
            value=sig_hex,
            signed_fields=[
                "version",
                "envelope_id",
                "timestamp",
                "nonce",
                "sender",
                "payload",
                "metadata",
            ],
        )
        return self

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PCIEnvelope":
        """
        Reconstruct a PCIEnvelope from a dictionary.

        Standing on Giants — Lamport:
        - Enables verification of received envelopes
        - Preserves signature chain for BFT validation
        """
        # Reconstruct nested dataclasses
        sender_data = d.get("sender", {})
        sender = EnvelopeSender(
            agent_type=AgentType(sender_data.get("agent_type", "PAT")),
            agent_id=sender_data.get("agent_id", ""),
            public_key=sender_data.get("public_key", ""),
        )

        payload_data = d.get("payload", {})
        payload = EnvelopePayload(
            action=payload_data.get("action", ""),
            data=payload_data.get("data", {}),
            policy_hash=payload_data.get("policy_hash", ""),
            state_hash=payload_data.get("state_hash", ""),
        )

        metadata_data = d.get("metadata", {})
        metadata = EnvelopeMetadata(
            ihsan_score=metadata_data.get("ihsan_score", 0.0),
            snr_score=metadata_data.get("snr_score", 0.0),
            urgency=metadata_data.get("urgency", "REAL_TIME"),
        )

        # Reconstruct signature if present
        sig_data = d.get("signature")
        signature = None
        if sig_data:
            signature = EnvelopeSignature(
                algorithm=sig_data.get("algorithm", "ed25519"),
                value=sig_data.get("value", ""),
                signed_fields=sig_data.get("signed_fields", []),
            )

        return cls(
            version=d.get("version", "1.0.0"),
            envelope_id=d.get("envelope_id", ""),
            timestamp=d.get("timestamp", ""),
            nonce=d.get("nonce", ""),
            sender=sender,
            payload=payload,
            metadata=metadata,
            signature=signature,
        )


class EnvelopeBuilder:
    """Builder for PCI Envelopes."""

    def __init__(self) -> None:
        self._sender: Optional[EnvelopeSender] = None
        self._payload: Optional[EnvelopePayload] = None
        self._metadata: Optional[EnvelopeMetadata] = None

    def with_sender(
        self, agent_type: str, agent_id: str, public_key: str
    ) -> "EnvelopeBuilder":
        self._sender = EnvelopeSender(AgentType(agent_type), agent_id, public_key)
        return self

    def with_payload(
        self, action: str, data: Dict[str, Any], policy_hash: str, state_hash: str
    ) -> "EnvelopeBuilder":
        self._payload = EnvelopePayload(action, data, policy_hash, state_hash)
        return self

    def with_metadata(
        self, ihsan: float, snr: float, urgency: str = "REAL_TIME"
    ) -> "EnvelopeBuilder":
        self._metadata = EnvelopeMetadata(ihsan, snr, urgency)
        return self

    def build(self) -> PCIEnvelope:
        if self._sender is None or self._payload is None or self._metadata is None:
            raise ValueError("Sender, Payload, and Metadata are required")

        return PCIEnvelope(
            version="1.0.0",
            envelope_id=str(uuid.uuid4()),
            timestamp=datetime_now_iso(),
            nonce=secrets.token_hex(32),
            sender=self._sender,
            payload=self._payload,
            metadata=self._metadata,
        )


def datetime_now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
